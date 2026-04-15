"""Tab 1 — Upload images into a named session folder under /data/uploads/.

Workflow (per-page):
  1. User picks / types a session name (new or existing).
  2. User drops images, a ZIP, or a folder. Files land in a staging
     directory ``.staging_<sanitized-session-name>`` — deterministic
     from the session name so a browser reload + retype of the same
     name resumes the same dir. Files whose name is already committed
     to the session (or already in the staging dir) are skipped
     client-side before tus.Upload even starts.
  3. User clicks **Save** — the staging directory is renamed to / merged
     into the chosen session, thumbnails move alongside, and the gallery
     renders.

Dot-prefixed staging directories are filtered out of ``list_sessions()``
so they never leak into pickers. If the user abandons a staging dir it
lingers until the session name is reused or it's manually cleaned up
(course tool, manual cleanup is fine).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
import time
import zipfile
from pathlib import Path

from nicegui import app as nicegui_app
from nicegui import ui

from ..config import THUMBS_DIR, UPLOADS_DIR
from ..sessions import list_sessions
from ..thumbs import make_thumbnail, thumb_path_for
from ..tus import register_tus_routes

_SAFE = re.compile(r"[^a-zA-Z0-9_-]+")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
# Staging dirs are named ``.staging_<sanitized-session-name>``. Deterministic
# by session name so a browser reload can resume into the same dir by just
# typing the same session name again — no UUID lookup needed. The character
# class matches ``_sanitize`` output (alnum + _ + -).
_STAGING_RE = re.compile(r"^\.staging_[A-Za-z0-9_-]+$")

# Per-staging-id SHA-256 → on-disk path. SHA-256 (not MD5) so the same
# hash serves both purposes:
#   1. Dedup — skip incoming files whose hash already exists in staging.
#   2. Ground-truth lookup — Lissl's CSV checksums are SHA-256, so we can
#      tag images by content hash inline at save time.
_STAGING_HASHES: dict[str, dict[str, Path]] = {}
_STAGING_HASHES_LOCK = threading.Lock()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hashes_for(staging_id: str, target_dir: Path) -> dict[str, Path]:
    """Return the cached {sha256: path} for a staging dir, populating it
    on first use by walking the directory. Reused across upload paths so
    a duplicate dropped after a partial folder upload is also caught."""
    with _STAGING_HASHES_LOCK:
        cache = _STAGING_HASHES.get(staging_id)
        if cache is not None:
            return cache
        cache = {}
        if target_dir.exists():
            for p in target_dir.iterdir():
                if p.is_file() and _is_image_name(p.name):
                    try:
                        cache[_sha256_file(p)] = p
                    except OSError:
                        continue
        _STAGING_HASHES[staging_id] = cache
        return cache


def _drop_staging_hashes(staging_id: str) -> None:
    """Drop the hash cache for a staging id (after commit / regenerate)."""
    with _STAGING_HASHES_LOCK:
        _STAGING_HASHES.pop(staging_id, None)


# Per-staging mapping of basename → Lissl ground-truth label. Built up as
# images are saved (we already have the SHA-256, one extra dict lookup).
# Persisted to the session's ``lissl_labels.json`` at commit time.
_STAGING_GT: dict[str, dict[str, str]] = {}


def _record_lissl(staging_id: str, basename: str, sha256_hex: str) -> str | None:
    """If the SHA-256 matches a Lissl row, record the label and return it."""
    from ..lissl_groundtruth import lookup as _lissl_lookup

    label = _lissl_lookup(sha256_hex)
    if label:
        _STAGING_GT.setdefault(staging_id, {})[basename] = label
    return label


def _sanitize(name: str) -> str:
    name = _SAFE.sub("_", name).strip("_")
    return name or f"session_{int(time.time())}"


def _is_image_name(name: str) -> bool:
    """True if ``name`` looks like a browser-/Pillow-decodable image file.

    Rejects macOS AppleDouble sidecars (``._foo.JPG``) and dotfiles like
    ``.DS_Store``. These carry image extensions but aren't decodable, so
    letting them through breaks thumbnailing and MD inference downstream.
    """
    base = Path(name).name
    if not base or base.startswith("."):
        return False
    return Path(base).suffix.lower() in IMAGE_EXTS


def _unique_path(d: Path, name: str) -> Path:
    """Return ``d/name`` if free, else ``d/<stem>_1.ext``, ``_2``, …

    Folder uploads flatten nested trees — two files named ``img.jpg`` in
    different subfolders would otherwise silently overwrite each other.
    """
    p = d / name
    if not p.exists():
        return p
    stem, ext = Path(name).stem, Path(name).suffix
    i = 1
    while True:
        p = d / f"{stem}_{i}{ext}"
        if not p.exists():
            return p
        i += 1


def _extract_zip_sync(
    zip_path: Path, target_dir: Path, staging_id: str
) -> tuple[int, int]:
    """Extract image entries from a ZIP into ``target_dir``.

    Defensive against zip-slip: ``Path(entry).name`` strips any ``../`` or
    absolute path components, flattening everything into the target dir.

    Skips entries whose SHA-256 is already present in the staging dir
    (either from earlier in this same ZIP or from a prior upload). Each
    new file's hash is also looked up in the Lissl ground-truth index and
    tagged inline if matched. Returns ``(saved, deduped)``.
    """
    hashes = _hashes_for(staging_id, target_dir)
    saved = 0
    deduped = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            basename = Path(info.filename).name
            if not basename or not _is_image_name(basename):
                continue
            with zf.open(info) as src:
                data = src.read()
            digest = _sha256_bytes(data)
            if digest in hashes:
                deduped += 1
                continue
            out_path = _unique_path(target_dir, basename)
            out_path.write_bytes(data)
            hashes[digest] = out_path
            _record_lissl(staging_id, out_path.name, digest)
            make_thumbnail(out_path)
            saved += 1
    return saved, deduped


# ── Resumable upload (tus.io) ───────────────────────────────────────────────


def _tus_complete(file_path: str, metadata: dict) -> None:
    """Called by the tus server when an upload completes.

    The file is staged at ``DATA_DIR/tus/<id>.bin`` with metadata carrying
    ``staging_id`` and ``filename``. We treat it like a single dropped
    item — extract ZIPs into the staging dir, dedup-and-move single
    images via SHA-256 + Lissl tagging, then clean up the temp pair.
    """
    staging_id = metadata.get("staging_id", "")
    fname = metadata.get("filename") or "upload.bin"
    src = Path(file_path)
    meta_sidecar = src.with_suffix(".meta.json")

    def _cleanup() -> None:
        for p in (src, meta_sidecar):
            try:
                p.unlink()
            except OSError:
                pass

    if not _STAGING_RE.match(staging_id):
        _cleanup()
        return

    target_dir = UPLOADS_DIR / staging_id
    target_dir.mkdir(parents=True, exist_ok=True)

    if fname.lower().endswith(".zip"):
        try:
            _extract_zip_sync(src, target_dir, staging_id)
        except Exception:  # noqa: BLE001
            pass
        _cleanup()
        return

    if not _is_image_name(fname):
        _cleanup()
        return

    # Single-image arrival via tus: hash, dedup against staging, move
    # into place, register Lissl ground-truth, generate thumbnail.
    digest = _sha256_file(src)
    hashes = _hashes_for(staging_id, target_dir)
    if digest in hashes:
        _cleanup()
        return
    out_path = _unique_path(target_dir, Path(fname).name)
    src.rename(out_path)
    hashes[digest] = out_path
    _record_lissl(staging_id, out_path.name, digest)
    try:
        make_thumbnail(out_path)
    except Exception:  # noqa: BLE001
        pass
    try:
        meta_sidecar.unlink()
    except OSError:
        pass


# Register tus routes once at module import; the on-complete callback is
# the bridge between the tus server (transport) and our staging logic.
register_tus_routes(nicegui_app, _tus_complete)


def _staging_id_for(session_name: str) -> str:
    """Deterministic staging id for a given session name.

    Stable across reloads — typing the same session name always points
    back at the same staging dir. That's the whole resume mechanism.
    """
    return f".staging_{_sanitize(session_name)}"


def _list_session_files(session_name: str) -> list[str]:
    """Return the image filenames already committed to ``session_name``.

    Used for client-side pre-upload filename dedup: the browser skips
    tus uploads for any file whose name is already in the target
    session. Prevents re-uploading gigabytes of data on resume.
    """
    session = _sanitize(session_name or "")
    if not session or session.startswith("."):
        return []
    d = UPLOADS_DIR / session
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_file() and _is_image_name(p.name))


@nicegui_app.get("/api/session-files")
async def _api_session_files(name: str = "") -> dict:
    """JSON ``{session, committed, staged}`` with the filenames the
    browser should skip when uploading. ``committed`` = already in the
    named session dir; ``staged`` = already in the matching staging
    dir (prior partial upload that's waiting for Save)."""
    session = _sanitize(name or "")
    committed = _list_session_files(session) if session else []
    staging_dir = UPLOADS_DIR / _staging_id_for(session) if session else None
    staged: list[str] = []
    if staging_dir is not None and staging_dir.exists():
        staged = sorted(
            p.name
            for p in staging_dir.iterdir()
            if p.is_file() and _is_image_name(p.name)
        )
    return {"session": session, "committed": committed, "staged": staged}


# ── Save (commit staging → named session) ───────────────────────────────────


def _count_images(d: Path) -> int:
    if not d.exists():
        return 0
    return sum(1 for p in d.iterdir() if p.is_file() and _is_image_name(p.name))


def _orphan_staging_sessions() -> list[tuple[str, int]]:
    """Return ``(session_name, image_count)`` for every non-empty
    ``.staging_*`` dir whose derived session name is recoverable.

    Used by the Upload tab to surface interrupted uploads so the user
    can re-attach the page to them — the whole point of making the
    staging id deterministic in session name.
    """
    out: list[tuple[str, int]] = []
    if not UPLOADS_DIR.exists():
        return out
    prefix = ".staging_"
    for d in UPLOADS_DIR.iterdir():
        if not (d.is_dir() and _STAGING_RE.match(d.name)):
            continue
        session = d.name[len(prefix):]
        if not session:
            continue
        n = _count_images(d)
        if n <= 0:
            continue
        out.append((session, n))
    out.sort(key=lambda x: x[0])
    return out


def _commit_staging(staging_id: str, session_name: str) -> int:
    """Rename the staging directory into ``session_name``.

    Fast-path: if the target doesn't exist, this is a single rename (O(1)).
    Slow-path (target exists): merge — rename files one by one with dedup.
    Returns the number of images committed.
    """
    staging_dir = UPLOADS_DIR / staging_id
    if not staging_dir.exists():
        return 0
    session_dir = UPLOADS_DIR / session_name
    staging_thumbs = THUMBS_DIR / staging_id
    session_thumbs = THUMBS_DIR / session_name

    if not session_dir.exists():
        staging_dir.rename(session_dir)
        if staging_thumbs.exists():
            session_thumbs.parent.mkdir(parents=True, exist_ok=True)
            staging_thumbs.rename(session_thumbs)
        _flush_lissl_labels(staging_id, session_name)
        return _count_images(session_dir)

    # Merge path — session already exists. Move each staged file, but
    # *skip* any whose name already lives in the session (same-filename
    # dedup — matches the client-side pre-upload filter). The existing
    # session copy wins: we treat its presence as "this image has
    # already been committed, don't re-introduce it".
    session_dir.mkdir(parents=True, exist_ok=True)
    session_thumbs.mkdir(parents=True, exist_ok=True)
    moved = 0
    for src in list(staging_dir.iterdir()):
        if not src.is_file() or not _is_image_name(src.name):
            continue
        dst = session_dir / src.name
        if dst.exists():
            # Drop the staged copy and its thumb — already committed.
            try:
                src.unlink()
            except OSError:
                pass
            stale_thumb = staging_thumbs / f"{src.stem}.jpg"
            if stale_thumb.exists():
                try:
                    stale_thumb.unlink()
                except OSError:
                    pass
            continue
        src.rename(dst)
        old_thumb = staging_thumbs / f"{src.stem}.jpg"
        if old_thumb.exists():
            new_thumb = session_thumbs / f"{dst.stem}.jpg"
            old_thumb.rename(new_thumb)
        moved += 1
    # Best-effort cleanup of emptied staging dirs.
    for d in (staging_dir, staging_thumbs):
        try:
            d.rmdir()
        except OSError:
            pass
    _flush_lissl_labels(staging_id, session_name)
    return moved


def _flush_lissl_labels(staging_id: str, session_name: str) -> None:
    """Persist any Lissl ground-truth labels collected during upload to
    the new session's ``lissl_labels.json``. Merges with any existing
    file so re-running into the same session keeps prior labels."""
    from ..lissl_groundtruth import labels_path

    new_labels = _STAGING_GT.pop(staging_id, None)
    if not new_labels:
        return
    p = labels_path(session_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if p.exists():
        try:
            existing = json.loads(p.read_text())
            if not isinstance(existing, dict):
                existing = {}
        except Exception:
            existing = {}
    existing.update(new_labels)
    p.write_text(json.dumps(existing, indent=2, sort_keys=True))


# ── Page render ──────────────────────────────────────────────────────────────


def render() -> None:
    with ui.column().classes("w-full gap-4"):
        ui.label("Upload images").classes("text-xl font-semibold")
        ui.label(
            "Pick (or type) a session name first, then drop images, a "
            "ZIP, or a folder below. Files whose name already lives in "
            "that session are skipped — so re-dropping the same batch "
            "after a browser reload only uploads what's missing."
        ).classes("text-sm text-gray-600")

        default_session = f"session_{time.strftime('%Y%m%d_%H%M%S')}"
        # Staging id is a pure function of session name — typing the
        # same name after a reload auto-resumes the prior batch. The
        # matching dir (if any) is discovered below.
        # ``processing`` flag drives the Save button's enabled state —
        # we don't want users committing while the worker is still
        # saving images / extracting ZIPs / generating thumbs.
        state: dict = {
            "staging_id": _staging_id_for(default_session),
            "staged": _count_images(UPLOADS_DIR / _staging_id_for(default_session)),
            "failed": 0,
            "deduped": 0,
            "processing": False,
            "save_btn": None,
            "progress_label": None,
            "session_summary": None,
        }

        def _render_progress(note: str = "") -> None:
            label = state["progress_label"]
            if label is None:
                return
            parts = [f"{state['staged']} file(s) staged"]
            if state["deduped"]:
                parts.append(f"{state['deduped']} duplicate(s) skipped")
            if state["failed"]:
                parts.append(f"{state['failed']} failed")
            if note:
                parts.append(note)
            label.set_text(" — ".join(parts))

        def _set_processing(busy: bool) -> None:
            """Toggle the processing flag, the label tint, and the Save
            button's enabled state. Both upload paths (drop-zone async
            handler + folder-upload JS via a hidden button) call this."""
            state["processing"] = busy
            btn = state["save_btn"]
            if btn is not None:
                btn.set_enabled(not busy)
            label = state["progress_label"]
            if label is not None:
                # Visual cue: orange while busy, gray when idle.
                label.classes(
                    remove="text-gray-700 text-orange-600 font-semibold"
                )
                label.classes(
                    "text-orange-600 font-semibold"
                    if busy
                    else "text-gray-700"
                )

        # ── Session name (first!) ─────────────────────────────────────────
        # Picking / typing the session name BEFORE uploading is what
        # enables filename dedup and automatic resume. The staging dir
        # is derived from the session name, so a reload + retype of the
        # same session name lands in the exact same staging dir.
        session_input = (
            ui.input(
                label="Session name",
                value=default_session,
                autocomplete=list_sessions(),
            )
            .classes("w-96")
            .props('input-class="session-raw"')
        )

        state["session_summary"] = (
            ui.label("").classes("text-sm text-gray-700")
        )

        def _refresh_session_summary() -> None:
            session = _sanitize(session_input.value or "")
            committed = len(_list_session_files(session)) if session else 0
            staged = _count_images(UPLOADS_DIR / state["staging_id"])
            state["staged"] = staged
            parts = []
            if session:
                parts.append(f"Session '{session}'")
                parts.append(f"{committed} already committed")
            if staged:
                parts.append(f"{staged} staged for next Save")
            lbl = state["session_summary"]
            if lbl is not None:
                lbl.set_text(" — ".join(parts) if parts else "")

        def _on_session_change() -> None:
            """Session name edits reset the staging id to match; the tus
            dropzone's data-attr is pushed to the client so the next
            upload lands in the right staging dir."""
            session = _sanitize(session_input.value or "") or default_session
            new_id = _staging_id_for(session)
            state["staging_id"] = new_id
            state["failed"] = 0
            state["deduped"] = 0
            ui.run_javascript(
                f"""
                const el = document.getElementById('tus-block');
                if (el) {{
                  el.dataset.stagingId = "{new_id}";
                  el.dataset.sessionName = "{session}";
                }}
                const summary = document.getElementById('tus-summary');
                if (summary) summary.innerText = '';
                const status = document.getElementById('tus-status');
                if (status) status.innerText = '';
                const wrap = document.getElementById('tus-progress');
                if (wrap) wrap.classList.add('hidden');
                """
            )
            _refresh_session_summary()
            _render_progress()
            render_upload_gallery()

        session_input.on(
            "change",
            lambda _e: (
                session_input.set_autocomplete(list_sessions()),
                _on_session_change(),
            ),
        )
        session_input.on("blur", lambda _e: _on_session_change())

        # ── Resume previous upload ─────────────────────────────────────────
        # Any ``.staging_<name>`` dir with images but no matching saved
        # session is an interrupted upload. Surface them here so a user
        # who closed the tab / reloaded the page can click to restore.
        orphans_row = ui.row().classes("items-center gap-2 w-full")

        def _resume_session(session: str) -> None:
            session_input.set_value(session)
            _on_session_change()
            _refresh_orphans()
            ui.notify(f"Resumed staging for '{session}'", type="info")

        def _refresh_orphans() -> None:
            orphans_row.clear()
            orphans = _orphan_staging_sessions()
            if not orphans:
                orphans_row.set_visibility(False)
                return
            orphans_row.set_visibility(True)
            with orphans_row:
                ui.label("Resume previous upload:").classes(
                    "text-sm text-gray-700"
                )
                for session, n in orphans:
                    ui.button(
                        f"'{session}' — {n} staged",
                        on_click=lambda _e=None, s=session: _resume_session(s),
                    ).props("flat dense color=primary")

        _refresh_orphans()

        # ── Resumable upload (tus.io) — sole upload entry point ───────────
        # For multi-GB ZIPs / images on flaky networks. Uploads in 5 MB
        # chunks via PATCH; resumes from the last completed chunk if the
        # connection drops mid-transfer. tus-js-client (CDN) drives the
        # client side; the server endpoints are at /api/tus.
        ui.html(
            f"""
            <div id="tus-block" class="w-full mt-2"
                 data-staging-id="{state['staging_id']}"
                 data-session-name="{_sanitize(default_session)}">
              <label for="tus-file-input"
                     class="block cursor-pointer border-2 border-dashed
                            border-blue-300 rounded p-3 text-center text-sm
                            hover:bg-blue-50">
                <i class="material-icons" style="font-size:18px;
                          vertical-align:middle;">cloud_upload</i>
                Resumable upload — click here for a single large ZIP /
                image (resumes after disconnects)
              </label>
              <input type="file" id="tus-file-input" multiple
                     accept=".jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff,.zip"
                     style="display:none">
              <div class="flex items-center gap-2 mt-1">
                <div id="tus-summary"
                     class="text-sm font-semibold text-gray-800
                            flex-1"></div>
                <button id="tus-abort"
                        class="hidden text-xs px-2 py-1 border
                               border-red-300 text-red-700 rounded
                               hover:bg-red-50">
                  Abort
                </button>
              </div>
              <div id="tus-status"
                   class="text-xs text-gray-600 mt-1 truncate"></div>
              <div id="tus-progress" class="hidden">
                <div class="w-full bg-gray-200 rounded h-2 mt-1
                            overflow-hidden">
                  <div id="tus-progress-bar"
                       class="bg-blue-500 h-2 transition-all"
                       style="width:0%"></div>
                </div>
              </div>
            </div>
            """
        )

        # Hidden buttons the tus JS clicks to signal start / done so
        # Python can disable Save during an upload and refresh state on
        # completion (re-counts staged files from disk).
        ui.button(
            "", on_click=lambda: _set_processing(True)
        ).style("display:none").props('id="folder-start-btn"')

        def _refresh_staged_from_disk() -> None:
            state["staged"] = _count_images(UPLOADS_DIR / state["staging_id"])
            _render_progress("ready to save")
            _set_processing(False)

        ui.button("", on_click=_refresh_staged_from_disk).style(
            "display:none"
        ).props('id="folder-refresh-btn"')

        # ── Tus dropzone wireup ────────────────────────────────────────────
        # Single upload entry point: tus.io for everything (single image,
        # single ZIP, walked folder). Drop or click on the blue dashed
        # zone above; per-file resume / retry handles flaky networks.
        ui.run_javascript(
            r"""
            (function wireFolderDrop() {
              if (window.__folderDropWired) return;
              window.__folderDropWired = true;
              const imgRe = /\.(jpg|jpeg|png|webp|bmp|tif|tiff)$/i;

              // ── tus.io resumable upload ────────────────────────────
              // Driven by tus-js-client loaded via CDN script tag. Each
              // file becomes a tus.Upload that PATCHes 5 MB chunks to
              // /api/tus/<id>. tus-js-client handles offset tracking,
              // retry, and resume across disconnects automatically.
              // Two entry points share the runTus() pipeline:
              //   - file input change (button click → OS picker)
              //   - drop on the dashed dropzone
              const tusInput = document.getElementById('tus-file-input');
              const tusBlock = document.getElementById('tus-block');
              const tusStatus = document.getElementById('tus-status');
              const tusSummary = document.getElementById('tus-summary');
              const tusProgressWrap = document.getElementById('tus-progress');
              const tusBar = document.getElementById('tus-progress-bar');

              function writeSummary(done, failed, total) {
                if (!tusSummary) return;
                const left = total - done - failed;
                let text = `${done} done · ${left} left · ${total} total`;
                if (failed) text += ` · ${failed} failed`;
                tusSummary.innerText = text;
              }

              // Throttled DOM writer — coalesces high-frequency progress
              // ticks (XHR upload events at ~200/s) into ~5 updates/s
              // so the main thread isn't drowning in style recalculation.
              // Critical for keeping Firefox responsive during long
              // uploads — without this the UI thread can starve and the
              // whole tab feels frozen.
              let _tusLastWrite = 0;
              function writeStatusThrottled(msg, pct, force) {
                const now = Date.now();
                if (!force && now - _tusLastWrite < 200) return;
                _tusLastWrite = now;
                if (tusStatus) tusStatus.innerText = msg;
                if (tusBar && pct != null) tusBar.style.width = pct + '%';
              }

              // Holds the in-flight tus.Upload so the Abort button can
              // call .abort() on it. Only one upload runs at a time.
              let _tusCurrent = null;
              const tusAbort = document.getElementById('tus-abort');
              if (tusAbort && !tusAbort.dataset.wired) {
                tusAbort.dataset.wired = '1';
                tusAbort.addEventListener('click', () => {
                  if (_tusCurrent) {
                    try { _tusCurrent.abort(true); }
                    catch (e) { console.warn('[tus] abort failed', e); }
                  }
                });
              }

              async function runTus(files) {
                if (!files.length) return;
                let waited = 0;
                while (typeof tus === 'undefined' && waited < 5000) {
                  await new Promise(r => setTimeout(r, 200));
                  waited += 200;
                }
                if (typeof tus === 'undefined') {
                  if (tusStatus) tusStatus.innerText =
                    'tus library failed to load — check your network.';
                  return;
                }
                const stagingId =
                  (tusBlock && tusBlock.dataset.stagingId) || '';
                const sessionName =
                  (tusBlock && tusBlock.dataset.sessionName) || '';
                // Pre-upload filename dedup: ask the server which names
                // are already committed to this session AND which are
                // already in the staging dir (resumed batch). Skip those
                // — same-name = "already have it", per user intent.
                // ZIP uploads bypass this filter since the names the ZIP
                // contains aren't visible until extraction.
                let skipSet = new Set();
                try {
                  const r = await fetch(
                    '/api/session-files?name='
                    + encodeURIComponent(sessionName)
                  );
                  if (r.ok) {
                    const js = await r.json();
                    for (const n of (js.committed || [])) skipSet.add(n);
                    for (const n of (js.staged || [])) skipSet.add(n);
                  }
                } catch (e) {
                  console.warn('[tus] session-files fetch failed', e);
                }
                const originalTotal = files.length;
                const skippedCount = files.filter(
                  f => !f.name.toLowerCase().endsWith('.zip')
                       && skipSet.has(f.name)
                ).length;
                files = files.filter(
                  f => f.name.toLowerCase().endsWith('.zip')
                       || !skipSet.has(f.name)
                );
                if (!files.length) {
                  if (tusStatus) tusStatus.innerText =
                    `All ${originalTotal} file(s) already uploaded `
                    + `to session '${sessionName}'.`;
                  const refreshBtnEarly =
                    document.getElementById('folder-refresh-btn');
                  if (refreshBtnEarly) refreshBtnEarly.click();
                  return;
                }
                const startBtn = document.getElementById('folder-start-btn');
                const refreshBtn = document.getElementById('folder-refresh-btn');
                if (startBtn) startBtn.click();
                if (tusProgressWrap) tusProgressWrap.classList.remove('hidden');
                if (tusAbort) tusAbort.classList.remove('hidden');
                let total = files.length, done = 0, failed = 0;
                writeSummary(done, failed, total);
                if (skippedCount > 0 && tusStatus) {
                  tusStatus.innerText =
                    `Skipping ${skippedCount} already-uploaded file(s); `
                    + `uploading ${total} new…`;
                }
                for (const file of files) {
                  // Adaptive chunkSize: a small image (≤ 10 MB) goes in
                  // a single PATCH — chunking adds HTTP overhead with
                  // no resume benefit. Larger files (multi-MB / GB ZIPs)
                  // get 5 MB chunks so a mid-transfer disconnect costs
                  // ≤ 5 MB of progress.
                  const fileChunk = file.size <= 10 * 1024 * 1024
                    ? Math.max(file.size, 1)
                    : 5 * 1024 * 1024;
                  await new Promise((resolve) => {
                    const upload = new tus.Upload(file, {
                      endpoint: '/api/tus',
                      chunkSize: fileChunk,
                      // Capped retry: 3 attempts max. If a chunk can't
                      // make it after 3 tries we give up on this file
                      // rather than spinning forever.
                      retryDelays: [1000, 3000, 8000],
                      metadata: {
                        filename: file.name,
                        filetype: file.type || '',
                        staging_id: stagingId,
                      },
                      onProgress: (sent, totalBytes) => {
                        const pct = totalBytes
                          ? Math.round(100 * sent / totalBytes)
                          : 0;
                        const mb = (sent / 1024 / 1024).toFixed(1);
                        const tot = (totalBytes / 1024 / 1024).toFixed(1);
                        writeStatusThrottled(
                          `${file.name}: ${pct}% `
                          + `(${mb} / ${tot} MB) `
                          + `· file ${done + 1} / ${total}`,
                          pct, false
                        );
                      },
                      onSuccess: () => {
                        done++;
                        writeSummary(done, failed, total);
                        writeStatusThrottled(
                          `${file.name} uploaded.`, 100, true
                        );
                        console.log('[tus] uploaded', file.name);
                        _tusCurrent = null;
                        resolve();
                      },
                      onError: (err) => {
                        failed++;
                        writeSummary(done, failed, total);
                        console.error('[tus] error on', file.name, err);
                        writeStatusThrottled(
                          `${file.name}: ${err.message || 'upload failed'}`,
                          null, true
                        );
                        _tusCurrent = null;
                        resolve();  // continue with next file
                      },
                    });
                    _tusCurrent = upload;
                    upload.start();
                  });
                }
                if (tusAbort) tusAbort.classList.add('hidden');
                writeSummary(done, failed, total);
                writeStatusThrottled(
                  `Finished: ${done} of ${total}`
                  + (failed ? ` (${failed} failed)` : ''),
                  null, true
                );
                if (refreshBtn) refreshBtn.click();
              }

              if (tusInput && !tusInput.dataset.wired) {
                tusInput.dataset.wired = '1';
                tusInput.addEventListener('change', async (ev) => {
                  const files = Array.from(ev.target.files);
                  ev.target.value = '';  // allow re-picking same file
                  await runTus(files);
                });
              }

              // ── Drop support on the tus dropzone ────────────────────
              // Local capture-phase listeners on the block; only intercept
              // when the drop actually lands here. Plain files (single
              // ZIP, single image) only — directories are routed to the
              // folder upload pipeline by the document-level interceptor.
              if (tusBlock && !tusBlock.dataset.dropWired) {
                tusBlock.dataset.dropWired = '1';
                tusBlock.addEventListener('dragover', (e) => {
                  if (e.dataTransfer && e.dataTransfer.types &&
                      Array.from(e.dataTransfer.types).includes('Files')) {
                    e.preventDefault();
                    e.stopPropagation();
                    tusBlock.classList.add('bg-blue-50');
                  }
                }, true);
                tusBlock.addEventListener('dragleave', () => {
                  tusBlock.classList.remove('bg-blue-50');
                }, true);
                tusBlock.addEventListener('drop', async (e) => {
                  if (!e.dataTransfer) return;
                  // Snapshot entries synchronously — DataTransferItemList
                  // becomes invalid after the handler returns.
                  const entries = Array.from(e.dataTransfer.items || [])
                    .map(i => (i.webkitGetAsEntry
                                ? i.webkitGetAsEntry() : null))
                    .filter(Boolean);
                  const hasDir = entries.some(en => en.isDirectory);
                  e.preventDefault();
                  e.stopPropagation();
                  tusBlock.classList.remove('bg-blue-50');

                  let files;
                  if (hasDir) {
                    // Folder drop on tus zone → walk recursively, then
                    // upload each image via tus.Upload (gives every
                    // file resume support).
                    if (tusStatus) tusStatus.innerText = 'Reading folder…';
                    const collected = [];
                    for (const en of entries) {
                      await safeWalk(en, collected, 0);
                    }
                    files = collected
                      .filter(({file}) =>
                        imgRe.test(file.name)
                        && !file.name.startsWith('.'))
                      .map(({file}) => file);
                    console.log('[tus] folder walked,',
                                files.length, 'image(s)');
                  } else {
                    // Plain file drop (single ZIP, single image, multi).
                    files = Array.from(e.dataTransfer.files || []);
                    console.log('[tus] dropped', files.length, 'file(s)');
                  }
                  if (!files.length) {
                    if (tusStatus) tusStatus.innerText =
                      'No image / ZIP files in the drop.';
                    return;
                  }
                  await runTus(files);
                }, true);
              }

              // Hardened recursive walk. Swallows per-entry errors so one
              // bad file (permission denied, never-resolves callback) can't
              // hang the whole drop handler. Per-call timeouts prevent
              // wedge on macOS files with extended attributes.
              async function safeWalk(entry, files, depth) {
                try {
                  if (entry.isFile) {
                    const f = await new Promise((res, rej) => {
                      const timer = setTimeout(
                        () => rej(new Error('file() timeout')), 30000
                      );
                      entry.file(
                        (file) => { clearTimeout(timer); res(file); },
                        (err) => { clearTimeout(timer); rej(err); }
                      );
                    });
                    files.push({file: f, path: entry.fullPath});
                    return;
                  }
                  if (entry.isDirectory && depth < 16) {
                    const reader = entry.createReader();
                    // readEntries returns at most ~100 per call — loop
                    // until empty to handle large folders.
                    let batch;
                    do {
                      batch = await new Promise((res, rej) => {
                        const timer = setTimeout(
                          () => rej(new Error('readEntries timeout')), 30000
                        );
                        reader.readEntries(
                          (list) => { clearTimeout(timer); res(list); },
                          (err) => { clearTimeout(timer); rej(err); }
                        );
                      });
                      for (const child of batch) {
                        await safeWalk(child, files, depth + 1);
                      }
                    } while (batch.length > 0);
                  }
                } catch (err) {
                  console.warn('[folder-drop] skipping', entry.fullPath, err);
                }
              }

              // Document-level dragover preventDefault stops the browser's
              // default "navigate to dropped file" behavior outside the
              // tus block — without it, a slightly-off-target drop would
              // open the file in the tab and replace the whole UI.
              document.addEventListener('dragover', (e) => {
                if (e.dataTransfer && e.dataTransfer.types &&
                    Array.from(e.dataTransfer.types).includes('Files')) {
                  e.preventDefault();
                }
              }, true);
              document.addEventListener('drop', (e) => {
                // Same idea for drop: swallow stray drops that miss the
                // tus block. The tus block's own capture-phase handler
                // runs first and stops propagation, so legitimate drops
                // never reach this listener.
                if (e.dataTransfer && e.dataTransfer.types &&
                    Array.from(e.dataTransfer.types).includes('Files')) {
                  e.preventDefault();
                }
              }, true);
            })();
            """
        )

        # ── Progress + Save ────────────────────────────────────────────────
        # Processing status sits right above the Save button so the user
        # always sees the current upload state before deciding to commit.
        # Tracked in ``state["progress_label"]`` so handlers defined
        # earlier (``_render_progress`` / ``_set_processing``) can find
        # it lazily.
        state["progress_label"] = (
            ui.label("Ready to upload.").classes("text-sm text-gray-700")
        )
        save_status = ui.label("").classes("text-sm text-green-700")

        def _reset_staging() -> None:
            """Clear per-batch counters and the hash cache after Save.

            The staging id stays pinned to the current session name, so
            a follow-up upload to the same session accumulates into a
            fresh ``.staging_<session>`` dir (the old one was just
            renamed/merged into the session by ``_commit_staging``).
            """
            _drop_staging_hashes(state["staging_id"])
            state["staged"] = 0
            state["failed"] = 0
            state["deduped"] = 0
            ui.run_javascript(
                """
                const summary = document.getElementById('tus-summary');
                if (summary) summary.innerText = '';
                const status = document.getElementById('tus-status');
                if (status) status.innerText = '';
                const wrap = document.getElementById('tus-progress');
                if (wrap) wrap.classList.add('hidden');
                """
            )

        def do_save() -> None:
            staging_dir = UPLOADS_DIR / state["staging_id"]
            if not staging_dir.exists() or _count_images(staging_dir) == 0:
                ui.notify(
                    "Nothing to save — drop some images first.", type="warning"
                )
                return
            session = _sanitize(session_input.value or "")
            if session.startswith("."):
                ui.notify(
                    "Session names can't start with a dot.", type="warning"
                )
                return
            moved = _commit_staging(state["staging_id"], session)
            _reset_staging()
            save_status.set_text(f"Saved {moved} image(s) to '{session}'.")
            session_input.set_value(session)
            session_input.set_autocomplete(list_sessions())
            _render_progress()
            _refresh_session_summary()
            _refresh_orphans()
            gallery_state["page"] = 0
            render_upload_gallery()

        with ui.row().classes("items-center gap-2"):
            save_btn = ui.button(
                "Save session", icon="save", on_click=do_save
            ).props("color=primary")
        state["save_btn"] = save_btn

        # ── Uploaded gallery (renders after Save or on session switch) ─────
        gallery_state = {"page": 0, "per_page": 50}
        gallery_container = ui.column().classes("w-full mt-4 gap-2")

        def render_upload_gallery() -> None:
            gallery_container.clear()
            session = _sanitize(session_input.value or "")
            src_dir = UPLOADS_DIR / session
            if not session or session.startswith(".") or not src_dir.exists():
                return
            images = sorted(
                (p for p in src_dir.iterdir() if _is_image_name(p.name)),
                key=lambda p: p.name,
            )
            total = len(images)
            if total == 0:
                with gallery_container:
                    ui.label(
                        f"Session '{session}' has no images yet."
                    ).classes("text-sm text-gray-500")
                return
            per_page = gallery_state["per_page"]
            pages = max(1, (total + per_page - 1) // per_page)
            gallery_state["page"] = max(0, min(gallery_state["page"], pages - 1))
            page = gallery_state["page"]
            start = page * per_page
            page_items = images[start : start + per_page]

            with gallery_container:
                ui.label(
                    f"Session '{session}' — {total} image(s), "
                    f"page {page + 1} / {pages}"
                ).classes("text-sm text-gray-600 font-semibold")

                with ui.row().classes("gap-2 flex-wrap"):
                    for src in page_items:
                        tp = thumb_path_for(src)
                        if tp.exists():
                            rel = tp.relative_to(THUMBS_DIR).as_posix()
                            url = f"/thumbs/{rel}"
                        else:
                            rel = src.relative_to(UPLOADS_DIR).as_posix()
                            url = f"/uploads/{rel}"
                        ui.image(url).classes(
                            "w-[120px] h-[120px] object-cover rounded"
                        ).tooltip(src.name)

                def go(delta: int) -> None:
                    gallery_state["page"] += delta
                    render_upload_gallery()

                with ui.row().classes("gap-2 items-center mt-2"):
                    ui.button("← Prev", on_click=lambda: go(-1)).props(
                        "flat"
                    ).set_enabled(page > 0)
                    ui.button("Next →", on_click=lambda: go(1)).props(
                        "flat"
                    ).set_enabled(page < pages - 1)

        # Gallery refresh on session switch is already wired via
        # ``_on_session_change`` above; no additional handler needed.
        # Render once so the picker immediately reflects any existing
        # images in the default session (usually none, but a user who
        # typed back into a pre-existing session should see them).
        _refresh_session_summary()
        render_upload_gallery()
