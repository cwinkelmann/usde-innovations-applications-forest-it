"""Tab 1 — Upload images into a named session folder under /data/uploads/.

Workflow (per-page):
  1. User drops images, a ZIP, or picks a folder. Files land in a staging
     directory ``.staging_<uuid>`` under /data/uploads/, isolated from real
     sessions.
  2. User types/edits the session name.
  3. User clicks **Save** — the staging directory is renamed to the chosen
     session name, thumbnails move alongside, and the gallery renders.

Dot-prefixed staging directories are filtered out of ``list_sessions()``
so they never leak into pickers. If the user abandons the page, the stage
lingers until a janitor removes ``.staging_*`` (not implemented — course
tool, manual cleanup is fine).
"""
from __future__ import annotations

import asyncio
import re
import time
import uuid
import zipfile
from pathlib import Path

from fastapi import File, Form, UploadFile
from nicegui import app as nicegui_app
from nicegui import events, ui

from ..config import THUMBS_DIR, UPLOADS_DIR
from ..sessions import list_sessions
from ..thumbs import make_thumbnail, thumb_path_for

_SAFE = re.compile(r"[^a-zA-Z0-9_-]+")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
_STAGING_RE = re.compile(r"^\.staging_[a-f0-9]{12}$")


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


def _extract_zip_sync(zip_path: Path, target_dir: Path) -> int:
    """Extract image entries from a ZIP into ``target_dir``. Returns count.

    Defensive against zip-slip: ``Path(entry).name`` strips any ``../`` or
    absolute path components, flattening everything into the target dir.
    """
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            basename = Path(info.filename).name
            if not basename or not _is_image_name(basename):
                continue
            out_path = _unique_path(target_dir, basename)
            with zf.open(info) as src, out_path.open("wb") as dst:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    dst.write(chunk)
            make_thumbnail(out_path)
            count += 1
    return count


# ── Folder-upload FastAPI route (registered at module import) ────────────────


@nicegui_app.post("/api/upload-folder")
async def upload_folder_endpoint(
    staging_id: str = Form(...),
    file: UploadFile = File(...),
) -> dict:
    """Accept one image into the page's staging directory.

    Validates ``staging_id`` format to prevent a malicious client from
    redirecting writes to arbitrary paths under UPLOADS_DIR.
    """
    if not _STAGING_RE.match(staging_id):
        return {"saved": 0, "error": "invalid staging_id"}
    filename = file.filename or ""
    if not _is_image_name(filename):
        return {"saved": 0}
    target_dir = UPLOADS_DIR / staging_id
    target_dir.mkdir(parents=True, exist_ok=True)
    basename = Path(filename).name or "upload.bin"
    out_path = _unique_path(target_dir, basename)
    data = await file.read()
    out_path.write_bytes(data)
    await asyncio.to_thread(make_thumbnail, out_path)
    return {"saved": 1}


# ── Save (commit staging → named session) ───────────────────────────────────


def _count_images(d: Path) -> int:
    if not d.exists():
        return 0
    return sum(1 for p in d.iterdir() if p.is_file() and _is_image_name(p.name))


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
        return _count_images(session_dir)

    # Merge path — session already exists. Move each staged file with dedup.
    session_dir.mkdir(parents=True, exist_ok=True)
    session_thumbs.mkdir(parents=True, exist_ok=True)
    moved = 0
    for src in list(staging_dir.iterdir()):
        if not src.is_file() or not _is_image_name(src.name):
            continue
        dst = _unique_path(session_dir, src.name)
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
    return moved


# ── Page render ──────────────────────────────────────────────────────────────


def render() -> None:
    with ui.column().classes("w-full gap-4"):
        ui.label("Upload images").classes("text-xl font-semibold")
        ui.label(
            "Drop images, a ZIP, or pick a folder. Files are staged locally; "
            "enter a session name and click Save to commit them."
        ).classes("text-sm text-gray-600")

        # One staging id per page render. Regenerated after every successful
        # Save so the next batch starts fresh. Must match the _STAGING_RE
        # checked by the folder-upload endpoint.
        # ``processing`` flag drives the Save button's enabled state — we
        # don't want users committing to a session name while the worker
        # is still saving images / extracting ZIPs / generating thumbs.
        # ``save_btn`` is filled in below once the button is created.
        state: dict = {
            "staging_id": f".staging_{uuid.uuid4().hex[:12]}",
            "staged": 0,
            "failed": 0,
            "processing": False,
            "save_btn": None,
            "progress_label": None,
        }

        def _render_progress(note: str = "") -> None:
            label = state["progress_label"]
            if label is None:
                return
            parts = [f"{state['staged']} file(s) staged"]
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

        async def handle_multi_upload(
            e: events.MultiUploadEventArguments,
        ) -> None:
            _set_processing(True)
            target_dir = UPLOADS_DIR / state["staging_id"]
            target_dir.mkdir(parents=True, exist_ok=True)
            _render_progress(f"processing {len(e.files)} file(s)…")

            try:
                for file in e.files:
                    fname = Path(file.name).name or "upload.bin"
                    try:
                        if fname.lower().endswith(".zip"):
                            tmp_zip = target_dir / f".{fname}.tmp"
                            await file.save(tmp_zip)
                            count = await asyncio.to_thread(
                                _extract_zip_sync, tmp_zip, target_dir
                            )
                            tmp_zip.unlink(missing_ok=True)
                            state["staged"] += count
                        elif _is_image_name(fname):
                            out_path = _unique_path(target_dir, fname)
                            await file.save(out_path)
                            await asyncio.to_thread(make_thumbnail, out_path)
                            state["staged"] += 1
                        else:
                            state["failed"] += 1
                    except Exception:
                        state["failed"] += 1
                    _render_progress()
            finally:
                _set_processing(False)
                _render_progress("ready to save")

        ui.upload(
            on_multi_upload=handle_multi_upload,
            multiple=True,
            auto_upload=True,
            label="Drop images or a ZIP archive here",
            max_file_size=10 * 1024 * 1024 * 1024,  # 10 GB — ZIPs get big
        ).props(
            'accept=".jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff,.zip" '
            "flat bordered"
        ).classes("w-full")

        # ── Folder upload ──────────────────────────────────────────────────
        #
        # Native <input webkitdirectory> with a <label for="…"> trigger to
        # preserve the user-gesture context (a server round-trip would lose
        # it and the picker wouldn't open). The staging_id lives in a
        # data-attribute the JS reads at pick time; rotating it after Save
        # is handled by the _reset_staging() Python helper below.

        ui.html(
            f"""
            <div id="folder-upload-block" class="flex items-center gap-3"
                 data-staging-id="{state['staging_id']}">
              <label for="folder-input"
                     class="cursor-pointer inline-flex items-center gap-2
                            px-4 py-2 border border-gray-300 rounded
                            hover:bg-gray-50 text-sm select-none"
                     style="line-height: 1;">
                <i class="material-icons" style="font-size:18px;">folder</i>
                <span>Upload folder</span>
              </label>
              <input type="file" id="folder-input"
                     webkitdirectory directory multiple
                     style="display:none">
              <span id="folder-total" class="text-sm text-gray-700"></span>
            </div>
            """
        )

        # Hidden buttons the folder-upload JS clicks to bracket its work.
        # The JS doesn't go through Python until these fire, so without
        # them the Save button wouldn't disable during a folder upload.
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

        ui.run_javascript(
            r"""
            (function wireFolderUpload() {
              const input = document.getElementById('folder-input');
              if (!input || input.dataset.wired) return;
              input.dataset.wired = '1';
              const totalEl = document.getElementById('folder-total');
              const block = document.getElementById('folder-upload-block');
              const imgRe = /\.(jpg|jpeg|png|webp|bmp|tif|tiff)$/i;

              input.addEventListener('change', async (ev) => {
                const stagingId = block.dataset.stagingId || '';
                const files = Array.from(ev.target.files).filter((f) => {
                  const base = (f.webkitRelativePath || f.name).split('/').pop();
                  if (!base || base.startsWith('.')) return false;
                  return imgRe.test(base);
                });
                const total = files.length;
                if (!total) {
                  totalEl.innerText = 'No image files in the selected folder.';
                  return;
                }
                // Tell Python we're starting so it disables Save.
                const startBtn = document.getElementById('folder-start-btn');
                if (startBtn) startBtn.click();
                let done = 0, failed = 0;
                totalEl.innerText = `Uploading 0 / ${total}...`;
                for (const f of files) {
                  const fd = new FormData();
                  fd.append('staging_id', stagingId);
                  fd.append('file', f, f.webkitRelativePath || f.name);
                  try {
                    const r = await fetch('/api/upload-folder', {
                      method: 'POST', body: fd,
                    });
                    if (r.ok) {
                      const js = await r.json();
                      done += (js.saved || 0);
                      if (!js.saved) failed++;
                    } else { failed++; }
                  } catch (e) { failed++; }
                  totalEl.innerText = `Uploading ${done} / ${total}`
                    + (failed ? ` (${failed} skipped/failed)` : '');
                }
                totalEl.innerText =
                  `Folder staged — ${done} / ${total}`
                  + (failed ? `, ${failed} skipped/failed` : '');
                const refresh = document.getElementById('folder-refresh-btn');
                if (refresh) refresh.click();
                input.value = '';
              });
            })();
            """
        )

        # ── Session name + Save ────────────────────────────────────────────
        default_name = f"session_{time.strftime('%Y%m%d_%H%M%S')}"
        session_input = (
            ui.input(
                label="Session name",
                value=default_name,
                autocomplete=list_sessions(),
            )
            .classes("w-96")
            .props('input-class="session-raw"')
        )
        session_input.on(
            "change",
            lambda _e: session_input.set_autocomplete(list_sessions()),
        )

        # Processing status sits *right above* the Save button so the user
        # always sees the current upload state before deciding to commit.
        # Tracked in `state["progress_label"]` so handlers defined earlier
        # (like _render_progress / _set_processing) can find it lazily.
        state["progress_label"] = (
            ui.label("Ready to upload.").classes("text-sm text-gray-700")
        )
        save_status = ui.label("").classes("text-sm text-green-700")

        def _reset_staging() -> None:
            """Regenerate the staging id so the next upload batch is fresh."""
            state["staging_id"] = f".staging_{uuid.uuid4().hex[:12]}"
            state["staged"] = 0
            state["failed"] = 0
            # Update the data-attribute in the DOM so the folder-upload JS
            # posts future files into the new staging dir.
            ui.run_javascript(
                f"""
                const el = document.getElementById('folder-upload-block');
                if (el) el.dataset.stagingId = "{state['staging_id']}";
                const total = document.getElementById('folder-total');
                if (total) total.innerText = '';
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

        session_input.on(
            "change",
            lambda _e: (gallery_state.update({"page": 0}), render_upload_gallery()),
        )
        # No initial gallery render — we only show it after Save, or when the
        # user explicitly switches to an existing session via the input.
