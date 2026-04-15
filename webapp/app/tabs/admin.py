"""Tab — Admin: camera-trap dataset maintenance.

Assign GPS coordinates + description to a session (surfaces on the Map tab),
permanently delete a dataset (removes uploads, thumbnails, outputs), and
download the demo datasets used by the course practicals (Snapshot
Serengeti, Caltech Camera Traps) as ready-to-use sessions.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from nicegui import run, ui

from ..camera_meta import delete_session, load, save
from ..config import DATA_DIR, OUTPUTS_DIR, UPLOADS_DIR
from ..job_manager import JobManager
from ..sessions import list_sessions
from ..thumbs import make_thumbnail

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Order matters: prefer the richer outputs (species enrichment) but fall
# back to plain MD detections. Animal counts are identical across these
# files for a given session, so first-existing wins.
_DETECTION_SOURCES = (
    "md_speciesnet/predictions.json",
    "md_deepfaune/predictions.json",
    "md/detections.json",
)


def _count_session_images(session: str) -> int:
    d = UPLOADS_DIR / session
    if not d.exists():
        return 0
    return sum(
        1 for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        and not p.name.startswith(".")
    )


def _count_session_animals(session: str) -> int | None:
    """Total animal detections across the session's MD output. Returns
    ``None`` if no inference has been run yet (so the table can show
    ``—`` instead of a misleading ``0``)."""
    for rel in _DETECTION_SOURCES:
        p = OUTPUTS_DIR / session / rel
        if p.exists():
            try:
                data = json.loads(p.read_text())
            except Exception:
                return None
            return sum(
                1
                for item in data
                for d in (item.get("detections") or [])
                if d.get("category_id") == 0
            )
    return None


def _collect_stats() -> list[dict]:
    """Per-session row: ``{session, images, animals, lat, lng, name}``."""
    rows: list[dict] = []
    for session in list_sessions():
        meta = load(session)
        rows.append(
            {
                "session": session,
                "name": meta.get("name") or "",
                "images": _count_session_images(session),
                "animals": _count_session_animals(session),
                "lat": meta.get("latitude"),
                "lng": meta.get("longitude"),
            }
        )
    return rows


def _install_to_session(src_dir: Path, session: str) -> int:
    """Copy images from ``src_dir`` into ``UPLOADS_DIR/<session>`` and
    generate thumbnails. Returns the number of images installed.
    Idempotent: existing files are not re-copied.
    """
    dest = UPLOADS_DIR / session
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in src_dir.iterdir():
        if not src.is_file() or src.suffix.lower() not in _IMAGE_EXTS:
            continue
        target = dest / src.name
        if not target.exists():
            shutil.copy2(src, target)
        make_thumbnail(target)
        n += 1
    return n


def _download_serengeti_subset(n_images: int) -> Path:
    """Wrapper so we can ``run.io_bound`` it. Returns the source dir."""
    from wildlife_detection.download_data import download_serengeti
    return download_serengeti(
        n_images=n_images, output_dir=DATA_DIR / "downloads"
    )


def _download_caltech_subset(n_images: int) -> Path:
    from wildlife_detection.download_data import download_caltech
    return download_caltech(
        n_images=n_images, output_dir=DATA_DIR / "downloads"
    )


def _session_has_output(session: str, mode_key: str) -> bool:
    """True if the session already has the named pipeline's output JSON."""
    fname = "detections.json" if mode_key == "md" else "predictions.json"
    return (OUTPUTS_DIR / session / mode_key / fname).exists()


def _sessions_needing(mode_key: str) -> list[str]:
    """Sessions with images but no output for ``mode_key``."""
    out: list[str] = []
    for session in list_sessions():
        if (
            _count_session_images(session) > 0
            and not _session_has_output(session, mode_key)
        ):
            out.append(session)
    return out


def render(jm: JobManager) -> None:
    with ui.column().classes("w-full gap-4"):
        ui.label("Admin").classes("text-xl font-semibold")

        # ── Worker pool / GPU status ──────────────────────────────────────
        # Live view of each worker's pinned GPU, free memory, and pending
        # jobs. Lets you see why the scheduler routed where it did —
        # GPUs below 50 % free are deprioritized, so external processes
        # hogging a card won't drag the pool's throughput.
        ui.label("Worker pool").classes("text-lg font-semibold mt-2")
        ui.label(
            f"{jm.num_workers} worker(s) on device='{jm.device}'. GPUs with "
            "<50% free memory are deprioritized — the scheduler picks the "
            "least-busy eligible GPU on each submit."
        ).classes("text-sm text-gray-600")
        gpu_container = ui.column().classes("w-full gap-1")

        def _refresh_gpus() -> None:
            gpu_container.clear()
            snap = jm.gpu_snapshot()
            with gpu_container:
                if not snap:
                    ui.label("No workers registered.").classes(
                        "text-sm text-gray-500"
                    )
                    return
                for row in snap:
                    w = row["worker"]
                    free = row["free_mib"]
                    total = row["total_mib"]
                    frac = row["free_frac"]
                    pending = row["pending"]
                    if total <= 0:
                        # Non-CUDA or unknown: just show pending count.
                        with ui.row().classes("items-center gap-2 w-full"):
                            ui.label(f"worker {w}").classes(
                                "text-xs font-mono w-24"
                            )
                            ui.label(f"{pending} pending job(s)").classes(
                                "text-xs flex-1"
                            )
                        continue
                    used = total - free
                    free_pct = (frac or 0.0) * 100
                    deprioritized = (frac or 0.0) < 0.5
                    label_text = (
                        f"cuda:{w}  ·  {used:,} / {total:,} MiB used  "
                        f"·  {free_pct:.0f}% free  ·  {pending} pending"
                    )
                    if deprioritized:
                        label_text += "  ·  ⚠ deprioritized"
                    with ui.row().classes("items-center gap-2 w-full"):
                        ui.label(label_text).classes(
                            "text-xs font-mono w-[36rem] truncate"
                            + (" text-orange-700" if deprioritized else "")
                        )
                        # Bar shows memory *used*, not free — so a fat
                        # bar means the GPU is busy (and likely skipped).
                        ui.linear_progress(
                            value=used / total if total else 0.0,
                            show_value=False,
                        ).classes("flex-1")

        _refresh_gpus()
        with ui.row().classes("gap-2 items-center"):
            ui.button(
                "Refresh GPUs", icon="refresh", on_click=_refresh_gpus
            ).props("flat")
        # Auto-refresh every 3 s so a submission tick is reflected quickly
        # without hammering nvidia-smi.
        ui.timer(3.0, _refresh_gpus)

        ui.separator().classes("mt-2")

        # ── Dataset overview ──────────────────────────────────────────────
        ui.label("Datasets").classes("text-lg font-semibold mt-2")
        ui.label(
            "All sessions on disk. Animals = total animal detections from "
            "MegaDetector (— means no inference run yet). Coordinates "
            "come from the metadata you set below."
        ).classes("text-sm text-gray-600")

        stats_container = ui.column().classes("w-full")

        def _refresh_stats() -> None:
            stats_container.clear()
            with stats_container:
                rows_data = [
                    {
                        "session": r["session"],
                        "name": r["name"] or "—",
                        "images": r["images"],
                        # None passed straight through; the column's
                        # :format directive handles null → en-dash.
                        # JSON has no NaN, so don't try float('nan').
                        "animals": r["animals"],
                        "lat": r["lat"],
                        "lng": r["lng"],
                    }
                    for r in _collect_stats()
                ]
                if not rows_data:
                    ui.label(
                        "No sessions yet — go to the Upload tab to create one."
                    ).classes("text-sm text-gray-500")
                    return
                total_images = sum(r["images"] for r in rows_data)
                total_animals = sum(
                    r["animals"] for r in rows_data if r["animals"]
                )
                ui.label(
                    f"{len(rows_data)} session(s) · {total_images:,} image(s) · "
                    f"{total_animals:,} total animal detection(s)"
                ).classes("text-sm text-gray-700 font-semibold")
                cols = [
                    {"name": "session", "label": "Session", "field": "session",
                     "align": "left", "sortable": True},
                    {"name": "name", "label": "Camera", "field": "name",
                     "align": "left", "sortable": True},
                    {"name": "images", "label": "Images", "field": "images",
                     "align": "right", "sortable": True},
                    {"name": "animals", "label": "Animals", "field": "animals",
                     "align": "right", "sortable": True,
                     ":format": "v => (v == null) ? '—' : v.toLocaleString()"},
                    {"name": "lat", "label": "Latitude", "field": "lat",
                     "align": "right", "sortable": True,
                     ":format": "v => (v == null) ? '—' : Number(v).toFixed(5)"},
                    {"name": "lng", "label": "Longitude", "field": "lng",
                     "align": "right", "sortable": True,
                     ":format": "v => (v == null) ? '—' : Number(v).toFixed(5)"},
                ]
                ui.table(columns=cols, rows=rows_data, row_key="session").classes(
                    "w-full"
                )

        with ui.row().classes("gap-2 items-center"):
            ui.button(
                "Refresh stats", icon="refresh", on_click=_refresh_stats
            ).props("flat")
        _refresh_stats()

        ui.separator().classes("mt-2")

        # ── Bulk inference (run on every session that lacks output) ───────
        ui.label("Bulk inference").classes("text-lg font-semibold mt-2")
        ui.label(
            "Submit MegaDetector / SpeciesNet / DeepFaune jobs for every "
            "session that doesn't already have those outputs. Progress for "
            "the submitted batch is shown in-line below."
        ).classes("text-sm text-gray-600")

        bulk_status = ui.label("").classes("text-sm text-gray-700 font-mono")

        # Overall progress (hidden until a batch is submitted) — shows
        # "done / total" across the whole submitted batch, regardless of
        # pipeline. Polled every 0.5s from ``_poll_bulk`` alongside the
        # per-job rows below.
        overall_row = ui.row().classes("items-center gap-2 mt-2 w-full")
        with overall_row:
            overall_label = ui.label("").classes(
                "text-sm text-gray-700 font-semibold w-48"
            )
            overall_bar = ui.linear_progress(value=0, show_value=False).classes(
                "flex-1"
            )
        overall_row.set_visibility(False)

        # Per-job rows. Each row carries a label (state / stage) plus a
        # linear progress bar fed from ``jm.get_status(job_id)``. Scrolls
        # if there are many jobs so the Admin tab doesn't explode.
        jobs_container = ui.column().classes(
            "w-full mt-1 gap-1 max-h-96 overflow-auto"
        )

        # Mutable state shared between enqueue + poll. ``jobs`` holds
        # dicts with UI element refs so the poll tick can update them in
        # place (no re-render, no flicker). ``timer`` is the ui.timer
        # we deactivate once everything is done.
        bulk_state: dict = {"jobs": [], "timer": None}

        def _refresh_bulk() -> None:
            n_md = len(_sessions_needing("md"))
            n_sn = len(_sessions_needing("md_speciesnet"))
            n_df = len(_sessions_needing("md_deepfaune"))
            n_snf = len(_sessions_needing("speciesnet"))
            bulk_status.set_text(
                f"Pending — MegaDetector: {n_md} · "
                f"MD+SpeciesNet: {n_sn} · "
                f"MD+DeepFaune: {n_df} · "
                f"SpeciesNet (full): {n_snf}"
            )

        def _poll_bulk() -> None:
            """Tick every 0.5s while jobs are outstanding. Pulls each
            job's latest status and updates its row + the overall bar."""
            jobs = bulk_state["jobs"]
            if not jobs:
                return
            done = 0
            for j in jobs:
                if j["done"]:
                    done += 1
                    continue
                s = jm.get_status(j["id"])
                state = s.get("state", "unknown")
                progress = float(s.get("progress", 0.0))
                stage = s.get("stage")
                processed = s.get("processed")
                total = s.get("total")
                worker = s.get("worker", "-")
                j["gpu_lbl"].text = worker
                if (
                    state == "running"
                    and stage
                    and processed is not None
                    and total
                ):
                    j["status_lbl"].text = f"{stage}: {processed}/{total}"
                else:
                    j["status_lbl"].text = state
                j["bar"].value = progress
                if state in ("done", "error"):
                    j["done"] = True
                    j["bar"].value = 1.0 if state == "done" else progress
                    done += 1
            total_n = len(jobs)
            overall_bar.value = done / total_n if total_n else 0.0
            overall_label.text = f"{done} / {total_n} done"
            if done == total_n:
                t = bulk_state["timer"]
                if t is not None:
                    try:
                        t.deactivate()
                    except Exception:  # noqa: BLE001
                        pass
                    bulk_state["timer"] = None
                _refresh_bulk()
                _refresh_stats()

        def _start_bulk_polling(submitted: list[tuple[str, str, str]]) -> None:
            """Build per-job rows for ``submitted`` = [(id, mode, session)]
            and (re)start the poll timer."""
            jobs_container.clear()
            bulk_state["jobs"] = []
            with jobs_container:
                for jid, mode, session in submitted:
                    # Assigned GPU is recorded by ``JobManager.submit`` so
                    # we can show it even while the job is still queued.
                    assigned = jm.get_status(jid).get("worker", "-")
                    with ui.row().classes("items-center gap-2 w-full"):
                        ui.label(mode).classes(
                            "text-xs font-mono w-32 truncate"
                        )
                        ui.label(session).classes(
                            "text-xs truncate flex-1"
                        )
                        gpu_lbl = ui.label(assigned).classes(
                            "text-xs font-mono text-gray-700 w-20 truncate"
                        )
                        status_lbl = ui.label("queued").classes(
                            "text-xs text-gray-600 w-56 truncate"
                        )
                        bar = ui.linear_progress(
                            value=0, show_value=False
                        ).classes("w-40")
                    bulk_state["jobs"].append(
                        {
                            "id": jid,
                            "mode": mode,
                            "session": session,
                            "status_lbl": status_lbl,
                            "gpu_lbl": gpu_lbl,
                            "bar": bar,
                            "done": False,
                        }
                    )
            overall_row.set_visibility(True)
            overall_label.text = f"0 / {len(submitted)} done"
            overall_bar.value = 0.0
            old_t = bulk_state["timer"]
            if old_t is not None:
                try:
                    old_t.deactivate()
                except Exception:  # noqa: BLE001
                    pass
            bulk_state["timer"] = ui.timer(0.5, _poll_bulk)

        def _enqueue(mode: str, mode_key: str, **extra) -> None:
            sessions = _sessions_needing(mode_key)
            if not sessions:
                ui.notify(
                    f"Nothing to do for {mode} — all sessions are up to date.",
                    type="info",
                )
                return
            submitted = [
                (jm.submit(mode=mode, session=s, **extra), mode, s)
                for s in sessions
            ]
            ui.notify(
                f"Queued {len(submitted)} {mode} job(s).", type="positive"
            )
            _start_bulk_polling(submitted)
            _refresh_bulk()

        def _enqueue_all() -> None:
            submitted: list[tuple[str, str, str]] = []
            for mode, mode_key, extra in (
                ("md", "md", {"conf": 0.2, "imgsz": 1280}),
                ("md+speciesnet", "md_speciesnet", {"conf": 0.2}),
                ("md+deepfaune", "md_deepfaune", {"conf": 0.2}),
                ("speciesnet", "speciesnet", {}),
            ):
                for s in _sessions_needing(mode_key):
                    submitted.append(
                        (jm.submit(mode=mode, session=s, **extra), mode, s)
                    )
            if submitted:
                ui.notify(
                    f"Queued {len(submitted)} job(s) across all three pipelines.",
                    type="positive",
                )
                _start_bulk_polling(submitted)
            else:
                ui.notify(
                    "All sessions already have all three outputs — nothing to do.",
                    type="info",
                )
            _refresh_bulk()

        with ui.row().classes("gap-2 mt-2 flex-wrap"):
            ui.button(
                "Run MD on all missing",
                icon="play_arrow",
                on_click=lambda: _enqueue(
                    "md", "md", conf=0.2, imgsz=1280
                ),
            ).props("flat bordered")
            ui.button(
                "Run MD + SpeciesNet on all missing",
                icon="play_arrow",
                on_click=lambda: _enqueue(
                    "md+speciesnet", "md_speciesnet", conf=0.2
                ),
            ).props("flat bordered")
            ui.button(
                "Run MD + DeepFaune on all missing",
                icon="play_arrow",
                on_click=lambda: _enqueue(
                    "md+deepfaune", "md_deepfaune", conf=0.2
                ),
            ).props("flat bordered")
            ui.button(
                "Run SpeciesNet (full) on all missing",
                icon="play_arrow",
                on_click=lambda: _enqueue("speciesnet", "speciesnet"),
            ).props("flat bordered")
            ui.button(
                "Run everything missing",
                icon="auto_fix_high",
                on_click=_enqueue_all,
            ).props("color=primary")
            ui.button(
                "Refresh", icon="refresh", on_click=_refresh_bulk
            ).props("flat")

        _refresh_bulk()

        ui.separator().classes("mt-2")

        # ── Per-session camera metadata ───────────────────────────────────
        ui.label("Camera metadata").classes("text-lg font-semibold mt-2")
        ui.label(
            "Per-session camera metadata. Latitude / longitude lets the "
            "Map tab plot the trap location and its detected species. "
            "'Delete dataset' permanently removes uploads, thumbnails, "
            "and all pipeline outputs for the selected session."
        ).classes("text-sm text-gray-600")

        session_select = ui.select(list_sessions(), label="Session").classes("w-96")
        session_select.on(
            "popup-show", lambda _e: session_select.set_options(list_sessions())
        )

        name_input = ui.input("Camera name").classes("w-96")
        # format="%.6f" gives ~11 cm precision at the equator — plenty for
        # camera-trap deployment, and 6 decimals is the conventional sweet
        # spot for GPS coordinates in scientific workflows.
        lat_input = ui.number(
            "Latitude", value=None, format="%.6f", min=-90.0, max=90.0
        ).classes("w-48")
        lng_input = ui.number(
            "Longitude", value=None, format="%.6f", min=-180.0, max=180.0
        ).classes("w-48")
        desc_input = ui.textarea(
            "Description",
            placeholder="Habitat, mounting height, target species…",
        ).classes("w-[32rem]")

        status_label = ui.label("").classes("text-sm text-gray-700")

        def _fill_from_session() -> None:
            session = session_select.value
            if not session:
                name_input.set_value("")
                lat_input.set_value(None)
                lng_input.set_value(None)
                desc_input.set_value("")
                status_label.set_text("")
                return
            meta = load(session)
            name_input.set_value(meta.get("name") or "")
            lat_input.set_value(meta.get("latitude"))
            lng_input.set_value(meta.get("longitude"))
            desc_input.set_value(meta.get("description") or "")
            if meta:
                status_label.set_text(f"Loaded metadata for '{session}'.")
            else:
                status_label.set_text(
                    f"No metadata yet for '{session}'. Fill in the form and Save."
                )

        session_select.on_value_change(lambda _e: _fill_from_session())

        def do_save() -> None:
            session = session_select.value
            if not session:
                ui.notify("Pick a session first", type="warning")
                return
            save(
                session,
                name=name_input.value or "",
                latitude=lat_input.value,
                longitude=lng_input.value,
                description=desc_input.value or "",
            )
            status_label.set_text(f"Saved metadata for '{session}'.")
            ui.notify("Metadata saved", type="positive")
            _refresh_stats()

        async def do_delete() -> None:
            session = session_select.value
            if not session:
                ui.notify("Pick a session first", type="warning")
                return
            with ui.dialog() as dialog, ui.card():
                ui.label(
                    f"Permanently delete '{session}'?"
                ).classes("text-lg font-semibold")
                ui.label(
                    "This removes all uploaded images, thumbnails, "
                    "MegaDetector / SpeciesNet outputs, EXIF metadata "
                    "cache, and user labels. Cannot be undone."
                ).classes("text-sm text-gray-700")
                with ui.row().classes("gap-2 mt-3"):
                    ui.button(
                        "Delete", on_click=lambda: dialog.submit("delete")
                    ).props("color=negative")
                    ui.button(
                        "Cancel", on_click=lambda: dialog.submit("cancel")
                    ).props("flat")
            choice = await dialog
            if choice != "delete":
                return
            summary = delete_session(session)
            session_select.set_options(list_sessions())
            session_select.set_value(None)
            status_label.set_text(
                f"Deleted '{session}' — "
                f"{summary['uploads']} upload(s), "
                f"{summary['thumbs']} thumb(s), "
                f"{summary['outputs']} output file(s)."
            )
            ui.notify(f"Deleted '{session}'", type="positive")
            _refresh_stats()

        with ui.row().classes("gap-2 mt-2"):
            ui.button(
                "Save metadata", icon="save", on_click=do_save
            ).props("color=primary")
            ui.button(
                "Delete dataset", icon="delete", on_click=do_delete
            ).props("color=negative")

        # ── Demo datasets ──────────────────────────────────────────────────
        ui.separator().classes("mt-6")
        ui.label("Demo datasets").classes("text-lg font-semibold mt-2")
        ui.label(
            "Course practicals use these datasets. Downloading installs "
            "them as sessions you can browse, classify, and label like any "
            "other upload. Source: LILA BC. Re-clicking is safe — the "
            "downloader caches and skips files that already exist."
        ).classes("text-sm text-gray-600")

        demo_status = ui.label("").classes("text-sm text-gray-700")

        async def _do_install(
            label: str,
            session_name: str,
            downloader,
            n_images: int,
            buttons: list,
        ) -> None:
            for b in buttons:
                b.set_enabled(False)
            demo_status.set_text(f"Downloading {label} ({n_images} images)…")
            try:
                src_dir = await run.io_bound(downloader, n_images)
                # File copy + Pillow thumbnail are CPU/IO; offload too.
                installed = await run.io_bound(
                    _install_to_session, src_dir, session_name
                )
                demo_status.set_text(
                    f"Installed {installed} image(s) into session "
                    f"'{session_name}'. Refresh the Session dropdown above "
                    f"or any other tab's session picker to see it."
                )
                # Refresh the Session selector so the new dataset shows up
                # without a manual refresh.
                session_select.set_options(list_sessions())
                ui.notify(
                    f"{label} installed as '{session_name}'",
                    type="positive",
                )
                _refresh_stats()
            except Exception as e:  # noqa: BLE001
                demo_status.set_text(f"Error: {e}")
                ui.notify(f"{label} download failed: {e}", type="negative")
            finally:
                for b in buttons:
                    b.set_enabled(True)

        # Hold button refs so each handler can disable both during work.
        demo_buttons: list = []

        # Serengeti row: count input + button. The `int(...)` cast at
        # click time means the user's latest typed value wins.
        with ui.row().classes("gap-2 mt-2 items-center"):
            srg_n = ui.number(
                "Serengeti images",
                value=100,
                min=10,
                max=2000,
                step=10,
                format="%.0f",
            ).classes("w-44")
            srg_btn = ui.button(
                "Download Serengeti",
                icon="cloud_download",
                on_click=lambda: _do_install(
                    "Snapshot Serengeti",
                    "demo_serengeti",
                    _download_serengeti_subset,
                    int(srg_n.value or 100),
                    demo_buttons,
                ),
            ).props("flat bordered")

        with ui.row().classes("gap-2 mt-2 items-center"):
            cal_n = ui.number(
                "Caltech images",
                value=50,
                min=10,
                max=2000,
                step=10,
                format="%.0f",
            ).classes("w-44")
            cal_btn = ui.button(
                "Download Caltech",
                icon="cloud_download",
                on_click=lambda: _do_install(
                    "Caltech Camera Traps",
                    "demo_caltech",
                    _download_caltech_subset,
                    int(cal_n.value or 50),
                    demo_buttons,
                ),
            ).props("flat bordered")

        demo_buttons.extend([srg_btn, cal_btn])
