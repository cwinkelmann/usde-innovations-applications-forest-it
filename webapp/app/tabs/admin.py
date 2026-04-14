"""Tab — Admin: camera-trap dataset maintenance.

Assign GPS coordinates + description to a session (surfaces on the Map tab),
permanently delete a dataset (removes uploads, thumbnails, outputs), and
download the demo datasets used by the course practicals (Snapshot
Serengeti, Caltech Camera Traps) as ready-to-use sessions.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from nicegui import run, ui

from ..camera_meta import delete_session, load, save
from ..config import DATA_DIR, UPLOADS_DIR
from ..sessions import list_sessions
from ..thumbs import make_thumbnail

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


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


def render() -> None:
    with ui.column().classes("w-full gap-4"):
        ui.label("Admin").classes("text-xl font-semibold")
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
