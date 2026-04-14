"""Tab 1 — Upload images into a named session folder under /data/uploads/."""
from __future__ import annotations

import re
import time
from pathlib import Path

from nicegui import events, ui

from ..config import UPLOADS_DIR
from ..sessions import list_sessions

_SAFE = re.compile(r"[^a-zA-Z0-9_-]+")


def _sanitize(name: str) -> str:
    name = _SAFE.sub("_", name).strip("_")
    return name or f"session_{int(time.time())}"


def _thumb_paths(session: str, limit: int = 60) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    d = UPLOADS_DIR / session
    if not d.exists():
        return []
    files = sorted(
        (p for p in d.iterdir() if p.suffix.lower() in exts),
        key=lambda p: p.stat().st_mtime,
    )
    return files[-limit:]


def render() -> None:
    with ui.column().classes("w-full gap-4"):
        ui.label("Upload images").classes("text-xl font-semibold")
        ui.label(
            "Sessions are just folders under /data/uploads/. Pick an existing "
            "session to add to it, or type a new name."
        ).classes("text-sm text-gray-600")

        default_name = f"session_{time.strftime('%Y%m%d_%H%M%S')}"
        session_input = ui.input(
            label="Session name",
            value=default_name,
            autocomplete=list_sessions(),
        ).classes("w-96")

        thumb_grid = ui.row().classes("w-full gap-2 flex-wrap")

        def load_thumbs_for(session: str) -> None:
            thumb_grid.clear()
            for p in _thumb_paths(session):
                rel = p.relative_to(UPLOADS_DIR).as_posix()
                with thumb_grid:
                    ui.image(f"/uploads/{rel}").classes(
                        "w-[120px] h-[120px] object-cover rounded"
                    ).tooltip(p.name)

        load_thumbs_for(session_input.value or "")
        session_input.on(
            "change",
            lambda _e: (
                session_input.set_autocomplete(list_sessions()),
                load_thumbs_for(_sanitize(session_input.value or "")),
            ),
        )

        upload_log = ui.log(max_lines=20).classes("h-32 w-full")

        async def handle_upload(e: events.UploadEventArguments) -> None:
            session = _sanitize(session_input.value or "")
            target_dir = UPLOADS_DIR / session
            target_dir.mkdir(parents=True, exist_ok=True)
            fname = Path(e.file.name).name or "upload.bin"
            dest = target_dir / fname
            await e.file.save(dest)
            upload_log.push(f"Saved {dest.relative_to(UPLOADS_DIR)}")
            rel = dest.relative_to(UPLOADS_DIR).as_posix()
            with thumb_grid:
                ui.image(f"/uploads/{rel}").classes(
                    "w-[120px] h-[120px] object-cover rounded"
                ).tooltip(dest.name)
            session_input.set_autocomplete(list_sessions())

        ui.upload(
            on_upload=handle_upload,
            multiple=True,
            auto_upload=True,
            label="Drop JPG/PNG/TIFF here",
        ).props('accept="image/*" flat bordered').classes("w-full")
