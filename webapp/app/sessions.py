"""Single source of truth for session listing — disk is the database."""
from __future__ import annotations

from .config import UPLOADS_DIR


def list_sessions() -> list[str]:
    """Return existing session folder names, newest first."""
    if not UPLOADS_DIR.exists():
        return []
    dirs = [p for p in UPLOADS_DIR.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in dirs]
