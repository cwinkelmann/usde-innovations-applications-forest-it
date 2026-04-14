"""Per-session camera metadata (GPS + description) + dataset deletion.

Camera metadata lives at ``/data/outputs/<session>/camera_meta.json`` so
that deleting a session's output directory (via ``delete_session``) wipes
its metadata automatically — no separate global registry to keep in sync.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .config import OUTPUTS_DIR, THUMBS_DIR, UPLOADS_DIR


def _meta_path(session: str) -> Path:
    return OUTPUTS_DIR / session / "camera_meta.json"


def load(session: str) -> dict:
    p = _meta_path(session)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save(
    session: str,
    *,
    name: str = "",
    latitude: float | None = None,
    longitude: float | None = None,
    description: str = "",
) -> None:
    """Write ``camera_meta.json`` for a session. Empty / None fields are
    stored as-is so the admin form can clear them."""
    payload = {
        "name": name or "",
        "latitude": float(latitude) if latitude is not None else None,
        "longitude": float(longitude) if longitude is not None else None,
        "description": description or "",
    }
    p = _meta_path(session)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))


def delete_session(session: str) -> dict[str, int]:
    """Remove all per-session directories. Returns a summary of files
    deleted under each root. Never raises on missing paths.
    """
    summary = {"uploads": 0, "thumbs": 0, "outputs": 0}
    for root, key in (
        (UPLOADS_DIR / session, "uploads"),
        (THUMBS_DIR / session, "thumbs"),
        (OUTPUTS_DIR / session, "outputs"),
    ):
        if root.exists():
            summary[key] = sum(1 for p in root.rglob("*") if p.is_file())
            shutil.rmtree(root, ignore_errors=True)
    return summary
