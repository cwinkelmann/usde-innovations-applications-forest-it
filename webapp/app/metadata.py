"""EXIF metadata extraction for uploaded session images.

Reads per-image timestamps from JPEG EXIF (``DateTimeOriginal``, falling
back to ``DateTime``) and caches them to
``/data/outputs/<session>/metadata.json`` so subsequent reads are cheap.
Used by the Series tab to group images into time-based bursts.

Reconyx HyperFire cameras (the target hardware) write timestamps in the
standard EXIF format ``"YYYY:MM:DD HH:MM:SS"``. Images without a valid
timestamp are included in the cache with ``datetime=None`` so the Series
tab can still bucket them into a "no-timestamp" pseudo-series.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from PIL import Image

from .config import OUTPUTS_DIR, UPLOADS_DIR

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# EXIF tag numeric IDs — used instead of Pillow's TAGS dict so we don't
# depend on the Pillow version's naming conventions.
_TAG_DATETIME_ORIGINAL = 36867  # ExifIFD:DateTimeOriginal
_TAG_DATETIME = 306             # IFD0:DateTime (fallback)
_TAG_MAKE = 271
_TAG_MODEL = 272
_EXIF_IFD = 0x8769

_DT_FORMATS = ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S")


def _parse_dt(raw: str) -> datetime | None:
    for fmt in _DT_FORMATS:
        try:
            return datetime.strptime(raw.strip(), fmt)
        except (ValueError, TypeError):
            continue
    return None


def extract_image_metadata(img_path: Path) -> dict:
    """Read ``file``, ``datetime`` (ISO string or None), and camera
    make/model from an image. Non-fatal — returns a minimal record on IO
    or decode errors so a single corrupt file can't derail a whole-session
    scan.
    """
    rec: dict = {
        "file": img_path.name,
        "datetime": None,
        "make": None,
        "model": None,
    }
    try:
        with Image.open(img_path) as im:
            exif = im.getexif()
            raw_dt = None
            try:
                exif_ifd = exif.get_ifd(_EXIF_IFD)
                raw_dt = exif_ifd.get(_TAG_DATETIME_ORIGINAL)
            except Exception:
                pass
            if not raw_dt:
                raw_dt = exif.get(_TAG_DATETIME)
            if raw_dt:
                dt = _parse_dt(str(raw_dt))
                if dt:
                    rec["datetime"] = dt.isoformat()
            rec["make"] = str(exif.get(_TAG_MAKE, "") or "").strip() or None
            rec["model"] = str(exif.get(_TAG_MODEL, "") or "").strip() or None
    except Exception:
        pass
    return rec


def _metadata_path(session: str) -> Path:
    return OUTPUTS_DIR / session / "metadata.json"


def load_metadata(session: str) -> list[dict] | None:
    """Return the cached metadata list, or None if it doesn't exist yet."""
    p = _metadata_path(session)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _iter_session_images(session: str) -> list[Path]:
    src_dir = UPLOADS_DIR / session
    if not src_dir.exists():
        return []
    return sorted(
        p
        for p in src_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and not p.name.startswith(".")
    )


def build_metadata(session: str) -> list[dict]:
    """Walk the session directory, extract EXIF from each image, write
    ``metadata.json``, and return the list.

    Synchronous — callers should wrap in ``asyncio.to_thread`` for
    sessions where the accumulated PIL opens take more than a second.
    ~30 s for 3000 images on a modern laptop.
    """
    images = _iter_session_images(session)
    records = [extract_image_metadata(p) for p in images]
    out_path = _metadata_path(session)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2))
    return records
