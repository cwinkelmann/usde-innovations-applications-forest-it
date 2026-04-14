"""JPEG thumbnail generation for uploaded images.

Thumbnails mirror the uploads tree under THUMBS_DIR:

    /data/uploads/<session>/<name>.TIF  ->  /data/thumbs/<session>/<name>.jpg

Design choices:
- Always JPEG output: browsers render them natively, and the gallery doesn't
  care about the source format. TIFF/BMP/WEBP all become JPEG here.
- `Image.thumbnail()` resizes in-place and uses JPEG's DCT downscaling via
  `.draft()` — cheap even on multi-MB drone imagery.
- Failures are swallowed: the original upload already succeeded; a missing
  thumb is a UI glitch, not a data-loss event.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFile

from .config import THUMBS_DIR, UPLOADS_DIR

# Drone orthomosaics can exceed Pillow's default 178 MP decompression-bomb
# guard. Raise it for our controlled inputs; don't disable entirely.
Image.MAX_IMAGE_PIXELS = 500_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True

THUMB_SIZE = 256
THUMB_QUALITY = 85


def thumb_path_for(src: Path) -> Path:
    """Map an upload path to its thumbnail path.

    Resolves both paths so symlinked bind mounts still compute a correct
    relative subpath. Raises ValueError if `src` isn't under UPLOADS_DIR.
    """
    rel = src.resolve().relative_to(UPLOADS_DIR.resolve())
    return (THUMBS_DIR / rel).with_suffix(".jpg")


def make_thumbnail(src: Path, dest: Path | None = None, size: int = THUMB_SIZE) -> Path | None:
    """Generate a JPEG thumbnail for `src`. Returns the dest path or None on failure.

    Safe to call from a thread pool (`asyncio.to_thread(make_thumbnail, p)`) —
    Pillow releases the GIL for decode/encode.
    """
    dest = dest or thumb_path_for(src)
    try:
        if dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime:
            return dest  # already up to date
    except OSError:
        pass  # race with another writer; fall through and regenerate
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(src) as im:
            # draft() is a no-op for non-JPEG inputs but lets JPEG decode at a
            # reduced resolution, which is ~4x faster on big source JPEGs.
            im.draft("RGB", (size * 2, size * 2))
            im = im.convert("RGB")
            im.thumbnail((size, size))
            im.save(dest, "JPEG", quality=THUMB_QUALITY, optimize=True)
        return dest
    except Exception:
        return None