"""Lissl ground-truth wolf / no-wolf labels via SHA-256 lookup.

Lissl's CSV files (placed in ``LISSL_DIR``) carry per-image SHA-256
checksums in a ``checksum`` column. Each CSV is associated with a
binary label — "wolf" for files in the wolf set, "no_wolf" for files
in the no-wolf set. Loading them into one in-memory dict keyed by
checksum lets us look up any uploaded image by content hash, totally
independent of filename or path.

Per-session results (``{filename: label}``) are persisted to
``OUTPUTS_DIR/<session>/lissl_labels.json`` so the Evaluation tab and
the effective-labels resolver can reuse them without re-hashing.

The CSV index is loaded once per process and cached. ~45 MB of RAM and
~50 s cold-start for the full ~600K-row no-wolf set; lookups are O(1).
"""
from __future__ import annotations

import csv
import hashlib
import json
import threading
from pathlib import Path

from .config import LISSL_DIR, OUTPUTS_DIR, UPLOADS_DIR

# CSV → label. wolf is loaded *after* no_wolf so any contested checksum
# lands as wolf — the rarer, more interesting class.
_CSV_LABELS: dict[str, str] = {
    "no_wolf.csv": "no_wolf",
    "no_wolf_old.csv": "no_wolf",
    "wb_wolf_dt.csv": "wolf",
    "missing_wolf.csv": "wolf",
}

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

_index: dict[str, str] = {}
_index_loaded = False
_index_lock = threading.Lock()


def _load_index() -> dict[str, str]:
    """Build ``{sha256: label}`` from the CSVs in ``LISSL_DIR``. Memoised."""
    global _index_loaded
    with _index_lock:
        if _index_loaded:
            return _index
        for fname, label in _CSV_LABELS.items():
            path = LISSL_DIR / fname
            if not path.exists():
                continue
            with path.open(newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or "checksum" not in reader.fieldnames:
                    continue
                for row in reader:
                    cs = (row.get("checksum") or "").strip()
                    if cs:
                        _index[cs] = label
        _index_loaded = True
        return _index


def index_size() -> int:
    return len(_load_index())


def lookup(sha256_hex: str) -> str | None:
    """Return the ground-truth label for a content hash, or None."""
    return _load_index().get(sha256_hex)


def sha256_file(path: Path) -> str:
    """Stream-compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def labels_path(session: str) -> Path:
    return OUTPUTS_DIR / session / "lissl_labels.json"


def load_session_labels(session: str) -> dict[str, str]:
    """Cached per-session labels written by ``tag_session``."""
    p = labels_path(session)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def tag_session(session: str) -> dict[str, int]:
    """Hash every image in the session, look up in the Lissl index, write
    ``lissl_labels.json``. Returns a summary count dict.

    Synchronous and CPU/IO-bound — wrap in ``run.io_bound`` from a tab
    handler. ~600 images/min on a typical SSD + modest CPU.
    """
    src = UPLOADS_DIR / session
    summary = {"images": 0, "tagged": 0, "wolf": 0, "no_wolf": 0}
    if not src.exists():
        return summary
    index = _load_index()
    out: dict[str, str] = {}
    for p in sorted(src.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        if p.suffix.lower() not in _IMAGE_EXTS:
            continue
        summary["images"] += 1
        try:
            digest = sha256_file(p)
        except OSError:
            continue
        label = index.get(digest)
        if label:
            out[p.name] = label
            summary["tagged"] += 1
            summary[label] = summary.get(label, 0) + 1
    target = labels_path(session)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2, sort_keys=True))
    return summary
