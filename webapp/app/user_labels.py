"""User-assigned species labels per session.

Labels are stored keyed by image basename, not by series id, because the
Series tab's series boundaries change when the user tunes the gap slider.
A per-image store is invariant to that — a series that later splits in
two still shows both halves as labelled, and a series that later merges
into a neighbour still shows its half as labelled.

On-disk layout: ``/data/outputs/<session>/user_labels.json``
Shape:           ``{"RCNX0001.JPG": "grey wolf", "RCNX0002.JPG": "grey wolf", ...}``
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import OUTPUTS_DIR


def _labels_path(session: str) -> Path:
    return OUTPUTS_DIR / session / "user_labels.json"


def load_labels(session: str) -> dict[str, str]:
    p = _labels_path(session)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items() if v}
    except Exception:
        pass
    return {}


def _save(session: str, labels: dict[str, str]) -> None:
    out = _labels_path(session)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Drop empty strings so "" never round-trips as a real label.
    clean = {k: v for k, v in labels.items() if v}
    out.write_text(json.dumps(clean, indent=2, sort_keys=True))


def label_images(session: str, filenames: list[str], species: str) -> None:
    """Assign ``species`` to every filename in the list (merged into existing)."""
    labels = load_labels(session)
    for fn in filenames:
        if fn:
            labels[fn] = species
    _save(session, labels)


def clear_images(session: str, filenames: list[str]) -> None:
    """Remove any label for the given filenames."""
    labels = load_labels(session)
    for fn in filenames:
        labels.pop(fn, None)
    _save(session, labels)


def series_assigned_species(
    series_filenames: list[str], labels: dict[str, str]
) -> str | None:
    """Return the single user-assigned species if *all* images in the series
    share it; otherwise None. Mixed labels within a series mean the user
    hasn't made a consistent call, so the UI shows nothing."""
    if not series_filenames:
        return None
    seen = {labels.get(fn) for fn in series_filenames}
    seen.discard(None)
    if len(seen) == 1:
        return next(iter(seen))
    return None
