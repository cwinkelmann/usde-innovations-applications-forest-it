"""One-stop "what species is in this image" resolver.

Reconciles four sources of per-image labels with this precedence:

  1. **Label Studio export** (highest) — biologist-corrected annotations,
     loaded from ``labelstudio_speciesnet_export.json`` / ``labelstudio_export.json``.
  2. **Series tab user labels** — manual assignments from ``user_labels.json``.
  3. **Lissl ground-truth wolf labels** — only the *positive* "wolf" rows
     from ``lissl_labels.json``. The "no_wolf" rows are uninformative for
     species (they say what the image *isn't*) and are ignored here.
  4. **SpeciesNet top-1** — model fallback from ``md_speciesnet/predictions.json``.

Higher-priority sources overwrite lower-priority ones on a per-filename basis.
Any consumer that wants the "current best label" should call
``effective_labels(session)`` rather than reading the raw files directly.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .config import OUTPUTS_DIR
from .lissl_groundtruth import load_session_labels as _load_lissl_labels
from .user_labels import load_labels

# LS prefixes uploaded files as ``<8+ hex chars>-<original_name>``. Strip
# that so the keys line up with on-disk basenames everywhere else.
_LS_PREFIX_RE = re.compile(r"^[a-f0-9]{8,}-(.+)$")


def _strip_ls_prefix(name: str) -> str:
    base = Path(name).name
    m = _LS_PREFIX_RE.match(base)
    return m.group(1) if m else base


def _speciesnet_labels(session: str) -> dict[str, str]:
    p = OUTPUTS_DIR / session / "md_speciesnet" / "predictions.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except Exception:
        return {}
    out: dict[str, str] = {}
    for item in data:
        species_list = item.get("species") or []
        if species_list:
            top = species_list[0].get("common_name")
            if top:
                out[Path(item.get("file", "")).name] = top
    return out


def _ls_labels(session: str) -> dict[str, str]:
    """Parse any Label Studio COCO export for the session.

    Returns ``{basename: species}`` keyed by stripped (no UUID prefix)
    filenames. When an image has multiple annotations with different
    categories, the most-common category wins. Both the SpeciesNet-flavoured
    and the plain-MD-flavoured exports are checked; the SpeciesNet one
    takes precedence because it carries finer-grained labels.
    """
    candidates = [
        OUTPUTS_DIR / session / "labelstudio_speciesnet_export.json",
        OUTPUTS_DIR / session / "labelstudio_export.json",
    ]
    out: dict[str, str] = {}
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        cats = {
            c["id"]: c.get("name") or "?"
            for c in data.get("categories", [])
        }
        img_files = {
            img["id"]: _strip_ls_prefix(img.get("file_name") or "")
            for img in data.get("images", [])
            if img.get("id") is not None
        }
        per_image: dict[str, Counter[str]] = {}
        for ann in data.get("annotations", []):
            img_id = ann.get("image_id")
            cat_id = ann.get("category_id")
            fn = img_files.get(img_id)
            cat = cats.get(cat_id)
            if not fn or not cat:
                continue
            per_image.setdefault(fn, Counter())[cat] += 1
        for fn, counter in per_image.items():
            # Most common annotation wins (tie-broken by Counter's
            # internal insertion order — deterministic enough for this).
            top = counter.most_common(1)[0][0]
            # SpeciesNet export overwrites the plain one because it's
            # checked first in `candidates`. Use setdefault to lock in
            # the higher-priority source.
            out.setdefault(fn, top)
    return out


def _lissl_wolf_labels(session: str) -> dict[str, str]:
    """Only the ``wolf`` side of Lissl's binary ground truth — ``no_wolf``
    is uninformative for species and is intentionally dropped here."""
    return {
        fn: lbl
        for fn, lbl in _load_lissl_labels(session).items()
        if lbl == "wolf"
    }


def effective_labels(session: str) -> dict[str, str]:
    """Return ``{basename: species}`` for every image with a label,
    applying the priority chain described in the module docstring.
    """
    merged = _speciesnet_labels(session)
    merged.update(_lissl_wolf_labels(session))  # GT wolf overrides model
    merged.update(load_labels(session))         # user_labels overrides GT
    merged.update(_ls_labels(session))          # LS export overrides all
    return merged


def species_counts(session: str) -> dict[str, int]:
    """Convenience: ``{species: n_images}`` from ``effective_labels``."""
    counts: dict[str, int] = {}
    for sp in effective_labels(session).values():
        if sp:
            counts[sp] = counts.get(sp, 0) + 1
    return counts
