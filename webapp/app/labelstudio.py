"""Label Studio bridge for the webapp.

Converts webapp `detections.json` entries (pixel xyxy, category_id 0/1/2)
into the normalised-bbox format expected by
`wildlife_detection.label_studio.LabelStudioProject.upload_with_megadetector`,
and delegates the HTTP work to that helper.

Runs inside the NiceGUI main process under `run.io_bound(...)` so the
asyncio event loop isn't blocked by uploads.
"""
from __future__ import annotations

import json
from pathlib import Path

_MD_CATEGORY = {0: "1", 1: "2", 2: "3"}  # webapp id → MD category string


def _webapp_to_md(detections: list[dict]) -> list[dict]:
    out: list[dict] = []
    for item in detections:
        w, h = item["width"], item["height"]
        dets = []
        for d in item.get("detections", []):
            x1, y1, x2, y2 = d["bbox_xyxy"]
            dets.append(
                {
                    "category": _MD_CATEGORY.get(d["category_id"], "1"),
                    "conf": d["conf"],
                    "bbox": [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h],
                }
            )
        # Forward width/height so upload_with_megadetector can skip the
        # per-image PIL decode used by get_image_size().
        out.append(
            {
                "file": Path(item["file"]).name,
                "width": w,
                "height": h,
                "detections": dets,
            }
        )
    return out


def build_species_map(predictions_path: Path) -> dict[str, str]:
    """Build {filename: species_label} from a SpeciesNet predictions.json.

    Species labels use underscores (LS label-config friendly). Falls back to
    'animal' if an image has no species prediction.
    """
    data = json.loads(Path(predictions_path).read_text())
    sm: dict[str, str] = {}
    for item in data:
        species = item.get("species") or []
        fname = Path(item["file"]).name
        if species:
            top = species[0]
            raw = (top.get("species") or top.get("common_name") or "animal").strip()
            sm[fname] = raw.replace(" ", "_") or "animal"
    return sm


def export_session(
    ls_url: str,
    token: str,
    project_name: str,
    session_dir: Path,
    detections_path: Path,
    species_map: dict[str, str] | None = None,
    file_allowlist: set[str] | None = None,
) -> str:
    """Create/reuse the LS project, upload session images + MD pre-annotations.

    If ``species_map`` is provided, animal boxes are labelled with the per-image
    species name instead of the generic 'animal', and the LS project config is
    populated with the full label set (species ∪ person ∪ vehicle).

    If ``file_allowlist`` is provided, only images whose basename is in the
    set are uploaded. Typically fed by the gallery's current filter so the
    export reflects what the user sees on screen (e.g. occupied-only, or a
    specific class).
    """
    from wildlife_detection.label_studio import (
        LabelStudioProject,
        make_bbox_config,
        make_session,
    )

    http = make_session(token, url=ls_url)
    md_results = _webapp_to_md(json.loads(Path(detections_path).read_text()))

    if file_allowlist is not None:
        md_results = [r for r in md_results if r["file"] in file_allowlist]

    if species_map:
        labels = sorted(set(species_map.values()) | {"person", "vehicle"})
    else:
        labels = ["animal", "person", "vehicle"]
    config = make_bbox_config(labels)

    proj = LabelStudioProject.get_or_create(http, ls_url, project_name, config=config)
    proj.upload_with_megadetector(
        Path(session_dir), md_results, species_map=species_map
    )
    return f"{ls_url.rstrip('/')}/projects/{proj.id}"


def import_session(
    ls_url: str,
    token: str,
    project_name: str,
    output_path: Path,
) -> int:
    """Export the LS project's COCO annotations to `output_path`. Returns the
    number of annotations written (0 if nothing has been submitted yet)."""
    from wildlife_detection.label_studio import LabelStudioProject, make_session

    http = make_session(token, url=ls_url)
    proj = LabelStudioProject.get_or_create(http, ls_url, project_name)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    proj.export(out, fmt="COCO")
    if out.exists():
        try:
            data = json.loads(out.read_text())
            return len(data.get("annotations", []))
        except Exception:
            return 0
    return 0
