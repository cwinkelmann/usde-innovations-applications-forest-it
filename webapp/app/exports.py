"""CSV converters for downloads from the MegaDetector tab.

Both converters produce one row per box; an image with zero detections emits
a single row with empty bbox cells so it still appears in the CSV.
"""
from __future__ import annotations

import csv
import io
import json
import re
from collections import defaultdict
from pathlib import Path

_LS_UUID = re.compile(r"^[0-9a-f]{6,}-(.+)$")


def _strip_ls_prefix(name: str) -> str:
    m = _LS_UUID.match(name)
    return m.group(1) if m else name


def md_detections_to_csv(md_json: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(
        ["file", "width", "height", "det_idx", "label", "conf", "x1", "y1", "x2", "y2"]
    )
    for item in md_json:
        fname = Path(item["file"]).name
        width = item.get("width", "")
        height = item.get("height", "")
        dets = item.get("detections", [])
        if not dets:
            w.writerow([fname, width, height, 0, "", "", "", "", "", ""])
            continue
        for i, d in enumerate(dets):
            x1, y1, x2, y2 = d["bbox_xyxy"]
            w.writerow(
                [
                    fname,
                    width,
                    height,
                    i,
                    d.get("label", ""),
                    f"{d.get('conf', 0):.4f}",
                    f"{x1:.1f}",
                    f"{y1:.1f}",
                    f"{x2:.1f}",
                    f"{y2:.1f}",
                ]
            )
    return buf.getvalue()


def coco_annotations_to_csv(coco_json: dict) -> str:
    cats = {c["id"]: c["name"] for c in coco_json.get("categories", [])}
    images = {img["id"]: img for img in coco_json.get("images", [])}
    ann_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco_json.get("annotations", []):
        ann_by_image[ann["image_id"]].append(ann)

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["file", "width", "height", "ann_idx", "label", "x1", "y1", "x2", "y2"])
    for img_id, img in images.items():
        fname = _strip_ls_prefix(Path(img["file_name"]).name)
        width = img.get("width", "")
        height = img.get("height", "")
        anns = ann_by_image.get(img_id, [])
        if not anns:
            w.writerow([fname, width, height, 0, "", "", "", "", ""])
            continue
        for i, a in enumerate(anns):
            x, y, ww, hh = a["bbox"]
            w.writerow(
                [
                    fname,
                    width,
                    height,
                    i,
                    cats.get(a["category_id"], "?"),
                    f"{x:.1f}",
                    f"{y:.1f}",
                    f"{x + ww:.1f}",
                    f"{y + hh:.1f}",
                ]
            )
    return buf.getvalue()


def md_json_to_csv_file(in_path: Path, out_path: Path) -> None:
    out_path.write_text(md_detections_to_csv(json.loads(Path(in_path).read_text())))


def coco_json_to_csv_file(in_path: Path, out_path: Path) -> None:
    out_path.write_text(
        coco_annotations_to_csv(json.loads(Path(in_path).read_text()))
    )


MD_CLASSES = ("animal", "person", "vehicle")


def md_per_image_csv(md_json: list[dict], with_species: bool = False) -> str:
    """One row per image; columns: file, animal, person, vehicle, empty
    (+ top_species, top_species_score when `with_species`).

    animal/person/vehicle = count of detections of that class.
    empty = 1 if the image has zero detections, else 0.
    """
    buf = io.StringIO()
    w = csv.writer(buf)
    header = ["file", *MD_CLASSES, "empty"]
    if with_species:
        header += ["top_species", "top_species_score"]
    w.writerow(header)
    for item in md_json:
        fname = Path(item["file"]).name
        counts = {cls: 0 for cls in MD_CLASSES}
        for d in item.get("detections", []):
            if d.get("label") in counts:
                counts[d["label"]] += 1
        total = sum(counts.values())
        row = [fname, *(counts[c] for c in MD_CLASSES), int(total == 0)]
        if with_species:
            species = item.get("species") or []
            if species:
                top = species[0]
                row += [top.get("common_name", ""), f"{top.get('score', 0):.4f}"]
            else:
                row += ["", ""]
        w.writerow(row)
    return buf.getvalue()


def images_by_class(md_json: list[dict]) -> dict[str, list[str]]:
    """Map each class (plus 'empty') → list of absolute image paths.

    An image can appear under multiple classes (e.g., animal + vehicle).
    """
    result: dict[str, list[str]] = {cls: [] for cls in MD_CLASSES}
    result["empty"] = []
    for item in md_json:
        labels = {d.get("label") for d in item.get("detections", [])}
        if not labels:
            result["empty"].append(item["file"])
            continue
        for cls in MD_CLASSES:
            if cls in labels:
                result[cls].append(item["file"])
    return result


def build_class_zip(md_json: list[dict], class_name: str) -> bytes:
    """Return a zip (in-memory) of every image whose MD detections include
    ``class_name`` (or that are detection-free when ``class_name=='empty'``)."""
    import zipfile

    by_class = images_by_class(md_json)
    paths = by_class.get(class_name, [])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        # ZIP_STORED: the images are already JPEG/PNG-compressed; re-DEFLATE
        # would burn CPU for near-zero space savings.
        for p in paths:
            src = Path(p)
            if src.exists():
                zf.write(src, arcname=src.name)
    return buf.getvalue()
