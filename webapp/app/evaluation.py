"""Compare MegaDetector predictions against human-corrected Label Studio
annotations. Returns precision / recall / F1 at a given IoU threshold.

Matching is greedy per image: each human box is matched to the predicted
box of compatible class with highest IoU (above threshold). Unmatched
human boxes are false negatives; unmatched predictions are false positives.

Class compatibility rule:
  - Exact label match always counts
  - Any non-{person, vehicle} human label also matches a predicted 'animal'
    (so MD's coarse 'animal' class is considered correct when the reviewer
    refined it to a species)
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


_LS_UUID_PREFIX = re.compile(r"^[0-9a-f]{6,}-(.+)$")


def _strip_ls_prefix(name: str) -> str:
    """Label Studio names every upload '<uuid8-or-more>-<original>.ext'."""
    m = _LS_UUID_PREFIX.match(name)
    return m.group(1) if m else name


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _classes_match(pred_label: str, human_label: str) -> bool:
    if pred_label == human_label:
        return True
    if pred_label == "animal" and human_label not in {"person", "vehicle"}:
        return True
    return False


def evaluate(
    md_detections: list[dict],
    ls_coco: dict,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    cats = {c["id"]: c["name"] for c in ls_coco.get("categories", [])}
    images = {img["id"]: img for img in ls_coco.get("images", [])}

    human_by_file: dict[str, list[dict]] = defaultdict(list)
    for ann in ls_coco.get("annotations", []):
        img = images.get(ann["image_id"])
        if not img:
            continue
        name = _strip_ls_prefix(Path(img["file_name"]).name)
        x, y, w, h = ann["bbox"]
        human_by_file[name].append(
            {
                "label": cats.get(ann["category_id"], "?"),
                "bbox_xyxy": [float(x), float(y), float(x + w), float(y + h)],
            }
        )

    md_by_file: dict[str, list[dict]] = {}
    for r in md_detections:
        name = Path(r["file"]).name
        md_by_file[name] = [
            {"label": d["label"], "bbox_xyxy": list(d["bbox_xyxy"])}
            for d in r.get("detections", [])
        ]

    tp = fp = fn = 0
    per_class: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )

    for fname in set(md_by_file) | set(human_by_file):
        preds = md_by_file.get(fname, [])
        humans = human_by_file.get(fname, [])
        matched_preds: set[int] = set()

        for h in humans:
            best_i, best_iou = -1, 0.0
            for pi, p in enumerate(preds):
                if pi in matched_preds:
                    continue
                if not _classes_match(p["label"], h["label"]):
                    continue
                i = _iou(h["bbox_xyxy"], p["bbox_xyxy"])
                if i >= iou_threshold and i > best_iou:
                    best_iou, best_i = i, pi
            if best_i >= 0:
                matched_preds.add(best_i)
                tp += 1
                per_class[h["label"]]["tp"] += 1
            else:
                fn += 1
                per_class[h["label"]]["fn"] += 1

        for pi, p in enumerate(preds):
            if pi not in matched_preds:
                fp += 1
                per_class[p["label"]]["fp"] += 1

    def _pr(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f

    precision, recall, f1 = _pr(tp, fp, fn)

    per_class_rows = []
    for cls, c in sorted(per_class.items()):
        p, r, f = _pr(c["tp"], c["fp"], c["fn"])
        per_class_rows.append(
            {
                "class": cls,
                "tp": c["tp"],
                "fp": c["fp"],
                "fn": c["fn"],
                "precision": p,
                "recall": r,
                "f1": f,
            }
        )

    return {
        "iou_threshold": iou_threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_predicted_boxes": sum(len(v) for v in md_by_file.values()),
        "n_human_boxes": sum(len(v) for v in human_by_file.values()),
        "n_images_md": len(md_by_file),
        "n_images_human": len(human_by_file),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class": per_class_rows,
    }


def evaluate_paths(
    md_detections_path: Path, ls_coco_path: Path, iou_threshold: float = 0.5
) -> dict:
    md = json.loads(Path(md_detections_path).read_text())
    ls = json.loads(Path(ls_coco_path).read_text())
    return evaluate(md, ls, iou_threshold)


def diff_boxes(
    md_detections: list[dict],
    ls_coco: dict,
    iou_threshold: float = 0.5,
) -> list[dict]:
    """Return per-image diffs for visualisation.

    Each item:
        {
          "file": "<absolute path to image>",
          "width": int, "height": int,
          "tp": [bbox_xyxy],   # matched — both predicted and kept by human
          "fp": [bbox_xyxy],   # predicted but human removed / disagreed
          "fn": [bbox_xyxy],   # human box MD missed
        }
    Only images present in both MD and LS are returned (union isn't useful
    for side-by-side inspection).
    """
    cats = {c["id"]: c["name"] for c in ls_coco.get("categories", [])}
    images = {img["id"]: img for img in ls_coco.get("images", [])}

    # Index MD by basename + keep the absolute path and dims so the gallery
    # can build a URL and draw overlays.
    md_by_name: dict[str, dict] = {}
    for r in md_detections:
        name = Path(r["file"]).name
        md_by_name[name] = {
            "file": r["file"],
            "width": r["width"],
            "height": r["height"],
            "dets": [
                {"label": d["label"], "bbox_xyxy": list(d["bbox_xyxy"])}
                for d in r.get("detections", [])
            ],
        }

    human_by_name: dict[str, list[dict]] = defaultdict(list)
    for ann in ls_coco.get("annotations", []):
        img = images.get(ann["image_id"])
        if not img:
            continue
        name = _strip_ls_prefix(Path(img["file_name"]).name)
        x, y, w, h = ann["bbox"]
        human_by_name[name].append(
            {
                "label": cats.get(ann["category_id"], "?"),
                "bbox_xyxy": [float(x), float(y), float(x + w), float(y + h)],
            }
        )

    out: list[dict] = []
    for name, md_info in md_by_name.items():
        if name not in human_by_name:
            continue
        preds = md_info["dets"]
        humans = human_by_name[name]
        matched_preds: set[int] = set()
        tp_boxes: list[list[float]] = []
        fn_boxes: list[list[float]] = []

        for h in humans:
            best_i, best_iou = -1, 0.0
            for pi, p in enumerate(preds):
                if pi in matched_preds:
                    continue
                if not _classes_match(p["label"], h["label"]):
                    continue
                i = _iou(h["bbox_xyxy"], p["bbox_xyxy"])
                if i >= iou_threshold and i > best_iou:
                    best_iou, best_i = i, pi
            if best_i >= 0:
                matched_preds.add(best_i)
                tp_boxes.append(preds[best_i]["bbox_xyxy"])
            else:
                fn_boxes.append(h["bbox_xyxy"])

        fp_boxes = [
            preds[pi]["bbox_xyxy"] for pi in range(len(preds)) if pi not in matched_preds
        ]

        if tp_boxes or fp_boxes or fn_boxes or humans or preds:
            out.append(
                {
                    "file": md_info["file"],
                    "width": md_info["width"],
                    "height": md_info["height"],
                    "tp": tp_boxes,
                    "fp": fp_boxes,
                    "fn": fn_boxes,
                }
            )
    return out


def diff_paths(
    md_detections_path: Path, ls_coco_path: Path, iou_threshold: float = 0.5
) -> list[dict]:
    md = json.loads(Path(md_detections_path).read_text())
    ls = json.loads(Path(ls_coco_path).read_text())
    return diff_boxes(md, ls, iou_threshold)
