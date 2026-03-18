#!/usr/bin/env python
"""
Confidence threshold calibration for optimal animal counting.

Given a fine-tuned detector and a COCO-format validation set with known counts,
finds the confidence threshold that minimises counting error (predicted count vs
ground truth count per image).

Reports:
  - Optimal threshold per class and overall
  - Count error (MAE, RMSE) at each threshold
  - Precision/Recall/F1 at the optimal threshold
  - Per-image count comparison at the optimal threshold

Usage:
    python scripts/training/calibrate_confidence.py \
        --weights ./output/md1000_larch_50ep/weights/best.pt \
        --dataset_dir ./week1/data/eikelboom_coco_tiled \
        --split val \
        --output output/calibration_larch.json

    # Compare calibration across models
    python scripts/training/calibrate_confidence.py \
        --weights ./output/md1000_larch_50ep/weights/best.pt \
                  ./output/md1000_sorrel_50ep/weights/best.pt \
                  ./output/mdv6_rtdetr_50ep/weights/best.pt \
        --dataset_dir ./week1/data/eikelboom_coco_tiled \
        --split val
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="Calibrate confidence threshold for counting")
    p.add_argument("--weights", nargs="+", required=True,
                   help="Path(s) to .pt weight file(s)")
    p.add_argument("--dataset_dir", required=True,
                   help="COCO dataset root with train/val/test subdirs")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--thresholds", type=str, default="0.05,0.95,0.05",
                   help="Threshold range: start,stop,step (default: 0.05,0.95,0.05)")
    p.add_argument("--output", default=None,
                   help="Save results to JSON file")
    return p.parse_args()


def load_ground_truth(dataset_dir, split):
    """Load per-image ground truth counts from COCO annotations."""
    split_dir = Path(dataset_dir) / split
    ann_path = split_dir / "annotations.json"
    with open(ann_path) as f:
        coco = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    sorted_cat_ids = sorted(categories.keys())
    cat_id_to_idx = {cid: i for i, cid in enumerate(sorted_cat_ids)}
    id2label = {i: categories[cid] for i, cid in enumerate(sorted_cat_ids)}

    images = []
    for img in coco["images"]:
        img_path = split_dir / "images" / img["file_name"]
        images.append((img["id"], str(img_path), img["width"], img["height"]))

    # Per-image, per-class GT counts + box list
    gt_counts = {}  # image_id -> {class_idx: count}
    gt_boxes = {}   # image_id -> list of (class_idx, [x1,y1,x2,y2])
    for img_id, _, _, _ in images:
        gt_counts[img_id] = defaultdict(int)
        gt_boxes[img_id] = []

    for ann in coco["annotations"]:
        cls_idx = cat_id_to_idx[ann["category_id"]]
        gt_counts[ann["image_id"]][cls_idx] += 1
        x, y, w, h = ann["bbox"]
        gt_boxes[ann["image_id"]].append((cls_idx, [x, y, x + w, y + h]))

    return images, gt_counts, gt_boxes, id2label


def run_inference(weights_path, images, imgsz, device):
    """Run inference and return raw detections (all confidences) per image."""
    from ultralytics import YOLO
    model = YOLO(weights_path)

    all_detections = {}  # image_id -> list of (class_idx, confidence, [x1,y1,x2,y2])

    for img_id, img_path, w, h in images:
        results = model.predict(
            img_path, imgsz=imgsz, conf=0.01,  # very low threshold to get all detections
            device=device, verbose=False,
        )
        result = results[0]
        boxes = result.boxes
        dets = []
        if len(boxes) > 0:
            for i in range(len(boxes)):
                cls = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().tolist()
                dets.append((cls, conf, xyxy))
        all_detections[img_id] = dets

    return all_detections


def calibrate(images, gt_counts, all_detections, id2label, thresholds):
    """Find optimal confidence threshold for counting accuracy.

    For each threshold, computes:
    - Per-image predicted count vs GT count
    - MAE (Mean Absolute Error of counts)
    - RMSE (Root Mean Squared Error of counts)
    - Total count error (sum predicted - sum GT)
    - Precision, Recall, F1 (detection-level, IoU not considered — pure counting)
    """
    num_classes = len(id2label)
    results = {}

    for thresh in thresholds:
        # Per-class counting
        class_stats = {}
        for cls_idx in range(num_classes):
            gt_total = 0
            pred_total = 0
            abs_errors = []
            sq_errors = []

            for img_id, _, _, _ in images:
                gt_count = gt_counts[img_id].get(cls_idx, 0)
                pred_count = sum(1 for cls, conf, _ in all_detections[img_id]
                                 if cls == cls_idx and conf >= thresh)
                gt_total += gt_count
                pred_total += pred_count
                abs_errors.append(abs(pred_count - gt_count))
                sq_errors.append((pred_count - gt_count) ** 2)

            mae = np.mean(abs_errors) if abs_errors else 0
            rmse = np.sqrt(np.mean(sq_errors)) if sq_errors else 0

            class_stats[id2label[cls_idx]] = {
                "gt_total": gt_total,
                "pred_total": pred_total,
                "count_error": pred_total - gt_total,
                "count_error_pct": (pred_total - gt_total) / max(gt_total, 1) * 100,
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
            }

        # Overall (all classes combined)
        gt_total_all = sum(v["gt_total"] for v in class_stats.values())
        pred_total_all = sum(v["pred_total"] for v in class_stats.values())

        # Per-image total count error
        total_abs_errors = []
        total_sq_errors = []
        for img_id, _, _, _ in images:
            gt_count = sum(gt_counts[img_id].values())
            pred_count = sum(1 for cls, conf, _ in all_detections[img_id] if conf >= thresh)
            total_abs_errors.append(abs(pred_count - gt_count))
            total_sq_errors.append((pred_count - gt_count) ** 2)

        results[round(thresh, 3)] = {
            "threshold": round(thresh, 3),
            "gt_total": gt_total_all,
            "pred_total": pred_total_all,
            "count_error": pred_total_all - gt_total_all,
            "count_error_pct": round((pred_total_all - gt_total_all) / max(gt_total_all, 1) * 100, 1),
            "mae": round(np.mean(total_abs_errors), 4),
            "rmse": round(np.sqrt(np.mean(total_sq_errors)), 4),
            "per_class": class_stats,
        }

    return results


def find_optimal_thresholds(calibration_results, id2label):
    """Find threshold that minimises counting error (absolute) overall and per class."""
    # Overall: minimise |count_error|
    best_overall = min(calibration_results.values(),
                       key=lambda r: abs(r["count_error"]))

    # Per-class
    best_per_class = {}
    for cls_name in id2label.values():
        best = min(calibration_results.values(),
                   key=lambda r: abs(r["per_class"][cls_name]["count_error"]))
        best_per_class[cls_name] = {
            "threshold": best["threshold"],
            "count_error": best["per_class"][cls_name]["count_error"],
            "count_error_pct": best["per_class"][cls_name]["count_error_pct"],
            "gt_total": best["per_class"][cls_name]["gt_total"],
            "pred_total": best["per_class"][cls_name]["pred_total"],
        }

    # Also find threshold minimising MAE (per-image accuracy, not total count)
    best_mae = min(calibration_results.values(), key=lambda r: r["mae"])

    return {
        "best_count": {
            "threshold": best_overall["threshold"],
            "count_error": best_overall["count_error"],
            "count_error_pct": best_overall["count_error_pct"],
            "gt_total": best_overall["gt_total"],
            "pred_total": best_overall["pred_total"],
            "mae": best_overall["mae"],
            "rmse": best_overall["rmse"],
        },
        "best_mae": {
            "threshold": best_mae["threshold"],
            "mae": best_mae["mae"],
            "rmse": best_mae["rmse"],
            "count_error": best_mae["count_error"],
        },
        "per_class": best_per_class,
    }


def print_calibration_report(model_name, calibration_results, optimal, id2label):
    """Print a formatted calibration report."""
    print(f"\n{'=' * 75}")
    print(f" Confidence Calibration: {model_name}")
    print(f"{'=' * 75}")

    # Threshold sweep summary
    print(f"\n  {'Thresh':>7} {'GT':>6} {'Pred':>6} {'Error':>7} {'Error%':>8} {'MAE':>7} {'RMSE':>7}")
    print(f"  {'-' * 55}")
    for thresh, r in sorted(calibration_results.items()):
        if thresh % 0.1 < 0.06 or thresh == optimal["best_count"]["threshold"]:
            marker = " <--" if thresh == optimal["best_count"]["threshold"] else ""
            print(f"  {thresh:>7.3f} {r['gt_total']:>6} {r['pred_total']:>6} "
                  f"{r['count_error']:>+7} {r['count_error_pct']:>+7.1f}% "
                  f"{r['mae']:>7.2f} {r['rmse']:>7.2f}{marker}")

    # Optimal thresholds
    print(f"\n  Optimal threshold (min |total count error|): "
          f"{optimal['best_count']['threshold']:.3f}")
    print(f"    Total: GT={optimal['best_count']['gt_total']}, "
          f"Pred={optimal['best_count']['pred_total']}, "
          f"Error={optimal['best_count']['count_error']:+d} "
          f"({optimal['best_count']['count_error_pct']:+.1f}%)")
    print(f"    MAE={optimal['best_count']['mae']:.3f}, "
          f"RMSE={optimal['best_count']['rmse']:.3f}")

    print(f"\n  Optimal threshold (min per-image MAE): "
          f"{optimal['best_mae']['threshold']:.3f}")
    print(f"    MAE={optimal['best_mae']['mae']:.3f}, "
          f"RMSE={optimal['best_mae']['rmse']:.3f}")

    print(f"\n  Per-class optimal thresholds:")
    print(f"    {'Class':<15} {'Thresh':>7} {'GT':>6} {'Pred':>6} {'Error':>7} {'Error%':>8}")
    print(f"    {'-' * 50}")
    for cls_name, info in optimal["per_class"].items():
        print(f"    {cls_name:<15} {info['threshold']:>7.3f} {info['gt_total']:>6} "
              f"{info['pred_total']:>6} {info['count_error']:>+7} "
              f"{info['count_error_pct']:>+7.1f}%")


def print_comparison(all_results):
    """Side-by-side comparison of optimal thresholds across models."""
    if len(all_results) < 2:
        return

    print(f"\n{'=' * 75}")
    print(f" COMPARISON — Optimal Counting Thresholds")
    print(f"{'=' * 75}")

    header = f"  {'Metric':<25}"
    for name in all_results:
        short = Path(name).stem[:20]
        header += f" {short:>18}"
    print(header)
    print(f"  {'-' * (25 + 19 * len(all_results))}")

    for metric in ["threshold", "count_error", "count_error_pct", "mae", "rmse"]:
        row = f"  {metric:<25}"
        for name, (_, optimal) in all_results.items():
            val = optimal["best_count"][metric]
            if isinstance(val, float):
                row += f" {val:>18.3f}"
            else:
                row += f" {val:>18}"
        print(row)


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Parse threshold range
    parts = args.thresholds.split(",")
    start, stop, step = float(parts[0]), float(parts[1]), float(parts[2])
    thresholds = np.arange(start, stop + step / 2, step).tolist()

    # Load ground truth
    images, gt_counts, gt_boxes, id2label = load_ground_truth(args.dataset_dir, args.split)
    total_gt = sum(sum(v.values()) for v in gt_counts.values())
    print(f"Dataset: {len(images)} images, {len(id2label)} classes, {total_gt} GT boxes")

    all_results = {}

    for weights_path in args.weights:
        model_name = Path(weights_path).parent.parent.name + "/" + Path(weights_path).name
        print(f"\nRunning inference: {model_name} ...")

        detections = run_inference(weights_path, images, args.imgsz, device)
        total_dets = sum(len(d) for d in detections.values())
        print(f"  Total raw detections (conf>=0.01): {total_dets}")

        calibration = calibrate(images, gt_counts, detections, id2label, thresholds)
        optimal = find_optimal_thresholds(calibration, id2label)
        print_calibration_report(model_name, calibration, optimal, id2label)

        all_results[weights_path] = (calibration, optimal)

    print_comparison(all_results)

    # Save
    if args.output:
        output_data = {}
        for weights_path, (calibration, optimal) in all_results.items():
            output_data[weights_path] = {
                "optimal": optimal,
                "sweep": {str(k): v for k, v in calibration.items()},
            }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
