#!/usr/bin/env python
"""
Evaluate and compare detection models on a COCO-format test/val set.

Reports per-class and overall: Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95.
Supports:
  - MDV6 RT-DETR (PytorchWildlife checkpoint)
  - Ultralytics YOLO/RTDETR (.pt checkpoint)

Usage:
    # Evaluate MDV6 fine-tuned model
    python scripts/training/evaluate_detectors.py \
        --model mdv6 \
        --weights ./output/mdv6_finetune/best_epoch*.pth \
        --model_version MDV6-apa-rtdetr-e \
        --dataset_dir ./week1/data/eikelboom_coco_tiled \
        --split val

    # Evaluate Ultralytics YOLO model
    python scripts/training/evaluate_detectors.py \
        --model yolo \
        --weights ./runs/detect/train/weights/best.pt \
        --dataset_dir ./week1/data/eikelboom_coco_tiled \
        --split val

    # Compare multiple models
    python scripts/training/evaluate_detectors.py \
        --model mdv6 yolo \
        --weights ./output/mdv6_finetune/best_epoch5.pth ./runs/detect/train/weights/best.pt \
        --model_version MDV6-apa-rtdetr-e \
        --dataset_dir ./week1/data/eikelboom_coco_tiled \
        --split val --conf_thres 0.3
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
from torchmetrics.detection import MeanAveragePrecision


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate and compare detection models")
    p.add_argument("--model", nargs="+", required=True,
                   choices=["mdv6", "yolo"],
                   help="Model type(s) to evaluate")
    p.add_argument("--weights", nargs="+", required=True,
                   help="Weight file(s), one per model")
    p.add_argument("--model_version", default="MDV6-apa-rtdetr-e",
                   choices=["MDV6-apa-rtdetr-e", "MDV6-apa-rtdetr-c"],
                   help="MDV6 variant (only used for mdv6 model type)")
    p.add_argument("--dataset_dir", required=True,
                   help="COCO dataset root with train/val/test subdirs")
    p.add_argument("--split", default="val", choices=["val", "test"],
                   help="Which split to evaluate on")
    p.add_argument("--conf_thres", type=float, default=0.25,
                   help="Confidence threshold for predictions")
    p.add_argument("--image_size", type=int, default=640)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--output", default=None,
                   help="Save results to JSON file")
    return p.parse_args()


# ─── Load COCO ground truth ─────────────────────────────────────────────────

def load_coco_gt(dataset_dir: str, split: str):
    """Load ground truth from COCO annotations.json.

    Returns:
        images: list of (image_id, file_path, width, height)
        gt_by_image: dict mapping image_id to list of (label, [x1,y1,x2,y2])
        categories: dict mapping contiguous_idx to name
        cat_id_to_idx: dict mapping original cat_id to contiguous idx
    """
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

    gt_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        x1, y1, x2, y2 = x, y, x + w, y + h
        label = cat_id_to_idx[ann["category_id"]]
        gt_by_image[ann["image_id"]].append((label, [x1, y1, x2, y2]))

    return images, dict(gt_by_image), id2label, cat_id_to_idx


# ─── MDV6 inference ──────────────────────────────────────────────────────────

def predict_mdv6(weights: str, model_version: str, images: list, image_size: int,
                 conf_thres: float, num_classes: int, device: torch.device):
    """Run MDV6 RT-DETR inference on all images."""
    import torchvision.transforms as T

    # Import model loading from training script
    from pathlib import Path as P
    import PytorchWildlife
    pw_root = P(PytorchWildlife.__file__).parent
    rtdetr_root = pw_root / "models" / "detection" / "rtdetr_apache"
    sys.path.insert(0, str(rtdetr_root))
    from rtdetrv2_pytorch.src.core import YAMLConfig

    model_configs = {
        "MDV6-apa-rtdetr-e": "rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_6x_megadetector.yml",
        "MDV6-apa-rtdetr-c": "rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_megadetector.yml",
    }

    config_path = str(rtdetr_root / model_configs[model_version])

    # We need a valid checkpoint for YAMLConfig init — use the provided weights
    # or the default cached weights just to build the model architecture
    default_ckpt = os.path.join(torch.hub.get_dir(), "checkpoints",
                                f"{model_version}.pth")
    init_ckpt = default_ckpt if os.path.exists(default_ckpt) else weights

    cfg = YAMLConfig(config_path, resume=init_ckpt)
    model = cfg.model
    postprocessor = cfg.postprocessor

    # Load fine-tuned weights
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    if "ema" in ckpt:
        state = ckpt["ema"]["module"]
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    postprocessor.to(device)

    transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])

    predictions = {}
    with torch.no_grad():
        for img_id, img_path, w, h in images:
            img = Image.open(img_path).convert("RGB")
            img_t = transform(img).unsqueeze(0).to(device)
            orig_size = torch.tensor([[w, h]], device=device, dtype=torch.float32)

            outputs = model(img_t)
            results = postprocessor(outputs, orig_size)

            # results is list of dicts
            result = results[0]
            scores = result["scores"]
            keep = scores > conf_thres
            predictions[img_id] = {
                "boxes": result["boxes"][keep].cpu(),
                "scores": scores[keep].cpu(),
                "labels": result["labels"][keep].cpu().long(),
            }

    return predictions


# ─── YOLO inference ──────────────────────────────────────────────────────────

def predict_yolo(weights: str, images: list, image_size: int,
                 conf_thres: float, device: torch.device):
    """Run Ultralytics YOLO/RTDETR inference on all images."""
    from ultralytics import YOLO
    model = YOLO(weights)

    predictions = {}

    # Process one image at a time to avoid OOM with large models (RT-DETR)
    for img_id, img_path, w, h in images:
        results = model.predict(
            img_path, imgsz=image_size, conf=conf_thres,
            device=device, verbose=False,
        )
        result = results[0]
        boxes = result.boxes
        if len(boxes) > 0:
            predictions[img_id] = {
                "boxes": boxes.xyxy.cpu(),
                "scores": boxes.conf.cpu(),
                "labels": boxes.cls.cpu().long(),
            }
        else:
            predictions[img_id] = {
                "boxes": torch.zeros((0, 4)),
                "scores": torch.zeros(0),
                "labels": torch.zeros(0, dtype=torch.long),
            }

    return predictions


# ─── Metrics computation ─────────────────────────────────────────────────────

def compute_metrics(predictions: dict, gt_by_image: dict, images: list,
                    id2label: dict, conf_thres: float):
    """Compute mAP and per-class precision/recall/F1.

    Returns dict with overall and per-class metrics.
    """
    num_classes = len(id2label)

    # ── torchmetrics mAP ──────────────────────────────────────────────────
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    preds_list = []
    targets_list = []

    for img_id, img_path, w, h in images:
        pred = predictions.get(img_id, {
            "boxes": torch.zeros((0, 4)),
            "scores": torch.zeros(0),
            "labels": torch.zeros(0, dtype=torch.long),
        })
        preds_list.append(pred)

        gt_items = gt_by_image.get(img_id, [])
        if gt_items:
            gt_labels = torch.tensor([g[0] for g in gt_items], dtype=torch.long)
            gt_boxes = torch.tensor([g[1] for g in gt_items], dtype=torch.float32)
        else:
            gt_labels = torch.zeros(0, dtype=torch.long)
            gt_boxes = torch.zeros((0, 4), dtype=torch.float32)

        targets_list.append({"boxes": gt_boxes, "labels": gt_labels})

    metric.update(preds_list, targets_list)
    map_result = metric.compute()

    # ── Per-class P/R/F1 at IoU=0.5 ──────────────────────────────────────
    # Simple matching: for each GT box, find best matching pred (highest IoU > 0.5)
    per_class = {}
    for cls_idx in range(num_classes):
        tp, fp, fn = 0, 0, 0

        for img_id, img_path, w, h in images:
            pred = predictions.get(img_id)
            gt_items = gt_by_image.get(img_id, [])

            # Filter to this class
            gt_boxes_cls = [g[1] for g in gt_items if g[0] == cls_idx]
            if pred is not None and len(pred["boxes"]) > 0:
                cls_mask = pred["labels"] == cls_idx
                pred_boxes_cls = pred["boxes"][cls_mask]
                pred_scores_cls = pred["scores"][cls_mask]
            else:
                pred_boxes_cls = torch.zeros((0, 4))
                pred_scores_cls = torch.zeros(0)

            matched_gt = set()
            # Sort predictions by score descending
            if len(pred_scores_cls) > 0:
                order = pred_scores_cls.argsort(descending=True)
                pred_boxes_cls = pred_boxes_cls[order]

            for pb in pred_boxes_cls:
                best_iou = 0
                best_gt_idx = -1
                for gi, gb in enumerate(gt_boxes_cls):
                    if gi in matched_gt:
                        continue
                    iou = _box_iou(pb, torch.tensor(gb))
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gi

                if best_iou >= 0.5 and best_gt_idx >= 0:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1

            fn += len(gt_boxes_cls) - len(matched_gt)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        per_class[id2label[cls_idx]] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }

    # Overall (micro-averaged)
    total_tp = sum(c["tp"] for c in per_class.values())
    total_fp = sum(c["fp"] for c in per_class.values())
    total_fn = sum(c["fn"] for c in per_class.values())
    overall_p = total_tp / max(total_tp + total_fp, 1)
    overall_r = total_tp / max(total_tp + total_fn, 1)
    overall_f1 = 2 * overall_p * overall_r / max(overall_p + overall_r, 1e-8)

    return {
        "mAP@0.5": float(map_result["map_50"]),
        "mAP@0.5:0.95": float(map_result["map"]),
        "mAR@100": float(map_result["mar_100"]),
        "precision": round(overall_p, 4),
        "recall": round(overall_r, 4),
        "f1": round(overall_f1, 4),
        "per_class": per_class,
        "conf_thres": conf_thres,
    }


def _box_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0].item(), box2[0].item())
    y1 = max(box1[1].item(), box2[1].item())
    x2 = min(box1[2].item(), box2[2].item())
    y2 = min(box1[3].item(), box2[3].item())

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]).item() * (box1[3] - box1[1]).item()
    area2 = (box2[2] - box2[0]).item() * (box2[3] - box2[1]).item()
    union = area1 + area2 - inter

    return inter / max(union, 1e-8)


# ─── Pretty printing ────────────────────────────────────────────────────────

def print_results(model_name: str, metrics: dict):
    """Print evaluation results as a formatted table."""
    print(f"\n{'=' * 70}")
    print(f" Model: {model_name}")
    print(f" Confidence threshold: {metrics['conf_thres']}")
    print(f"{'=' * 70}")
    print(f"\n  Overall metrics:")
    print(f"    mAP@0.5:     {metrics['mAP@0.5']:.4f}")
    print(f"    mAP@0.5:0.95:{metrics['mAP@0.5:0.95']:.4f}")
    print(f"    mAR@100:     {metrics['mAR@100']:.4f}")
    print(f"    Precision:   {metrics['precision']:.4f}")
    print(f"    Recall:      {metrics['recall']:.4f}")
    print(f"    F1:          {metrics['f1']:.4f}")

    print(f"\n  Per-class (IoU=0.5):")
    print(f"    {'Class':<15} {'Prec':>8} {'Recall':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    print(f"    {'-' * 60}")
    for cls_name, cls_metrics in metrics["per_class"].items():
        print(f"    {cls_name:<15} {cls_metrics['precision']:>8.4f} "
              f"{cls_metrics['recall']:>8.4f} {cls_metrics['f1']:>8.4f} "
              f"{cls_metrics['tp']:>6} {cls_metrics['fp']:>6} {cls_metrics['fn']:>6}")


def print_comparison(all_results: dict):
    """Print side-by-side comparison of all models."""
    if len(all_results) < 2:
        return

    print(f"\n{'=' * 70}")
    print(f" COMPARISON")
    print(f"{'=' * 70}")

    header = f"  {'Metric':<20}"
    for name in all_results:
        header += f" {name:>15}"
    print(header)
    print(f"  {'-' * (20 + 16 * len(all_results))}")

    for metric_key in ["mAP@0.5", "mAP@0.5:0.95", "mAR@100", "precision", "recall", "f1"]:
        row = f"  {metric_key:<20}"
        values = []
        for name, metrics in all_results.items():
            val = metrics[metric_key]
            values.append(val)
            row += f" {val:>15.4f}"
        # Highlight best
        best_val = max(values)
        print(row + ("  *" if values.count(best_val) == 1 else ""))


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if len(args.model) != len(args.weights):
        print(f"Error: {len(args.model)} model types but {len(args.weights)} weight files")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load ground truth
    images, gt_by_image, id2label, cat_id_to_idx = load_coco_gt(
        args.dataset_dir, args.split
    )
    num_classes = len(id2label)
    print(f"Dataset: {len(images)} images, {num_classes} classes: {list(id2label.values())}")
    total_gt = sum(len(v) for v in gt_by_image.values())
    print(f"Total GT boxes: {total_gt}")

    all_results = {}

    for model_type, weights_path in zip(args.model, args.weights):
        model_name = f"{model_type}:{Path(weights_path).stem}"
        print(f"\nEvaluating {model_name}...")

        if model_type == "mdv6":
            predictions = predict_mdv6(
                weights_path, args.model_version, images,
                args.image_size, args.conf_thres, num_classes, device,
            )
        elif model_type == "yolo":
            predictions = predict_yolo(
                weights_path, images, args.image_size, args.conf_thres, device,
            )

        metrics = compute_metrics(predictions, gt_by_image, images, id2label, args.conf_thres)
        all_results[model_name] = metrics
        print_results(model_name, metrics)

    print_comparison(all_results)

    # Save results
    if args.output:
        # Convert tensors for JSON serialization
        serializable = {}
        for name, metrics in all_results.items():
            serializable[name] = {k: v for k, v in metrics.items()}
        with open(args.output, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
