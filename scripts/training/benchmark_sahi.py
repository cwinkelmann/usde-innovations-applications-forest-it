#!/usr/bin/env python
"""
SAHI tiled inference benchmark on full-sized drone images.

Runs SAHI (Slicing Aided Hyper Inference) with multiple fine-tuned models on
full-resolution test images and compares detection accuracy against COCO ground
truth annotations.

Supports:
  - Ultralytics models (.pt) — YOLO, RT-DETR, MegaDetector
  - D-FINE models (.pth) via HuggingFace transformers
  - Configurable slice size, overlap, confidence threshold

Usage:
    # Benchmark MegaDetector models
    python scripts/training/benchmark_sahi.py \
        --models \
            yolo:./output/md1000-larch-50ep/weights/best.pt \
            yolo:./output/md1000-sorrel-50ep/weights/best.pt \
            yolo:./output/mdv6-rtdetr-50ep/weights/best.pt \
        --test_images ./week1/data/eikelboom/test \
        --test_ann ./week1/data/eikelboom_coco/test/annotations.json \
        --slice_size 640 --overlap 0.2

    # Include D-FINE
    python scripts/training/benchmark_sahi.py \
        --models \
            yolo:./output/md1000-larch-50ep/weights/best.pt \
            huggingface:ustc-community/dfine-nano-coco:./output/dfine_nano_eikelboom/best.pth \
        --test_images ./week1/data/eikelboom/test \
        --test_ann ./week1/data/eikelboom_coco/test/annotations.json
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="SAHI benchmark on full-sized images")
    p.add_argument("--models", nargs="+", required=True,
                   help="Model specs: type:weights_path (type=yolo|huggingface)")
    p.add_argument("--test_images", required=True,
                   help="Directory with full-sized test images")
    p.add_argument("--test_ann", required=True,
                   help="COCO annotations JSON for test images")
    p.add_argument("--slice_size", type=int, default=640)
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--conf", type=float, default=0.25,
                   help="Confidence threshold")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default=None,
                   help="Save results to JSON")
    return p.parse_args()


def load_ground_truth(ann_path):
    """Load COCO ground truth with 0-indexed category mapping."""
    with open(ann_path) as f:
        coco = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    sorted_cat_ids = sorted(categories.keys())
    cat_id_to_idx = {cid: i for i, cid in enumerate(sorted_cat_ids)}
    id2label = {i: categories[cid] for i, cid in enumerate(sorted_cat_ids)}

    # GT boxes per image filename
    img_id_to_fname = {img["id"]: img["file_name"] for img in coco["images"]}
    gt_by_fname = defaultdict(list)
    for ann in coco["annotations"]:
        fname = img_id_to_fname[ann["image_id"]]
        cls_idx = cat_id_to_idx[ann["category_id"]]
        x, y, w, h = ann["bbox"]
        gt_by_fname[fname].append({
            "label": cls_idx,
            "bbox": [x, y, x + w, y + h],  # xyxy
        })

    return gt_by_fname, id2label


def build_sahi_model(model_spec, device):
    """Build a SAHI AutoDetectionModel from a model spec string.

    Formats:
        yolo:/path/to/weights.pt
        dfine:/path/to/config.yml:/path/to/checkpoint.pth
    """
    from sahi import AutoDetectionModel

    parts = model_spec.split(":", 1)
    model_type = parts[0]

    if model_type == "yolo":
        weights_path = parts[1]
        model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=weights_path,
            confidence_threshold=0.01,  # We filter later
            device=device,
        )
        name = Path(weights_path).parent.parent.name
    elif model_type == "dfine":
        # dfine:config.yml:checkpoint.pth — returns a DFineSahiWrapper
        rest = parts[1]
        config_path, ckpt_path = rest.split(":", 1)
        model = DFineSahiWrapper(config_path, ckpt_path, device)
        name = f"dfine({Path(ckpt_path).stem})"
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'yolo' or 'dfine'")

    return model, name


class DFineSahiWrapper:
    """Wraps D-FINE model to work with SAHI's get_sliced_prediction.

    SAHI calls model.perform_inference() and model.convert_original_predictions(),
    so we implement those methods.
    """

    def __init__(self, config_path, ckpt_path, device):
        import sys as _sys
        # D-FINE repo root: walk up from config until we find src/
        config_resolved = Path(config_path).resolve()
        dfine_root = None
        for parent in config_resolved.parents:
            if (parent / "src" / "core").exists():
                dfine_root = str(parent)
                break
        if dfine_root is None:
            raise FileNotFoundError(f"Cannot find D-FINE repo root from {config_path}")
        if dfine_root not in _sys.path:
            _sys.path.insert(0, dfine_root)
        from src.core import YAMLConfig

        cfg = YAMLConfig(config_path, resume=ckpt_path)
        self.model = cfg.model
        self.postprocessor = cfg.postprocessor
        self.device = device

        # Load weights (EMA if available)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "ema" in ckpt:
            state = ckpt["ema"]["module"]
        else:
            state = ckpt["model"]
        self.model.load_state_dict(state)
        self.model.eval().to(device)
        self.postprocessor.to(device)

        self.num_classes = cfg.yaml_cfg.get("num_classes", 3)
        self.image_size = 640
        self._original_predictions = None
        self.category_mapping = {str(i): str(i) for i in range(self.num_classes)}

    def perform_inference(self, image_as_np_array):
        """Run D-FINE inference on a numpy image array."""
        import torchvision.transforms as T
        h, w = image_as_np_array.shape[:2]
        transform = T.Compose([T.ToPILImage(), T.Resize((self.image_size, self.image_size)), T.ToTensor()])
        img_t = transform(image_as_np_array).unsqueeze(0).to(self.device)
        orig_sizes = torch.tensor([[w, h]], device=self.device)

        with torch.no_grad():
            outputs = self.model(img_t)
            results = self.postprocessor(outputs, orig_sizes)

        self._original_predictions = results
        self._original_image_size = (h, w)

    def convert_original_predictions(self, shift_amount=None, full_shape=None):
        """Convert D-FINE predictions to SAHI ObjectPrediction format."""
        from sahi.prediction import ObjectPrediction

        if shift_amount is None:
            shift_amount = [0, 0]

        predictions = []
        if self._original_predictions is None:
            return predictions

        result = self._original_predictions[0]
        labels = result["labels"]
        boxes = result["boxes"]
        scores = result["scores"]

        for i in range(len(scores)):
            score = scores[i].item()
            label = int(labels[i].item())
            bbox = boxes[i].tolist()  # [x1, y1, x2, y2]

            # Apply shift for SAHI tile stitching
            bbox[0] += shift_amount[1]
            bbox[1] += shift_amount[0]
            bbox[2] += shift_amount[1]
            bbox[3] += shift_amount[0]

            predictions.append(
                ObjectPrediction(
                    bbox=bbox,
                    category_id=label,
                    score=score,
                    category_name=str(label),
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
            )

        self.object_prediction_list = predictions
        return predictions


def run_sahi_inference(model, image_dir, slice_size, overlap, conf_threshold):
    """Run SAHI tiled inference on all images in directory.

    Returns dict: filename -> list of {label, score, bbox [xyxy]}
    """
    from sahi.predict import get_sliced_prediction

    image_dir = Path(image_dir)
    image_files = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    )

    # For DFineSahiWrapper, we need to pass the model directly
    is_dfine = isinstance(model, DFineSahiWrapper)

    predictions = {}
    t0 = time.time()

    for i, img_path in enumerate(image_files):
        if is_dfine:
            # Manual tiled inference for D-FINE wrapper
            preds = _run_dfine_tiled(model, img_path, slice_size, overlap, conf_threshold)
        else:
            result = get_sliced_prediction(
                str(img_path),
                model,
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_height_ratio=overlap,
                overlap_width_ratio=overlap,
                verbose=0,
            )
            preds = []
            for pred in result.object_prediction_list:
                if pred.score.value >= conf_threshold:
                    bbox = pred.bbox.to_xyxy()
                    preds.append({
                        "label": pred.category.id,
                        "score": pred.score.value,
                        "bbox": bbox,
                    })

        predictions[img_path.name] = preds

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{i + 1}/{len(image_files)}] {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"  Done: {len(image_files)} images in {elapsed:.0f}s "
          f"({elapsed / len(image_files):.1f}s/img)")
    return predictions


def _run_dfine_tiled(model, img_path, slice_size, overlap, conf_threshold):
    """Manual tiled inference for D-FINE (no SAHI integration needed)."""
    from torchvision.ops import nms

    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    stride = int(slice_size * (1 - overlap))

    all_boxes, all_scores, all_labels = [], [], []

    for y0 in range(0, H, stride):
        for x0 in range(0, W, stride):
            x1 = min(x0 + slice_size, W)
            y1 = min(y0 + slice_size, H)
            if (x1 - x0) < slice_size // 4 or (y1 - y0) < slice_size // 4:
                continue

            tile = np.array(img.crop((x0, y0, x1, y1)))
            model.perform_inference(tile)
            model.convert_original_predictions(shift_amount=[y0, x0], full_shape=[H, W])

            for pred in model.object_prediction_list:
                if pred.score.value >= conf_threshold:
                    bbox = pred.bbox.to_xyxy()
                    all_boxes.append(bbox)
                    all_scores.append(pred.score.value)
                    all_labels.append(pred.category.id)

    if not all_boxes:
        return []

    # NMS to merge overlapping detections from adjacent tiles
    boxes_t = torch.tensor(all_boxes, dtype=torch.float32)
    scores_t = torch.tensor(all_scores, dtype=torch.float32)
    labels_t = torch.tensor(all_labels, dtype=torch.long)

    preds = []
    for cls_id in labels_t.unique():
        mask = labels_t == cls_id
        cls_boxes = boxes_t[mask]
        cls_scores = scores_t[mask]
        keep = nms(cls_boxes, cls_scores, iou_threshold=0.5)
        for idx in keep:
            preds.append({
                "label": cls_id.item(),
                "score": cls_scores[idx].item(),
                "bbox": cls_boxes[idx].tolist(),
            })

    return preds


def compute_metrics(predictions, gt_by_fname, id2label, iou_threshold=0.5):
    """Compute per-class and overall P, R, F1, count accuracy."""
    num_classes = len(id2label)

    per_class = {}
    for cls_idx in range(num_classes):
        tp, fp, fn = 0, 0, 0

        for fname in set(list(predictions.keys()) + list(gt_by_fname.keys())):
            preds = [p for p in predictions.get(fname, []) if p["label"] == cls_idx]
            gts = [g for g in gt_by_fname.get(fname, []) if g["label"] == cls_idx]

            # Sort preds by score descending
            preds = sorted(preds, key=lambda p: p["score"], reverse=True)

            matched_gt = set()
            for pred in preds:
                best_iou = 0
                best_gt = -1
                for gi, gt in enumerate(gts):
                    if gi in matched_gt:
                        continue
                    iou = _box_iou(pred["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gi
                if best_iou >= iou_threshold and best_gt >= 0:
                    tp += 1
                    matched_gt.add(best_gt)
                else:
                    fp += 1
            fn += len(gts) - len(matched_gt)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        gt_count = sum(len([g for g in gt_by_fname.get(f, []) if g["label"] == cls_idx])
                       for f in gt_by_fname)
        pred_count = sum(len([p for p in predictions.get(f, []) if p["label"] == cls_idx])
                         for f in predictions)

        per_class[id2label[cls_idx]] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
            "gt_count": gt_count,
            "pred_count": pred_count,
            "count_error": pred_count - gt_count,
        }

    # Overall
    total_tp = sum(c["tp"] for c in per_class.values())
    total_fp = sum(c["fp"] for c in per_class.values())
    total_fn = sum(c["fn"] for c in per_class.values())
    overall_p = total_tp / max(total_tp + total_fp, 1)
    overall_r = total_tp / max(total_tp + total_fn, 1)
    overall_f1 = 2 * overall_p * overall_r / max(overall_p + overall_r, 1e-8)
    total_gt = sum(c["gt_count"] for c in per_class.values())
    total_pred = sum(c["pred_count"] for c in per_class.values())

    return {
        "precision": round(overall_p, 4),
        "recall": round(overall_r, 4),
        "f1": round(overall_f1, 4),
        "gt_count": total_gt,
        "pred_count": total_pred,
        "count_error": total_pred - total_gt,
        "count_error_pct": round((total_pred - total_gt) / max(total_gt, 1) * 100, 1),
        "per_class": per_class,
    }


def _box_iou(box1, box2):
    """IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / max(area1 + area2 - inter, 1e-8)


def print_results(model_name, metrics, conf):
    """Print formatted results for one model."""
    print(f"\n{'=' * 70}")
    print(f" {model_name} (SAHI, conf={conf})")
    print(f"{'=' * 70}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Count:     GT={metrics['gt_count']}, Pred={metrics['pred_count']}, "
          f"Error={metrics['count_error']:+d} ({metrics['count_error_pct']:+.1f}%)")

    print(f"\n  {'Class':<15} {'P':>8} {'R':>8} {'F1':>8} {'GT':>6} {'Pred':>6} {'Err':>6}")
    print(f"  {'-' * 58}")
    for cls_name, c in metrics["per_class"].items():
        print(f"  {cls_name:<15} {c['precision']:>8.4f} {c['recall']:>8.4f} "
              f"{c['f1']:>8.4f} {c['gt_count']:>6} {c['pred_count']:>6} "
              f"{c['count_error']:>+6}")


def print_comparison(all_results):
    """Print side-by-side comparison table."""
    if len(all_results) < 2:
        return

    print(f"\n{'=' * 70}")
    print(f" SAHI BENCHMARK COMPARISON")
    print(f"{'=' * 70}")

    header = f"  {'Metric':<20}"
    for name in all_results:
        short = name[:18]
        header += f" {short:>18}"
    print(header)
    print(f"  {'-' * (20 + 19 * len(all_results))}")

    for metric_key in ["precision", "recall", "f1", "count_error", "count_error_pct"]:
        row = f"  {metric_key:<20}"
        values = []
        for name, metrics in all_results.items():
            val = metrics[metric_key]
            values.append(val)
            if isinstance(val, float):
                row += f" {val:>18.4f}"
            else:
                row += f" {val:>18}"
        print(row)


def main():
    args = parse_args()
    print(f"SAHI Benchmark: slice={args.slice_size}, overlap={args.overlap}, conf={args.conf}")

    # Load GT
    gt_by_fname, id2label = load_ground_truth(args.test_ann)
    total_gt = sum(len(v) for v in gt_by_fname.values())
    print(f"Ground truth: {len(gt_by_fname)} images, {total_gt} boxes, "
          f"{len(id2label)} classes: {list(id2label.values())}")

    all_results = {}

    for model_spec in args.models:
        print(f"\nLoading: {model_spec}")
        model, name = build_sahi_model(model_spec, args.device)
        print(f"Running SAHI inference: {name}")

        predictions = run_sahi_inference(
            model, args.test_images,
            args.slice_size, args.overlap, args.conf,
        )

        metrics = compute_metrics(predictions, gt_by_fname, id2label)
        print_results(name, metrics, args.conf)
        all_results[name] = metrics

        # Free GPU memory between models
        del model
        torch.cuda.empty_cache()

    print_comparison(all_results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
