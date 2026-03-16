#!/usr/bin/env python
"""
Local object detection trainer — works on CUDA, MPS (Apple Silicon), and CPU.
Supports: RTDETRv2, YOLOS, DETR and any AutoModelForObjectDetection-compatible checkpoint.
Dataset format: COCO JSON (train/ and val/ splits with images/ and annotations.json).
Logging: WandB and/or TensorBoard.

Usage:
    python train_detection.py \
        --model PekingU/rtdetr_r50vd_coco_o365 \
        --dataset_dir ./data/my_dataset \
        --output_dir ./output/detection \
        --epochs 50 \
        --batch_size 8 \
        --log wandb --wandb_project wildlife-detection
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    Trainer,
    TrainingArguments,
)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local detection trainer")
    p.add_argument("--model", default="PekingU/rtdetr_r50vd_coco_o365",
                   help="HF Hub model ID or local path")
    p.add_argument("--dataset_dir", required=True,
                   help="Root dir with train/ and val/ subdirs")
    p.add_argument("--output_dir", default="./output/detection")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8,
                   help="Per-device train batch size")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--grad_accum", type=int, default=1,
                   help="Gradient accumulation steps")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--image_size", type=int, default=640)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log", nargs="+", choices=["wandb", "tensorboard"],
                   default=[], help="Logging backends")
    p.add_argument("--wandb_project", default="detection-training")
    p.add_argument("--wandb_run", default=None)
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


# ─── AUGMENTATION ────────────────────────────────────────────────────────────

def build_transforms(size: int, augment: bool):
    """Build albumentations transforms compatible with albumentations >= 2.0."""
    import albumentations as A

    bbox_params = A.BboxParams(
        format="coco",
        label_fields=["category_ids"],
        min_visibility=0.3,
    )
    if augment:
        return A.Compose([
            A.RandomResizedCrop(size=(size, size), scale=(0.5, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, p=0.6),
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        ], bbox_params=bbox_params)
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(min_height=size, min_width=size,
                          border_mode=0, fill=(128, 128, 128)),
        ], bbox_params=bbox_params)


# ─── DATASET ──────────────────────────────────────────────────────────────────

class CocoDetectionDataset(Dataset):
    """
    COCO-format detection dataset.
    Expected layout:
        split_dir/
            images/      ← image files
            annotations.json  ← standard COCO JSON
    """

    def __init__(self, split_dir: str, processor, image_size: int, augment: bool = False):
        self.split_dir = Path(split_dir)
        self.processor = processor
        self.augment = augment

        ann_path = self.split_dir / "annotations.json"
        if not ann_path.exists():
            raise FileNotFoundError(f"annotations.json not found in {split_dir}")

        with open(ann_path) as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

        # Build 0-indexed label mappings (model output indices)
        sorted_cat_ids = sorted(self.categories.keys())
        self._cat_id_to_contiguous = {cid: i for i, cid in enumerate(sorted_cat_ids)}
        self.id2label = {i: self.categories[cid] for i, cid in enumerate(sorted_cat_ids)}
        self.label2id = {v: k for k, v in self.id2label.items()}

        # Group annotations by image id
        self.annotations: dict[int, list] = {img_id: [] for img_id in self.images}
        for ann in coco["annotations"]:
            self.annotations[ann["image_id"]].append(ann)

        self.image_ids = sorted(self.images.keys())

        self.transforms = build_transforms(image_size, augment)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        img_path = self.split_dir / "images" / image_info["file_name"]

        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)

        anns = self.annotations[image_id]
        bboxes = [a["bbox"] for a in anns]          # [x, y, w, h]
        category_ids = [a["category_id"] for a in anns]

        # Filter out zero-area boxes
        valid = [(b, c) for b, c in zip(bboxes, category_ids)
                 if b[2] > 1 and b[3] > 1]
        if valid:
            bboxes, category_ids = zip(*valid)
            bboxes, category_ids = list(bboxes), list(category_ids)
        else:
            bboxes, category_ids = [], []

        if bboxes:
            transformed = self.transforms(
                image=image_np,
                bboxes=bboxes,
                category_ids=category_ids,
            )
            image_np = transformed["image"]
            bboxes = list(transformed["bboxes"])
            category_ids = list(transformed["category_ids"])

        # Build COCO-style target for the processor.
        # Remap category_ids to 0-indexed contiguous indices.
        target = {
            "image_id": image_id,
            "annotations": [
                {
                    "image_id": image_id,
                    "category_id": self._cat_id_to_contiguous[cat_id],
                    "bbox": list(bbox),
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
                for cat_id, bbox in zip(category_ids, bboxes)
            ],
        }

        encoding = self.processor(
            images=Image.fromarray(image_np),
            annotations=target,
            return_tensors="pt",
        )

        # The processor returns pixel_values as [1, C, H, W] and labels as a list
        # of dicts. Squeeze the batch dim from tensors, keep labels as-is.
        result = {}
        for k, v in encoding.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.squeeze(0)
            elif isinstance(v, list) and len(v) == 1:
                result[k] = v[0]  # unwrap single-element list (labels)
            else:
                result[k] = v
        return result


def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels}


# ─── METRICS ──────────────────────────────────────────────────────────────────

def build_compute_metrics(processor):
    from torchmetrics.detection import MeanAveragePrecision
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    def compute_metrics(eval_pred):
        (logits_boxes, logits_labels), labels = eval_pred

        preds_list, targets_list = [], []
        for i in range(len(labels["class_labels"])):
            scores = torch.tensor(logits_labels[i]).softmax(-1)
            pred_scores, pred_labels = scores.max(-1)

            # Filter out the no-object class (last class index)
            no_obj_idx = scores.shape[-1] - 1
            keep = pred_labels != no_obj_idx

            pred_boxes_xyxy = box_cxcywh_to_xyxy(torch.tensor(logits_boxes[i]))

            preds_list.append({
                "boxes": pred_boxes_xyxy[keep],
                "scores": pred_scores[keep],
                "labels": pred_labels[keep],
            })
            targets_list.append({
                "boxes": box_cxcywh_to_xyxy(torch.tensor(labels["boxes"][i])),
                "labels": torch.tensor(labels["class_labels"][i]),
            })

        metric.update(preds_list, targets_list)
        result = metric.compute()
        metric.reset()

        return {
            "mAP@0.5": float(result["map_50"]),
            "mAP@0.5:0.95": float(result["map"]),
            "mAR@100": float(result["mar_100"]),
        }

    return compute_metrics


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [cx, cy, w, h] (normalised) to [x1, y1, x2, y2]."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h,
                        cx + 0.5 * w, cy + 0.5 * h], dim=-1)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Detect device ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU: {device_name} ({vram:.1f} GB VRAM)")
        use_bf16 = True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("GPU: Apple MPS (Metal Performance Shaders)")
        use_bf16 = False  # MPS does not support bf16
        # Some ops (e.g. grid_sampler_2d_backward in RT-DETR) are not implemented
        # on MPS. Enable CPU fallback for those ops.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    else:
        print("No GPU found — training on CPU (will be slow)")
        use_bf16 = False

    # ── Load processor & model ────────────────────────────────────────────────
    print(f"\nLoading processor and model: {args.model}")
    processor = AutoImageProcessor.from_pretrained(
        args.model,
        size={"height": args.image_size, "width": args.image_size},
        use_fast=False,
    )

    # Load dataset first to get category mapping
    train_dir = os.path.join(args.dataset_dir, "train")
    val_dir   = os.path.join(args.dataset_dir, "val")

    train_ds = CocoDetectionDataset(train_dir, processor, args.image_size, augment=True)
    val_ds   = CocoDetectionDataset(val_dir,   processor, args.image_size, augment=False)

    print(f"   Train: {len(train_ds)} images | Val: {len(val_ds)} images")
    print(f"   Classes ({len(train_ds.id2label)}): {list(train_ds.id2label.values())}")

    model = AutoModelForObjectDetection.from_pretrained(
        args.model,
        id2label=train_ds.id2label,
        label2id=train_ds.label2id,
        ignore_mismatched_sizes=True,   # allows head replacement for new categories
    )

    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            print("   Gradient checkpointing: ON")
        except ValueError:
            print("   Gradient checkpointing: not supported by this model, skipping")

    # ── Logging setup ─────────────────────────────────────────────────────────
    report_to = args.log if args.log else ["none"]

    if "wandb" in report_to:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config=vars(args),
        )

    # ── TrainingArguments ─────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,

        # Hardware — auto-adapts to CUDA / MPS / CPU
        bf16=use_bf16,
        fp16=False,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),

        # Schedule
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",

        # Eval & saving
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="mAP@0.5",
        greater_is_better=True,

        # Logging
        logging_steps=10,
        report_to=report_to,
        run_name=args.wandb_run,

        # Reproducibility
        seed=args.seed,
        data_seed=args.seed,

        # No HF Hub push
        push_to_hub=False,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=build_compute_metrics(processor),
        processing_class=processor,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nStarting training — {args.epochs} epochs, "
          f"batch {args.batch_size} x {args.grad_accum} grad_accum\n")
    trainer.train(resume_from_checkpoint=args.resume)

    # ── Save final model ──────────────────────────────────────────────────────
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    print(f"\nModel saved to {final_dir}")

    # ── Final eval ────────────────────────────────────────────────────────────
    metrics = trainer.evaluate()
    print("\nFinal validation metrics:")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")

    if "wandb" in report_to:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
