#!/usr/bin/env python
"""
Local segmentation trainer — RTX 4080 optimised.

Supports:
  --task instance   Mask2Former on COCO polygon format
  --task semantic   SegFormer (or Mask2Former) on image/mask-PNG format

Model families:
  Mask2Former — facebook/mask2former-swin-{tiny,small,base,large}-{coco-instance,ade-semantic,...}
  SegFormer   — nvidia/segformer-b{0..5}-finetuned-ade-512-512

Dataset formats:
  Instance (COCO polygon):
    dataset_dir/
      train/images/  + train/annotations.json  (COCO with 'segmentation' polygons)
      val/images/    + val/annotations.json

  Semantic (mask PNGs):
    dataset_dir/
      train/images/   img001.jpg ...
      train/masks/    img001.png ...   (single-channel, pixel value = class id)
      val/images/
      val/masks/

Usage:
    # Instance segmentation
    python train_segmentation.py \\
        --task instance \\
        --model facebook/mask2former-swin-base-coco-instance \\
        --dataset_dir ./data/my_coco_dataset \\
        --output_dir ./output/seg_instance \\
        --epochs 50 --batch_size 4

    # Semantic segmentation
    python train_segmentation.py \\
        --task semantic \\
        --model nvidia/segformer-b2-finetuned-ade-512-512 \\
        --dataset_dir ./data/my_semantic_dataset \\
        --output_dir ./output/seg_semantic \\
        --num_classes 8 \\
        --epochs 80 --batch_size 8 \\
        --log wandb --wandb_project wildlife-seg
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    Trainer,
    TrainingArguments,
)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local segmentation trainer")
    p.add_argument("--task", required=True, choices=["instance", "semantic"],
                   help="instance = COCO polygon masks | semantic = per-pixel class masks")
    p.add_argument("--model",
                   default=None,
                   help="HF Hub model ID. Defaults: instance→mask2former-swin-base-coco-instance, "
                        "semantic→segformer-b2-finetuned-ade-512-512")
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--output_dir", default="./output/segmentation")
    p.add_argument("--num_classes", type=int, default=None,
                   help="Required for semantic seg when using a new dataset. "
                        "Inferred from mask values if omitted.")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log", nargs="+", choices=["wandb", "tensorboard"], default=[])
    p.add_argument("--wandb_project", default="segmentation-training")
    p.add_argument("--wandb_run", default=None)
    p.add_argument("--resume", default=None)
    return p.parse_args()


# ─── INSTANCE SEGMENTATION — COCO POLYGON FORMAT ──────────────────────────────

def polygon_to_mask(polygon: list, height: int, width: int) -> np.ndarray:
    """Convert a COCO polygon (flat list of [x,y,x,y,...]) to a binary mask."""
    from PIL import ImageDraw
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygon:
        if len(poly) >= 6:
            pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)]
            draw.polygon(pts, fill=1)
    return np.array(mask, dtype=np.uint8)


class CocoInstanceDataset(Dataset):
    """
    COCO-format instance segmentation dataset.
    Expects 'segmentation' polygons in annotations.json.
    Falls back to bbox-derived masks if polygons are absent (crowd=0 only).
    """

    def __init__(self, split_dir: str, processor, image_size: int, augment: bool = False):
        self.split_dir = Path(split_dir)
        self.processor = processor
        self.image_size = image_size

        ann_path = self.split_dir / "annotations.json"
        if not ann_path.exists():
            raise FileNotFoundError(f"annotations.json not found in {split_dir}")

        with open(ann_path) as f:
            coco = json.load(f)

        self.images      = {img["id"]: img for img in coco["images"]}
        self.categories  = {cat["id"]: cat for cat in coco["categories"]}
        self.id2label    = {int(cat["id"]): cat["name"] for cat in coco["categories"]}
        self.label2id    = {v: k for k, v in self.id2label.items()}

        # Group annotations per image, skip crowd annotations
        self.ann_by_image: dict[int, list] = {img_id: [] for img_id in self.images}
        for ann in coco["annotations"]:
            if ann.get("iscrowd", 0) == 0:
                self.ann_by_image[ann["image_id"]].append(ann)

        self.image_ids = sorted(self.images.keys())

        # Augmentation pipeline — geometric transforms applied identically to image + masks
        if augment:
            self.transforms = A.Compose([
                A.RandomResizedCrop(height=image_size, width=image_size,
                                    scale=(0.5, 1.0), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, p=0.6),
                A.GaussNoise(var_limit=(5, 25), p=0.2),
            ], additional_targets={"masks": "masks"})
        else:
            self.transforms = A.Compose([
                A.LongestMaxSize(max_size=image_size),
                A.PadIfNeeded(min_height=image_size, min_width=image_size,
                              border_mode=0, value=128, mask_value=0),
            ], additional_targets={"masks": "masks"})

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_id   = self.image_ids[idx]
        image_info = self.images[image_id]
        img_path   = self.split_dir / "images" / image_info["file_name"]
        H, W       = image_info["height"], image_info["width"]

        image = np.array(Image.open(img_path).convert("RGB"))
        anns  = self.ann_by_image[image_id]

        # Build per-instance binary masks and label list
        instance_masks, instance_labels = [], []
        for ann in anns:
            seg = ann.get("segmentation", [])
            if isinstance(seg, dict):
                # RLE format — requires pycocotools
                try:
                    from pycocotools import mask as coco_mask
                    binary = coco_mask.decode(seg).astype(np.uint8)
                except ImportError:
                    continue
            elif isinstance(seg, list) and seg:
                binary = polygon_to_mask(seg, H, W)
            else:
                # Fallback: bounding box as rough mask
                x, y, bw, bh = [int(v) for v in ann["bbox"]]
                binary = np.zeros((H, W), dtype=np.uint8)
                binary[y:y + bh, x:x + bw] = 1

            if binary.sum() > 0:
                instance_masks.append(binary)
                instance_labels.append(ann["category_id"])

        # Apply spatial augmentations
        if instance_masks:
            transformed = self.transforms(image=image, masks=instance_masks)
            image = transformed["image"]
            instance_masks = transformed["masks"]
            # Filter out masks zeroed-out by crop
            valid = [(m, l) for m, l in zip(instance_masks, instance_labels)
                     if m.sum() > 0]
            if valid:
                instance_masks, instance_labels = zip(*valid)
            else:
                instance_masks, instance_labels = [], []
        else:
            transformed = self.transforms(image=image, masks=[np.zeros((H, W), np.uint8)])
            image = transformed["image"]

        # Build target dict for Mask2FormerImageProcessor
        target = {
            "masks": (np.stack(instance_masks).astype(np.uint8)
                      if instance_masks else np.zeros((0, self.image_size, self.image_size), np.uint8)),
            "class_labels": (torch.tensor(instance_labels, dtype=torch.long)
                             if instance_labels else torch.zeros(0, dtype=torch.long)),
        }

        encoding = self.processor(
            images=Image.fromarray(image),
            segmentation_maps=None,
            instance_id_to_semantic_id=None,
            return_tensors="pt",
        )

        # Attach ground-truth masks and labels
        encoding["mask_labels"]  = target["masks"]
        encoding["class_labels"] = target["class_labels"]

        return {k: v.squeeze(0) if isinstance(v, torch.Tensor) and v.ndim > 1
                else v for k, v in encoding.items()}


# ─── SEMANTIC SEGMENTATION — IMAGE / MASK PNG FORMAT ──────────────────────────

class SemanticSegDataset(Dataset):
    """
    Paired images + single-channel mask PNGs.
    Pixel value = class id (0-indexed, 255 = ignore).
    Layout:
        split/images/img001.jpg
        split/masks/img001.png
    """

    def __init__(self, split_dir: str, processor, image_size: int,
                 augment: bool = False, ignore_index: int = 255):
        self.split_dir    = Path(split_dir)
        self.processor    = processor
        self.image_size   = image_size
        self.ignore_index = ignore_index

        img_dir  = self.split_dir / "images"
        mask_dir = self.split_dir / "masks"

        if not img_dir.exists():
            raise FileNotFoundError(f"images/ not found in {split_dir}")
        if not mask_dir.exists():
            raise FileNotFoundError(f"masks/ not found in {split_dir}")

        # Pair images with their masks by stem name
        img_files = {f.stem: f for f in img_dir.iterdir()
                     if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}}
        mask_files = {f.stem: f for f in mask_dir.iterdir()
                      if f.suffix.lower() in {".png", ".tif", ".tiff"}}

        self.pairs = sorted([
            (img_files[stem], mask_files[stem])
            for stem in img_files if stem in mask_files
        ])
        if not self.pairs:
            raise RuntimeError(f"No matching image/mask pairs found in {split_dir}. "
                               "Check that filenames (without extension) match between "
                               "images/ and masks/.")

        # Build augmentation pipeline
        if augment:
            self.transforms = A.Compose([
                A.RandomResizedCrop(height=image_size, width=image_size,
                                    scale=(0.5, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, p=0.6),
                A.GaussNoise(var_limit=(5, 25), p=0.2),
            ])
        else:
            self.transforms = A.Compose([
                A.LongestMaxSize(max_size=image_size),
                A.PadIfNeeded(min_height=image_size, min_width=image_size,
                              border_mode=0, value=128, mask_value=ignore_index),
            ])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_path, mask_path = self.pairs[idx]

        image = np.array(Image.open(img_path).convert("RGB"))
        mask  = np.array(Image.open(mask_path))        # single channel, class ids

        transformed = self.transforms(image=image, mask=mask)
        image = transformed["image"]
        mask  = transformed["mask"]

        encoding = self.processor(
            images=Image.fromarray(image),
            segmentation_maps=Image.fromarray(mask.astype(np.uint8)),
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


# ─── COLLATORS ────────────────────────────────────────────────────────────────

def collate_instance(batch: list[dict]) -> dict:
    """Mask2Former instance: pixel_values stacked, masks/labels kept as lists."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    pixel_mask   = torch.stack([b["pixel_mask"]   for b in batch])
    mask_labels  = [b["mask_labels"]  for b in batch]
    class_labels = [b["class_labels"] for b in batch]
    return {
        "pixel_values": pixel_values,
        "pixel_mask":   pixel_mask,
        "mask_labels":  mask_labels,
        "class_labels": class_labels,
    }


def collate_semantic(batch: list[dict]) -> dict:
    """SegFormer / Mask2Former semantic: standard stack."""
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ─── METRICS ──────────────────────────────────────────────────────────────────

def build_instance_metrics():
    """mAP using torchmetrics (mask IoU)."""
    from torchmetrics.detection import MeanAveragePrecision

    metric = MeanAveragePrecision(iou_type="segm", box_format="xyxy")

    def compute_metrics(eval_pred):
        # eval_pred.predictions: tuple of (class_queries_logits, masks_queries_logits)
        # This is complex to decode without the processor — we compute a simplified
        # mask IoU instead using argmax on the predicted masks.
        # For full COCO mAP, run coco_eval separately on saved predictions.
        logits_per_pixel, _ = eval_pred.predictions
        labels = eval_pred.label_ids

        # Simplified: compute pixel accuracy as a proxy during training
        preds = np.argmax(logits_per_pixel, axis=1).flatten()
        valid = labels.flatten() != 255
        acc = (preds[valid] == labels.flatten()[valid]).mean() if valid.any() else 0.0
        return {"pixel_accuracy_proxy": float(acc)}

    return compute_metrics


def build_semantic_metrics(num_classes: int, ignore_index: int = 255):
    """Mean IoU over all classes (ignoring ignore_index)."""
    import evaluate
    metric = evaluate.load("mean_iou")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # logits: (B, num_classes, H/4, W/4) — upsample to match labels
        logits_t = torch.tensor(logits)
        labels_t = torch.tensor(labels)

        upsampled = F.interpolate(
            logits_t.float(),
            size=labels_t.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        preds = upsampled.argmax(dim=1).numpy()

        result = metric.compute(
            predictions=preds,
            references=labels,
            num_labels=num_classes,
            ignore_index=ignore_index,
            reduce_labels=False,
        )
        return {
            "mean_iou":       float(result["mean_iou"]),
            "mean_accuracy":  float(result["mean_accuracy"]),
        }

    return compute_metrics


# ─── INSTANCE TRAINING PIPELINE ───────────────────────────────────────────────

def train_instance(args: argparse.Namespace):
    model_id = args.model or "facebook/mask2former-swin-base-coco-instance"
    print(f"\n📦  Instance segmentation | model: {model_id}")

    from transformers import Mask2FormerImageProcessor
    processor = Mask2FormerImageProcessor.from_pretrained(
        model_id,
        size={"shortest_edge": args.image_size, "longest_edge": args.image_size},
        do_resize=True,
        do_normalize=True,
        ignore_index=255,
        reduce_labels=False,
    )

    train_dir = os.path.join(args.dataset_dir, "train")
    val_dir   = os.path.join(args.dataset_dir, "val")
    train_ds  = CocoInstanceDataset(train_dir, processor, args.image_size, augment=True)
    val_ds    = CocoInstanceDataset(val_dir,   processor, args.image_size, augment=False)

    print(f"   Train: {len(train_ds)}  |  Val: {len(val_ds)}")
    print(f"   Classes ({len(train_ds.id2label)}): {list(train_ds.id2label.values())}")

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_id,
        id2label=train_ds.id2label,
        label2id=train_ds.label2id,
        ignore_mismatched_sizes=True,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    training_args = _base_training_args(args, metric_for_best="pixel_accuracy_proxy")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_instance,
        compute_metrics=build_instance_metrics(),
        tokenizer=processor,
    )

    _run_training(trainer, args)


# ─── SEMANTIC TRAINING PIPELINE ───────────────────────────────────────────────

def train_semantic(args: argparse.Namespace):
    model_id = args.model or "nvidia/segformer-b2-finetuned-ade-512-512"
    print(f"\n📦  Semantic segmentation | model: {model_id}")

    # Auto-detect num_classes from mask values if not provided
    num_classes = args.num_classes
    if num_classes is None:
        print("   ⏳  Auto-detecting num_classes from training masks...")
        mask_dir = Path(args.dataset_dir) / "train" / "masks"
        unique_ids: set[int] = set()
        for mp in list(mask_dir.iterdir())[:100]:
            if mp.suffix.lower() in {".png", ".tif", ".tiff"}:
                m = np.array(Image.open(mp))
                unique_ids.update(int(v) for v in np.unique(m) if v != 255)
        num_classes = max(unique_ids) + 1
        print(f"   Detected {num_classes} classes (ids: {sorted(unique_ids)})")

    # Build id2label mapping from a label map file if present, otherwise numeric
    label_map_path = Path(args.dataset_dir) / "label_map.json"
    if label_map_path.exists():
        with open(label_map_path) as f:
            id2label = {int(k): v for k, v in json.load(f).items()}
        print(f"   Label map loaded: {id2label}")
    else:
        id2label = {i: str(i) for i in range(num_classes)}
        print("   ℹ️   No label_map.json found — using numeric class names.")
        print("      Create dataset_dir/label_map.json: {\"0\": \"background\", \"1\": \"zebra\", ...}")

    label2id = {v: k for k, v in id2label.items()}

    is_segformer = "segformer" in model_id.lower()

    if is_segformer:
        processor = SegformerImageProcessor.from_pretrained(
            model_id,
            size={"height": args.image_size, "width": args.image_size},
            do_resize=True,
            do_normalize=True,
            reduce_labels=False,
        )
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_id,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
    else:
        # Mask2Former semantic variant
        from transformers import Mask2FormerImageProcessor
        processor = Mask2FormerImageProcessor.from_pretrained(
            model_id,
            size={"shortest_edge": args.image_size},
            do_resize=True,
            do_normalize=True,
            reduce_labels=False,
            ignore_index=255,
        )
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_id,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_dir = os.path.join(args.dataset_dir, "train")
    val_dir   = os.path.join(args.dataset_dir, "val")
    train_ds  = SemanticSegDataset(train_dir, processor, args.image_size, augment=True)
    val_ds    = SemanticSegDataset(val_dir,   processor, args.image_size, augment=False)

    print(f"   Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    training_args = _base_training_args(args, metric_for_best="mean_iou")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_semantic,
        compute_metrics=build_semantic_metrics(num_classes),
        tokenizer=processor,
    )

    _run_training(trainer, args)


# ─── SHARED HELPERS ───────────────────────────────────────────────────────────

def _base_training_args(args: argparse.Namespace, metric_for_best: str) -> TrainingArguments:
    report_to = args.log if args.log else ["none"]
    return TrainingArguments(
        output_dir=args.output_dir,

        # Hardware — RTX 4080, bf16
        bf16=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,

        # Schedule
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="polynomial",  # common choice for seg models
        lr_end=1e-6,

        # Eval & save
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best,
        greater_is_better=True,

        # Logging
        logging_dir=os.path.join(args.output_dir, "runs"),
        logging_steps=10,
        report_to=report_to,
        run_name=args.wandb_run,

        seed=args.seed,
        data_seed=args.seed,
        push_to_hub=False,
    )


def _run_training(trainer: Trainer, args: argparse.Namespace):
    eff_batch = args.batch_size * args.grad_accum
    print(f"\n🚀  Training — {args.epochs} epochs, "
          f"batch {args.batch_size} × {args.grad_accum} grad_accum = {eff_batch} effective\n")

    trainer.train(resume_from_checkpoint=args.resume)

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    trainer.processing_class.save_pretrained(final_dir)
    print(f"\n✅  Model saved to {final_dir}")

    metrics = trainer.evaluate()
    print("\n📊  Final validation metrics:")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")

    if "wandb" in (args.log or []):
        import wandb
        wandb.finish()


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not torch.cuda.is_available():
        sys.exit("❌  No CUDA GPU found.")
    print(f"✅  GPU: {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB VRAM)")

    if "wandb" in (args.log or []):
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config=vars(args),
        )

    if args.task == "instance":
        train_instance(args)
    else:
        train_semantic(args)


if __name__ == "__main__":
    main()
