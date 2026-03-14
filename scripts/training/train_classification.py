#!/usr/bin/env python
"""
Local image classification trainer — RTX 4080 optimised.
Supports any AutoModelForImageClassification-compatible checkpoint:
  ViT, DINOv2, MobileViT, ResNet, EfficientNet, Swin, etc.
Dataset format: ImageFolder (class-named subdirs) or a Hugging Face dataset ID.
Logging: WandB and/or TensorBoard.

Usage:
    python train_classification.py \
        --model google/vit-base-patch16-224 \
        --dataset_dir ./data/wildlife_crops \
        --output_dir ./output/classification \
        --epochs 30 \
        --batch_size 32 \
        --log wandb tensorboard --wandb_project wildlife-clf
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import classification_report
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)
import evaluate


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local classification trainer")
    p.add_argument("--model", default="google/vit-base-patch16-224",
                   help="HF Hub model ID or local path")
    p.add_argument("--dataset_dir", required=True,
                   help="ImageFolder root dir OR a HF dataset ID (username/dataset)")
    p.add_argument("--output_dir", default="./output/classification")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.0,
                   help="Classifier dropout (0 = model default)")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze feature extractor, train only the head")
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log", nargs="+", choices=["wandb", "tensorboard"],
                   default=[], help="Logging backends")
    p.add_argument("--wandb_project", default="classification-training")
    p.add_argument("--wandb_run", default=None)
    p.add_argument("--resume", default=None,
                   help="Checkpoint path to resume from")
    return p.parse_args()


# ─── DATASET ──────────────────────────────────────────────────────────────────

class ImageFolderWithProcessor(Dataset):
    """Wraps torchvision ImageFolder and applies the HF processor + Albumentations."""

    def __init__(self, root: str, processor, augment: bool = False):
        self.folder = ImageFolder(root)
        self.processor = processor
        self.classes = self.folder.classes
        self.class_to_idx = self.folder.class_to_idx

        size = processor.size.get("shortest_edge",
               processor.size.get("height", 224))
        self.transforms = self._build_transforms(size, augment)

    def _build_transforms(self, size: int, augment: bool) -> A.Compose:
        if augment:
            return A.Compose([
                A.RandomResizedCrop(height=size, width=size, scale=(0.7, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.3, contrast=0.3,
                              saturation=0.2, hue=0.05, p=0.7),
                A.GaussNoise(var_limit=(5, 30), p=0.2),
                A.Rotate(limit=15, p=0.3),
                A.CoarseDropout(max_holes=4, max_height=size//8,
                                max_width=size//8, p=0.2),
            ])
        else:
            return A.Compose([
                A.SmallestMaxSize(max_size=size),
                A.CenterCrop(height=size, width=size),
            ])

    def __len__(self) -> int:
        return len(self.folder)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img, label = self.folder[idx]
        img_np = np.array(img.convert("RGB"))
        img_np = self.transforms(image=img_np)["image"]
        encoding = self.processor(
            images=Image.fromarray(img_np), return_tensors="pt"
        )
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class HFDatasetWrapper(Dataset):
    """Wraps a Hugging Face dataset split for classification."""

    def __init__(self, hf_split, processor, augment: bool = False):
        self.data = hf_split
        self.processor = processor
        size = processor.size.get("shortest_edge",
               processor.size.get("height", 224))
        self.augment = augment
        self.transforms = self._build_transforms(size, augment)

        # Detect image column
        self.image_col = next(
            (c for c in ["image", "img", "pixel_values"] if c in hf_split.column_names),
            hf_split.column_names[0],
        )
        self.label_col = next(
            (c for c in ["label", "labels", "class"] if c in hf_split.column_names),
            hf_split.column_names[1],
        )

    def _build_transforms(self, size: int, augment: bool) -> A.Compose:
        if augment:
            return A.Compose([
                A.RandomResizedCrop(height=size, width=size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
            ])
        return A.Compose([
            A.SmallestMaxSize(max_size=size),
            A.CenterCrop(height=size, width=size),
        ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.data[idx]
        img = row[self.image_col]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img_np = np.array(img.convert("RGB"))
        img_np = self.transforms(image=img_np)["image"]
        encoding = self.processor(
            images=Image.fromarray(img_np), return_tensors="pt"
        )
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": torch.tensor(row[self.label_col], dtype=torch.long),
        }


def load_datasets(dataset_dir: str, processor):
    """
    Auto-detect: ImageFolder on disk vs. HF Hub dataset.
    Returns (train_ds, val_ds, id2label, label2id).
    """
    train_path = os.path.join(dataset_dir, "train")
    val_path   = os.path.join(dataset_dir, "val")

    if os.path.isdir(train_path) and os.path.isdir(val_path):
        # ── ImageFolder ───────────────────────────────────────────────────────
        train_ds = ImageFolderWithProcessor(train_path, processor, augment=True)
        val_ds   = ImageFolderWithProcessor(val_path,   processor, augment=False)
        id2label = {i: c for c, i in train_ds.class_to_idx.items()}
        label2id = train_ds.class_to_idx
    else:
        # ── HF Hub dataset ────────────────────────────────────────────────────
        raw = load_dataset(dataset_dir)
        train_split = raw["train"]
        val_split   = raw.get("validation", raw.get("val", raw.get("test")))
        if val_split is None:
            raise ValueError(
                f"Dataset '{dataset_dir}' has no 'validation' or 'val' split. "
                "Use --dataset_dir with an ImageFolder instead."
            )

        # Build label map from dataset features
        label_feature = train_split.features.get("label")
        if hasattr(label_feature, "names"):
            id2label = {i: n for i, n in enumerate(label_feature.names)}
        else:
            # Infer from unique values
            all_labels = sorted(set(train_split["label"]))
            id2label = {i: str(i) for i in all_labels}
        label2id = {v: k for k, v in id2label.items()}

        train_ds = HFDatasetWrapper(train_split, processor, augment=True)
        val_ds   = HFDatasetWrapper(val_split,   processor, augment=False)

    return train_ds, val_ds, id2label, label2id


# ─── METRICS ──────────────────────────────────────────────────────────────────

def build_compute_metrics(id2label: dict):
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        acc = accuracy_metric.compute(predictions=preds, references=labels)
        f1  = f1_metric.compute(predictions=preds, references=labels,
                                average="macro")
        return {
            "accuracy": acc["accuracy"],
            "f1_macro": f1["f1"],
        }

    return compute_metrics


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Validate GPU ──────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        sys.exit("❌  No CUDA GPU found.")
    print(f"✅  GPU: {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB VRAM)")

    # ── Processor ─────────────────────────────────────────────────────────────
    print(f"\n⬇  Loading processor: {args.model}")
    processor = AutoImageProcessor.from_pretrained(args.model)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds, val_ds, id2label, label2id = load_datasets(args.dataset_dir, processor)
    num_classes = len(id2label)
    print(f"   Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"   Classes ({num_classes}): {list(id2label.values())}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\n⬇  Loading model: {args.model}")
    model_kwargs = dict(
        pretrained_model_name_or_path=args.model,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    if args.dropout > 0:
        model_kwargs["hidden_dropout_prob"] = args.dropout
        model_kwargs["attention_probs_dropout_prob"] = args.dropout

    model = AutoModelForImageClassification.from_pretrained(**model_kwargs)

    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Backbone frozen. Trainable params: {trainable:,}")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("   Gradient checkpointing: ON")

    # ── Logging ───────────────────────────────────────────────────────────────
    report_to = args.log if args.log else ["none"]

    if "wandb" in report_to:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config=vars(args),
        )
        wandb.config.update({"num_classes": num_classes,
                             "classes": list(id2label.values())})

    # ── TrainingArguments ─────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,

        # Hardware
        bf16=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,

        # Schedule
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        label_smoothing_factor=args.label_smoothing,

        # Eval & saving
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,

        # Logging
        logging_dir=os.path.join(args.output_dir, "runs"),
        logging_steps=10,
        report_to=report_to,
        run_name=args.wandb_run,

        # Reproducibility
        seed=args.seed,
        data_seed=args.seed,

        push_to_hub=False,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=build_compute_metrics(id2label),
        tokenizer=processor,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n🚀  Training {num_classes}-class classifier — "
          f"{args.epochs} epochs, batch {args.batch_size}\n")
    trainer.train(resume_from_checkpoint=args.resume)

    # ── Save ──────────────────────────────────────────────────────────────────
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    print(f"\n✅  Model saved to {final_dir}")

    # ── Per-class report ──────────────────────────────────────────────────────
    print("\n📊  Generating per-class classification report...")
    preds_output = trainer.predict(val_ds)
    preds = np.argmax(preds_output.predictions, axis=-1)
    labels_true = preds_output.label_ids
    class_names = [id2label[i] for i in sorted(id2label.keys())]
    print(classification_report(labels_true, preds, target_names=class_names))

    if "wandb" in report_to:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
