#!/usr/bin/env python3
"""CLI wrapper for wildlife_detection.training.train_yolo_combined.

Supports YOLO (larch, sorrel) and RT-DETR (MDV6-rtdetr-c) models.
RT-DETR-specific settings (amp=False, lower LR) are applied automatically.

Usage:
    # YOLO11L (MegaDetector larch)
    python scripts/training/train_combined_yolo11.py \
        --weights weights/md_v1000.0.0-larch.pt \
        --epochs 50 --batch 16 --freeze 10 --name larch-freeze10

    # RT-DETR (auto-detects, sets amp=False + lr0=0.0001)
    python scripts/training/train_combined_yolo11.py \
        --weights ~/.cache/torch/hub/checkpoints/MDV6-rtdetr-c.pt \
        --epochs 50 --batch 4 --freeze 10 --name rtdetr-freeze10

    # Resume interrupted training
    python scripts/training/train_combined_yolo11.py \
        --resume wildlife-detection/larch-freeze10/weights/last.pt
"""

import argparse
from pathlib import Path

from wildlife_detection.training.train_yolo_combined import resume_training, train_combined

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Train YOLO/RT-DETR on wildlife detection dataset")
    parser.add_argument("--data", type=str,
                        default="/data/mnt/storage/Datasets/combined_aerial_yolo_640/dataset.yaml")
    parser.add_argument("--weights", type=str,
                        default=str(REPO_ROOT / "weights" / "md_v1000.0.0-larch.pt"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--freeze", type=int, default=10)
    parser.add_argument("--lr0", type=float, default=None,
                        help="Learning rate (default: 0.001 for YOLO, 0.0001 for RT-DETR)")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="wildlife-detection")
    parser.add_argument("--name", type=str, default="larch-freeze10-combined")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to last.pt to resume training")
    args = parser.parse_args()

    if args.resume:
        resume_training(args.resume)
    else:
        train_combined(
            data=args.data, weights=args.weights, epochs=args.epochs,
            batch=args.batch, imgsz=args.imgsz, freeze=args.freeze,
            lr0=args.lr0, patience=args.patience, device=args.device,
            project=args.project, name=args.name,
            warmup_epochs=args.warmup_epochs, weight_decay=args.weight_decay,
        )


if __name__ == "__main__":
    main()
