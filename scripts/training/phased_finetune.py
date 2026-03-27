#!/usr/bin/env python3
"""3-phase YOLO fine-tuning: freeze backbone → partial unfreeze → full fine-tune.

Preserves MegaDetector backbone knowledge through progressive unfreezing.
Each phase loads the best weights from the previous phase.

Usage:
    # Default 3-phase on Eikelboom with MegaDetector larch
    python scripts/training/phased_finetune.py \
        --weights weights/md_v1000.0.0-larch.pt \
        --data week1/data/eikelboom_yolo_tiled/dataset.yaml

    # On combined dataset with custom epochs per phase
    python scripts/training/phased_finetune.py \
        --weights weights/md_v1000.0.0-larch.pt \
        --data /data/mnt/storage/Datasets/combined_aerial_yolo_640/dataset.yaml \
        --epochs-p1 15 --epochs-p2 30 --epochs-p3 30

    # Start from MMLA model instead of MegaDetector
    python scripts/training/phased_finetune.py \
        --weights ~/.cache/huggingface/hub/models--imageomics--mmla/.../best.pt \
        --data week1/data/eikelboom_yolo_tiled/dataset.yaml
"""

import argparse
import logging
from pathlib import Path

from wildlife_detection.training.phased_finetune import TrainConfig, run_phased_training

log = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(
        description="3-phase YOLO fine-tuning for aerial wildlife detection"
    )
    p.add_argument("--weights", type=str,
                   default=str(REPO_ROOT / "weights" / "md_v1000.0.0-larch.pt"),
                   help="Path to pretrained .pt weights")
    p.add_argument("--data", type=str,
                   default="week1/data/eikelboom_yolo_tiled/dataset.yaml",
                   help="Path to dataset YAML")
    p.add_argument("--project", type=str, default="wildlife-detection")
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs-p1", type=int, default=None,
                   help="Override Phase 1 epochs (default: 10)")
    p.add_argument("--epochs-p2", type=int, default=None,
                   help="Override Phase 2 epochs (default: 20)")
    p.add_argument("--epochs-p3", type=int, default=None,
                   help="Override Phase 3 epochs (default: 20)")
    args = p.parse_args()

    cfg = TrainConfig(
        weights=args.weights,
        data=args.data,
        project=args.project,
        device=args.device,
        imgsz=args.imgsz,
        batch=args.batch,
    )

    overrides = [args.epochs_p1, args.epochs_p2, args.epochs_p3]
    for phase, override in zip(cfg.phases, overrides):
        if override is not None:
            log.info(f"Overriding {phase.name} epochs: {phase.epochs} -> {override}")
            phase.epochs = override

    log.info("3-Phase Training Configuration:")
    log.info(f"  Weights: {cfg.weights}")
    log.info(f"  Data:    {cfg.data}")
    log.info(f"  Project: {cfg.project}")
    for phase in cfg.phases:
        log.info(f"  {phase.name}: freeze={phase.freeze}, lr0={phase.lr0}, epochs={phase.epochs}")

    run_phased_training(cfg)


if __name__ == "__main__":
    main()
