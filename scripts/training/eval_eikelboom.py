#!/usr/bin/env python3
"""CLI wrapper for wildlife_detection.training.eval_eikelboom.

Usage:
    python scripts/training/eval_eikelboom.py \
        --weights output/combined_yolo11/larch-freeze10-combined/weights/best.pt

    # Only create the remapped eval dataset (run once):
    python scripts/training/eval_eikelboom.py --setup-only
"""

import argparse
from pathlib import Path

from wildlife_detection.training.eval_eikelboom import evaluate, setup_eval_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
EIKELBOOM_SRC = REPO_ROOT / "week1" / "data" / "eikelboom_yolo_tiled"
EVAL_DIR = Path("/data/mnt/storage/Datasets/eikelboom_eval_megadetector")


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO on Eikelboom test set")
    parser.add_argument("--weights", type=str,
                        default="output/combined_yolo11/larch-freeze10-combined/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--setup-only", action="store_true",
                        help="Only create the remapped eval dataset")
    args = parser.parse_args()

    if args.setup_only:
        setup_eval_dataset(EIKELBOOM_SRC, EVAL_DIR)
        return

    evaluate(
        weights_path=args.weights,
        eval_dir=EVAL_DIR,
        eikelboom_src=EIKELBOOM_SRC,
        conf=args.conf,
        iou=args.iou,
    )


if __name__ == "__main__":
    main()
