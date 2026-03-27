#!/usr/bin/env python3
"""CLI wrapper for wildlife_detection.training.prepare_combined_dataset.

Usage:
    python scripts/training/prepare_combined_dataset.py \
        --output /data/mnt/storage/Datasets/combined_aerial_yolo_640 \
        --tile-size 640 --overlap 120 \
        --sources eikelboom,koger_ungulates,koger_geladas,liege,mmla
"""

import argparse
from pathlib import Path

from wildlife_detection.training.prepare_combined_dataset import (
    get_default_paths,
    prepare_combined_dataset,
)


def main():
    parser = argparse.ArgumentParser(description="Prepare combined aerial wildlife YOLO dataset")
    parser.add_argument("--output", type=str,
                        default="/data/mnt/storage/Datasets/combined_aerial_yolo_640",
                        help="Output directory for combined dataset")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=int, default=120)
    parser.add_argument("--sources", type=str,
                        default="eikelboom,koger_ungulates,koger_geladas,liege,mmla",
                        help="Comma-separated list of sources to include")
    parser.add_argument("--download-mmla", action="store_true",
                        help="Download MMLA Wilds from HuggingFace before conversion")
    args = parser.parse_args()

    prepare_combined_dataset(
        output_dir=Path(args.output),
        sources=[s.strip() for s in args.sources.split(",")],
        tile_size=args.tile_size,
        overlap=args.overlap,
        do_download_mmla=args.download_mmla,
    )


if __name__ == "__main__":
    main()
