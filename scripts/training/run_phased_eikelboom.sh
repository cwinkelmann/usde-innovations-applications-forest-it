#!/usr/bin/env bash
# 3-phase progressive unfreezing on Eikelboom with MegaDetector larch
# Total: 50 epochs (10 + 20 + 20) for comparability with single-phase runs
#
# Phase 1: freeze=10 (head only),     lr=1e-3, warmup=5, 15 epochs
# Phase 2: freeze=6  (partial),       lr=1e-4, warmup=3, 20 epochs
# Phase 3: freeze=0  (full finetune), lr=1e-5, warmup=1, 15 epochs
# Total: 50 epochs for comparability with single-phase runs
#
# Usage: bash scripts/training/run_phased_eikelboom.sh

set -e
cd "$(dirname "$0")/../.."

python scripts/training/phased_finetune.py \
  --weights weights/md_v1000.0.0-larch.pt \
  --data week1/data/eikelboom_yolo_tiled/dataset.yaml \
  --project wildlife-detection \
  --batch 16 --imgsz 640 \
  --epochs-p1 15 --epochs-p2 20 --epochs-p3 15
