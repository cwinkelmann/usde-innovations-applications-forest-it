#!/usr/bin/env bash
# Reproduce the 4 previous WandB training runs from 2026-03-18
# All log to WandB project: wildlife-detection
# Dataset: Eikelboom tiled 640px (3 classes: Elephant, Giraffe, Zebra)
#
# Usage: bash scripts/training/reproduce_runs.sh

set -e
cd "$(dirname "$0")/../.."

DATA="week1/data/eikelboom_yolo_tiled/dataset.yaml"
PROJECT="wildlife-detection"

SORREL="/home/christian/.cache/torch/hub/checkpoints/md_v1000.0.0-sorrel.pt"
LARCH="weights/md_v1000.0.0-larch.pt"
RTDETR="/home/christian/.cache/torch/hub/checkpoints/MDV6-rtdetr-c.pt"

echo "============================================================"
echo "Run 1/4: MD1000-sorrel (no freeze)"
echo "============================================================"
python scripts/training/train_combined_yolo11.py \
  --data "$DATA" \
  --weights "$SORREL" \
  --epochs 50 --batch 16 --imgsz 640 --freeze 0 \
  --weight-decay 0.0001 \
  --project "$PROJECT" \
  --name md1000-sorrel-50ep

echo ""
echo "============================================================"
echo "Run 2/4: MD1000-sorrel (freeze=10, warmup=5)"
echo "============================================================"
python scripts/training/train_combined_yolo11.py \
  --data "$DATA" \
  --weights "$SORREL" \
  --epochs 50 --batch 16 --imgsz 640 --freeze 10 \
  --warmup-epochs 5 --weight-decay 0.0001 \
  --project "$PROJECT" \
  --name sorrel-freeze10-warmup5

echo ""
echo "============================================================"
echo "Run 3/4: MD1000-larch (freeze=10, warmup=5)"
echo "============================================================"
python scripts/training/train_combined_yolo11.py \
  --data "$DATA" \
  --weights "$LARCH" \
  --epochs 50 --batch 16 --imgsz 640 --freeze 10 \
  --warmup-epochs 5 --weight-decay 0.0001 \
  --project "$PROJECT" \
  --name larch-freeze10-warmup5

echo ""
echo "============================================================"
echo "Run 4/4: MDV6-rtdetr-c — SKIPPED (not priority)"
echo "============================================================"
# Uncomment to run RT-DETR:
# python scripts/training/train_combined_yolo11.py \
#   --data "$DATA" \
#   --weights "$RTDETR" \
#   --epochs 50 --batch 4 --imgsz 640 --freeze 10 \
#   --warmup-epochs 5 --weight-decay 0.0001 \
#   --project "$PROJECT" \
#   --name rtdetr-freeze10-warmup5

echo ""
echo "============================================================"
echo "All runs complete! Check https://wandb.ai/karisu/wildlife-detection"
echo "============================================================"
