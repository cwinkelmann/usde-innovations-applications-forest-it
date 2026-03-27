# scripts/ — Training & Evaluation CLI

All scripts are thin CLI wrappers around modules in `src/wildlife_detection/training/`.
Training logs to WandB project `wildlife-detection` (https://wandb.ai/karisu/wildlife-detection).

## Training Scripts

### `training/train_combined_yolo11.py`
Fine-tune YOLO or RT-DETR models. Auto-detects RT-DETR weights and applies `amp=False` + lower LR.

```bash
# YOLO11L (MegaDetector larch) on Eikelboom
python scripts/training/train_combined_yolo11.py \
    --weights weights/md_v1000.0.0-larch.pt \
    --data week1/data/eikelboom_yolo_tiled/dataset.yaml \
    --epochs 50 --batch 16 --freeze 10 --name larch-freeze10

# RT-DETR (auto-detects, sets amp=False + lr0=0.0001)
python scripts/training/train_combined_yolo11.py \
    --weights ~/.cache/torch/hub/checkpoints/MDV6-rtdetr-c.pt \
    --batch 4 --name rtdetr-freeze10
```

**Module:** `wildlife_detection.training.train_yolo_combined`

### `training/phased_finetune.py`
3-phase progressive unfreezing (Phase 1: head only → Phase 2: partial unfreeze → Phase 3: full).
Each phase loads `best.pt` from the previous. Based on strategy in `doc/fine_tuning_yolo11.md`.

```bash
python scripts/training/phased_finetune.py \
    --weights weights/md_v1000.0.0-larch.pt \
    --data week1/data/eikelboom_yolo_tiled/dataset.yaml \
    --epochs-p1 10 --epochs-p2 20 --epochs-p3 20
```

**Module:** `wildlife_detection.training.phased_finetune`

### `training/reproduce_runs.sh`
Re-runs the 4 original WandB experiments from 2026-03-18 (sorrel no-freeze, sorrel freeze=10,
larch freeze=10, rtdetr freeze=10). All on Eikelboom tiled, all log to `wildlife-detection`.

```bash
bash scripts/training/reproduce_runs.sh
```

## Data Preparation

### `training/prepare_combined_dataset.py`
Converts 5 aerial wildlife datasets to unified YOLO 640px tiles with MegaDetector classes
(animal=0, person=1). Tiles large images, remaps species classes, merges into one directory.

```bash
python scripts/training/prepare_combined_dataset.py \
    --output /data/mnt/storage/Datasets/combined_aerial_yolo_640 \
    --sources eikelboom,koger_ungulates,koger_geladas,liege,mmla \
    --download-mmla
```

**Module:** `wildlife_detection.training.prepare_combined_dataset`

## Evaluation

### `training/eval_eikelboom.py`
Evaluate any trained model on the Eikelboom test set. Remaps species labels to MegaDetector
class 0 (animal) so models trained with MegaDetector classes can be evaluated against the
original species-annotated test tiles.

```bash
python scripts/training/eval_eikelboom.py \
    --weights wildlife-detection/larch-freeze10/weights/best.pt
```

**Module:** `wildlife_detection.training.eval_eikelboom`

## History

- `evaluate_detectors.py` — model comparison script, created in commit `e1cb40e` (2026-03-18),
  deleted in `bd1e81c` (repo cleanup). Produced `output/comparison_2ep.json` with per-class
  P/R/F1/mAP for RT-DETR 2-epoch baseline on Eikelboom val (conf=0.3). Still in git history.
- `train_megadetector.py` — original MegaDetector fine-tuning script, created in `e1cb40e`,
  deleted in `bd1e81c`. Supported MD1000-larch/sorrel and MDV6-rtdetr-c with auto-download
  and RT-DETR gotcha handling. Replaced by `train_combined_yolo11.py`.
- `convert_dataset.py` — format converter (CSV/COCO/YOLO), created in `e1cb40e`, deleted in
  `bd1e81c`. Replaced by conversion logic in `prepare_combined_dataset.py`.

## Available Model Weights

| Weight | Path | Architecture |
|--------|------|-------------|
| MD1000-larch | `weights/md_v1000.0.0-larch.pt` | YOLO11L (25M params, MIT) |
| MD1000-sorrel | `~/.cache/torch/hub/checkpoints/md_v1000.0.0-sorrel.pt` | YOLO11S (9M params, MIT) |
| MDV6-rtdetr-c | `~/.cache/torch/hub/checkpoints/MDV6-rtdetr-c.pt` | RT-DETR-L (32M params, AGPL) |
| MMLA YOLOv11m | `~/.cache/huggingface/hub/models--imageomics--mmla/.../best.pt` | YOLO11M (20M params) |

## WandB Runs

All runs log to project **`wildlife-detection`** on wandb.ai. The `project` CLI arg doubles
as the WandB project name (ultralytics convention).

Previous runs from 2026-03-18 are stored locally in `wandb/run-20260318_*`. Their exact CLI
args can be recovered from `wandb/run-*/files/wandb-metadata.json` → `args` field.
