---
name: local-vision-trainer
description: >
  Train and fine-tune computer vision models locally on a CUDA GPU (optimised for RTX 4080 / 16 GB VRAM).
  Supports MegaDetector fine-tuning (MD1000-larch, MD1000-sorrel, MDV6-rtdetr-c via train_megadetector.py),
  YOLOv8 detection, Transformers-based detection (RTDETRv2, YOLOS, DETR),
  image classification (ViT, DINOv2, MobileViT, ResNet via timm/transformers), and segmentation
  (Mask2Former, SegFormer). Logs to WandB and/or TensorBoard.
  No cloud infrastructure — runs entirely on the local machine.
  Use when the user mentions training a vision model, fine-tuning a detector, MegaDetector,
  YOLO, YOLOv8, COCO dataset, bounding box training, image classification, or local GPU training.
---

# Local Vision Trainer Skill

Train object detection and image classification models on your local CUDA GPU.
No Hugging Face Jobs, no cloud costs — everything runs on your machine.

---

## Hardware context

**Target GPU:** RTX 4080 — 16 GB VRAM, Ada Lovelace (supports `bf16`).

| Model family         | Recommended batch size | VRAM usage |
|----------------------|------------------------|------------|
| YOLOv8-nano          | 32–64                  | ~2–4 GB    |
| YOLOv8-small         | 16–32                  | ~4–6 GB    |
| YOLOv8-medium        | 8–16                   | ~6–10 GB   |
| YOLOv8-large         | 4–8                    | ~10–14 GB  |
| YOLOv8-xlarge        | 2–4                    | ~14–16 GB  |
| RTDETRv2-small       | 8–16                   | ~8–10 GB   |
| RTDETRv2-large       | 4–6                    | ~13–15 GB  |
| MD1000-larch (YOLO11-L)  | 8–16               | ~6–8 GB    |
| MD1000-sorrel (YOLO11-S) | 16–32              | ~3–5 GB    |
| MDV6-rtdetr-c (.pt)  | 4–8                    | ~10–12 GB  |
| YOLOS-tiny           | 16–32                  | ~4–6 GB    |
| YOLOS-base           | 8–12                   | ~9–11 GB   |
| DETR-resnet50        | 8–12                   | ~8–10 GB   |
| ViT-base (cls)       | 32–64                  | ~6–8 GB    |
| DINOv2-base (cls)    | 32–48                  | ~8–10 GB   |
| MobileViT-small (cls)| 64–128                 | ~3–5 GB    |
| Mask2Former-swin-tiny (seg)  | 4–6             | ~6–8 GB    |
| Mask2Former-swin-base (seg)  | 2–4             | ~10–12 GB  |
| Mask2Former-swin-large (seg) | 1–2 + grad_ckpt | ~14–16 GB  |
| SegFormer-B2 (semantic)      | 8–16            | ~5–7 GB    |
| SegFormer-B5 (semantic)      | 4–8             | ~9–11 GB   |

Always use `bf16=True` for Transformers trainers (RTX 4080 supports bfloat16 natively).
Use `gradient_checkpointing=True` when VRAM is tight.

---

## Execution — LOCAL ONLY

> **NEVER use `hf_jobs()`.** Do NOT submit to Hugging Face cloud infrastructure.
> All training MUST run locally via `python` or `uv run`.

### Dataset format conversion

Before training, convert annotations to the format your trainer expects.
Run `convert_dataset.py` first, then `validate_dataset.py`, then train.

```bash
# Eikelboom CSV → COCO JSON (for train_detection.py)
python scripts/training/convert_dataset.py \
  --from eikelboom-csv \
  --src  ./week1/data/eikelboom \
  --dst  ./week1/data/eikelboom_coco

# Generic Pascal VOC CSV (with header) → COCO JSON
python scripts/training/convert_dataset.py \
  --from pascal-voc-csv \
  --src  ./data/my_dataset \
  --dst  ./data/my_dataset_coco \
  --csv-train annotations_train.csv \
  --csv-val   annotations_val.csv \
  --header

# COCO JSON → YOLO .txt labels
python scripts/training/convert_dataset.py \
  --from coco --to yolo \
  --src ./data/my_coco_dataset \
  --dst ./data/my_yolo_dataset

# YOLO .txt → COCO JSON
python scripts/training/convert_dataset.py \
  --from yolo --to coco \
  --src ./data/my_yolo_dataset \
  --dst ./data/my_coco_dataset

# Pascal VOC CSV → YOLO (two-step: CSV → COCO → YOLO)
python scripts/training/convert_dataset.py \
  --from pascal-voc-csv --to yolo \
  --src ./data/my_dataset \
  --dst ./data/my_yolo_dataset

# Use --symlink to avoid copying images (saves disk space)
python scripts/training/convert_dataset.py \
  --from eikelboom-csv \
  --src ./week1/data/eikelboom \
  --dst ./week1/data/eikelboom_coco \
  --symlink
```

Supported format pairs:

| Source | Target | Notes |
|--------|--------|-------|
| `eikelboom-csv` | `coco` | No-header CSV: `filepath,x1,y1,x2,y2,class` |
| `pascal-voc-csv` | `coco` | Same as above, supports `--header` flag |
| `coco` | `yolo` | COCO `[x,y,w,h]` → YOLO normalised `cx cy w h` |
| `yolo` | `coco` | Requires `dataset.yaml` with class names |
| `eikelboom-csv` | `yolo` | Two-step: CSV → temp COCO → YOLO |
| `pascal-voc-csv` | `yolo` | Two-step: CSV → temp COCO → YOLO |

### MegaDetector fine-tuning (recommended for wildlife detection)

Fine-tune MegaDetector models using `train_megadetector.py`.
Weights are auto-downloaded on first use. Dataset must be in YOLO format.
All MegaDetector models detect 3 classes: **animal, person, vehicle**.

#### Three models verified to fine-tune with ultralytics

| Model | Architecture | Params | License | Input | Speed | Accuracy |
|-------|-------------|--------|---------|-------|-------|----------|
| **MD1000-larch** | YOLO11-L | 25M | MIT | 640px | Fast (3.5ms) | Best overall |
| **MD1000-sorrel** | YOLO11-S | 9M | MIT | 960px | Fastest (1.6ms) | Good, highest precision |
| **MDV6-rtdetr-c** | RT-DETR-L | 32M | AGPL | 1280px | Slower (8.7ms) | Best recall/mAP, NMS-free |

**2-epoch benchmark on Eikelboom aerial wildlife data (val, conf=0.3):**

| Model | Precision | Recall | F1 | mAP@0.5 |
|-------|-----------|--------|-----|---------|
| MD1000-sorrel | **0.714** | 0.566 | **0.632** | 0.555 |
| MD1000-larch | 0.650 | 0.588 | 0.617 | 0.555 |
| MDV6-rtdetr-c | 0.411 | **0.795** | 0.542 | **0.696** |

**Recommendation:** Use **MD1000-larch** for general fine-tuning (MIT license, stable training,
best balance). Use **MDV6-rtdetr-c** only if you need maximum recall or NMS-free inference.
Use **MD1000-sorrel** for edge/CPU deployment or when precision matters most.

#### Other models in the registry (not all trainable)

| Model | Status | Notes |
|-------|--------|-------|
| `MD1000-cedar` | ⚠️ Needs `yolov9pip` | YOLOv9c, not plain ultralytics |
| `MD1000-redwood` | ❌ YOLOv5 format | Cannot train with ultralytics v8 |
| `MD1000-spruce` | ❌ YOLOv5 format | Cannot train with ultralytics v8 |
| `MDV6-yolov9-c/e` | ⚠️ Untested | PytorchWildlife AGPL |
| `MDV6-yolov10-c/e` | ⚠️ Untested | PytorchWildlife AGPL |
| `MDv5a/b` | ❌ YOLOv5 format | Cannot train with ultralytics v8 |

```bash
# Fine-tune MD1000-larch (recommended)
python scripts/training/train_megadetector.py \
  --model MD1000-larch \
  --data ./week1/data/eikelboom_yolo_tiled/dataset.yaml \
  --epochs 50 --batch 16 --imgsz 640 \
  --log wandb --wandb_project wildlife-detection

# Fine-tune MD1000-sorrel (small/fast)
python scripts/training/train_megadetector.py \
  --model MD1000-sorrel \
  --data ./week1/data/eikelboom_yolo_tiled/dataset.yaml \
  --epochs 50 --batch 16 --imgsz 640

# Fine-tune MDV6 RT-DETR (amp=False set automatically)
python scripts/training/train_megadetector.py \
  --model MDV6-rtdetr-c \
  --data ./week1/data/eikelboom_yolo_tiled/dataset.yaml \
  --epochs 50 --batch 8 --imgsz 640

# Evaluate a fine-tuned model
python scripts/training/train_megadetector.py \
  --model ./output/md1000_larch_finetune/weights/best.pt \
  --data ./week1/data/eikelboom_yolo_tiled/dataset.yaml \
  --eval_only

# Compare multiple models (P, R, F1, mAP)
python scripts/training/evaluate_detectors.py \
  --model yolo yolo yolo \
  --weights ./output/sorrel/best.pt ./output/larch/best.pt ./output/rtdetr/best.pt \
  --dataset_dir ./week1/data/eikelboom_coco_tiled \
  --split val --conf_thres 0.3
```

**RT-DETR gotchas** (handled automatically by `train_megadetector.py`):
- `amp=False` — AMP causes NaN losses in RT-DETR bipartite matching
- `deterministic=False` — `F.grid_sample` rejects deterministic mode
- Lower LR required (`lr0=0.0001` vs `0.001` for YOLO)
- `freeze` has no effect on RT-DETR (backbone not cleanly separable)

### Transformers-based detection

```bash
python scripts/training/train_detection.py \
  --model PekingU/rtdetr_r50vd_coco_o365 \
  --dataset_dir ./data/my_coco_dataset \
  --output_dir ./output/detection \
  --epochs 50 \
  --batch_size 8
```

### Classification

```bash
python scripts/training/train_classification.py \
  --model google/vit-base-patch16-224 \
  --dataset_dir ./data/my_images \
  --output_dir ./output/classification \
  --epochs 30 \
  --batch_size 32
```

### Instance segmentation (COCO polygons)

```bash
python scripts/training/train_segmentation.py \
  --task instance \
  --model facebook/mask2former-swin-base-coco-instance \
  --dataset_dir ./data/my_coco_dataset \
  --output_dir ./output/seg_instance \
  --epochs 50 \
  --batch_size 4 --grad_accum 2
```

### Semantic segmentation (image/mask PNGs)

```bash
python scripts/training/train_segmentation.py \
  --task semantic \
  --model nvidia/segformer-b2-finetuned-ade-512-512 \
  --dataset_dir ./data/my_semantic_dataset \
  --num_classes 8 \
  --output_dir ./output/seg_semantic \
  --epochs 80 \
  --batch_size 8
```

### When the user asks to train a model:

1. Determine task type: **megadetector** / detection / classification / seg_instance / seg_semantic
2. For wildlife detection, default to **MegaDetector fine-tuning** via `train_megadetector.py` (MD1000-larch recommended)
2.1. The MMLA variant is good for aerial imagery: https://imageomics.github.io/mmla/
2.2. For combined multi-dataset training, use `prepare_combined_dataset.py` + `train_combined_yolo11.py`
     (see "Combined aerial wildlife dataset pipeline" section below)
2.3. Dataset research and strategy documented in `doc/fine_tuning_yolo11.md`
3. For generic detection, use YOLOv8 or Transformers models
4. Ensure dataset is in correct format (YOLO .txt for MegaDetector/YOLO, COCO JSON for Transformers)
5. Run the appropriate training script with `--log wandb` if requested
6. After training, compare models with `evaluate_detectors.py`

---

## Supported models

### YOLOv8 (Ultralytics) — detection

| Model         | Size  | Params | Notes                       |
|---------------|-------|--------|-----------------------------|
| `yolov8n.pt`  | 3 MB  | 3.2M   | Nano — fastest, great for iteration |
| `yolov8s.pt`  | 11 MB | 11.2M  | Small — good speed/accuracy balance |
| `yolov8m.pt`  | 26 MB | 25.9M  | Medium — production quality  |
| `yolov8l.pt`  | 44 MB | 43.7M  | Large — high accuracy        |
| `yolov8x.pt`  | 68 MB | 68.2M  | XLarge — maximum accuracy    |

All variants are pre-trained on COCO (80 classes). Fine-tuning replaces the
classification head for your custom classes automatically.

Custom weights from a previous training run can be used as `--model path/to/best.pt`.

### Transformers — detection

| Model ID (HF Hub)                          | Architecture  | Notes                        |
|--------------------------------------------|---------------|------------------------------|
| `PekingU/rtdetr_r50vd_coco_o365`           | RTDETRv2-s    | Best speed/accuracy balance  |
| `PekingU/rtdetr_r101vd_coco_o365`          | RTDETRv2-l    | Higher accuracy, more VRAM   |
| `hustvl/yolos-tiny`                        | YOLOS-tiny    | Lightest, fast iteration     |
| `hustvl/yolos-base`                        | YOLOS-base    | Good baseline                |
| `facebook/detr-resnet-50`                  | DETR          | Classic, widely cited        |
| `facebook/detr-resnet-101`                 | DETR-101      | Stronger backbone            |

### Image classification

| Model ID (HF Hub)                          | Architecture  | Notes                        |
|--------------------------------------------|---------------|------------------------------|
| `google/vit-base-patch16-224`              | ViT-B/16      | Strong general baseline      |
| `facebook/dinov2-base`                     | DINOv2-B      | Excellent features           |
| `facebook/dinov2-large`                    | DINOv2-L      | Best accuracy, ~12 GB        |
| `apple/mobilevit-small`                    | MobileViT-S   | Lightweight, fast            |
| `microsoft/resnet-50`                      | ResNet-50     | timm-compatible              |

### Segmentation — instance

| Model ID (HF Hub)                                          | Architecture        | Notes                           |
|------------------------------------------------------------|---------------------|---------------------------------|
| `facebook/mask2former-swin-tiny-coco-instance`             | Mask2Former-T       | Fastest, lowest VRAM            |
| `facebook/mask2former-swin-small-coco-instance`            | Mask2Former-S       | Good speed/accuracy             |
| `facebook/mask2former-swin-base-coco-instance`             | Mask2Former-B       | Best default choice             |
| `facebook/mask2former-swin-large-coco-instance`            | Mask2Former-L       | Highest accuracy, needs grad_ckpt |

### Segmentation — semantic

| Model ID (HF Hub)                                          | Architecture  | Notes                           |
|------------------------------------------------------------|---------------|---------------------------------|
| `nvidia/segformer-b0-finetuned-ade-512-512`                | SegFormer-B0  | Tiny, very fast                 |
| `nvidia/segformer-b2-finetuned-ade-512-512`                | SegFormer-B2  | Best default choice             |
| `nvidia/segformer-b5-finetuned-ade-512-512`                | SegFormer-B5  | Highest accuracy                |
| `facebook/mask2former-swin-base-ade-semantic`              | Mask2Former-B | Universal model, semantic mode  |

---

## Dataset formats

### YOLOv8 — YOLO .txt format

```
dataset_dir/
  dataset.yaml              ← references paths, class names
  images/
    train/   img001.jpg ...
    val/     img002.jpg ...
  labels/
    train/   img001.txt ...  ← one .txt per image
    val/     img002.txt ...
```

Each `.txt` label file has one line per object: `class_id cx cy w h` (all normalised 0–1):
```
0  0.512  0.344  0.231  0.189     ← class 0, centre x/y, width, height
```

`dataset.yaml`:
```yaml
path: /absolute/path/to/dataset_dir
train: images/train
val: images/val

nc: 3
names: ['animal', 'person', 'vehicle']
```

Run the validator:
```bash
python scripts/training/validate_dataset.py --task yolo --dataset_dir ./data/yolo_dataset
```

**COCO JSON auto-conversion:** If you have COCO format instead, use `--dataset_dir`
instead of `--data` and `train_yolo.py` will convert automatically. It creates
a `yolo_converted/` directory with symlinked images and generated .txt labels.

### Detection — COCO JSON (Transformers trainers)

```
dataset_dir/
  train/
    images/          # .jpg / .png files
    annotations.json # COCO format
  val/
    images/
    annotations.json
```

COCO JSON schema (standard):
```json
{
  "images": [{"id": 1, "file_name": "img001.jpg", "width": 1280, "height": 720}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 2,
                   "bbox": [x, y, w, h], "area": 1234, "iscrowd": 0}],
  "categories": [{"id": 1, "name": "background"}, {"id": 2, "name": "zebra"}]
}
```

Run the validator:
```bash
python scripts/training/validate_dataset.py --task detection --dataset_dir ./data/my_coco_dataset
```

### Classification — ImageFolder

```
dataset_dir/
  train/
    class_a/  img001.jpg  img002.jpg ...
    class_b/  img003.jpg  ...
  val/
    class_a/  ...
    class_b/  ...
```

Or a Hugging Face dataset ID: `--dataset_dir username/my-hf-dataset`

Run the validator:
```bash
python scripts/training/validate_dataset.py --task classification --dataset_dir ./data/my_images
```

### Instance segmentation — COCO polygons

Same directory layout as detection, but annotations must include `segmentation` polygons:

```json
{
  "annotations": [{
    "id": 1, "image_id": 1, "category_id": 2,
    "bbox": [x, y, w, h],
    "segmentation": [[x1,y1,x2,y2,x3,y3,...]],
    "area": 1234, "iscrowd": 0
  }]
}
```

Run the validator:
```bash
python scripts/training/validate_dataset.py --task seg_instance --dataset_dir ./data/my_coco_dataset
```

### Semantic segmentation — image/mask PNGs

```
dataset_dir/
  label_map.json          ← {"0": "background", "1": "zebra", "2": "elephant"}
  train/
    images/   img001.jpg ...
    masks/    img001.png ...   ← single-channel PNG, pixel value = class id, 255 = ignore
  val/
    images/
    masks/
```

Masks must be **single-channel** (mode `L` or `P`) PNGs. Pixel value = integer class id.
Use `255` for unlabelled / ignore regions. Filenames must match between `images/` and `masks/` (extension can differ).

Run the validator:
```bash
python scripts/training/validate_dataset.py --task seg_semantic --dataset_dir ./data/my_semantic_dataset
```

---

## SAHI — tiled inference for large images

After training, use SAHI (Slicing Aided Hyper Inference) for inference on large images.
Essential for drone/satellite imagery where objects are small relative to the image.
See the `sahi-inference` skill for full details.

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path="./output/md1000_larch_finetune/weights/best.pt",
    confidence_threshold=0.2,
    device="cuda:0",
)
result = get_sliced_prediction(
    "large_image.jpg", model,
    slice_height=640, slice_width=640, overlap_height_ratio=0.2, overlap_width_ratio=0.2,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sahi_slice` | 640 | Tile height/width in pixels |
| `--sahi_overlap` | 0.2 | Overlap ratio between tiles |
| `--sahi_conf` | 0.2 | Confidence threshold |

SAHI results are saved as JSON in the training output directory.

Requires: `pip install sahi`

---

## Logging

### WandB
```bash
pip install wandb
wandb login   # paste API key once — stored in ~/.netrc

# Enable wandb in ultralytics settings (one-time, persisted)
python -c "from ultralytics import settings; settings.update(wandb=True)"
```

For MegaDetector: `--log wandb --wandb_project my-project --wandb_run my-run`
For Transformers: `--log wandb --wandb_project my-project --wandb_run my-run`

MegaDetector training logs P, R, mAP, loss curves (built-in ultralytics) + F1 (custom callback).

### TensorBoard
```bash
pip install tensorboard
tensorboard --logdir ./output/detection/runs &
```
For MegaDetector: `--log tensorboard`
For Transformers: `--log tensorboard`

Both can be enabled simultaneously: `--log wandb tensorboard`.

---

## Metrics

### YOLOv8 (Ultralytics)
- **mAP50(B)** — primary metric
- **mAP50-95(B)** — stricter, averaged over IoU thresholds
- **precision(B)** / **recall(B)**
- Computed by Ultralytics internally; reported per epoch

### Detection (Transformers)
- **mAP@0.5** — COCO primary metric
- **mAP@0.5:0.95** — stricter, averaged over IoU thresholds
- **mAR@100** — recall at max 100 detections per image
- Computed with `torchmetrics.detection.MeanAveragePrecision`

### Classification
- **Accuracy** (top-1)
- **F1** (macro, weighted)
- **Per-class accuracy** printed at end of each epoch

### Segmentation — instance
- **pixel_accuracy_proxy** — used during training for checkpoint selection
- For full COCO mask mAP, run `pycocotools` evaluation on saved predictions post-training

### Segmentation — semantic
- **mean_iou** — primary metric, averaged over all classes (ignoring id 255)
- **mean_accuracy** — per-class pixel accuracy averaged over classes

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | Halve `--batch` / `--batch_size`, use smaller model variant |
| `NaN loss on first step` | Lower `--lr0` / `--lr` to `1e-5`, check for corrupt images |
| YOLO mAP stuck at 0 | Verify .txt labels are `class_id cx cy w h` (normalised 0–1) |
| Detection mAP stuck at 0 | Verify COCO bbox is `[x, y, w, h]` not `[x1,y1,x2,y2]` |
| Classification overfitting | Add `--dropout 0.2`, reduce epochs, check class balance |
| Slow data loading | Increase `--workers` / `--num_workers` to 8 |
| `Assertion failed in NMS` | Ensure no zero-area bounding boxes in annotations |
| Instance mAP = 0 from start | Check polygons have ≥ 3 points, not all iscrowd=1 |
| Semantic mIoU stuck at 0 | Confirm mask pixel values are class ids, not RGB colours |
| Mask size mismatch error | Set `--image_size` consistently and re-run validator |
| YOLO "no labels found" | Ensure labels/ mirrors images/ directory structure |
| SAHI import error | Run `pip install sahi` |
| RT-DETR NaN loss | Set `amp=False` (done automatically by `train_megadetector.py`) |
| RT-DETR mAP drops while loss drops | Overfitting — add `patience=10`, reduce epochs |
| MD1000-spruce/redwood won't train | YOLOv5 format — incompatible with ultralytics v8 training |
| WandB not logging | Run `python -c "from ultralytics import settings; settings.update(wandb=True)"` |

---

## YOLOv8 augmentation flags

| Flag | Default | Description |
|------|---------|-------------|
| `--mosaic` | 1.0 | Mosaic augmentation probability |
| `--mixup` | 0.0 | MixUp augmentation probability |
| `--degrees` | 0.0 | Rotation range (degrees) |
| `--flipud` | 0.0 | Vertical flip probability |
| `--fliplr` | 0.5 | Horizontal flip probability |
| `--no_augment` | off | Disable all augmentation |

---

## File index

```
scripts/training/                           (CLI wrappers — logic in src/wildlife_detection/training/)
  prepare_combined_dataset.py ← ✅ merge 5 datasets into unified YOLO 640px tiles
  train_combined_yolo11.py    ← ✅ fine-tune YOLO or RT-DETR (auto-detects model type)
  phased_finetune.py          ← ✅ 3-phase progressive unfreezing (head → partial → full)
  eval_eikelboom.py           ← ✅ evaluate on Eikelboom test set (remaps to MegaDetector classes)
  reproduce_runs.sh           ← ✅ re-run all 4 original WandB experiments from 2026-03-18

src/wildlife_detection/training/            (implementation modules)
  prepare_combined_dataset.py ← dataset conversion, tiling, class remapping
  train_yolo_combined.py      ← YOLO/RT-DETR training with auto RT-DETR detection
  phased_finetune.py          ← 3-phase TrainConfig/PhaseConfig dataclasses + runner
  eval_eikelboom.py           ← Eikelboom test set setup + evaluation

scripts/training/  (deleted in repo cleanup, recoverable from git commit e1cb40e)
  train_megadetector.py       ← original MegaDetector fine-tuning (replaced by train_combined_yolo11.py)
  evaluate_detectors.py       ← model comparison P/R/F1/mAP (produced output/comparison_2ep.json)
  convert_dataset.py          ← format converter CSV/COCO/YOLO (replaced by prepare_combined_dataset.py)

scripts/training/  (not yet implemented)
  validate_dataset.py         ← pre-flight checker for all five task types
  train_detection.py          ← object detection (RTDETRv2, YOLOS, DETR — Transformers)
  train_classification.py     ← image classification (ViT, DINOv2, MobileViT, ResNet)
  train_segmentation.py       ← instance seg (Mask2Former) + semantic seg (SegFormer)
  calibrate_confidence.py     ← find optimal confidence threshold for animal counting
  diagnose_training.py        ← detect early dip, overfitting, LR issues; suggest fixes
```

### Training artifacts in output/

| File | Origin | Contents |
|------|--------|----------|
| `output/comparison_2ep.json` | `evaluate_detectors.py` (git `e1cb40e`, now deleted) | Per-class P/R/F1 + mAP for RT-DETR 2-epoch run on Eikelboom val, conf=0.3 |
| `output/dfine_nano_smoke/` | Transformers Trainer | D-FINE nano smoke test (3 classes, HGNetV2 backbone) |
| `output/eikelboom_rtdetr_tiled/` | Transformers Trainer | RTDETRv2 on Eikelboom tiles (checkpoints 246, 12054, 12300) |
| `output/mdv6_finetune_test/` | Transformers Trainer | MDV6 RT-DETR test run |

### WandB run history

Previous runs from 2026-03-18 in `wandb/run-20260318_*`. CLI args recoverable from
`wandb/run-*/files/wandb-metadata.json` → `args` field. Config values in
`wandb/run-*/files/config.yaml`.

### Combined aerial wildlife dataset pipeline (implemented)

Prepares and trains on a unified dataset from 5 aerial wildlife sources, all mapped to
MegaDetector classes (animal=0, person=1):

```bash
# Step 1: Prepare combined dataset (tiles all sources to 640×640 YOLO format)
python scripts/training/prepare_combined_dataset.py \
  --output /data/mnt/storage/Datasets/combined_aerial_yolo_640 \
  --tile-size 640 --overlap 120 \
  --sources eikelboom,koger_ungulates,koger_geladas,liege,mmla \
  --download-mmla  # downloads MMLA Wilds from HuggingFace if missing

# Step 2: Train YOLO11L (MegaDetector larch weights, frozen backbone)
python scripts/training/train_combined_yolo11.py \
  --data /data/mnt/storage/Datasets/combined_aerial_yolo_640/dataset.yaml \
  --weights weights/md_v1000.0.0-larch.pt \
  --epochs 50 --batch 16 --freeze 10

# Step 3: Evaluate on Eikelboom test set
python scripts/training/eval_eikelboom.py \
  --weights output/combined_yolo11/larch-freeze10-combined/weights/best.pt
```

**Datasets included:**

| Dataset | Source Format | Class Mapping |
|---------|-------------|---------------|
| Eikelboom (tiled) | YOLO 640px | Elephant/Giraffe/Zebra → animal |
| MMLA Wilds | YOLO native (HuggingFace) | zebra/giraffe/onager/dog → animal |
| Koger Ungulates | COCO JSON | zebra/gazelle/wbuck/buffalo/other → animal |
| Koger Geladas | COCO JSON | gelada/adult_male → animal, human → person |
| Liège Multispecies | CSV + COCO JSON | 6 African species → animal |

**MegaDetector larch weights:** `weights/md_v1000.0.0-larch.pt` (auto-downloaded)

**Research reference:** `doc/fine_tuning_yolo11.md` — comprehensive analysis of the 5 aerial
wildlife datasets, the Eikelboom benchmark's oblique-aerial characteristics, domain gap
quantification (20–50 point mAP drop for COCO-pretrained models on aerial data), and
the evidence-backed 3-phase training strategy (freeze backbone → progressive unfreeze →
full fine-tune). Key findings:
- MMLA study showed 52-point mAP50 improvement from fine-tuning (30% → 82%)
- Dataset coherence matters more than raw diversity (WAID cross-dataset experiment)
- Eikelboom is oblique manned-aircraft imagery, not nadir drone — training data should cover both
- No existing paper directly optimises training mixtures for the Eikelboom test set

Install dependencies:
```bash
# Core (all trainers)
pip install torch torchvision pillow tqdm

# Ultralytics (MegaDetector + YOLOv8)
pip install ultralytics

# SAHI (optional, for tiled inference)
pip install sahi

# Transformers trainers
pip install transformers datasets timm torchmetrics pycocotools \
            albumentations evaluate scikit-learn

# Logging (optional)
pip install wandb tensorboard
```


### Experiment Tracking:
Log to wandb, enable it if necessary.

```
from ultralytics import settings
settings.update(wandb=True)

```