---
name: local-vision-trainer
description: >
  Train and fine-tune computer vision models locally on a CUDA GPU (optimised for RTX 4080 / 16 GB VRAM).
  Supports YOLOv8 detection (Ultralytics API), Transformers-based detection (RTDETRv2, YOLOS, DETR),
  image classification (ViT, DINOv2, MobileViT, ResNet via timm/transformers), and segmentation
  (Mask2Former, SegFormer). YOLOv8 training accepts YOLO .txt or COCO JSON format (auto-converted).
  Optional SAHI tiled inference after YOLO training for large images.
  Logs to WandB and/or TensorBoard. No cloud infrastructure — runs entirely on the local machine.
  Use when the user mentions training a vision model, fine-tuning a detector, YOLO, YOLOv8,
  COCO dataset, bounding box training, image classification, or local GPU training.
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

### YOLOv8 (Ultralytics) — simplest path for detection

```bash
# YOLO-format dataset (dataset.yaml already exists)
python scripts/training/train_yolo.py \
  --data ./data/yolo_dataset/dataset.yaml \
  --model yolov8n.pt \
  --epochs 50 --batch 16 --imgsz 640

# COCO-format dataset (auto-converts to YOLO)
python scripts/training/train_yolo.py \
  --dataset_dir ./data/my_coco_dataset \
  --model yolov8s.pt \
  --epochs 100 --batch 8

# Fine-tune from previous training run
python scripts/training/train_yolo.py \
  --data ./data/yolo_dataset/dataset.yaml \
  --model ./output/yolo/train/weights/best.pt \
  --epochs 30 --lr0 0.001

# With SAHI tiled inference after training
python scripts/training/train_yolo.py \
  --data ./data/yolo_dataset/dataset.yaml \
  --model yolov8n.pt --epochs 50 \
  --sahi --sahi_slice 640 --sahi_overlap 0.2

# With WandB logging
python scripts/training/train_yolo.py \
  --data ./data/yolo_dataset/dataset.yaml \
  --model yolov8m.pt --epochs 100 \
  --wandb_project wildlife-detection --wandb_run exp01
```

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

1. Determine task type: **yolo** / detection / classification / seg_instance / seg_semantic
2. For quick detection training, default to **YOLOv8** (Ultralytics) unless the user requests a Transformers model
3. Run the validator first: `python scripts/training/validate_dataset.py --task <task> --dataset_dir <path>`
4. Run the appropriate training script
5. Start a `wandb` or TensorBoard session if requested
6. Monitor training by tailing logs or reading WandB output

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

After YOLO training, use `--sahi` to run SAHI (Slicing Aided Hyper Inference) on the
validation set. Essential for drone/satellite imagery where objects are small relative
to the image.

```bash
python scripts/training/train_yolo.py \
  --data ./data/yolo_dataset/dataset.yaml \
  --model yolov8n.pt --epochs 50 \
  --sahi --sahi_slice 640 --sahi_overlap 0.2 --sahi_conf 0.2
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
wandb login   # paste API key once
```

For YOLO: `--wandb_project my-project --wandb_run my-run`
For Transformers: `--log wandb --wandb_project my-project --wandb_run my-run`

### TensorBoard
```bash
pip install tensorboard
tensorboard --logdir ./output/detection/runs &
```
For Transformers: `--log tensorboard`

Both can be enabled simultaneously for Transformers trainers: `--log wandb tensorboard`.

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
scripts/training/
  train_yolo.py              ← YOLOv8 detection (Ultralytics API, SAHI support)
  train_detection.py         ← object detection (RTDETRv2, YOLOS, DETR — Transformers)
  train_classification.py    ← image classification (ViT, DINOv2, MobileViT, ResNet)
  train_segmentation.py      ← instance seg (Mask2Former) + semantic seg (SegFormer)
  validate_dataset.py        ← pre-flight checker for all five task types
```

Install dependencies:
```bash
# Core (all trainers)
pip install torch torchvision pillow tqdm

# YOLOv8
pip install ultralytics

# SAHI (optional, for tiled inference)
pip install sahi

# Transformers trainers
pip install transformers datasets timm torchmetrics pycocotools \
            albumentations evaluate scikit-learn

# Logging (optional)
pip install wandb tensorboard
```
