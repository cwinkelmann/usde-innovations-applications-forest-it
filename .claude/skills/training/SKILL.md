---
name: local-vision-trainer
description: >
  Train and fine-tune computer vision models locally on a CUDA GPU (optimised for RTX 4080 / 16 GB VRAM).
  Supports object detection with bounding boxes (RTDETRv2, YOLOS, DETR) and image classification
  (ViT, DINOv2, MobileViT, ResNet via timm/transformers). Uses Transformers Trainer API.
  Accepts COCO JSON format for detection and ImageFolder / Hugging Face datasets for classification.
  Logs to WandB and/or TensorBoard. No cloud infrastructure — runs entirely on the local machine.
  Use when the user mentions training a vision model, fine-tuning a detector, COCO dataset,
  bounding box training, image classification, or local GPU training.
---

# Local Vision Trainer Skill

Train object detection and image classification models on your local CUDA GPU.
No Hugging Face Jobs, no cloud costs — everything runs on your machine.

---

## Hardware context

**Target GPU:** RTX 4080 — 16 GB VRAM, Ada Lovelace (supports `bf16`).

| Model family         | Recommended batch size | VRAM usage |
|----------------------|------------------------|------------|
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

Always use `bf16=True` (RTX 4080 supports bfloat16 natively).
Use `gradient_checkpointing=True` when VRAM is tight.

---

## Execution — LOCAL ONLY

> ⚠️ **NEVER use `hf_jobs()`.** Do NOT submit to Hugging Face cloud infrastructure.
> All training MUST run locally via `python` or `uv run`.

```bash
# Detection
python scripts/train_detection.py \
  --model PekingU/rtdetr_r50vd_coco_o365 \
  --dataset_dir ./data/my_coco_dataset \
  --output_dir ./output/detection \
  --epochs 50 \
  --batch_size 8

# Classification
python scripts/train_classification.py \
  --model google/vit-base-patch16-224 \
  --dataset_dir ./data/my_images \
  --output_dir ./output/classification \
  --epochs 30 \
  --batch_size 32

# Instance segmentation (COCO polygons)
python scripts/train_segmentation.py \
  --task instance \
  --model facebook/mask2former-swin-base-coco-instance \
  --dataset_dir ./data/my_coco_dataset \
  --output_dir ./output/seg_instance \
  --epochs 50 \
  --batch_size 4 --grad_accum 2

# Semantic segmentation (image/mask PNGs)
python scripts/train_segmentation.py \
  --task semantic \
  --model nvidia/segformer-b2-finetuned-ade-512-512 \
  --dataset_dir ./data/my_semantic_dataset \
  --num_classes 8 \
  --output_dir ./output/seg_semantic \
  --epochs 80 \
  --batch_size 8
```

When the user asks to train a model:
1. Ask for dataset path, task type (detection / classification / seg_instance / seg_semantic), and base model if not specified
2. Run the validator first: `python scripts/validate_dataset.py --task <task> --dataset_dir <path>`
3. Run the appropriate training script with the correct arguments
4. Start a `wandb` or TensorBoard session if requested
5. Monitor training by tailing logs or reading WandB output

---

## Supported models

### Object detection
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

### Detection — COCO JSON

Expected directory layout:
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

Run the dataset validator before training:
```bash
python scripts/validate_dataset.py --task detection --dataset_dir ./data/my_coco_dataset
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
python scripts/validate_dataset.py --task classification --dataset_dir ./data/my_images
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
python scripts/validate_dataset.py --task seg_instance --dataset_dir ./data/my_coco_dataset
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
python scripts/validate_dataset.py --task seg_semantic --dataset_dir ./data/my_semantic_dataset
```

---

## Logging

### WandB
```bash
pip install wandb
wandb login   # paste API key once
```
Then use `--log wandb --wandb_project my-project --wandb_run my-run`.

### TensorBoard
```bash
pip install tensorboard
tensorboard --logdir ./output/detection/runs &
```
Then use `--log tensorboard`.

Both can be enabled simultaneously: `--log wandb tensorboard`.

---

## Metrics

### Detection
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
| `CUDA out of memory` | Halve `--batch_size`, add `--gradient_checkpointing` |
| `NaN loss on first step` | Lower `--lr` to `1e-5`, check for corrupt images |
| Detection mAP stuck at 0 | Verify COCO bbox is `[x, y, w, h]` not `[x1,y1,x2,y2]` |
| Classification overfitting | Add `--dropout 0.2`, reduce epochs, check class balance |
| Slow data loading | Increase `--num_workers 8` |
| `Assertion failed in NMS` | Ensure no zero-area bounding boxes in annotations |
| Instance mAP = 0 from start | Check polygons have ≥ 3 points, not all iscrowd=1 |
| Semantic mIoU stuck at 0 | Confirm mask pixel values are class ids, not RGB colours |
| Mask size mismatch error | Set `--image_size` consistently and re-run validator |

---

## File index

```
local-vision-trainer/
  SKILL.md                       ← this file (agent instructions)
  scripts/
    train_detection.py           ← object detection (RTDETRv2, YOLOS, DETR)
    train_classification.py      ← image classification (ViT, DINOv2, MobileViT, ResNet)
    train_segmentation.py        ← instance seg (Mask2Former) + semantic seg (SegFormer)
    validate_dataset.py          ← pre-flight checker for all four task types
```

Install dependencies once:
```bash
pip install torch torchvision transformers datasets timm \
            torchmetrics pycocotools albumentations wandb \
            tensorboard pillow tqdm evaluate scikit-learn
```
