# Iguana Training Case Study

Walk-through of the actual iguana classification training code from the Iguanas From Above project. This documents the real scripts at `/Users/christian/PycharmProjects/hnee/pytorch-image-models/` -- `iguana_train.py`, `run_training_iguana.sh`, and `timm/data/dataset_iguana.py`.

---

## Project Context

The Iguanas From Above project uses drone imagery to count marine iguanas (*Amblyrhynchus cristatus*) on the Galapagos Islands. The classification task is binary: determine whether a 512x512 or 518x518 pixel tile extracted from a drone orthomosaic contains at least one iguana.

**Sites:** Floreana, Fernandina South (Fernandina_s), Fernandina Medium (Fernandina_m)
**Tile sizes:** 512x512 for CNNs, 518x518 for DINOv2 ViTs (518 = 37 x 14 patches)

---

## Training Script: `iguana_train.py`

This is a modified version of timm's standard `train.py` with the full training pipeline. Key imports:

```python
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import NativeScaler
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy
```

The script uses timm's standard argparse interface with arguments like `--model`, `--data-dir`, `--train-split`, `--val-split`, `--batch-size`, `--lr`, `--opt`, `--weight-decay`, `--epochs`, `--input-size`, `--pretrained`, `--amp`.

---

## Shell Launcher: `run_training_iguana.sh`

The shell script systematically trains multiple model architectures on multiple island datasets.

### CNN Models (512x512)

```bash
# Models tested with 512x512 input:
MODEL_BATCH_SIZES["timm/dla34.in1k"]=200
MODEL_BATCH_SIZES["timm/dla102x.in1k"]=80
MODEL_BATCH_SIZES["timm/resnet34.a1_in1k"]=100
MODEL_BATCH_SIZES["timm/resnet50.a1_in1k"]=128
MODEL_BATCH_SIZES["timm/resnet152.tv2_in1k"]=50
MODEL_BATCH_SIZES["timm/efficientnet_b0.ra_in1k"]=200
MODEL_BATCH_SIZES["timm/convnext_tiny.in12k_ft_in1k"]=64
MODEL_BATCH_SIZES["timm/convnextv2_huge.fcmae_ft_in1k"]=8

# Training command pattern for CNNs:
python train.py \
    --model "timm/resnet152.tv2_in1k" \
    --data-dir ./training_data/iguana/labels_512_overlap_0/Floreana_classification/ \
    --train-split classification_train_0_512 \
    --val-split classification_val_0_512 \
    --batch-size 50 \
    --opt adamw \
    --lr 0.00001 \                    # 1e-5 for CNNs
    --weight-decay 0.05 \
    --epochs 100 \
    --input-size 3 512 512 \
    --pretrained \
    --amp \
    --device cuda:1
```

**Key observation:** CNNs use `lr=1e-5` (0.00001) and `input-size 3 512 512`.

### DINOv2 ViT Models (518x518)

```bash
# DINOv2 models with 518x518 input:
MODEL_BATCH_SIZES["timm/vit_base_patch14_dinov2.lvd142m"]=20
MODEL_BATCH_SIZES["timm/vit_large_patch14_dinov2.lvd142m"]=10

# Training command pattern for DINOv2:
python train.py \
    --model "timm/vit_base_patch14_dinov2.lvd142m" \
    --data-dir ./training_data/iguana/labels_518_overlap_0/Floreana_classification/ \
    --train-split classification_train_0_518 \
    --val-split classification_val_0_518 \
    --batch-size 20 \
    --opt adamw \
    --lr 0.000001 \                   # 1e-6 for DINOv2 -- 10x lower than CNNs!
    --weight-decay 0.05 \
    --epochs 100 \
    --input-size 3 518 518 \
    --pretrained \
    --amp \
    --checkpoint-hist 1 \             # Keep only best checkpoint (save disk)
    --device cuda:0
```

**Key observations:**
- DINOv2 models use `lr=1e-6` (0.000001) -- 10x lower than CNNs
- Input size is 518x518 (= 37 x 14 patches for ViT-14 models)
- Batch size is 20 for ViT-B, 10 for ViT-L
- `--checkpoint-hist 1` saves only the best model
- Training runs on `cuda:0` (separate GPU from CNNs on `cuda:1`)

### Multi-Site Training Pattern

The script trains each model on three site datasets independently:
1. `Floreana_classification/` -- Floreana island
2. `Fernandina_s_classification/` -- Fernandina south
3. `Fernandina_m_classification/` -- Fernandina medium

Plus an additional "old Floreana" dataset at 512x512 for the ViT models:
```
training_data/iguana_512/classification_train_0_512
```

---

## Custom Dataset: `dataset_iguana.py`

Located at `timm/data/dataset_iguana.py`, this provides two dataset classes integrated with timm:

### IguanaPresenceDataset (Training)

Random crop sampling with configurable positive/negative ratio:

- **Input:** Full orthomosaic images + CSV annotations with `[images, x, y]` columns
- **Crop size:** 518x518 (configurable)
- **Positive sampling:** Centers a random crop around an annotation point (the point can be anywhere within the crop, not just centered)
- **Negative sampling:** Random crop with no annotations, up to 50 attempts
- **Augmentation:** HorizontalFlip, VerticalFlip, RandomRotate90, RandomBrightnessContrast, HueSaturationValue, GaussianBlur, GaussNoise, CoarseDropout, RandomShadow (all appropriate for nadir drone imagery)
- **Normalization:** ImageNet standard [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]
- **Output:** `(crop_tensor, label_float)` where label is 0.0 (no iguana) or 1.0 (iguana present)

```python
dataset = IguanaPresenceDataset(
    root='/path/to/images/',
    csv_path='/path/to/annotations.csv',
    crop_size=518,
    crops_per_image=8,
    positive_ratio=0.5,
    augment=True,
)
```

### IguanaTiledDataset (Evaluation)

Deterministic non-overlapping tile extraction for reproducible evaluation:

- **Input:** Same as above
- **Tiling:** Non-overlapping grid with configurable stride and overlap
- **Padding:** Zero-padding for tiles that extend beyond image boundaries
- **Output:** `(crop_tensor, target_dict)` where target_dict contains:
  - `label`: binary presence label
  - `patch_labels`: per-patch (14x14 grid) binary labels
  - `points`: annotation coordinates relative to tile
  - `name`, `crop_x`, `crop_y`: tile metadata

```python
dataset = IguanaTiledDataset(
    root='/path/to/images/',
    csv_path='/path/to/annotations.csv',
    crop_size=518,
    overlap=0,
    patch_size=14,
    point_radius=1,
)
```

### timm Integration

The module provides `patch_timm_create_dataset()` to register iguana datasets with timm's factory:

```python
from timm.data.dataset_iguana import patch_timm_create_dataset
patch_timm_create_dataset()

# Now works with timm's create_dataset:
dataset = create_dataset('iguana/presence', root=..., csv_path=...)
dataset = create_dataset('iguana/tiled', root=..., csv_path=...)
```

---

## Data Directory Structure

```
training_data/
  iguana/
    labels_512_overlap_0/
      Floreana_classification/
        classification_train_0_512/
          iguana/            # Positive tiles
          background/        # Negative tiles
        classification_val_0_512/
          iguana/
          background/
      Fernandina_s_classification/
        ...
      Fernandina_m_classification/
        ...
    labels_518_overlap_0/
      Floreana_classification/
        classification_train_0_518/
          iguana/
          background/
        classification_val_0_518/
          iguana/
          background/
      ...
  iguana_512/
    classification_train_0_512/
    classification_val_0_512/
```

---

## Key Lessons from the Iguana Training

1. **LR matters enormously for ViTs:** 1e-6 for DINOv2 vs 1e-5 for CNNs. Using a CNN-level LR on a ViT causes catastrophic forgetting.

2. **Site-based splits prevent leakage:** Train on one island's data, validate on another. Never mix tiles from the same orthomosaic across splits.

3. **Binary classification is a pragmatic choice:** Rather than species-level classification (the dataset has only one species), the task is presence/absence -- is there at least one iguana in this tile?

4. **Multiple models, same pipeline:** The shell script tests 10+ architectures with identical hyperparameters (except batch size and LR). This enables fair comparison.

5. **518 is not arbitrary:** For ViT models with patch size 14, input 518 = 37 x 14 gives an integer number of patches. This avoids interpolation artifacts in positional embeddings.
