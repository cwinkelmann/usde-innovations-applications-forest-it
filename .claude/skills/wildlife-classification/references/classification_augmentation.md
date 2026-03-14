# Classification Augmentation Reference

Data augmentation strategies for wildlife image classification, with domain-specific guidance for drone nadir imagery, camera trap photos, and general wildlife images. Includes timm's `create_transform` API and custom albumentations pipelines.

---

## timm create_transform API

```python
from timm.data import create_transform

# Standard augmentation for training
train_transform = create_transform(
    input_size=518,
    is_training=True,
    auto_augment='rand-m9-mstd0.5-inc1',  # RandAugment policy
    re_prob=0.25,                            # Random erasing probability
    re_mode='pixel',                         # Random erasing mode
    interpolation='bicubic',
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)

# Validation transform (deterministic)
val_transform = create_transform(
    input_size=518,
    is_training=False,
    interpolation='bicubic',
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    crop_pct=1.0,  # No center crop -- use full image
)
```

**create_transform parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_size` | 224 | Target input size (int or tuple) |
| `is_training` | False | Enable training augmentations |
| `auto_augment` | None | Auto augment policy string |
| `re_prob` | 0.0 | Random erasing probability |
| `re_mode` | 'const' | Random erasing fill mode: 'const', 'pixel', 'rand' |
| `interpolation` | 'bilinear' | Resize interpolation: 'bicubic', 'bilinear', 'nearest' |
| `mean` | IMAGENET_DEFAULT_MEAN | Normalization mean |
| `std` | IMAGENET_DEFAULT_STD | Normalization std |
| `crop_pct` | 0.875 | Center crop percentage (validation only) |

---

## Nadir Drone Imagery Augmentation

Nadir (top-down) drone images have unique properties that affect augmentation choices:

**Valid augmentations (no gravity bias):**
- Horizontal flip
- Vertical flip (animals look the same from above when flipped vertically)
- 90/180/270 degree rotations (no preferred orientation)
- Brightness/contrast variation (lighting changes)
- Hue/saturation shift (different times of day)
- Gaussian noise (sensor noise)
- Gaussian blur (focus variation at altitude)
- Coarse dropout / cutout (simulate partial occlusion)
- Shadow simulation (vegetation shadows)

**Questionable augmentations:**
- Aggressive scale changes (GSD is relatively fixed for a given altitude)
- Perspective transforms (nadir = no perspective)
- Elastic deformation (not physically meaningful for aerial imagery)

**Recommended pipeline (from iguana dataset_iguana.py):**

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

nadir_train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(
        brightness_limit=0.3,
        contrast_limit=0.3,
        p=0.5,
    ),
    A.HueSaturationValue(
        hue_shift_limit=20,
        sat_shift_limit=30,
        val_shift_limit=20,
        p=0.3,
    ),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.GaussNoise(var_limit=(10, 30), p=0.2),
    A.CoarseDropout(
        max_holes=8, max_height=64, max_width=64,
        min_holes=1, min_height=16, min_width=16,
        fill_value=0,
        p=0.3,
    ),
    A.RandomShadow(
        shadow_roi=(0, 0, 1, 1),
        num_shadows_lower=1,
        num_shadows_upper=3,
        shadow_dimension=5,
        p=0.2,
    ),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    ToTensorV2(),
])
```

---

## Camera Trap Imagery Augmentation

Camera trap images have a fixed perspective with strong gravity orientation:

**Valid augmentations:**
- Horizontal flip (left-right mirror is valid)
- Brightness/contrast (day/night variation is extreme)
- Hue/saturation (IR filter effects, seasonal color changes)
- Gaussian noise (sensor noise, especially at night)
- Shadow simulation (vegetation shadows are common)
- JPEG compression artifacts (camera quality varies)

**Invalid augmentations:**
- Vertical flip (animals do not appear upside down)
- 90/180 degree rotations (camera has fixed orientation)
- Heavy geometric distortion (camera is fixed-position)

**Recommended pipeline:**

```python
camera_trap_train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    # NO VerticalFlip, NO RandomRotate90
    A.RandomBrightnessContrast(
        brightness_limit=0.4,  # Stronger -- day/night variation
        contrast_limit=0.4,
        p=0.5,
    ),
    A.HueSaturationValue(
        hue_shift_limit=20,
        sat_shift_limit=30,
        val_shift_limit=20,
        p=0.3,
    ),
    A.RandomShadow(
        shadow_roi=(0, 0, 1, 1),
        num_shadows_lower=1,
        num_shadows_upper=3,
        shadow_dimension=5,
        p=0.3,
    ),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.ImageCompression(quality_lower=50, quality_upper=100, p=0.2),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    ToTensorV2(),
])
```

---

## Augmentation for Small Datasets

When you have fewer than 100 images per class, augmentation becomes critical:

**Aggressive augmentation pipeline:**

```python
small_dataset_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),  # If nadir; omit for camera trap
    A.RandomRotate90(p=0.5),  # If nadir; omit for camera trap
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.15,
        rotate_limit=15,
        p=0.5,
    ),
    A.OneOf([
        A.RandomBrightnessContrast(0.3, 0.3),
        A.CLAHE(clip_limit=4.0),
        A.ColorJitter(0.3, 0.3, 0.3, 0.1),
    ], p=0.7),
    A.OneOf([
        A.GaussianBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.MotionBlur(blur_limit=5),
    ], p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.CoarseDropout(
        max_holes=12, max_height=48, max_width=48,
        min_holes=1, min_height=8, min_width=8,
        fill_value=0,
        p=0.4,
    ),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    ToTensorV2(),
])
```

**Additional strategies for small datasets:**
- Mixup/CutMix (via timm's `Mixup` class)
- Test-time augmentation (TTA) for evaluation
- Repeated augmentation (multiple augmented copies per epoch)

---

## Mixup and CutMix (via timm)

```python
from timm.data import Mixup

mixup_fn = Mixup(
    mixup_alpha=0.8,     # Mixup interpolation strength
    cutmix_alpha=1.0,    # CutMix interpolation strength
    prob=0.5,            # Probability of applying mix
    switch_prob=0.5,     # Probability of switching between mixup and cutmix
    mode='batch',        # 'batch', 'pair', or 'elem'
    label_smoothing=0.1,
    num_classes=5,
)

# In training loop:
images, targets = mixup_fn(images, targets)
outputs = model(images)
loss = criterion(outputs, targets)  # Use SoftTargetCrossEntropy with mixup
```

**Note:** When using Mixup/CutMix, replace standard CrossEntropyLoss with `timm.loss.SoftTargetCrossEntropy`.

---

## Test-Time Augmentation (TTA)

Apply augmentations at inference time and average predictions:

```python
def predict_with_tta(model, image, transforms_list, device):
    """
    Apply multiple augmentations to an image and average predictions.

    transforms_list: list of augmentation pipelines to apply
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for transform in transforms_list:
            augmented = transform(image=image)['image']
            augmented = augmented.unsqueeze(0).to(device)
            probs = torch.softmax(model(augmented), dim=1)
            all_probs.append(probs)

    # Average probabilities
    avg_probs = torch.mean(torch.stack(all_probs), dim=0)
    return avg_probs

# Standard TTA transforms for nadir imagery:
tta_transforms = [
    val_transform,                                    # Original
    A.Compose([A.HorizontalFlip(p=1.0)] + val_transform.transforms),  # H-flip
    A.Compose([A.VerticalFlip(p=1.0)] + val_transform.transforms),    # V-flip
    A.Compose([A.RandomRotate90(p=1.0)] + val_transform.transforms),  # 90 degrees
]
```

---

## Normalization Constants

**Always use ImageNet normalization for pretrained models:**

```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
```

These values are correct for:
- All timm pretrained models
- DeepFaune (uses [0.4850, 0.4560, 0.4060] / [0.2290, 0.2240, 0.2250] -- equivalent)
- Any model pretrained on ImageNet

**Do NOT compute dataset-specific normalization** unless training from scratch (no pretraining). Using different normalization constants with a pretrained model will degrade performance.
