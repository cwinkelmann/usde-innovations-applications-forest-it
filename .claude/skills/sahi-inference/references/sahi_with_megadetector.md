# Using SAHI with MegaDetector

## Why Combine Them?

MegaDetector (MDv5a) is trained at 1280×1280 input resolution. When processing
large drone images or high-resolution camera trap panoramics, the resize to 1280
causes small animals to shrink below the detection threshold. SAHI preserves
the native resolution by processing overlapping tiles.

**When to use SAHI + MegaDetector:**
- Drone orthomosaics (>5000px per side)
- High-resolution camera trap images where animals are far from the camera
- Panoramic stitched images
- Any image where target animals are <30px at model input resolution

**When NOT to use SAHI + MegaDetector:**
- Standard camera trap images (2048×1536) with animals filling >5% of frame
- Already-cropped images
- Video frames at typical camera trap resolution

---

## Setup

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# MegaDetector v5a is a YOLOv5 model
md_model = AutoDetectionModel.from_pretrained(
    model_type='yolov5',
    model_path='md_v5a.0.0.pt',
    confidence_threshold=0.2,    # MegaDetector's standard threshold
    device='cuda:0',
)
```

**Important:** Use `model_type='yolov5'` for MDv5a. MegaDetector v5 uses the
Ultralytics YOLOv5 format.

---

## MegaDetector Category Mapping

MegaDetector outputs three categories:

| ID | Name | Description |
|----|------|-------------|
| 0 | animal | Any animal (species-agnostic) |
| 1 | person | Human |
| 2 | vehicle | Car, truck, etc. |

To filter for animals only after inference:

```python
result = get_sliced_prediction(...)

animal_detections = [
    pred for pred in result.object_prediction_list
    if pred.category.id == 0
]
```

---

## Recommended Parameters for MegaDetector + SAHI

### Drone Imagery (GSD 0.5-2.0 cm/px)

```python
result = get_sliced_prediction(
    image='drone_image.tif',
    detection_model=md_model,
    slice_height=1280,              # Match MD's training resolution
    slice_width=1280,
    overlap_height_ratio=0.2,       # 256px overlap at 1280 slice
    overlap_width_ratio=0.2,
    postprocess_type='NMS',
    postprocess_match_metric='IOS',
    postprocess_match_threshold=0.5,
    perform_standard_pred=False,    # Skip full-image pass (animals too small)
)
```

**Why slice_size=1280?** MegaDetector was trained at 1280×1280. Using this as
the slice size means each tile is processed at the model's native resolution
with no further resizing, giving optimal detection performance.

### High-Resolution Camera Trap (4096×3072)

```python
result = get_sliced_prediction(
    image='hires_camera_trap.jpg',
    detection_model=md_model,
    slice_height=1280,
    slice_width=1280,
    overlap_height_ratio=0.15,      # Smaller overlap — fewer tiles needed
    overlap_width_ratio=0.15,
    postprocess_type='NMS',
    postprocess_match_metric='IOU',
    postprocess_match_threshold=0.5,
    perform_standard_pred=True,     # Also try full-image — may catch large animals
)
```

### Satellite Imagery (Very Large, Small Targets)

```python
result = get_sliced_prediction(
    image='satellite_tile.tif',
    detection_model=md_model,
    slice_height=640,               # Smaller slices for very small targets
    slice_width=640,
    overlap_height_ratio=0.3,       # Higher overlap for tiny targets
    overlap_width_ratio=0.3,
    postprocess_type='NMM',         # Merging works better for uncertain boxes
    postprocess_match_metric='IOS',
    postprocess_match_threshold=0.4,
    perform_standard_pred=False,
)
```

---

## SAHI + MegaDetector + Species Classifier Pipeline

A common pipeline: MegaDetector (via SAHI) detects animals, then a species
classifier runs on the cropped detections.

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from PIL import Image
import torch
import timm

# Step 1: Detection with SAHI + MegaDetector
md_model = AutoDetectionModel.from_pretrained(
    model_type='yolov5',
    model_path='md_v5a.0.0.pt',
    confidence_threshold=0.2,
    device='cuda:0',
)

result = get_sliced_prediction(
    image='drone_image.tif',
    detection_model=md_model,
    slice_height=1280,
    slice_width=1280,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

# Step 2: Crop detections and classify species
classifier = timm.create_model('vit_base_patch14_dinov2.lvd142m',
                                pretrained=False, num_classes=5)
ckpt = torch.load('species_classifier.pth', map_location='cuda:0',
                   weights_only=False)
classifier.load_state_dict(ckpt['model_state_dict'])
classifier = classifier.to('cuda:0').eval()

full_image = Image.open('drone_image.tif')

for pred in result.object_prediction_list:
    if pred.category.id != 0:  # Skip non-animals
        continue

    x1, y1, x2, y2 = [int(c) for c in pred.bbox.to_xyxy()]

    # Add 10% padding around the crop
    pad_x = int((x2 - x1) * 0.1)
    pad_y = int((y2 - y1) * 0.1)
    crop = full_image.crop((
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(full_image.width, x2 + pad_x),
        min(full_image.height, y2 + pad_y),
    ))

    # Classify the crop
    # (apply appropriate transforms before inference)
    species = classify_crop(classifier, crop)
    print(f"Detection at ({x1},{y1})-({x2},{y2}): {species}, conf={pred.score.value:.3f}")
```

---

## Performance Notes

- SAHI adds overhead from tile generation, per-tile inference, and NMS merging
- For a 15000×12000 image with 1280px tiles and 0.2 overlap: ~100 tiles
- Each tile requires a separate forward pass through MegaDetector
- Total inference time: ~100 × (single_image_time) + NMS overhead
- On a modern GPU (RTX 3090), expect ~0.02-0.05s per tile for YOLOv5
- Full image processing: ~2-5 seconds for 100 tiles

**Optimization tip:** If many tiles are empty (ocean, bare rock), consider
a pre-filter that skips tiles with very low variance (likely no animals):

```python
import numpy as np

def has_content(tile_array, variance_threshold=500):
    """Skip tiles that are likely empty (low variance)."""
    return np.var(tile_array) > variance_threshold
```
