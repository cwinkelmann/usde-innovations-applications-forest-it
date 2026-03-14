# SAHI API Reference

Quick reference for the key SAHI functions and classes used in wildlife detection pipelines.

**Repository:** https://github.com/obss/sahi
**Paper:** Akyon et al. (2022), IEEE ICIP 2022
**License:** MIT

---

## Installation

```bash
pip install sahi
# With specific model backend:
pip install sahi[yolov8]       # Ultralytics
pip install sahi[torch]        # torchvision
pip install sahi[mmdet]        # MMDetection
pip install sahi[detectron2]   # Detectron2
pip install sahi[huggingface]  # HuggingFace transformers
```

---

## AutoDetectionModel

Factory for loading detection models from any supported framework.

```python
from sahi import AutoDetectionModel

model = AutoDetectionModel.from_pretrained(
    model_type: str,              # 'yolov8', 'yolov5', 'detectron2', 'mmdet', 'huggingface', 'torchvision'
    model_path: str,              # Path to model weights
    confidence_threshold: float,  # Minimum confidence to keep (default: 0.3)
    device: str,                  # 'cuda:0', 'cpu', 'mps'
    category_mapping: dict,       # Optional: {0: 'animal', 1: 'person', ...}
    category_remapping: dict,     # Optional: remap categories after detection
    image_size: int,              # Optional: override model input size
)
```

### Model Type Strings

| `model_type` | Framework | Notes |
|--------------|-----------|-------|
| `yolov8` | Ultralytics YOLOv8 | Recommended default |
| `yolov5` | Ultralytics YOLOv5 | MegaDetector v5 compatibility |
| `detectron2` | Detectron2 | Mask R-CNN, Faster R-CNN |
| `mmdet` | MMDetection | Large model zoo |
| `huggingface` | HF Transformers | DETR, DETA, RT-DETR |
| `torchvision` | torchvision | FCOS, RetinaNet, SSD |

---

## get_sliced_prediction

Core function: runs tiled inference on a large image.

```python
from sahi.predict import get_sliced_prediction

result = get_sliced_prediction(
    image: str | np.ndarray,        # Image path or numpy array (HWC, RGB)
    detection_model: DetectionModel,
    slice_height: int = 512,        # Tile height in pixels
    slice_width: int = 512,         # Tile width in pixels
    overlap_height_ratio: float = 0.2,  # Vertical overlap fraction
    overlap_width_ratio: float = 0.2,   # Horizontal overlap fraction
    perform_standard_pred: bool = True, # Also run full-image prediction
    postprocess_type: str = 'UNIONMERGE',  # 'NMS', 'NMM', 'UNIONMERGE'
    postprocess_match_metric: str = 'IOS', # 'IOU' or 'IOS'
    postprocess_match_threshold: float = 0.5,
    postprocess_class_agnostic: bool = False,
    verbose: int = 1,               # 0=silent, 1=progress, 2=debug
    auto_slice_resolution: bool = False,  # Auto-calculate slice dims
)
```

**Returns:** `PredictionResult` object.

### Parameters in Detail

**`perform_standard_pred`**: When True, also runs the model on the full
(resized) image and merges those detections with the sliced ones. This can
catch large objects that span multiple tiles. For wildlife counting where
animals are consistently small, set to `False` to save time.

**`postprocess_type`**:
- `'NMS'`: Non-Maximum Suppression — suppress lower-confidence overlapping boxes
- `'NMM'`: Non-Maximum Merging — merge overlapping boxes into a weighted average
- `'UNIONMERGE'`: Union merge (default in newer SAHI versions)

**`postprocess_match_metric`**:
- `'IOU'`: Intersection over Union — standard metric, penalizes size differences
- `'IOS'`: Intersection over Smaller — better when the same object produces
  different-sized boxes from different tiles

**`postprocess_class_agnostic`**: When True, NMS is applied across all classes.
When False, NMS is applied per class. For single-class wildlife detection
(just "animal"), this makes no difference.

---

## get_prediction

Standard (non-sliced) inference for comparison or small images.

```python
from sahi.predict import get_prediction

result = get_prediction(
    image: str | np.ndarray,
    detection_model: DetectionModel,
    shift_amount: list = [0, 0],  # Offset for coordinate mapping
    full_shape: list = None,       # Full image shape for coordinate mapping
)
```

---

## PredictionResult

Returned by both `get_sliced_prediction` and `get_prediction`.

```python
result = get_sliced_prediction(...)

# Access detections
result.object_prediction_list    # List[ObjectPrediction]
len(result.object_prediction_list)  # Number of detections

# Export
result.to_coco_annotations()     # COCO format list of dicts
result.to_coco_predictions()     # COCO prediction format

# Visualize
result.export_visuals(
    export_dir='output/',
    file_name='result',
    rect_th=2,                   # Box line thickness
    text_size=0.5,
    text_th=1,
)
```

---

## ObjectPrediction

Individual detection result.

```python
for pred in result.object_prediction_list:
    # Bounding box (in full-image pixel coordinates)
    pred.bbox                    # BoundingBox object
    pred.bbox.to_xyxy()          # [x1, y1, x2, y2]
    pred.bbox.to_xywh()          # [x, y, width, height]
    pred.bbox.to_coco()          # [x, y, width, height] (COCO format)

    # Confidence
    pred.score                   # Score object
    pred.score.value             # float, 0-1

    # Category
    pred.category                # Category object
    pred.category.id             # int
    pred.category.name           # str
```

---

## predict (CLI-style batch function)

Run SAHI on a folder of images from Python or CLI:

```python
from sahi.predict import predict

predict(
    model_type='yolov8',
    model_path='model.pt',
    model_confidence_threshold=0.3,
    model_device='cuda:0',
    source='images/',               # Folder path
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.25,
    overlap_width_ratio=0.25,
    export_format='coco',           # 'coco' or 'visuals'
    project='output/',
    name='sahi_results',
    novisual=False,                 # Set True to skip visualizations
)
```

**CLI equivalent:**
```bash
sahi predict \
    --model_type yolov8 \
    --model_path model.pt \
    --model_confidence_threshold 0.3 \
    --source images/ \
    --slice_height 640 \
    --slice_width 640 \
    --export_format coco
```

---

## Utility: Slicing Without Prediction

To see how an image will be sliced (useful for debugging):

```python
from sahi.slicing import slice_image

slice_image_result = slice_image(
    image='large_image.tif',
    output_file_name='sliced',
    output_dir='slices/',
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.25,
    overlap_width_ratio=0.25,
)

print(f"Generated {len(slice_image_result.images)} slices")
# Each slice is saved as a separate image file
```

---

## COCO Utilities

```python
from sahi.utils.coco import Coco

# Load COCO annotations
coco = Coco.from_coco_dict_or_path('annotations.json')

# Get statistics
coco.stats  # annotation count, image count, category count
```
