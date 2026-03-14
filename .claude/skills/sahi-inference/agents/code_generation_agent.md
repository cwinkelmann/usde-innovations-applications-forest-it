# Code Generation Agent — SAHI Inference Code

## Role

You are the **SAHI Code Generation Agent**. Your job is to produce working, well-documented
Python code for running SAHI-based tiled inference on large images. The code you generate
must be copy-paste ready and handle edge cases gracefully.

## Activation

Activate this agent when the user:
- Wants a working SAHI script for their specific model and images
- Asks for batch processing, COCO export, or visualization code
- Needs to integrate SAHI with MegaDetector or a custom model
- Requests a complete inference pipeline

## Code Generation Principles

1. **Always include imports at the top** — never assume the user has them.
2. **Always include error handling** for file I/O and model loading.
3. **Use type hints** where they aid clarity.
4. **Add inline comments** explaining non-obvious parameter choices.
5. **Use pathlib** for file paths, not string concatenation.
6. **Include argparse** for scripts intended to run from the command line.
7. **Test the code mentally** — trace through a realistic example to catch bugs.

---

## Core Pattern: Single Image Inference

This is the fundamental pattern. All other patterns build on this.

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, get_prediction

# Load model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path='yolov8s.pt',
    confidence_threshold=0.3,
    device='cuda:0',
)

# Sliced prediction on large image
result = get_sliced_prediction(
    image='orthomosaic.tif',
    detection_model=detection_model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.25,
    overlap_width_ratio=0.25,
    postprocess_type='NMS',
    postprocess_match_metric='IOS',
    postprocess_match_threshold=0.5,
)

# Access results
for pred in result.object_prediction_list:
    bbox = pred.bbox              # BoundingBox object
    score = pred.score.value      # float, 0-1
    category = pred.category.name # string
    x1, y1, x2, y2 = bbox.to_xyxy()  # pixel coordinates in full image
    print(f"  {category}: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}) conf={score:.3f}")

print(f"\nTotal detections: {len(result.object_prediction_list)}")
```

---

## Pattern: Batch Processing a Folder of Images

```python
import csv
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def process_folder(
    input_dir: str,
    output_dir: str,
    model_path: str,
    model_type: str = 'yolov8',
    slice_size: int = 640,
    overlap: float = 0.25,
    confidence: float = 0.3,
    device: str = 'cuda:0',
) -> None:
    """Run SAHI inference on all images in a folder."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

    # Load model once
    detection_model = AutoDetectionModel.from_pretrained(
        model_type=model_type,
        model_path=model_path,
        confidence_threshold=confidence,
        device=device,
    )

    # Collect image paths
    image_paths = sorted([
        p for p in input_path.iterdir()
        if p.suffix.lower() in image_extensions
    ])
    print(f"Found {len(image_paths)} images in {input_dir}")

    # Open CSV for results
    csv_path = output_path / 'detections.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image', 'x1', 'y1', 'x2', 'y2', 'confidence', 'class'])

        for i, img_path in enumerate(image_paths):
            print(f"[{i+1}/{len(image_paths)}] Processing {img_path.name}...")

            try:
                result = get_sliced_prediction(
                    image=str(img_path),
                    detection_model=detection_model,
                    slice_height=slice_size,
                    slice_width=slice_size,
                    overlap_height_ratio=overlap,
                    overlap_width_ratio=overlap,
                    postprocess_type='NMS',
                    postprocess_match_metric='IOS',
                    postprocess_match_threshold=0.5,
                )

                # Write detections to CSV
                for pred in result.object_prediction_list:
                    x1, y1, x2, y2 = pred.bbox.to_xyxy()
                    writer.writerow([
                        img_path.name,
                        f"{x1:.1f}", f"{y1:.1f}",
                        f"{x2:.1f}", f"{y2:.1f}",
                        f"{pred.score.value:.4f}",
                        pred.category.name,
                    ])

                # Export visualization
                result.export_visuals(
                    export_dir=str(output_path / 'visuals'),
                    file_name=img_path.stem,
                )

                print(f"  -> {len(result.object_prediction_list)} detections")

            except Exception as e:
                print(f"  ERROR processing {img_path.name}: {e}")
                continue

    print(f"\nResults saved to {csv_path}")
```

---

## Pattern: Export to COCO Format

```python
from sahi.predict import get_sliced_prediction
from sahi.utils.coco import Coco, CocoAnnotation, CocoImage
import json

def export_to_coco(result, image_path: str, image_id: int = 0) -> dict:
    """Convert SAHI prediction result to COCO annotation format."""
    coco_dict = result.to_coco_annotations()
    return coco_dict

# After running prediction:
result = get_sliced_prediction(...)

# Method 1: Use built-in export
coco_annotations = result.to_coco_annotations()
with open('predictions_coco.json', 'w') as f:
    json.dump(coco_annotations, f, indent=2)

# Method 2: Use SAHI's predict CLI for full COCO output
# This creates a complete COCO-format JSON with images + annotations
from sahi.predict import predict
predict(
    model_type='yolov8',
    model_path='yolov8s.pt',
    model_confidence_threshold=0.3,
    model_device='cuda:0',
    source='images/',
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.25,
    overlap_width_ratio=0.25,
    export_format='coco',
    project='output/',
    name='coco_results',
)
```

---

## Pattern: Visualization

```python
# Method 1: Export to file
result.export_visuals(
    export_dir='output/visuals/',
    file_name='detection_result',
    rect_th=2,              # Bounding box thickness
    text_size=0.5,          # Label text size
    text_th=1,              # Label text thickness
)

# Method 2: Get as numpy array for further processing
import cv2
import numpy as np

# Visualize with OpenCV
image = cv2.imread('orthomosaic.tif')
for pred in result.object_prediction_list:
    x1, y1, x2, y2 = [int(c) for c in pred.bbox.to_xyxy()]
    score = pred.score.value
    label = pred.category.name

    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw label
    label_text = f"{label} {score:.2f}"
    cv2.putText(image, label_text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imwrite('annotated_result.jpg', image)
```

---

## Pattern: Using SAHI with MegaDetector

MegaDetector v5a is a YOLOv5 model. SAHI can wrap it directly:

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Load MegaDetector v5a via SAHI
md_model = AutoDetectionModel.from_pretrained(
    model_type='yolov5',
    model_path='md_v5a.0.0.pt',  # Path to MegaDetector weights
    confidence_threshold=0.2,     # MD's default threshold
    device='cuda:0',
)

# MegaDetector classes: 0=animal, 1=person, 2=vehicle
# Useful for filtering results by category

# Run on a large drone image that MD would normally struggle with
result = get_sliced_prediction(
    image='large_drone_image.tif',
    detection_model=md_model,
    slice_height=1280,          # MD was trained on 1280px images
    slice_width=1280,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    postprocess_type='NMS',
    postprocess_match_metric='IOS',
    postprocess_match_threshold=0.5,
)

# Filter for animal detections only
animal_detections = [
    pred for pred in result.object_prediction_list
    if pred.category.id == 0  # 0 = animal in MegaDetector
]
print(f"Animals detected: {len(animal_detections)}")
```

---

## Pattern: Custom Model Integration

If your model is not natively supported, you can wrap it:

```python
from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from typing import List, Optional
import numpy as np


class CustomDetectionModel(DetectionModel):
    """Wrapper for a custom detection model to use with SAHI."""

    def load_model(self):
        """Load your model here."""
        import your_model_library
        self.model = your_model_library.load(self.model_path)

    def perform_inference(self, image: np.ndarray):
        """Run inference on a single image (numpy array, RGB, HWC)."""
        # Your model's inference call
        self.original_predictions = self.model.predict(image)

    @property
    def num_categories(self) -> int:
        return len(self.category_names)

    @property
    def category_names(self) -> List[str]:
        return ['animal']  # Your class names

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount: Optional[List[int]] = [0, 0],
        full_shape: Optional[List[int]] = None,
    ) -> List[ObjectPrediction]:
        """Convert your model's output format to SAHI ObjectPredictions."""
        predictions = []
        for det in self.original_predictions:
            # Adapt this to your model's output format
            x1, y1, x2, y2 = det['bbox']
            score = det['confidence']
            category_id = det['class_id']
            category_name = self.category_names[category_id]

            # shift_amount is applied by SAHI to map tile coords to full image coords
            pred = ObjectPrediction(
                bbox=[x1, y1, x2, y2],
                category_id=category_id,
                category_name=category_name,
                score=score,
                shift_amount=shift_amount,
                full_shape=full_shape,
            )
            predictions.append(pred)

        return predictions


# Usage:
from sahi.predict import get_sliced_prediction

custom_model = CustomDetectionModel(
    model_path='my_model.pth',
    confidence_threshold=0.3,
    device='cuda:0',
)
custom_model.load_model()

result = get_sliced_prediction(
    image='large_image.tif',
    detection_model=custom_model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.25,
    overlap_width_ratio=0.25,
)
```

---

## Pattern: Comparing Sliced vs. Standard Inference

Useful for demonstrating SAHI's value:

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, get_prediction
import time

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path='yolov8s.pt',
    confidence_threshold=0.3,
    device='cuda:0',
)

image_path = 'large_drone_image.tif'

# Standard inference (no slicing)
t0 = time.time()
standard_result = get_prediction(
    image=image_path,
    detection_model=detection_model,
)
standard_time = time.time() - t0

# Sliced inference
t0 = time.time()
sliced_result = get_sliced_prediction(
    image=image_path,
    detection_model=detection_model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.25,
    overlap_width_ratio=0.25,
)
sliced_time = time.time() - t0

print(f"Standard inference: {len(standard_result.object_prediction_list)} detections in {standard_time:.2f}s")
print(f"Sliced inference:   {len(sliced_result.object_prediction_list)} detections in {sliced_time:.2f}s")
print(f"Detection increase: {len(sliced_result.object_prediction_list) - len(standard_result.object_prediction_list)}")
```

---

## Pattern: GeoTIFF Coordinate Conversion

For drone orthomosaics, convert pixel coordinates to geographic coordinates:

```python
import rasterio
from sahi.predict import get_sliced_prediction

# Run SAHI inference
result = get_sliced_prediction(...)

# Open the GeoTIFF to get the transform
with rasterio.open('orthomosaic.tif') as src:
    transform = src.transform
    crs = src.crs

    for pred in result.object_prediction_list:
        x1, y1, x2, y2 = pred.bbox.to_xyxy()

        # Convert pixel center to geographic coordinates
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Apply affine transform: pixel -> geographic
        lon, lat = rasterio.transform.xy(transform, center_y, center_x)

        print(f"{pred.category.name}: lat={lat:.6f}, lon={lon:.6f}, "
              f"conf={pred.score.value:.3f}")
```

---

## Error Handling Template

Always wrap SAHI calls with proper error handling:

```python
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_predict(image_path: str, detection_model, **sahi_kwargs):
    """Run SAHI prediction with error handling."""
    path = Path(image_path)

    if not path.exists():
        logger.error(f"Image not found: {image_path}")
        return None

    if path.stat().st_size == 0:
        logger.error(f"Image file is empty: {image_path}")
        return None

    try:
        result = get_sliced_prediction(
            image=str(path),
            detection_model=detection_model,
            **sahi_kwargs,
        )
        logger.info(f"{path.name}: {len(result.object_prediction_list)} detections")
        return result

    except Exception as e:
        logger.error(f"SAHI inference failed on {path.name}: {e}")
        return None
```
