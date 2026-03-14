# YOLOv8 Fine-Tuning Exercise Template

## Usage

This template defines the cell structure for a hands-on YOLOv8 fine-tuning exercise on wildlife data. The `detection_bridge_agent` uses this template to generate the complete exercise content.

---

## Exercise Metadata

```
Title: Object Detection for Wildlife: YOLOv8 Fine-Tuning
Module: New (fills PCV Module 8 gap)
Estimated time: 2-3 hours
Prerequisites: PCV Module 5 (fine-tuning), PCV Module 8 (conceptual detection)
Dataset: Mini wildlife detection dataset (camera trap subset)
```

---

## Cell Structure

### Cell 1 [markdown]: Title and Learning Objectives

```markdown
# Object Detection for Wildlife: YOLOv8 Fine-Tuning

In PCV Module 8, you learned what object detection is conceptually: finding objects in images by predicting bounding boxes and class labels. In this module, you will implement detection hands-on using YOLOv8 on wildlife data.

## Learning Objectives

After completing this exercise, you will be able to:
1. Prepare a wildlife dataset in YOLO annotation format
2. Configure YOLOv8 training with a `data.yaml` file
3. Fine-tune YOLOv8s on a small wildlife dataset
4. Evaluate detection performance using mAP, AP50, and AP75
5. Visualize detections and analyze failure cases
6. Compare YOLOv8 results with MegaDetector as a baseline
```

### Cell 2 [markdown]: Prerequisites

```markdown
## Prerequisites

- **PCV Module 3:** Training loops, DataLoaders, evaluation metrics
- **PCV Module 5:** Fine-tuning pretrained models (you will fine-tune YOLOv8, which is pretrained on COCO)
- **PCV Module 8:** Conceptual understanding of bounding boxes, anchor boxes, and detection as a task

**New concepts introduced here:**
- IoU (Intersection over Union)
- Non-Maximum Suppression (NMS)
- mAP, AP50, AP75 metrics
- YOLOv8 architecture (anchor-free detection)

> **Why YOLOv8?** It is the most widely used detection framework, has excellent documentation, and runs efficiently on consumer GPUs. The Miesner thesis used HerdNet (point-based detection) instead -- we will discuss why at the end of this module.
```

### Cell 3 [code]: Setup and Installation

```python
# Setup: Install ultralytics and download mini wildlife dataset

# Install ultralytics (YOLOv8)
# !pip install ultralytics

from ultralytics import YOLO
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import json
import yaml

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path("./data/wildlife_detection")
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"ultralytics version: {__import__('ultralytics').__version__}")
```

### Cell 4 [code]: Download and Prepare Dataset

```python
# Download mini wildlife detection dataset
# [Agent fills in: specific dataset download code]
# Options:
#   - Caltech Camera Traps subset (10 species, 1000 images)
#   - Custom curated subset with YOLO-format annotations
#   - Pre-converted dataset from wildlife_datasets_guide.md

# Dataset structure:
# data/wildlife_detection/
#   images/
#     train/
#     val/
#     test/
#   labels/
#     train/
#     val/
#     test/
#   data.yaml

# [Download and preparation code here]

# Verify dataset
train_images = list((DATA_DIR / "images" / "train").glob("*.jpg"))
val_images = list((DATA_DIR / "images" / "val").glob("*.jpg"))
print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")
```

### Cell 5 [code]: Explore Dataset

```python
# Explore: Visualize sample images with annotations

def plot_annotations(image_path, label_path, class_names):
    """Display an image with its YOLO-format bounding box annotations."""
    img = plt.imread(image_path)
    h, w = img.shape[:2]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(img)

    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                cls_id = int(parts[0])
                x_center, y_center, bw, bh = map(float, parts[1:5])

                # Convert from normalized to pixel coordinates
                x1 = (x_center - bw / 2) * w
                y1 = (y_center - bh / 2) * h
                box_w = bw * w
                box_h = bh * h

                rect = patches.Rectangle((x1, y1), box_w, box_h,
                                          linewidth=2, edgecolor='lime',
                                          facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, class_names[cls_id],
                        color='lime', fontsize=10, weight='bold')

    ax.set_title(image_path.name)
    ax.axis('off')
    plt.show()

# Visualize a few samples
# [Agent fills in: class_names list and sample visualization]
```

### Cell 6 [markdown]: Understanding Detection Metrics

```markdown
## Part 1: Detection Metrics

Before training, let us understand how detection performance is measured.

### IoU (Intersection over Union)

IoU measures how well a predicted box overlaps with the ground truth box:

$$
\text{IoU} = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)}
$$

### mAP (mean Average Precision)

1. Sort predictions by confidence
2. At each confidence threshold, compute precision and recall
3. Plot the Precision-Recall curve
4. AP = area under the PR curve
5. mAP = mean AP across all classes

Common variants:
- **AP50:** IoU threshold = 0.50 (lenient)
- **AP75:** IoU threshold = 0.75 (strict)
- **mAP:** Average across IoU thresholds [0.50, 0.55, ..., 0.95]
```

### Cell 7 [code]: IoU Implementation

```python
# Demonstration: Compute IoU between two boxes

def compute_iou(box_a, box_b):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0

# Example
gt_box = [100, 100, 200, 250]   # Ground truth
pred_box = [110, 105, 210, 260] # Slightly shifted prediction
iou = compute_iou(gt_box, pred_box)
print(f"IoU = {iou:.3f}")  # Should be ~0.75
```

### Cell 8 [code]: NMS Implementation TODO (Level 1)

```python
# TODO Level 1: Implement Non-Maximum Suppression

def nms(boxes, scores, iou_threshold=0.5):
    """Non-Maximum Suppression.

    Args:
        boxes: list of [x1, y1, x2, y2] boxes
        scores: list of confidence scores
        iou_threshold: IoU threshold for suppression

    Returns:
        List of indices of kept detections
    """
    boxes = np.array(boxes)
    scores = np.array(scores)

    # Sort by confidence (descending)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        # TODO: Take the highest-confidence detection
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        # TODO: Compute IoU of this detection with all remaining detections
        # TODO: Keep only detections with IoU <= iou_threshold
        # Hint: use compute_iou() and np.where()

        raise NotImplementedError("Complete the NMS loop")

    return keep
```

### Cell 9 [code]: NMS Validation

```python
# Validation: Test NMS
test_boxes = [
    [100, 100, 200, 200],  # High confidence, keep
    [105, 105, 205, 205],  # Overlaps with first, lower confidence, suppress
    [300, 300, 400, 400],  # No overlap, different object, keep
    [305, 305, 405, 405],  # Overlaps with third, suppress
]
test_scores = [0.9, 0.8, 0.85, 0.7]

kept = nms(test_boxes, test_scores, iou_threshold=0.5)
assert len(kept) == 2, f"Expected 2 detections after NMS, got {len(kept)}"
assert kept[0] == 0, "First kept detection should be index 0 (highest confidence)"
print(f"NMS kept indices: {kept}")
print("NMS test passed!")
```

### Cell 10 [markdown]: YOLOv8 Training Section

```markdown
## Part 2: YOLOv8 Fine-Tuning

Now we will fine-tune a YOLOv8 model on our wildlife dataset.

### YOLOv8 Architecture

YOLOv8 is an **anchor-free** detector:
- **Backbone:** CSPNet (Cross-Stage Partial Network) for feature extraction
- **Neck:** PAN (Path Aggregation Network) for multi-scale feature fusion
- **Head:** Decoupled head with separate classification and regression branches

We will use **YOLOv8s** (small, 11.2M parameters) -- a good balance between speed and accuracy for teaching.
```

### Cell 11 [code]: data.yaml TODO (Level 1)

```python
# TODO Level 1: Create the data.yaml configuration file

data_config = {
    'path': str(DATA_DIR.absolute()),
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/test',
    # TODO: Set the number of classes
    'nc': None,  # <-- Replace with the correct number
    # TODO: Set the class names list
    'names': None,  # <-- Replace with the correct class names
}

# Write data.yaml
with open(DATA_DIR / 'data.yaml', 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False)

print("data.yaml created:")
print(yaml.dump(data_config, default_flow_style=False))
```

### Cell 12 [code]: Training TODO (Level 2)

```python
# TODO Level 2: Train YOLOv8s on the wildlife dataset

# Load pretrained YOLOv8s
model = YOLO("yolov8s.pt")

# TODO: Configure and launch training
# Use these parameters:
#   data: path to data.yaml
#   epochs: 50 (or fewer for quick iteration)
#   imgsz: 640
#   batch: 16 (reduce if GPU memory is limited)
#   patience: 10 (early stopping)
#   device: DEVICE
#
# results = model.train(...)

raise NotImplementedError("Configure and launch YOLOv8 training")
```

### Cell 13 [code]: Training Validation

```python
# Validation: Verify training completed
assert results is not None, "Training did not complete"
print(f"Training completed in {results.speed['train']:.1f}s")
print(f"Best mAP50: {results.box.map50:.3f}")
print(f"Best mAP50-95: {results.box.map:.3f}")
```

### Cell 14 [code]: Evaluation TODO (Level 2)

```python
# TODO Level 2: Evaluate the trained model

# Load the best weights
best_model = YOLO(results.save_dir / "weights" / "best.pt")

# TODO: Run validation and print per-class metrics
# metrics = best_model.val(...)
# Print: mAP50, mAP50-95, per-class AP50

raise NotImplementedError("Evaluate the trained model")
```

### Cell 15 [code]: Visualize Predictions TODO (Level 2)

```python
# TODO Level 2: Visualize predictions on test images

# TODO: Run prediction on 5 test images
# TODO: Display images with predicted boxes (green) and ground truth (red)
# TODO: Print confidence scores for each detection

raise NotImplementedError("Visualize predictions")
```

### Cell 16 [markdown]: Comparison Section

```markdown
## Part 3: Detection in the Wild

### YOLOv8 vs MegaDetector

MegaDetector is a YOLOv5 model pretrained to detect three categories: animal, person, vehicle. It does not classify species. A common pipeline:

1. Run MegaDetector on camera trap images
2. Crop each "animal" detection
3. Pass crops through a species classifier

### YOLOv8 vs HerdNet (Thesis Comparison)

The Miesner thesis chose HerdNet (point-based detection) over YOLO because:
- Marine iguanas are small (25-37 px) and densely packed
- Bounding boxes overlap heavily in dense colonies
- NMS merges adjacent iguanas (undercounting)
- Point annotations are faster than bounding box annotations

> **Key insight:** YOLO is excellent for camera traps (distinct, separated animals). Point-based methods like HerdNet are better for aerial surveys of dense colonies.
```

### Cell 17 [code]: MegaDetector Comparison TODO (Level 3)

```python
# TODO Level 3: Compare your YOLOv8 model with MegaDetector baseline
#
# This is an open-ended exercise. You should:
# 1. Run MegaDetector on the test images (if available)
#    OR simulate by using a generic COCO-pretrained YOLOv8
# 2. Compare: Does your fine-tuned model detect more species-specific details?
# 3. Analyze: On which images does fine-tuning help most?
# 4. Discuss: What are the trade-offs of fine-tuning vs using a general detector?

# Your analysis here
```

### Cell 18 [markdown]: Summary and Case Study Connection

```markdown
## Summary

In this module, you:
1. Implemented IoU and NMS from scratch -- the building blocks of detection evaluation
2. Fine-tuned YOLOv8s on wildlife data -- your first hands-on detection model
3. Evaluated with mAP metrics -- now you can read and interpret detection benchmarks
4. Compared approaches -- YOLO for camera traps, HerdNet for aerial dense counting

**Case study connection:** The Miesner thesis reports F1 = 0.934 for marine iguana detection on Floreana. This was achieved with HerdNet (point-based), not YOLO. The dense colony problem that NMS struggles with is exactly why point-based methods exist.

**Next:** In the Tile Inference module, you will learn how to apply detection models to images much larger than 640x640 -- like the full orthomosaics from drone surveys.
```

### Cell 19 [markdown]: Solutions

```markdown
## Solutions
```

### Cell 20 [code]: All Solutions

```python
# ---------- SOLUTIONS ----------
# [Complete solutions for all TODO exercises]
```
