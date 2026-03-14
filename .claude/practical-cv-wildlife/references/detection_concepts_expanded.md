# Detection Concepts Expanded -- From Conceptual to Practical

## Purpose

This reference expands PCV Module 8 from conceptual detection/segmentation overview to practical implementation knowledge. It provides the technical depth needed by the `detection_bridge_agent` and `exercise_generator_agent` to produce hands-on content.

---

## Anchor Boxes and Anchor-Free Detection

### Anchor-Based Detection (YOLO v3/v5, Faster R-CNN)

Anchor-based detectors predefine a set of reference boxes at each spatial location in the feature map:

```
For each feature map cell (i, j):
  For each anchor box k (e.g., k=3 anchors of different aspect ratios):
    Predict: dx, dy, dw, dh (offsets from anchor)
    Predict: confidence (objectness score)
    Predict: class probabilities
```

Anchor boxes are defined by (width, height) pairs, typically determined by k-means clustering on the training dataset's bounding box dimensions.

**Problem:** Anchor design requires dataset-specific tuning. Wrong anchors = poor detection of unusual aspect ratios.

### Anchor-Free Detection (YOLOv8, FCOS, CenterNet)

Anchor-free detectors predict object properties directly without reference boxes:

```
For each feature map cell (i, j):
  Predict: x_offset, y_offset (object center offset from cell center)
  Predict: width, height (absolute or relative to input size)
  Predict: confidence
  Predict: class probabilities
```

**YOLOv8 specifics:**
- Each grid cell predicts object center offset and dimensions directly
- No anchor box clustering needed
- Task-Aligned Assigner (TAL) for positive sample assignment during training
- Distribution Focal Loss (DFL) for box regression

**Why anchor-free matters for wildlife:** Wildlife objects have highly variable aspect ratios (iguana from above is roughly circular; iguana from the side is elongated). Anchor-free detection adapts without manual anchor design.

---

## IoU (Intersection over Union)

### Formula

```
IoU(A, B) = Area(A intersection B) / Area(A union B)
         = Area(A intersection B) / (Area(A) + Area(B) - Area(A intersection B))
```

### Implementation

```python
def compute_iou(box_a, box_b):
    """
    Compute IoU between two boxes.
    Each box is [x1, y1, x2, y2] (top-left and bottom-right corners).
    """
    # Intersection coordinates
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    # Intersection area (0 if no overlap)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Union area
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0
```

### IoU Interpretation

| IoU Range | Interpretation |
|-----------|---------------|
| 0.0 | No overlap |
| 0.0 - 0.3 | Poor overlap |
| 0.3 - 0.5 | Partial overlap |
| 0.5 | Standard threshold for "correct" detection (AP50) |
| 0.5 - 0.75 | Good overlap |
| 0.75 | Strict threshold (AP75) |
| 0.75 - 1.0 | Excellent overlap |
| 1.0 | Perfect overlap |

---

## NMS (Non-Maximum Suppression)

### Algorithm

```
Input: List of detections D, each with (box, confidence, class)
       NMS threshold T (typically 0.45-0.65)

Output: Filtered list of detections F

Algorithm:
1. Sort D by confidence (descending)
2. Initialize F = []
3. While D is not empty:
   a. Take the highest-confidence detection d from D, add to F
   b. Remove d from D
   c. For each remaining detection d' in D:
      - Compute IoU(d.box, d'.box)
      - If IoU > T and d.class == d'.class: remove d' from D
4. Return F
```

### Implementation

```python
def nms(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression.
    boxes: Nx4 array of [x1, y1, x2, y2]
    scores: N array of confidence scores
    Returns: indices of kept detections
    """
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        ious = np.array([compute_iou(boxes[i], boxes[j]) for j in order[1:]])
        remaining = np.where(ious <= iou_threshold)[0]
        order = order[remaining + 1]

    return keep
```

### NMS Considerations for Wildlife

- **Dense colonies:** Marine iguanas cluster tightly. NMS threshold too low (e.g., 0.3) merges adjacent iguanas into one detection. Threshold too high (e.g., 0.7) keeps duplicate detections.
- **Thesis approach:** HerdNet uses point-based detection, which avoids NMS entirely. Each prediction is a point, not a box, so overlap is not defined.
- **Cross-tile NMS:** When tiling large images, detections near tile boundaries may be duplicated. NMS must be applied globally after stitching tile-local detections.

---

## Detection Metrics

### Precision-Recall Curve

Given a set of predictions sorted by confidence:

```
For each detection (in order of decreasing confidence):
  If IoU with a ground truth box >= threshold AND that GT box is not yet matched:
    -> True Positive (TP)
  Else:
    -> False Positive (FP)

Ground truth boxes with no matching prediction:
  -> False Negative (FN)

At each confidence threshold:
  Precision = cumulative_TP / (cumulative_TP + cumulative_FP)
  Recall = cumulative_TP / total_GT
```

### AP (Average Precision)

AP is the area under the Precision-Recall curve, computed using the 101-point interpolation method (COCO standard):

```
recall_points = [0.0, 0.01, 0.02, ..., 1.0]  # 101 points
for each r in recall_points:
    p_interp(r) = max(precision(r') for r' >= r)
AP = mean(p_interp(r) for r in recall_points)
```

### Common AP Metrics

| Metric | IoU Threshold | Description |
|--------|--------------|-------------|
| AP50 | 0.50 | Lenient -- does the model find the object approximately? |
| AP75 | 0.75 | Strict -- does the model localize the object precisely? |
| mAP (COCO) | avg over [0.50:0.05:0.95] | Overall performance across localization precisions |
| mAP (VOC) | 0.50 | VOC-style mAP (equivalent to AP50 averaged over classes) |
| AP_small | COCO, area < 32^2 px | Small object performance (critical for wildlife) |
| AP_medium | COCO, 32^2 < area < 96^2 px | Medium object performance |
| AP_large | COCO, area > 96^2 px | Large object performance |

### Counting Metrics (Point-Based)

For point-based detection (HerdNet), box-based metrics are not applicable. Instead:

```
MAE = (1/N) * sum(|predicted_count - true_count|)    for each image
RMSE = sqrt((1/N) * sum((predicted_count - true_count)^2))
Relative Error = |predicted - true| / true
```

The thesis reports point-level F1 by matching predicted points to ground truth points within a distance threshold.

---

## YOLOv8 Architecture

### Backbone: CSPNet (Cross-Stage Partial Network)

```
Input (640x640x3)
  -> Stem: Conv 3x3, stride 2 (320x320x32)
  -> Stage 1: C2f block (160x160x64)
  -> Stage 2: C2f block (80x80x128)
  -> Stage 3: C2f block (40x40x256)       -> P3 features
  -> Stage 4: C2f block (20x20x512)       -> P4 features
  -> Stage 5: SPPF (20x20x512)            -> P5 features
```

C2f (Cross-Stage 2 with Flow) blocks use partial channel connections to reduce computation while maintaining gradient flow.

### Neck: PAN (Path Aggregation Network)

```
P5 (20x20) -> Upsample -> Concat with P4 -> C2f -> N4 (40x40)
N4 (40x40) -> Upsample -> Concat with P3 -> C2f -> N3 (80x80)
N3 (80x80) -> Conv stride 2 -> Concat with N4 -> C2f -> N4' (40x40)
N4'(40x40) -> Conv stride 2 -> Concat with P5 -> C2f -> N5' (20x20)
```

PAN fuses features from multiple scales, enabling detection of small objects (P3 features) and large objects (P5 features).

### Head: Decoupled Detection Head

```
For each scale (N3, N4', N5'):
  Classification branch:
    -> Conv 3x3 -> Conv 3x3 -> Conv 1x1 (num_classes outputs)
  Regression branch:
    -> Conv 3x3 -> Conv 3x3 -> Conv 1x1 (4 * reg_max outputs for DFL)
```

Decoupled head separates classification and localization, which converge at different rates.

### YOLOv8 Training with Ultralytics API

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolov8s.pt")

# Train on custom dataset
results = model.train(
    data="wildlife_data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,        # early stopping
    lr0=0.01,           # initial learning rate
    lrf=0.01,           # final learning rate (fraction of lr0)
    augment=True,       # mosaic, mixup, etc.
    device=0,           # GPU
)

# Evaluate
metrics = model.val()
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")

# Predict
results = model.predict("test_image.jpg", conf=0.25, iou=0.45)
```

### data.yaml Format

```yaml
path: /data/wildlife_dataset
train: images/train
val: images/val
test: images/test

nc: 10  # number of classes
names:
  0: zebra
  1: wildebeest
  2: elephant
  3: giraffe
  4: lion
  5: buffalo
  6: impala
  7: warthog
  8: hyena
  9: ostrich
```

### YOLO Annotation Format

One .txt file per image, same name as image file. Each line:
```
class_id x_center y_center width height
```
All values normalized to [0, 1] relative to image dimensions.

---

## Detection vs Point-Based Counting Comparison

| Criterion | YOLOv8 (Box Detection) | HerdNet (Point Detection) |
|-----------|----------------------|--------------------------|
| Annotation type | Bounding boxes (4 coords per object) | Points (2 coords per object) |
| Annotation time | Slower (draw boxes) | Faster (click centers) |
| Dense scenes | Boxes overlap, NMS may merge | Points handle density naturally |
| Small objects (<20 px) | Box regression noisy | Point prediction more stable |
| Counting accuracy | Count = len(boxes after NMS) | Count = len(points after thresholding) |
| Localization precision | Box gives size estimate | Point gives center only |
| Available tools | ultralytics, Detectron2, mmdet | HerdNet, P2PNet, CounTR |
| Pretrained models | COCO, Objects365 | Few pretrained options |
| Recommendation | Camera traps (distinct individuals) | Aerial surveys (dense colonies) |

**Thesis choice rationale:** Marine iguanas in aerial imagery are small (15-40 px), densely packed, and often overlapping. Point-based detection with FIDT maps avoids the NMS problems that plague box-based detection in these conditions.
