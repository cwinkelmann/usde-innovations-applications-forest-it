# SAHI Parameter Tuning Guide for Wildlife Detection

## Parameter Interaction Map

SAHI has several interacting parameters. Changing one often requires adjusting others.

```
Image Size ──┬── Slice Size ──┬── Overlap Ratio ── Tile Count ── Speed
             │                │
             │                └── Object Coverage ── Recall
             │
             └── Object Size ──── Confidence Threshold ── Precision/Recall
                                       │
                                       └── Postprocess Type ── NMS/NMM
                                              │
                                              └── Match Threshold ── Duplicates
```

---

## Step-by-Step Tuning Protocol

### Step 1: Determine Object Size in Pixels

Before tuning anything, measure your target objects:

```python
# From ground truth annotations (COCO format)
import json
import numpy as np

with open('annotations.json') as f:
    coco = json.load(f)

widths = [ann['bbox'][2] for ann in coco['annotations']]
heights = [ann['bbox'][3] for ann in coco['annotations']]

print(f"Object width:  min={min(widths):.0f}, median={np.median(widths):.0f}, max={max(widths):.0f}")
print(f"Object height: min={min(heights):.0f}, median={np.median(heights):.0f}, max={max(heights):.0f}")
```

Or compute from GSD and physical size:
```
object_size_px = physical_size_cm / GSD_cm_per_px
```

### Step 2: Set Slice Size

**Rule:** slice_size should be 2-3× the largest object dimension, and at least
as large as the model's training resolution.

```python
slice_size = max(
    model_input_size,           # e.g., 640 for YOLOv8
    2 * max_object_size_px      # 2× largest object
)
# Round to nearest 32 (for YOLO compatibility)
slice_size = ((slice_size + 31) // 32) * 32
# Clamp
slice_size = max(320, min(slice_size, 2048))
```

### Step 3: Set Overlap Ratio

**Rule:** overlap must be large enough that every object is fully contained
in at least one tile.

```python
min_overlap_ratio = max_object_size_px / slice_size
# Add safety margin
overlap_ratio = max(min_overlap_ratio * 1.5, 0.2)
# Clamp
overlap_ratio = min(overlap_ratio, 0.5)
```

### Step 4: Set Confidence Threshold

Start with the detector's default and adjust based on the task:

| Task | Starting threshold | Adjust direction |
|------|--------------------|------------------|
| Population census (recall matters) | 0.15-0.25 | Lower if missing animals |
| Presence/absence survey | 0.3-0.4 | Balance FP/FN |
| Individual identification | 0.4-0.5 | Higher to reduce FP |
| Real-time monitoring | 0.3-0.4 | Balance speed and accuracy |

### Step 5: Select Postprocess Strategy

| Situation | postprocess_type | match_metric | match_threshold |
|-----------|-----------------|--------------|-----------------|
| Sparse, varied sizes | NMS | IOS | 0.5 |
| Sparse, similar sizes | NMS | IOU | 0.5 |
| Dense, similar sizes | NMM | IOS | 0.3-0.4 |
| Dense, varied sizes | NMM | IOS | 0.3 |
| Mixed density | NMS | IOS | 0.4-0.5 |

### Step 6: Validate on Test Set

Run the configured pipeline on a test set with ground truth and compute:

```python
from sahi.utils.coco import get_coco_stats

# Requires ground truth COCO JSON and prediction COCO JSON
stats = get_coco_stats(
    coco_ground_truth_path='ground_truth.json',
    coco_prediction_path='predictions.json',
)
```

Key metrics to check:
- **AP@0.5** (mAP at IoU=0.5): Overall detection quality
- **AR@100** (recall with up to 100 detections): Are we finding all animals?
- **AP_small, AP_medium, AP_large**: Performance by object size

---

## Common Tuning Scenarios

### Scenario 1: "I'm missing small animals"

**Diagnosis:** Objects are too small relative to the tile size.

**Fixes (try in order):**
1. Decrease `slice_size` (e.g., 640 → 480 or 320)
2. Lower `confidence_threshold` (e.g., 0.3 → 0.15)
3. Increase `overlap_ratio` (e.g., 0.2 → 0.3)
4. Use a model trained at smaller object sizes

### Scenario 2: "I'm getting too many duplicate detections"

**Diagnosis:** Postprocessing is not merging duplicates from adjacent tiles.

**Fixes:**
1. Lower `postprocess_match_threshold` (e.g., 0.5 → 0.3)
2. Switch `postprocess_match_metric` from IOU to IOS
3. Increase `overlap_ratio` (counterintuitively, more overlap can help NMS
   because both tiles see the full object, producing more similar boxes)

### Scenario 3: "Dense clusters are under-counted"

**Diagnosis:** NMS is suppressing valid adjacent animals.

**Fixes:**
1. Switch from NMS to NMM
2. Raise `postprocess_match_threshold` (e.g., 0.5 → 0.6-0.7)
3. Consider switching to a point-based approach (HerdNet)

### Scenario 4: "Inference is too slow"

**Diagnosis:** Too many tiles being generated.

**Fixes:**
1. Increase `slice_size` (fewer tiles)
2. Decrease `overlap_ratio` (fewer tiles)
3. Set `perform_standard_pred=False` (skip full-image pass)
4. Use a smaller model (YOLOv8n instead of YOLOv8x)
5. Pre-filter empty tiles (variance check)

### Scenario 5: "Out of memory"

**Diagnosis:** Tiles or the full image exceed GPU memory.

**Fixes:**
1. Decrease `slice_size` (smaller tiles = less GPU memory per forward pass)
2. Process image in spatial chunks (see optimization_agent for details)
3. Use FP16 inference
4. Switch to a smaller model

---

## Wildlife-Specific Parameter Presets

### Marine Iguanas (GSD ~1 cm/px, dense colonies)
```python
slice_height=640, slice_width=640,
overlap_height_ratio=0.3, overlap_width_ratio=0.3,
postprocess_type='NMM', postprocess_match_metric='IOS',
postprocess_match_threshold=0.4, confidence_threshold=0.25
```

### Elephants from Satellite (GSD ~50 cm/px)
```python
slice_height=640, slice_width=640,
overlap_height_ratio=0.2, overlap_width_ratio=0.2,
postprocess_type='NMS', postprocess_match_metric='IOU',
postprocess_match_threshold=0.5, confidence_threshold=0.3
```

### Seabird Nesting Colony (GSD ~2 cm/px, extremely dense)
```python
slice_height=640, slice_width=640,
overlap_height_ratio=0.35, overlap_width_ratio=0.35,
postprocess_type='NMM', postprocess_match_metric='IOS',
postprocess_match_threshold=0.3, confidence_threshold=0.2
```

### Camera Trap Panoramic (2× stitched, sparse animals)
```python
slice_height=1280, slice_width=1280,
overlap_height_ratio=0.15, overlap_width_ratio=0.15,
postprocess_type='NMS', postprocess_match_metric='IOU',
postprocess_match_threshold=0.5, confidence_threshold=0.3
```
