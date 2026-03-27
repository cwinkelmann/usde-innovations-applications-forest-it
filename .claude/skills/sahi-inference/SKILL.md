---
name: sahi-inference
description: >
  SAHI (Slicing Aided Hyper Inference) skill for running object detection on large
  images using tiled inference with overlapping tiles. Covers slice inference setup,
  AutoDetectionModel configuration, get_sliced_prediction usage, postprocess strategies
  including NMS merge and NMM, and integration with drone orthomosaic workflows.
  Handles large image detection via tile stitching for wildlife monitoring.
trigger_keywords:
  - SAHI
  - slicing
  - tiled inference
  - large image
  - overlapping tiles
  - slice inference
  - AutoDetectionModel
  - get_sliced_prediction
  - postprocess
  - NMS merge
  - drone orthomosaic
  - tile stitching
version: "1.0"
---

# SAHI Inference Skill

## Overview

SAHI (Slicing Aided Hyper Inference) is an open-source library that enables object
detection on arbitrarily large images by splitting them into overlapping tiles, running
a detector on each tile independently, and merging the per-tile detections back into
full-image coordinates using Non-Maximum Suppression (NMS) or Non-Maximum Merging (NMM).
This is critical for drone orthomosaics and satellite imagery where target animals are
tiny (often <50 pixels) relative to the full image (often >10,000 pixels per side).

**Repository:** https://github.com/obss/sahi
**Paper:** Akyon et al. (2022) — "Slicing Aided Hyper Inference and Fine-tuning for
Small Object Detection" (IEEE ICIP 2022)

---

## Installation

```bash
pip install sahi
```

For specific model backends:
```bash
pip install sahi[yolov8]       # Ultralytics YOLOv8
pip install sahi[torch]        # torchvision models
pip install sahi[mmdet]        # MMDetection
pip install sahi[detectron2]   # Detectron2
pip install sahi[huggingface]  # HuggingFace transformers
```

---

## Does NOT Trigger

| If the user wants... | Use this skill instead |
|---|---|
| Point-based heatmap detection (HerdNet Stitcher/LMDS) | herdnet-training |
| Model training or fine-tuning | herdnet-training or wildlife-classification |
| Single-image MegaDetector inference (no tiling) | megadetector |
| Course material from iguana case study | iguana-case-study |
| Active learning sample selection | active-learning-wildlife |

---

## Core API

### AutoDetectionModel

Factory for loading any supported detection model:

```python
from sahi import AutoDetectionModel

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',        # 'yolov5', 'detectron2', 'mmdet', 'huggingface', 'torchvision'
    model_path='yolov8s.pt',
    confidence_threshold=0.3,
    device='cuda:0',            # 'cpu', 'cuda:0', 'mps'
)
```

### get_sliced_prediction

Run tiled inference on a large image:

```python
from sahi.predict import get_sliced_prediction

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
```

### get_prediction

Standard (non-sliced) inference for comparison:

```python
from sahi.predict import get_prediction

result = get_prediction(
    image='image.jpg',
    detection_model=detection_model,
)
```

### Supported Model Types

| model_type     | Backend         | Notes                                |
|----------------|-----------------|--------------------------------------|
| `yolov8`       | Ultralytics     | Recommended for most wildlife tasks  |
| `yolov5`       | Ultralytics     | MegaDetector v5 uses this            |
| `detectron2`   | Detectron2      | Mask R-CNN, Faster R-CNN             |
| `mmdet`        | MMDetection     | Large model zoo                      |
| `huggingface`  | Transformers    | DETR, DETA, RT-DETR                  |
| `torchvision`  | torchvision     | FCOS, RetinaNet, SSD                 |

---

## Key Parameters

### Slice Dimensions

- **`slice_height` / `slice_width`**: Size of each tile in pixels. Typically 640-1280
  for wildlife applications.
- Rule of thumb: slice_size should be 1.5-3x the largest expected object dimension.

### Overlap Ratios

- **`overlap_height_ratio` / `overlap_width_ratio`**: Fraction of slice dimension
  that overlaps with adjacent tiles. Range: 0.0-0.5. Typical wildlife settings: 0.2-0.3.
- Overlap should be >= largest_object_size / slice_size to ensure every object is
  fully contained in at least one tile.

### Postprocessing

- **`postprocess_type`**: `'NMS'` (Non-Maximum Suppression) or `'NMM'` (Non-Maximum Merging).
- **`postprocess_match_metric`**: `'IOU'` (Intersection over Union) or `'IOS'`
  (Intersection over Smaller area). IOS is better for wildlife where boxes may have
  very different sizes.
- **`postprocess_match_threshold`**: 0.5 default. Lower (0.3) for dense animal
  colonies, higher (0.7) for sparse scenes.

---

## Agent Team

This skill uses a 4-agent team:

| Agent | Role | When to invoke |
|-------|------|----------------|
| `config_agent` | Determines optimal SAHI parameters based on image/object characteristics | User needs help choosing slice_size, overlap, postprocess settings |
| `code_generation_agent` | Generates working SAHI code for specific models and use cases | User wants runnable code for their pipeline |
| `optimization_agent` | Tunes SAHI parameters for speed, accuracy, and memory | User has a working pipeline but needs better performance |
| `exercise_designer_agent` | Creates learning exercises about tiled inference | User is learning about SAHI or teaching a workshop |

---

## Modes

### 1. `generate-code` (default)
Generate working SAHI inference code for a specific model and use case. Delegates
to `code_generation_agent` with config from `config_agent`.

### 2. `explain-concept`
Explain how SAHI works — slicing, overlap, NMS merging, coordinate mapping. Use
diagrams and step-by-step walkthroughs. Good for teaching contexts.

### 3. `optimize-pipeline`
Analyze an existing SAHI pipeline and suggest parameter improvements. Delegates
to `optimization_agent`. Requires user to provide current settings and performance
metrics.

### 4. `create-exercise`
Design a hands-on exercise about tiled inference. Delegates to `exercise_designer_agent`.
Specify difficulty level: basic, intermediate, or advanced.

### 5. `full-course-module`
Create a complete teaching module covering SAHI from theory to practice. Combines
all agents: concept explanation, parameter selection guide, code examples, and
exercises with solutions.

---

## Integration with Other Skills

### megadetector
SAHI can wrap MegaDetector as its underlying detection model. Use `model_type='yolov5'`
for MDv5a. This enables tiled inference on large drone images or panoramic camera
trap images where standard MegaDetector inference would miss small animals. See
`references/sahi_with_megadetector.md`.

### herdnet-training
HerdNet uses its own `HerdNetStitcher` for tiled inference on heatmaps, which is
conceptually similar but operates differently (Hann window blending on density maps
rather than NMS on bounding boxes). Compare SAHI's box-based approach with HerdNet's
heatmap-based approach in `references/sahi_vs_herdnet_stitcher.md`.

### practical-cv-wildlife
The tiled inference exercise in practical-cv-wildlife can use SAHI as the
implementation framework. Students can compare standard inference vs. SAHI-tiled
inference to see the improvement on large images with small objects.

---

## Common Failure Paths

### 1. Slice size too small
**Symptom:** Large animals are cut across many tiles, NMS struggles to merge fragments.
**Fix:** Increase `slice_height`/`slice_width` so the largest expected animal fits
fully within a single tile.

### 2. Slice size too large
**Symptom:** Small animals become tiny within tiles, detector misses them (same problem
as no slicing at all).
**Fix:** Decrease slice size. Target 1.5-3x the largest expected object dimension.

### 3. Overlap too small
**Symptom:** Animals near tile borders are clipped and either missed entirely or
detected as partial objects with low confidence.
**Fix:** Increase overlap ratios. Minimum overlap should be >= largest_object_size / slice_size.

### 4. Overlap too large
**Symptom:** Very slow inference (many redundant tiles). Excessive duplicate detections
that stress the postprocessing step.
**Fix:** Reduce overlap to the minimum needed. For sparse scenes, 0.2 is usually sufficient.

### 5. NMS threshold too aggressive (too low)
**Symptom:** Valid detections of nearby but distinct animals are suppressed. Clustered
animals (e.g., iguana colonies) have many missed individuals.
**Fix:** Raise `postprocess_match_threshold` (e.g., from 0.3 to 0.5-0.6) or switch
to NMM which merges rather than suppresses.

### 6. NMS threshold too permissive (too high)
**Symptom:** Many duplicate detections of the same animal from overlapping tiles.
Inflated count estimates.
**Fix:** Lower `postprocess_match_threshold` or switch match metric from IOU to IOS.

### 7. Wrong model_type string
**Symptom:** `ValueError` or model fails to load.
**Fix:** Ensure `model_type` matches the actual model framework. Check supported
types in the API reference.

### 8. Out of memory on very large orthomosaics
**Symptom:** CUDA OOM or system memory exhaustion when processing multi-gigapixel images.
**Fix:** Use `auto_slice_resolution=True` or process the image in spatial chunks,
writing intermediate results to disk.
