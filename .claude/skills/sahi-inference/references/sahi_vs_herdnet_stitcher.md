# SAHI vs. HerdNet Stitcher: Two Approaches to Tiled Inference

## Overview

Both SAHI and HerdNet's Stitcher solve the same fundamental problem: running
detection models on images too large for a single forward pass. But they take
fundamentally different approaches.

| Aspect | SAHI | HerdNet Stitcher |
|--------|------|------------------|
| **Output type** | Bounding boxes | Point locations (from density maps) |
| **Tile merging** | NMS/NMM on boxes | Hann window blending on heatmaps |
| **Model dependency** | Any detector (YOLO, Detectron2, etc.) | HerdNet only |
| **Boundary handling** | Overlap + suppression | Overlap + weighted blending |
| **Dense scenes** | NMM needed, can still miss | Native density estimation |
| **Ease of use** | `pip install sahi`, 5 lines | Part of animaloc package |
| **Coordinate output** | Bounding box corners | Single point (x, y) |

---

## How SAHI Works (Box-Based)

```
Large Image → Split into overlapping tiles → Run detector on each tile
→ Map per-tile boxes back to full-image coordinates → NMS/NMM to remove duplicates
→ Final list of bounding boxes
```

1. **Split**: Image divided into a grid of overlapping tiles
2. **Detect**: Standard object detector runs on each tile independently
3. **Remap**: Tile-local coordinates shifted to full-image coordinates
4. **Merge**: Overlapping boxes from adjacent tiles are deduplicated via NMS or NMM

**Strengths:**
- Works with any bounding box detector
- Simple, well-tested, widely used
- Produces bounding boxes (useful for downstream cropping/classification)

**Weaknesses:**
- NMS can suppress valid detections of adjacent animals
- Bounding boxes poorly represent irregular clusters
- Each tile is processed independently — no spatial context across tiles

---

## How HerdNet Stitcher Works (Heatmap-Based)

```
Large Image → Split into overlapping patches → Run HerdNet on each patch
→ Get per-patch density heatmap → Blend overlapping heatmaps using Hann window
→ Apply Local Maximum Detection Scheme (LMDS) → Final list of point locations
```

1. **Patch**: Image divided into overlapping patches (via `HerdNetPatcher`)
2. **Predict**: HerdNet produces a density heatmap for each patch
3. **Blend**: Overlapping heatmap regions are blended using a 2D Hann window
   that smoothly weights down values near patch edges
4. **Detect**: LMDS finds local maxima in the blended heatmap → point detections

**Strengths:**
- Hann window blending eliminates hard boundary artifacts
- Density maps handle dense clusters naturally (no NMS needed)
- Point output is appropriate for counting (no box ambiguity)

**Weaknesses:**
- Only works with HerdNet (not model-agnostic)
- Requires the full animaloc/HerdNet pipeline
- Point output lacks bounding box information for downstream tasks

---

## Hann Window Blending (Key Difference)

The Hann window is a bell-shaped weighting function that equals 1.0 at the center
of a patch and smoothly decreases to 0.0 at the edges:

```
Hann(x, y) = 0.5 * (1 - cos(2π * x/W)) * 0.5 * (1 - cos(2π * y/H))
```

When patches overlap, the Hann-weighted contributions from adjacent patches
sum to approximately 1.0 in the overlap region, creating a seamless blend.

**Why this matters for wildlife counting:**
- An iguana at a tile boundary gets a density response in both adjacent tiles
- SAHI approach: two boxes → NMS keeps one → correct (if threshold is right)
- HerdNet approach: two heatmap peaks → Hann blending averages them → one smooth
  peak → LMDS finds one point → correct (by construction)

The HerdNet approach is more robust because the blending is continuous and
deterministic, while NMS depends on an arbitrary threshold.

---

## When to Use Which

### Use SAHI when:
- You have a bounding box detector (YOLO, Faster R-CNN, DETR)
- You need bounding boxes for downstream tasks (cropping for classification)
- You want a quick, model-agnostic solution
- You're working with MegaDetector
- Animals are sparsely distributed (NMS works well)

### Use HerdNet Stitcher when:
- You're using HerdNet for point-based detection
- Animals are densely packed (colonies, herds, nesting sites)
- Accurate counting is more important than bounding boxes
- You're already in the animaloc/HerdNet ecosystem
- You're doing aerial wildlife surveys where point counts are the deliverable

### Use both for comparison when:
- Evaluating detection approaches for a new dataset
- Teaching a course module on tiled inference (show both paradigms)
- Developing a pipeline where you want to validate counts from one approach
  against the other

---

## Practical Comparison

For the marine iguana case study (Miesner 2025), HerdNet + Stitcher achieved:
- F1 = 0.934 on Floreana (dense colonies)
- F1 = 0.843 on Fernandina (challenging terrain)

These results are with the heatmap-based approach. Comparable YOLOv8 results
with SAHI on similar imagery would require careful NMM tuning to match the
dense-colony performance, as NMS-based approaches inherently struggle with
tightly packed animals.

---

## Code Comparison

### SAHI (5 lines of core code)
```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

model = AutoDetectionModel.from_pretrained(model_type='yolov8', model_path='yolov8s.pt',
                                            confidence_threshold=0.3, device='cuda:0')
result = get_sliced_prediction(image='orthomosaic.tif', detection_model=model,
                                slice_height=640, slice_width=640,
                                overlap_height_ratio=0.25, overlap_width_ratio=0.25)
```

### HerdNet Stitcher (conceptual — requires animaloc setup)
```python
from animaloc.eval import HerdNetStitcher, Patcher, LMDS

patcher = Patcher(image_path='orthomosaic.tif', patch_size=640, overlap=160)
stitcher = HerdNetStitcher(model=herdnet_model, patcher=patcher,
                            lmds_kwargs={'kernel_size': (5, 5)})
detections = stitcher.predict()  # Returns list of (x, y, class, confidence)
```
