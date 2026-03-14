# Confidence Threshold Selection Guide

## Default Thresholds

| Model | Typical (balanced) | Conservative (recall-oriented) |
|---|---|---|
| MDV5A / MDV5B | **0.2** | 0.05 |
| MDV4.1 | 0.8 | 0.3 |
| v1000 series | 0.2 | 0.05 |

## What Confidence Means

MegaDetector confidence is a softmax/sigmoid output from the YOLOv5 detection head. It does NOT represent a calibrated probability. A score of 0.8 does not mean "80% chance this is an animal." It means the detector is relatively confident compared to its training distribution.

## Choosing a Threshold

### For wildlife surveys (recall matters)
Use **0.1–0.2**. Accept more false positives to avoid missing animals. Review flagged images manually.

### For automated sorting (precision matters)
Use **0.5–0.8**. Only move images to "animal" folder if highly confident. Accept missing some animals.

### For HITL review pipeline
Use **two thresholds**:
- Auto-accept above **0.8** (high confidence animals)
- Send to review queue between **0.1–0.8**
- Auto-reject below **0.1** (very low confidence)

## Threshold Sweep on Your Data

Always validate on a small labeled subset before processing a full survey:

```python
import numpy as np
from sklearn.metrics import precision_recall_curve

# gt_labels: list of 0/1 (does image contain animal?)
# md_scores: list of max_detection_conf from MegaDetector output
precisions, recalls, thresholds = precision_recall_curve(gt_labels, md_scores)

# Find threshold for desired recall (e.g., 0.95)
target_recall = 0.95
idx = np.argmin(np.abs(recalls - target_recall))
optimal_threshold = thresholds[idx]
```

## Common Mistakes

1. **Using MDV4 threshold on MDV5.** V4 uses ~0.8; V5 uses ~0.2. Using 0.8 on V5 will miss most animals.
2. **Filtering before saving.** Always save all detections (threshold 0.005), then filter downstream. You can always raise the threshold later but can't recover discarded detections.
3. **Same threshold across sites.** Camera angle, lighting, and species appearance affect confidence distributions. Sweep per deployment site if possible.
