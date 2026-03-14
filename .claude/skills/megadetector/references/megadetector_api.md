# MegaDetector API Reference

Source: `/Users/christian/work/hnee/MegaDetector/megadetector/detection/`

## Model Loading

```python
from megadetector.detection.run_detector import load_detector

# Auto-download from GitHub releases
detector = load_detector('MDV5A')  # Recommended default
detector = load_detector('MDV5B')  # Alternative training data
detector = load_detector('default')  # Alias for MDV5A

# Or pass a local .pt file path
detector = load_detector('/path/to/md_v5a.0.1.pt')
```

**Auto-download:** `try_download_known_detector(model_name)` downloads to user cache directory.

**Known model strings:**
- `'MDV5A'` → `md_v5a.0.1.pt` (YOLOv5, image_size=1280, threshold=0.2)
- `'MDV5B'` → `md_v5b.0.1.pt` (same arch, different training split)
- `'v1000.0.0-redwood'` through `'v1000.0.0-spruce'` → 2025 models (5 speed/accuracy variants)

## Single-Image Inference

```python
from PIL import Image
import torch

img = Image.open('image.jpg')
# PTDetector.generate_detections_one_batch expects:
#   img_original: list of PIL/numpy images
#   image_id: list of string identifiers
#   detection_threshold: float (default 0.005 — filter later)
results = detector.generate_detections_one_batch(
    img_original=[img],
    image_id=['image.jpg'],
    detection_threshold=0.005
)
```

## Batch CLI Processing

```bash
python -m megadetector.detection.run_detector_batch \
  MDV5A \
  "/path/to/images/" \
  "/path/to/output.json" \
  --output_relative_filenames \
  --recursive \
  --checkpoint_frequency 10000 \
  --quiet
```

**Key CLI arguments:**
- `--output_relative_filenames` — paths relative to image folder
- `--recursive` — scan subdirectories
- `--checkpoint_frequency N` — save progress every N images (crash recovery)
- `--resume_from_checkpoint auto` — resume from latest checkpoint
- `--confidence_threshold` — minimum confidence to include (default 0.005)
- `--ncores` — CPU workers (GPU ignores this)

## Detection Label Map

```python
DEFAULT_DETECTOR_LABEL_MAP = {
    '1': 'animal',
    '2': 'person',
    '3': 'vehicle'  # v4+ only
}
```

## Confidence Thresholds by Model

| Model | Typical | Conservative |
|---|---|---|
| MDV5A/B | 0.2 | 0.05 |
| MDV4.1 | 0.8 | 0.3 |
| v1000 series | 0.2 | 0.05 |

## Postprocessing Utilities

```python
# Separate images into folders by detection type
python -m megadetector.postprocessing.separate_detections_into_folders \
  output.json "/path/to/images/" "/path/to/sorted/" --threshold 0.2

# Generate CSV report
python -m megadetector.postprocessing.generate_csv_report output.json

# Convert to COCO format
python -m megadetector.postprocessing.md_to_coco output.json coco_output.json
```

## GPU Test

```bash
python -m megadetector.utils.gpu_test
```
