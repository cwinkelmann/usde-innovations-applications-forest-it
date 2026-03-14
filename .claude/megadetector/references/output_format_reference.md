# MegaDetector Output Format Reference

Spec: https://lila.science/megadetector-output-format

## JSON Structure

```json
{
  "info": {
    "format_version": "1.5",
    "detector": "md_v5a.0.1.pt",
    "detection_completion_time": "2025-03-13 10:00:00",
    "detector_metadata": {
      "megadetector_version": "v5a.0.1",
      "typical_detection_threshold": 0.2,
      "conservative_detection_threshold": 0.05
    }
  },
  "detection_categories": {
    "1": "animal",
    "2": "person",
    "3": "vehicle"
  },
  "images": [
    {
      "file": "relative/path/to/image.jpg",
      "detections": [
        {
          "category": "1",
          "conf": 0.926,
          "bbox": [0.0, 0.2762, 0.1539, 0.2825]
        }
      ]
    },
    {
      "file": "empty_image.jpg",
      "detections": []
    },
    {
      "file": "corrupted_image.jpg",
      "failure": "Failure image access",
      "detections": null
    }
  ]
}
```

## Bounding Box Format

**`[x_min, y_min, width, height]`** — all values **normalized** to 0–1 range.

- Origin: top-left corner of the image
- `x_min`: left edge of box / image_width
- `y_min`: top edge of box / image_height
- `width`: box width / image_width
- `height`: box height / image_height

**Convert to pixel coordinates:**
```python
x_min_px = bbox[0] * img_width
y_min_px = bbox[1] * img_height
w_px = bbox[2] * img_width
h_px = bbox[3] * img_height
```

**This is NOT the same as:**
- COCO format (`[x, y, w, h]` in absolute pixels)
- YOLO format (`[cx, cy, w, h]` normalized, center-based)
- Pascal VOC (`[x_min, y_min, x_max, y_max]` absolute)

## Converting to DataFrame

```python
import json
import pandas as pd

with open('output.json') as f:
    results = json.load(f)

rows = []
for img in results['images']:
    if img.get('failure'):
        continue
    for det in img.get('detections', []):
        rows.append({
            'file': img['file'],
            'category': results['detection_categories'][det['category']],
            'confidence': det['conf'],
            'x_min': det['bbox'][0],
            'y_min': det['bbox'][1],
            'width': det['bbox'][2],
            'height': det['bbox'][3],
        })
df = pd.DataFrame(rows)
```

## Empty vs Failed Images

- **Empty** (no animals): `"detections": []` — valid result
- **Failed** (corrupt/unreadable): `"failure": "error message"`, `"detections": null`
- Always check for `failure` field before accessing detections
