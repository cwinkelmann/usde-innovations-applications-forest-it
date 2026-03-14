# MegaDetector → Species Classifier Pipeline

## Architecture

```
Camera trap image
        |
  [MegaDetector v5]
        |
  Detections: animal / person / vehicle
        |
  Filter: category == 'animal', conf >= 0.2
        |
  Crop bounding boxes (with 10% padding)
        |
  [Species Classifier]  ← timm / DeepFaune / SpeciesNet
        |
  Species label + confidence per detection
        |
  Merge: image_path + bbox + MD_conf + species + species_conf
```

## Why This Pipeline

MegaDetector answers: **"Is there an animal?"**
The classifier answers: **"Which species?"**

MegaDetector is trained on millions of camera trap images across all species — it generalizes well to "animal vs background." But it has only 3 classes (animal/person/vehicle) and cannot identify species.

Species classifiers (DeepFaune, SpeciesNet, custom timm models) are trained on cropped animal images. They expect the input to be a tight crop of a single animal, not a full scene.

## Crop Extraction

```python
from PIL import Image

def crop_detection(img, bbox, padding=0.1):
    """
    Extract animal crop from MegaDetector bbox.
    bbox: [x_min, y_min, width, height] normalized 0-1
    padding: fractional padding around box (0.1 = 10%)
    """
    w, h = img.size
    x1 = bbox[0] * w
    y1 = bbox[1] * h
    x2 = (bbox[0] + bbox[2]) * w
    y2 = (bbox[1] + bbox[3]) * h

    # Add padding
    pad_w = (x2 - x1) * padding
    pad_h = (y2 - y1) * padding
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)

    return img.crop((x1, y1, x2, y2))
```

## Classifier Options

| Classifier | Species Count | Region | Input Size | Notes |
|---|---|---|---|---|
| DeepFaune | 34 | Europe | 182×182 | DINOv2 ViT-L backbone; non-commercial license |
| SpeciesNet | 2000+ | Worldwide | varies | EfficientNetV2-M + MD ensemble; geographic filtering |
| Custom timm | User-defined | Any | Model-dependent | Fine-tuned on your own dataset |

## Handling Edge Cases

- **Multiple animals per image:** Each detection gets its own crop and classification
- **Overlapping detections:** Run NMS before cropping (MD already does this internally)
- **Very small detections:** Skip crops smaller than 32×32 pixels — classifier cannot meaningfully classify them
- **No detections:** Image is "empty" — no classifier needed
