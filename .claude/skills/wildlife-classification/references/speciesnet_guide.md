# SpeciesNet Guide

Reference for using SpeciesNet, Google's ensemble model for wildlife species classification from camera trap images. SpeciesNet combines MegaDetector with an EfficientNetV2-M classifier and supports geographic filtering across 2000+ species.

---

## Overview

SpeciesNet is an inference-only ensemble model that:
1. Detects animals using MegaDetector
2. Classifies detected crops using EfficientNetV2-M
3. Filters predictions using geographic priors (which species are plausible in a given country)

**Key constraint:** SpeciesNet does NOT support fine-tuning. It is a fixed inference pipeline.

---

## Installation

```bash
pip install speciesnet
```

---

## CLI Usage

### Basic Inference

```bash
python -m speciesnet.scripts.run_model \
    --folders "images/" \
    --predictions_json "output.json"
```

### With Geographic Filtering

```bash
python -m speciesnet.scripts.run_model \
    --folders "images/" \
    --predictions_json "output.json" \
    --country GBR      # ISO 3166-1 alpha-3 country code
```

**Common country codes for wildlife ecology:**

| Country | Code | Notes |
|---------|------|-------|
| United Kingdom | GBR | |
| Germany | DEU | |
| France | FRA | |
| Ecuador | ECU | Galapagos Islands |
| United States | USA | |
| Kenya | KEN | East African wildlife |
| South Africa | ZAF | |
| Australia | AUS | |
| Brazil | BRA | |
| Canada | CAN | |

### Processing Multiple Folders

```bash
python -m speciesnet.scripts.run_model \
    --folders "site_a/" "site_b/" "site_c/" \
    --predictions_json "all_sites.json"
```

---

## Model Variants

### v4.0.2a (Always-Crop)
- Always crops the detected animal before classification
- More robust for images with multiple animals
- Recommended for camera trap images with cluttered backgrounds

### v4.0.2b (Full-Image)
- Uses the full image when no animal is detected
- Falls back to full-image classification
- May be better for images where context matters

---

## Output Format

SpeciesNet produces a JSON file with predictions per image:

```json
{
  "images": [
    {
      "file": "images/IMG_0001.jpg",
      "detections": [
        {
          "bbox": [0.12, 0.08, 0.45, 0.72],
          "confidence": 0.95,
          "category": "animal"
        }
      ],
      "classifications": [
        {
          "species": "vulpes vulpes",
          "common_name": "Red Fox",
          "confidence": 0.87,
          "taxonomy": {
            "class": "Mammalia",
            "order": "Carnivora",
            "family": "Canidae"
          }
        }
      ]
    }
  ]
}
```

---

## Architecture Details

### Detection Stage: MegaDetector
- Architecture: YOLOv5-based (detection head)
- Categories: animal, person, vehicle
- Bounding box format: [x_min, y_min, width, height] normalized to [0, 1]

### Classification Stage: EfficientNetV2-M
- Architecture: EfficientNetV2-M (medium variant)
- Parameters: ~54M
- Input: Cropped animal region, resized
- Output: Softmax probabilities over 2000+ species
- Geographic filtering: Post-hoc probability adjustment based on species range

### Ensemble Logic
1. Run MegaDetector to detect animals
2. Crop each detected animal
3. Run EfficientNetV2-M on each crop
4. Apply geographic filter (if country code provided)
5. Return top predictions per crop

---

## Comparison with timm and DeepFaune

| Feature | SpeciesNet | timm/DINOv2 | DeepFaune |
|---------|-----------|-------------|-----------|
| Species count | 2000+ | Customizable | 34 (European) |
| Fine-tuning | No | Full support | Backbone only (non-European) |
| Detection included | Yes (MegaDetector) | No | Yes (YOLOv8s) |
| Geographic filtering | Yes | No | No |
| Input handling | Full image (auto-crop) | Pre-cropped | Full image (auto-crop) |
| Classifier backbone | EfficientNetV2-M | Any timm model | DINOv2 ViT-L |
| Deployment | CLI / Python API | Custom script | Custom script |
| Training data needed | None | Yes (labeled) | Yes (for new species) |
| Best for | Zero-shot baseline | Custom fine-tuning | European camera traps |

---

## When to Use SpeciesNet

### Good Use Cases
1. **Quick baseline** -- run SpeciesNet first to see if your species are already covered
2. **Zero-shot species ID** -- you have no training data but need species labels
3. **Geographic filtering** -- you know the country and want to narrow predictions
4. **Pseudo-labeling** -- use SpeciesNet predictions as noisy labels to bootstrap a training dataset
5. **Comparison benchmark** -- evaluate your fine-tuned model against SpeciesNet's zero-shot performance

### Not Suitable For
1. **Fine-tuning** -- SpeciesNet does not expose training APIs
2. **Novel species** -- if your species is not in the 2000+ list, predictions will be wrong
3. **Drone imagery** -- SpeciesNet is designed for camera trap images, not nadir aerial views
4. **Real-time inference** -- the ensemble pipeline is heavier than a single classifier
5. **Custom taxonomies** -- you cannot modify the species list or add new classes

---

## Using SpeciesNet as a Pseudo-Labeler

A practical workflow for bootstrapping training data:

```python
import json
import shutil
from pathlib import Path

# Step 1: Run SpeciesNet on unlabeled images
# python -m speciesnet.scripts.run_model --folders unlabeled/ --predictions_json preds.json --country ECU

# Step 2: Filter high-confidence predictions
with open('preds.json') as f:
    results = json.load(f)

CONFIDENCE_THRESHOLD = 0.85  # Only use high-confidence predictions

for img_result in results['images']:
    file_path = img_result['file']
    for cls in img_result.get('classifications', []):
        if cls['confidence'] >= CONFIDENCE_THRESHOLD:
            species = cls['species'].replace(' ', '_')
            # Step 3: Copy to ImageFolder structure
            dest_dir = Path(f'pseudo_labeled/{species}/')
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest_dir / Path(file_path).name)

# Step 4: Now fine-tune a timm model on pseudo_labeled/ with manual verification
```

**Warning:** Pseudo-labels contain errors. Always manually verify a random sample (at least 10% of images) before training on pseudo-labeled data.

---

## Python API Usage

```python
from speciesnet import SpeciesNet

# Initialize model
model = SpeciesNet()

# Predict on a single image
result = model.predict("path/to/image.jpg", country="ECU")

# Predict on a folder
results = model.predict_folder("path/to/images/", country="ECU")
```

**Note:** The Python API is less documented than the CLI. The CLI is the recommended interface for batch processing.
