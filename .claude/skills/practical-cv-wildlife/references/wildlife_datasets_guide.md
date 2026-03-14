# Wildlife Datasets Guide -- Datasets for Teaching and Exercises

## Purpose

This reference catalogs publicly available wildlife datasets suitable for teaching exercises in the practical-cv-wildlife skill. Each dataset entry includes size, format, access method, and suitability for specific exercises.

Used by: `wildlife_adapter_agent`, `exercise_generator_agent`

---

## Dataset Catalog

### 1. iNaturalist 2021 Mini

| Attribute | Value |
|-----------|-------|
| Full name | iNaturalist 2021 Competition Dataset (mini subset) |
| Size | ~500,000 images (mini: ~50,000) |
| Species/Classes | 10,000 species (mini: 10 common species) |
| Format | ImageFolder (class_name/image.jpg) |
| Access | Public, via torchvision or direct download |
| License | CC-BY-NC |
| Image size | Variable (typically 300-1000 px) |

**Best for:** Replacing MNIST/CIFAR in classification exercises. Multi-class species classification.

**Download code:**
```python
from torchvision.datasets import INaturalist
# Full dataset
dataset = INaturalist(root="./data", version="2021_train_mini", download=True)

# Or create a mini subset manually:
# Select 10 common species, 500 images each
```

**PCV swap target:** `Kaggle_Competition_LeNet5_Digit_Recognition.ipynb` (MNIST -> iNaturalist mini)

---

### 2. iWildCam 2022

| Attribute | Value |
|-----------|-------|
| Full name | iWildCam 2022 FGVC Competition Dataset |
| Size | ~260,000 images |
| Species/Classes | 300+ species |
| Format | COCO JSON annotations |
| Access | Kaggle (requires account) |
| License | Competition license |
| Image size | Variable (camera trap images, typically 1920x1080) |

**Best for:** Camera trap species classification with real-world challenges (class imbalance, location bias, day/night variation).

**Download:**
```bash
# Requires kaggle CLI
kaggle competitions download -c iwildcam2022-fgvc9
```

**PCV swap target:** `Finetuning_a_Resnet_for_Multilabel_Classification.ipynb` (multi-label with camera traps)

---

### 3. Caltech Camera Traps (CCT-20)

| Attribute | Value |
|-----------|-------|
| Full name | Caltech Camera Traps |
| Size | ~243,000 images |
| Species/Classes | 20+ species |
| Format | COCO JSON annotations |
| Access | Public via LILA BC (lila.science) |
| License | Community Data License |
| Image size | Variable |

**Best for:** Detection exercises (bounding boxes provided), species classification, domain adaptation studies.

**Download:**
```python
# Download from LILA BC
# https://lila.science/datasets/caltech-camera-traps/
import urllib.request
url = "https://lila.science/wp-content/uploads/2020/01/caltech_camera_traps.json"
```

**PCV swap target:** `Pet_Classification.ipynb` (Oxford Pets -> Caltech Camera Traps species subset)

---

### 4. Snapshot Serengeti

| Attribute | Value |
|-----------|-------|
| Full name | Snapshot Serengeti |
| Size | ~7.1 million images (use Season 1 subset: ~1.2M) |
| Species/Classes | 48 species |
| Format | CSV metadata + images |
| Access | Public via LILA BC |
| License | CC-BY |
| Image size | 2048x1536 typical |

**Best for:** Large-scale exercises, class imbalance handling (many empty frames), temporal analysis.

**Download:**
```python
# Season 1 subset (manageable size)
# https://lila.science/datasets/snapshot-serengeti/
# Download metadata CSV first, then selectively download images
```

**Teaching note:** Use a curated 10-species subset with balanced classes (~500 images per species) for exercises. The full dataset is too large for teaching.

**PCV swap target:** General classification exercises; also useful for DataLoader exercises with metadata

---

### 5. AID (Aerial Image Dataset)

| Attribute | Value |
|-----------|-------|
| Full name | Aerial Image Dataset for Scene Classification |
| Size | ~10,000 images |
| Species/Classes | 30 scene categories (airport, beach, forest, farmland, etc.) |
| Format | ImageFolder (scene_class/image.jpg) |
| Access | Public (request from authors) |
| License | Research use |
| Image size | 600x600 pixels |

**Best for:** Aerial scene classification exercises. Not species-specific but introduces aerial imagery perspective.

**Download:**
```python
# Available from: https://captain-whu.github.io/AID/
# ImageFolder format, ready for torchvision
```

**PCV swap target:** CIFAR-10 classification exercises -> aerial scene classification

---

### 6. DOTA (Dataset for Object Detection in Aerial Images)

| Attribute | Value |
|-----------|-------|
| Full name | DOTA v1.5/v2.0 |
| Size | ~11,000 images, ~1.8M instances |
| Classes | 18 categories (plane, ship, vehicle, etc.) |
| Format | Oriented bounding boxes (OBB) |
| Access | Public (registration required) |
| License | Research use |
| Image size | 800x800 to 20,000x20,000 pixels |

**Best for:** Tile inference exercises (large images requiring tiling), aerial object detection practice.

**Teaching note:** Not wildlife-specific but excellent for teaching tile-based inference mechanics. Large images require SAHI-style processing.

---

### 7. MegaDetector Test Images

| Attribute | Value |
|-----------|-------|
| Full name | MegaDetector sample/test dataset |
| Size | ~100 images |
| Classes | 3 (animal, person, vehicle) |
| Format | COCO JSON |
| Access | Public via Microsoft AI for Earth GitHub |
| License | MIT |
| Image size | Variable camera trap images |

**Best for:** Quick MegaDetector demonstration, two-stage pipeline exercises (detect -> classify).

**Download:**
```python
# https://github.com/microsoft/CameraTraps
# Sample images in the repository
```

---

### 8. Wildlife Aerial Tiles (Synthetic Teaching Set)

For exercises requiring aerial wildlife imagery where no public dataset exists, create a synthetic teaching set:

**Method:**
1. Take public aerial/satellite images (AID, DOTA, or Google Earth screenshots under fair use)
2. Paste wildlife silhouettes at known positions
3. Record ground truth point annotations
4. Use for tile inference and counting exercises only

**Important:** Clearly label as synthetic. Never present as real data. Only use when no real alternative exists.

---

## Format Conversion Code

### COCO JSON to YOLO Format

```python
import json
import os

def coco_to_yolo(coco_json_path, output_dir, image_width, image_height):
    """Convert COCO JSON annotations to YOLO .txt format."""
    with open(coco_json_path) as f:
        coco = json.load(f)

    # Build image_id -> annotations mapping
    img_anns = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)

    os.makedirs(output_dir, exist_ok=True)

    for img in coco["images"]:
        img_id = img["id"]
        w, h = img["width"], img["height"]
        txt_name = os.path.splitext(img["file_name"])[0] + ".txt"

        with open(os.path.join(output_dir, txt_name), "w") as f:
            for ann in img_anns.get(img_id, []):
                bbox = ann["bbox"]  # [x, y, w, h] in pixels
                cat_id = ann["category_id"]

                # Convert to YOLO format (normalized center x, y, w, h)
                x_center = (bbox[0] + bbox[2] / 2) / w
                y_center = (bbox[1] + bbox[3] / 2) / h
                bw = bbox[2] / w
                bh = bbox[3] / h

                f.write(f"{cat_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")
```

### ImageFolder to Classification CSV

```python
import os
import pandas as pd

def imagefolder_to_csv(root_dir, output_csv):
    """Create a CSV from an ImageFolder structure."""
    records = []
    for class_name in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            records.append({
                "filepath": os.path.join(class_name, img_name),
                "class": class_name
            })
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    return df
```

---

## Dataset Selection Guide

| Exercise Type | Recommended Dataset | Reason |
|---------------|-------------------|--------|
| Basic classification (MNIST replacement) | iNaturalist mini (10 species) | Similar complexity, species context |
| Transfer learning (Pets replacement) | Caltech Camera Traps subset | Direct animal-to-animal swap |
| Multi-label classification | iWildCam 2022 | Multi-species frames common |
| Aerial scene classification | AID | Introduces aerial perspective |
| Object detection (YOLOv8) | Caltech Camera Traps (bounding boxes) | Ready-to-use detection annotations |
| Tile inference | DOTA or synthetic aerial tiles | Large images requiring tiling |
| MegaDetector pipeline | MegaDetector test images | Quick two-stage demo |
| Embedding similarity | Snapshot Serengeti (10-species subset) | Diverse species for clustering |
| CLIP zero-shot | Any of the above | CLIP works with any image set |

---

## Data Ethics and Licensing Notes

1. **Always cite the dataset source** in notebooks and exercises.
2. **Respect license terms** -- some datasets are non-commercial use only.
3. **Camera trap location data** may be sensitive for endangered species. Never publish precise GPS coordinates.
4. **Synthetic data** must be clearly labeled as synthetic.
5. **Kaggle datasets** require a Kaggle account and acceptance of competition terms. Provide alternative access instructions for students without Kaggle accounts.
