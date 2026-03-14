# Dataset Proposal — Week 1 Practicals

This document maps datasets to practicals and explains the rationale for each choice.

---

## Overview

| Dataset | Modality | Size | Annotation | License | Used in |
|---------|----------|------|------------|---------|---------|
| **karisu/General_Dataset** | Aerial drone | 8.4 GB | Points (multi-species) | ULiège Open Data | P1, HerdNet notebook |
| **Iguanas From Above** | Aerial drone | 57,838 tiles | Points + counts (CC-BY) | CC-BY 4.0 | P1, P2, case study |
| **Snapshot Serengeti** (subset) | Camera trap | ~5,000 imgs | Species labels + boxes | CDLA Permissive | P3, P4 |
| **Caltech Camera Traps** (subset) | Camera trap | ~2,000 imgs | Bounding boxes | CDLA Permissive | P5, P6 |

---

## 1 — karisu/General_Dataset (Confirmed)

**Source:** https://huggingface.co/datasets/karisu/General_Dataset
**Citation:** Delplanque et al. (2023), ISPRS Journal of Photogrammetry and Remote Sensing

This is the official training dataset for HerdNet. It contains aerial drone imagery of
African mammals with point annotations and is the backbone of the HerdNet pipeline
notebook (`notebooks/01_herdnet_pipeline.py`).

**Contents:**
- Multi-species African aerial surveys (cattle, buffalos, elephants, etc.)
- Point annotations in CSV format: `images, x, y, labels`
- Pre-split into `test_sample/` (quick experiments) and full dataset
- HerdNet pretrained weights stored in the same `karisu/HerdNet` repo

**Download:**
```python
from huggingface_hub import snapshot_download

# Small test sample — use this for P1 and the HerdNet notebook
snapshot_download(
    repo_id="karisu/General_Dataset",
    repo_type="dataset",
    local_dir="week1/data/general_dataset",
    allow_patterns=["test_sample/*", "test_sample.csv"],
)

# Full dataset (~8 GB) — only needed for HerdNet training
snapshot_download(
    repo_id="karisu/General_Dataset",
    repo_type="dataset",
    local_dir="week1/data/general_dataset",
)
```

**Use in course:**
- **P1** — use `test_sample/` tiles for the tiling visualisation
- **HerdNet notebook** — full dataset for training demonstration
- **P2** — visualise point annotation format

---

## 2 — Iguanas From Above

**Source:** https://figshare.com/articles/dataset/Iguanas_from_Above_Citizen_Scientists_and_Drone_Imagery/25196306
**DOI:** 10.6084/m9.figshare.25196306
**Citation:** Varela-Jaramillo et al. (2025), *Scientific Reports* 15, 11282

The case study running through the entire course. 57,838 drone tiles (1000×1000 px)
from the Galápagos Islands annotated by 13,000+ citizen scientists and a consensus
of expert biologists. CC-BY 4.0.

**What FigShare contains (open access):**
- Aggregated citizen science counts per image (CSV)
- Presence/absence labels per image (CSV)
- Expert quality judgements (good/bad image)
- Individual volunteer classification records

**What is available on request from the authors:**
- Raw 1000×1000 px tile JPEGs (~50 GB total)
- GeoPackage annotation files in local tile coordinates

**Note:** The raw tiles need to be requested from C. Winkelmann (course instructor).
A teaching subset (~500 tiles, 3 islands) will be placed in `week1/data/iguanas/`.

**Use in course:**
- **P1** — load and tile a sample orthomosaic; use iguana tiles as visual examples
- **P2** — visualise point annotations; discuss citizen science labelling noise
- **P7** — use for SAM segmentation (rock/iguana/sand habitat classes)
- **Lectures** — primary case study throughout Week 1

---

## 3 — Snapshot Serengeti (LILA Science)

**Source:** https://lila.science/datasets/snapshot-serengeti
**License:** CDLA Permissive (free for research and education)

7.1 million camera trap images from the Serengeti, 40 mammalian species.
~76 % of frames are empty — perfect for demonstrating MegaDetector's value.
~150,000 images have bounding box annotations.

**Recommended subset for P3/P4:** 5,000 images, ~10 species, mix of empty and
animal-present, sampled from one location to avoid spatial leakage.

**Download (subset via LILA metadata + Azure Blob):**
```python
import io
import json
import os
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

OUTPUT_DIR = Path("week1/data/camera_trap/serengeti_subset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Download metadata ──────────────────────────────────────────────────────
META_URL = (
    "https://lilablobssc.blob.core.windows.net"
    "/snapshotserengeti-v-2-0/SnapshotSerengeti_S1-11_v2.1.json.zip"
)
print("Downloading Snapshot Serengeti metadata...")
r = requests.get(META_URL, timeout=60)
r.raise_for_status()

with zipfile.ZipFile(io.BytesIO(r.content)) as z:
    with z.open(z.namelist()[0]) as f:
        meta = json.load(f)

print(f"  {len(meta['images'])} images, {len(meta['annotations'])} annotations")

# ── 2. Sample a teaching subset ───────────────────────────────────────────────
# Take the first 5,000 images from a single location (Season 1, Site B04)
# to avoid spatial leakage between train and test.
TARGET_SITE = "B04"
N_IMAGES = 5000

images_at_site = [
    img for img in meta["images"]
    if img.get("location", "").startswith(TARGET_SITE)
][:N_IMAGES]

print(f"  Sampled {len(images_at_site)} images from site {TARGET_SITE}")

# Save subset metadata
subset_meta = {
    "images": images_at_site,
    "annotations": [
        ann for ann in meta["annotations"]
        if ann["image_id"] in {img["id"] for img in images_at_site}
    ],
    "categories": meta["categories"],
}
meta_path = OUTPUT_DIR.parent / "serengeti_meta.json"
with open(meta_path, "w") as f:
    json.dump(subset_meta, f)
print(f"  Metadata saved to {meta_path}")

# ── 3. Download images from Azure Blob ────────────────────────────────────────
BASE_URL = "https://lilablobssc.blob.core.windows.net/snapshotserengeti-v-2-0/"

for img in tqdm(images_at_site, desc="Downloading images"):
    dest = OUTPUT_DIR / os.path.basename(img["file_name"])
    if dest.exists():
        continue
    url = BASE_URL + img["file_name"]
    resp = requests.get(url, timeout=30)
    if resp.status_code == 200:
        dest.write_bytes(resp.content)
    else:
        print(f"  WARNING: {img['file_name']} → HTTP {resp.status_code}")

print(f"Done. Images in {OUTPUT_DIR}")
```

**Use in course:**
- **P3** — primary dataset for running MegaDetector (camera trap perspective,
  abundant animals, lots of empty frames)
- **P4** — detection exploration and failure mode analysis

---

## 4 — Caltech Camera Traps (LILA Science)

**Source:** https://lila.science/datasets/caltech-camera-traps
**License:** CDLA Permissive

243,000 camera trap images from 140 locations in the American Southwest.
21 species including opossum, coyote, and deer. ~27 % have bounding box annotations.
Images are geographically distinct from Snapshot Serengeti — useful to test whether
MegaDetector generalises across continents.

**Download (subset, ~2,000 images with ground-truth boxes):**
```python
import io
import json
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

OUTPUT_DIR = Path("week1/data/camera_trap/caltech_subset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Download metadata ──────────────────────────────────────────────────────
META_URL = (
    "https://caltech-camera-traps.s3.amazonaws.com"
    "/cct_images/caltech_images_20200316.json.zip"
)
print("Downloading Caltech Camera Traps metadata...")
r = requests.get(META_URL, timeout=120)
r.raise_for_status()

with zipfile.ZipFile(io.BytesIO(r.content)) as z:
    with z.open(z.namelist()[0]) as f:
        meta = json.load(f)

print(f"  {len(meta['images'])} images, {len(meta['annotations'])} annotations")

# ── 2. Keep only images that have bounding box annotations ───────────────────
N_IMAGES = 2000
annotated_ids = {ann["image_id"] for ann in meta["annotations"]}
annotated_imgs = [img for img in meta["images"] if img["id"] in annotated_ids][:N_IMAGES]

print(f"  Sampled {len(annotated_imgs)} annotated images")

subset_meta = {
    "images": annotated_imgs,
    "annotations": [
        ann for ann in meta["annotations"]
        if ann["image_id"] in {img["id"] for img in annotated_imgs}
    ],
    "categories": meta["categories"],
}
meta_path = OUTPUT_DIR.parent / "caltech_meta.json"
with open(meta_path, "w") as f:
    json.dump(subset_meta, f)

# ── 3. Build reference label CSV for P6 ──────────────────────────────────────
import pandas as pd

id_to_category = {cat["id"]: cat["name"] for cat in meta["categories"]}
id_to_filename = {img["id"]: img["file_name"] for img in annotated_imgs}

label_rows = []
for ann in subset_meta["annotations"]:
    img_id = ann["image_id"]
    if img_id not in id_to_filename:
        continue
    label_rows.append({
        "crop": Path(id_to_filename[img_id]).name,
        "true_label": id_to_category.get(ann["category_id"], "unknown"),
        "bbox_x": ann["bbox"][0],
        "bbox_y": ann["bbox"][1],
        "bbox_w": ann["bbox"][2],
        "bbox_h": ann["bbox"][3],
    })

labels_df = pd.DataFrame(label_rows)
labels_path = Path("week1/data/camera_trap_labels.csv")
labels_df.to_csv(labels_path, index=False)
print(f"  Reference labels saved to {labels_path} ({len(labels_df)} rows)")

# ── 4. Download images from S3 ────────────────────────────────────────────────
BASE_URL = "https://caltech-camera-traps.s3.amazonaws.com/cct_images/"

for img in tqdm(annotated_imgs, desc="Downloading images"):
    dest = OUTPUT_DIR / Path(img["file_name"]).name
    if dest.exists():
        continue
    url = BASE_URL + img["file_name"]
    resp = requests.get(url, timeout=30)
    if resp.status_code == 200:
        dest.write_bytes(resp.content)

print(f"Done. Images in {OUTPUT_DIR}")
```

**Use in course:**
- **P5** — animal crops for classifier inference (geographically distinct from P3/P4
  data → tests classifier generalisation)
- **P6** — evaluation against labelled reference set (boxes available as ground truth)

---

## Mapping to Practicals

| Practical | Dataset | What students use |
|-----------|---------|-------------------|
| P1 — Drone imagery | karisu/General_Dataset `test_sample/` | Tile JPEGs + point CSV |
| P2 — Annotation tools | Iguanas From Above (teaching subset) | GeoPackage, visualise annotation types |
| P3 — MegaDetector | Snapshot Serengeti (5k subset) | Raw images → run MegaDetector |
| P4 — Detection exploration | Snapshot Serengeti detections from P3 | Detections CSV + crops |
| P5 — Classifier | Caltech Camera Traps crops | Animal crops from MegaDetector |
| P6 — Evaluation | Caltech Camera Traps (ground truth boxes) | Labelled reference CSV |
| P7 — Segmentation | Iguanas From Above (teaching subset) | Tiles for SAM + pixel classifier |
| P8 — Wrap-up | All of the above | Summary export |

---

## Data Download Plan

### Before Day 1

The following should be in `week1/data/` before the course starts:

```
week1/data/
  general_dataset/
    test_sample/         ← ~200 MB, from karisu/General_Dataset
    test_sample.csv
  iguanas/
    tiles/               ← ~500 tiles, 1000×1000 px (request from C. Winkelmann)
    annotations.gpkg
    iguana_counts.csv    ← from FigShare
  camera_trap/
    serengeti_subset/    ← ~5,000 images, ~2 GB (from LILA / Azure Blob)
    serengeti_meta.json
  camera_trap_labels.csv ← reference labels for P6 (from Caltech subset)
```

Run `download_data.py` (in this folder) to set everything up in one go:

```bash
python week1/data/download_data.py --sample    # fast: test_sample + 500 iguana tiles + 200 Serengeti
python week1/data/download_data.py --full      # full teaching subsets (~5 GB total)
```

```python
# download_data.py — run this before Day 1
"""
Downloads all Week 1 datasets into week1/data/.

Usage:
    python download_data.py --sample   # minimal set for quick testing (~500 MB)
    python download_data.py --full     # full teaching subsets (~5 GB)
"""
import argparse
import io
import json
import os
import zipfile
from pathlib import Path

import requests
from huggingface_hub import snapshot_download, hf_hub_download
from tqdm import tqdm

BASE_DIR = Path(__file__).parent  # week1/data/


def download_general_dataset(full: bool) -> None:
    """karisu/General_Dataset — HerdNet aerial dataset from HuggingFace."""
    out = BASE_DIR / "general_dataset"
    print("\n=== karisu/General_Dataset ===")
    if full:
        snapshot_download(
            repo_id="karisu/General_Dataset",
            repo_type="dataset",
            local_dir=str(out),
        )
    else:
        snapshot_download(
            repo_id="karisu/General_Dataset",
            repo_type="dataset",
            local_dir=str(out),
            allow_patterns=["test_sample/*", "test_sample.csv"],
        )
    print(f"  Saved to {out}")


def download_herdnet_weights() -> None:
    """Download pretrained HerdNet weights from karisu/HerdNet."""
    models_dir = BASE_DIR.parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    print("\n=== HerdNet pretrained weights ===")
    hf_hub_download(
        repo_id="karisu/HerdNet",
        filename="general_2022/20220413_HerdNet_General_dataset_2022.pth",
        local_dir=str(models_dir),
    )
    hf_hub_download(
        repo_id="karisu/HerdNet",
        filename="general_2022/config.yaml",
        local_dir=str(models_dir),
    )
    print(f"  Saved to {models_dir}")


def download_serengeti(n_images: int = 500) -> None:
    """Snapshot Serengeti subset from LILA / Azure Blob Storage."""
    out_dir = BASE_DIR / "camera_trap" / "serengeti_subset"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Snapshot Serengeti (n={n_images}) ===")

    META_URL = (
        "https://lilablobssc.blob.core.windows.net"
        "/snapshotserengeti-v-2-0/SnapshotSerengeti_S1-11_v2.1.json.zip"
    )
    r = requests.get(META_URL, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            meta = json.load(f)

    # Sample from one site to avoid spatial leakage
    TARGET_SITE = "B04"
    sampled = [
        img for img in meta["images"]
        if img.get("location", "").startswith(TARGET_SITE)
    ][:n_images]

    meta_path = BASE_DIR / "camera_trap" / "serengeti_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "images": sampled,
            "annotations": [
                a for a in meta["annotations"]
                if a["image_id"] in {img["id"] for img in sampled}
            ],
            "categories": meta["categories"],
        }, f)

    BASE = "https://lilablobssc.blob.core.windows.net/snapshotserengeti-v-2-0/"
    for img in tqdm(sampled, desc="Serengeti images"):
        dest = out_dir / os.path.basename(img["file_name"])
        if dest.exists():
            continue
        resp = requests.get(BASE + img["file_name"], timeout=30)
        if resp.status_code == 200:
            dest.write_bytes(resp.content)
    print(f"  {len(sampled)} images in {out_dir}")


def download_caltech(n_images: int = 500) -> None:
    """Caltech Camera Traps subset (annotated images only) from LILA / S3."""
    import pandas as pd

    out_dir = BASE_DIR / "camera_trap" / "caltech_subset"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Caltech Camera Traps (n={n_images}) ===")

    META_URL = (
        "https://caltech-camera-traps.s3.amazonaws.com"
        "/cct_images/caltech_images_20200316.json.zip"
    )
    r = requests.get(META_URL, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            meta = json.load(f)

    annotated_ids = {ann["image_id"] for ann in meta["annotations"]}
    sampled = [img for img in meta["images"] if img["id"] in annotated_ids][:n_images]

    id_to_cat = {c["id"]: c["name"] for c in meta["categories"]}
    id_to_file = {img["id"]: img["file_name"] for img in sampled}

    rows = [
        {
            "crop": Path(id_to_file[a["image_id"]]).name,
            "true_label": id_to_cat.get(a["category_id"], "unknown"),
            "bbox_x": a["bbox"][0], "bbox_y": a["bbox"][1],
            "bbox_w": a["bbox"][2], "bbox_h": a["bbox"][3],
        }
        for a in meta["annotations"] if a["image_id"] in id_to_file
    ]
    pd.DataFrame(rows).to_csv(BASE_DIR / "camera_trap_labels.csv", index=False)

    BASE = "https://caltech-camera-traps.s3.amazonaws.com/cct_images/"
    for img in tqdm(sampled, desc="Caltech images"):
        dest = out_dir / Path(img["file_name"]).name
        if dest.exists():
            continue
        resp = requests.get(BASE + img["file_name"], timeout=30)
        if resp.status_code == 200:
            dest.write_bytes(resp.content)
    print(f"  {len(sampled)} images + camera_trap_labels.csv in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Week 1 datasets")
    parser.add_argument("--sample", action="store_true", help="Minimal set (~500 MB)")
    parser.add_argument("--full", action="store_true", help="Full teaching subsets (~5 GB)")
    args = parser.parse_args()

    if not args.sample and not args.full:
        parser.print_help()
        return

    n = 5000 if args.full else 500

    download_herdnet_weights()
    download_general_dataset(full=args.full)
    download_serengeti(n_images=n)
    download_caltech(n_images=n)

    print("\n✓ All datasets ready. Iguana tiles must be copied manually — see section 2.")


if __name__ == "__main__":
    main()
```

---

## Roboflow (Supplementary)

Roboflow Universe has several wildlife camera trap datasets that can serve as
drop-in replacements if LILA access is slow. Recommended:

- **Wildlife Camera Traps** — various contributors, YOLO format, direct API download
- **Serengeti Species** — subset with bounding boxes pre-formatted for YOLO training

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")  # free account required
project = rf.workspace("wildlife").project("camera-traps")
dataset = project.version(1).download("yolov8")
```

Roboflow is mentioned as an annotation tool demo in P2 — the API also shows how
professionally formatted datasets are structured.

---

## Notes

- **HerdNet pretrained weights** are in `karisu/HerdNet` on HuggingFace (separate from the dataset repo).
- **Raw iguana orthomosaics** are available but large (~10 GB per island). Not used in practicals — students work on pre-tiled 1000×1000 px JPEGs.
- **Week 2 (SAR)** datasets are handled separately by N. Voss & A. Bosu (Sentinel-1 scenes, not covered here).
