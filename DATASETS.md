# Datasets

## Datasets used in this course

All datasets below are downloaded automatically by `week1/data/download_data.py`
(or by the first cell in Practical 1). Run once:

```bash
python week1/data/download_data.py --sample   # ~500 MB, 50 images per camera trap dataset
python week1/data/download_data.py --full     # ~5 GB, 500 images per camera trap dataset
```

### Practical-to-dataset mapping

| Dataset | Used in | Format | Size (sample) |
|---------|---------|--------|---------------|
| **HerdNet General Dataset** (`karisu/General_Dataset`) | P1 (aerial exploration), P3 (SAHI demos) | Point CSV + JPEG tiles | ~200 MB |
| **HerdNet pretrained weights** (`karisu/HerdNet`) | HerdNet notebook | `.pth` checkpoint | ~300 MB |
| **Snapshot Serengeti subset** (LILA) | P1 (camera trap overview), P2 (bbox annotation), P3 (batch detection) | COCO JSON + JPEG | ~50 images |
| **Caltech Camera Traps subset** (LILA) | P1 (bbox demo), P2 (polygon annotation), P3–P6 (detection + classification) | CSV + JPEG (with bboxes) | ~50 images |
| **Eikelboom 2019** (`karisu/Eikelboom2019`) | P2 (Label Studio import demo) | Pascal VOC CSV + JPEG | ~3 species, train/val/test |

### Where data lands after download

```
week1/data/
  general_dataset/
    test_sample/          ← aerial JPEG tiles (P1, P3 SAHI)
    test_sample.csv       ← point annotations: images, x, y, labels
  camera_trap/
    serengeti_subset/     ← Serengeti JPEG images
    serengeti_meta.json   ← COCO JSON metadata for the subset
    caltech_subset/       ← Caltech JPEG images
  camera_trap_labels.csv  ← Caltech labels + bboxes (crop, true_label, bbox_x/y/w/h)
  eikelboom/
    train/ val/ test/     ← aerial drone images, 3 species
    annotations_train.csv ← Pascal VOC format bounding boxes
```

---

## Dataset details

### HerdNet General Dataset

Pre-tiled aerial images of African mammals from drone surveys, with **point annotations**
(one `(x, y)` pixel coordinate per animal). Hosted on HuggingFace as `karisu/General_Dataset`.

- **Annotation format:** CSV with columns `images, x, y, labels`
- **Tile size:** 512 x 512 px JPEG
- **Species:** buffalo, elephant, kob, topi, warthog, waterbuck (varies by subset)
- **Test sample:** ~100 tiles used in P1 for data exploration and in P3 for SAHI demos
- **Full dataset:** thousands of tiles across multiple African survey sites

This is the dataset format used throughout the HerdNet training pipeline.

### Snapshot Serengeti

LILA-hosted camera trap dataset from Serengeti National Park, Tanzania.
The full corpus spans 7.1 million images across 61 species. We use a small
subset (50–500 images) filtered for non-empty frames.

- **Full dataset:** ~78,000 images with manual bounding boxes
- **Our subset:** 50 images (sample) or 500 images (full) with species labels
- **Annotation format:** COCO Camera Traps JSON (`serengeti_meta.json`)
- **License:** Community Data License Agreement (CDLA)
- **Used in P1** to show camera trap imagery and COCO JSON structure
- **Used in P2** as the bbox annotation task (students draw boxes on species-labelled images)
- **Used in P3** for MegaDetector batch inference

### Caltech Camera Traps

LILA-hosted camera trap dataset from California with wildlife bounding boxes.
A separate bbox annotation file provides `[x, y, w, h]` COCO-format boxes.

- **Our subset:** 50 images sampled across species (round-robin, seed 42)
- **Annotation format:** CSV with columns `crop, true_label, bbox_x, bbox_y, bbox_w, bbox_h`
- **Bounding boxes:** available for most images (from `caltech_bboxes_20200316.json`)
- **License:** CDLA
- **Used in P1** to demonstrate bounding box formats (COCO, YOLO, Pascal VOC)
- **Used in P2** as the polygon annotation task (students trace outlines over existing boxes)
- **Used in P3–P6** for MegaDetector inference, crop extraction, classification, evaluation

### Eikelboom 2019

Aerial drone wildlife detection dataset with 3 species, hosted on HuggingFace
as `karisu/Eikelboom2019`. Contains train/val/test splits with Pascal VOC
format bounding box annotations.

- **Used in P2** as a Label Studio import demo (pre-annotation from CSV)

---

## Additional African wildlife datasets (reference)

The datasets below are not used directly in the practicals but are useful
for students who want to explore larger datasets or create custom subsets.

### Ready-made small datasets

**Ultralytics African Wildlife** is the single best turnkey option for a YOLO tutorial.
It contains **1,504 images** (1,052 train / 225 val / 227 test) across four classes
(buffalo, elephant, rhino, zebra) with per-image YOLO-format bounding box annotations.
Download is a single ~100 MB zip, or auto-fetched via `yolo train data=african-wildlife.yaml`.
License is permissive.

**Roboflow GRI Camera Trap Data v1** offers broader species coverage: **860 images
across 15 African wildlife classes** including elephant, lion, leopard, hyena, impala,
warthog, and wild dog. Exportable from Roboflow Universe in COCO JSON, YOLO, Pascal VOC,
or TFRecord format. Licensed under CC BY 4.0.

**Roboflow African-Wildlife** provides **1,463 images** with bounding boxes across
4 classes (buffalo, elephant, rhino, zebra) — same base as Ultralytics but available
through Roboflow's multi-format export. CC BY 4.0.

### Large-scale LILA datasets with bounding boxes

**Snapshot Serengeti (full bbox set):** ~78,000 images with manual bounding boxes
across 61 Tanzanian species. The bbox JSON is a compact download
(`SnapshotSerengetiBboxes_20190903.json.zip`). Images are individually downloadable
from Azure/GCP/AWS.

**WCS Camera Traps:** 12 countries, ~375,000 bounding boxes on ~300,000 images,
~675 species. African species include elephant, lion, gerenuk, oryx.

**Nkhotakota Camera Traps:** Pure-African dataset from Malawi — 33,813 images
with manual bounding boxes, 46 taxa (smaller antelope focus).

**Leopard ID 2022 / Hyena ID 2022:** Bounding boxes + individual IDs for
re-identification research.

### Datasets with species labels only (no bounding boxes)

These suit classification tasks or can gain detection boxes via MegaDetector.
LILA publishes pre-computed MegaDetector results for all hosted datasets.

| Dataset | Location | Images | Species |
|---------|----------|--------|---------|
| Snapshot Safari 2024 | 15 African sites | 4,029,374 | 151 |
| Snapshot Enonkishu | Kenya | 28,544 | 39 |
| Snapshot Kruger | South Africa | 10,072 | 46 |
| Snapshot Karoo | South Africa | 38,074 | 38 |
| Biome Health Project | Kenya | 37,075 | 100 |
| Desert Lion Conservation | Namibia | 65,959 | 46 |

---

## Tools for custom subsetting

For students who want to create balanced subsets from any LILA dataset:

- **pycocotools** — filter by category, extract matching images and annotations
- **FiftyOne** (Voxel51) — visual curation, import COCO, filter, sample, export to COCO/YOLO
- **MegaDetector repo utilities** — `coco_to_yolo.py`, `generate_crops_from_cct.py`, species-filtered LILA downloads
- **kwcoco** — CLI-based subset extraction
- **Roboflow** — no-code format conversion with augmentation and splitting

---

## Comparison table

| Dataset | African? | Images | Species | Bboxes? | Format | License |
|---------|----------|--------|---------|---------|--------|---------|
| **HerdNet General** | Yes | ~1,000+ tiles | 6+ | Points only | CSV + JPEG | HF |
| **Ultralytics African Wildlife** | Yes | 1,504 | 4 | Yes | YOLO | Permissive |
| **Roboflow GRI Camera Trap v1** | Yes | 860 | 15 | Yes | COCO/YOLO/VOC | CC BY 4.0 |
| **Snapshot Serengeti (bbox)** | Yes | 78,000 | 61 | Yes (manual) | COCO JSON | CDLA |
| **WCS Camera Traps (bbox)** | Partial | 300,000 | ~675 | Yes (manual) | COCO JSON | CDLA |
| **Nkhotakota Camera Traps** | Yes | 33,813 | 46 | Yes (manual) | COCO JSON | CDLA |
| **Eikelboom 2019** | Yes | ~1,000 | 3 | Yes | Pascal VOC CSV | HF |
| **Kaggle African Wildlife** | Yes | 1,504 | 4 | No (labels only) | ImageFolder | — |
