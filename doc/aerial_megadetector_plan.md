# Aerial MegaDetector: Implementation Plan

> An aerial/drone equivalent of MegaDetector that detects **animal**, **person**, and **vehicle** from overhead imagery, following the POLO architecture (May et al., arXiv:2410.11741).


---

## 1. POLO Paper Reference

POLO (Point-based, multi-class animal detection) modifies YOLOv8 for point-label training:

| Parameter | Value |
|-----------|-------|
| Base architecture | YOLOv8 |
| Output | Point detections (Gaussian heatmaps), not bounding boxes |
| Tile size | 640x640 px, 10% overlap |
| Batch size | 32 |
| Epochs | 300 (early stopping, patience 50) |
| Loss | MSE on Gaussian heatmaps (radius per class) |
| Post-processing | Degree of Response (DoR) thresholding + NMS |
| Empty tiles | 95% discarded, 5% kept as negatives |
| Dataset | Izembek Lagoon waterfowl (9,267 images, 521K annotations, 5 species) |
| Paper | May, Dalsasso, Kellenberger, Tuia. ECCV 2024 CV4Ecology Workshop |

---

## 2. Dataset Catalog

### 2A. Animal Datasets (19 datasets)

| # | Dataset | Images | Annotations | Classes | Ann. Type | Size | License | Priority |
|---|---------|--------|-------------|---------|-----------|------|---------|----------|
| A1 | **Izembek Lagoon Waterfowl** | 9,267 | 521K bbox / 631K pts | 5 (geese, gull, other) | Point/Bbox | 124 GB | Public domain | HIGH |
| A2 | **Eikelboom Savanna** | 561 | 4,305 | 3 (elephant, zebra, giraffe) | Bbox | 5.7 GB | CC0 | HIGH |
| A3 | **WAID** | 14,366 | ~tens of K | 6 (sheep, cattle, seal, camel, kiang, zebra) | Bbox (YOLO) | 1.5 GB | Unspecified | HIGH |
| A4 | **Delplanque African Mammals** | 1,297 | 10,239 | 6 (buffalo, kob, warthog, waterbuck, elephant, alcelaphinae) | Bbox (COCO) | 12 GB | CC-BY-NC-SA 4.0 | HIGH |
| A5 | **Weinstein Global Birds** | 23,765 | 386,638 | 1 (bird) | Bbox | 29 GB | CC BY 4.0 | HIGH |
| A6 | **Aerial Elephant Dataset (AED)** | 2,074 | 15,581 | 1 (elephant) | Point | 16.3 GB | CC0 | HIGH |
| A7 | **SAVMAP/Kuzikus** | 659 | ~7,500 | 1 (animal) | Polygon | 3 GB | AFL-3.0 | MEDIUM |
| A8 | **NOAA Arctic Seals 2019** | 44,185 pairs | 14,311 | 6 (seal species) | Bbox | ~1 TB | CDLA-perm. | MEDIUM |
| A9 | **BIRDSAI (Thermal)** | 62K + 100K synth | 166,221 | animal vs human | Bbox+Track | 3.7 GB | CDLA-perm. | MEDIUM |
| A10 | **Qian Penguins** | 753 | 137,365 | 1 (penguin) | Point | 300 MB | CC0 | MEDIUM |
| A11 | **Hayes Seabird Colonies** | 3,947 | 44,966 | 2 (albatross, penguin) | Bbox | 20.5 GB | CC0 | MEDIUM |
| A12 | **Aerial Seabirds West Africa** | orthomosaic | 21,516 | 6 (pelican, tern, gull, cormorant) | Point | 2.2 GB | CDLA-perm. | MEDIUM |
| A13 | **Big Bird** | 4,824 w/box | 49,490 | 100 bird species | Bbox | 45 GB | Reuse w/ack. | MEDIUM |
| A14 | **MMLA-Mpala** | 130,102 | ~617K | 2 (zebra, giraffe) | Bbox (YOLO) | 490 GB | CC0 | LOW |
| A15 | **BuckTales** | 320 (det.) | 18,400 | 6 (blackbuck, bird, etc.) | Bbox | 80 GB | CC BY-SA 4.0 | LOW |
| A16 | **AWIR** | 1,325 | 6,587 | 13 species | Bbox+Polygon | ~5 GB | Open access | LOW |
| A17 | **NOAA Arctic Seals 2016** | ~1M pairs | ~7,000 | seal species | Bbox | huge | CDLA-perm. | LOW |
| A18 | **SA Savanna (2026)** | TBD | TBD | multi-species | Bbox | ~450 GB | Public domain | LOW |
| A19 | **MMLA-OPC** | 29,268 | ~163K | 1 (zebra) | Bbox (YOLO) | 64 GB | CC0 | LOW |

### 2B. Person / Crowd Datasets (16 datasets)

| # | Dataset | Images | Annotations | Classes | Ann. Type | Size | License | Priority |
|---|---------|--------|-------------|---------|-----------|------|---------|----------|
| P1 | **VisDrone-DET** | 10,209 | 2.6M bbox | pedestrian, person + 8 others | Bbox | ~10 GB | Research only | HIGH |
| P2 | **DroneCrowd** | 33,600 frames | 4.8M head pts | person (head) | Point/Density | ~8 GB | Research only | HIGH |
| P3 | **TinyPerson** | 1,610 | 72,651 | person (sea/earth) | Bbox | ~2 GB | Not specified | HIGH |
| P4 | **DLR-ACD** | 33 large images | 226,291 pts | person | Point | ~5 GB | Research (DLR) | HIGH |
| P5 | **HERIDAL** | 500 + 68,750 patches | person bbox | person | Bbox | ~1 GB | CC BY 3.0 | HIGH |
| P6 | **SeaDronesSee** | 14,127+ | person bbox | person, boat, buoy, etc. | Bbox | ~10 GB | Not specified | MEDIUM |
| P7 | **SARD** | 1,981 | person bbox | person | Bbox (VOC) | ~2 GB | Public | MEDIUM |
| P8 | **NOMAD** | 42,825 frames | person bbox + visibility | person | Bbox + occlusion | ~15 GB | Not specified | MEDIUM |
| P9 | **HIT-UAV** | 2,898 IR images | 24,899 | person, car, bicycle | Bbox | ~2 GB | CC BY 4.0 | MEDIUM |
| P10 | **MOBDrone** | 126,170 frames | 180K+ bbox | person, boat, buoy, etc. | Bbox (COCO) | 243 GB | CC BY 4.0 | MEDIUM |
| P11 | **UAV-Human** | 67,428 seq. | multi-task | person (action, pose) | Bbox+skeleton | large | Research only | LOW |
| P12 | **Okutama-Action** | 77,365 frames | multi-action | person (12 actions) | Bbox+action | large | Academic | LOW |
| P13 | **Stanford Drone Dataset** | ~20K frames | tracking | pedestrian + 5 others | Bbox tracks | ~5 GB | CC BY-NC-SA 3.0 | LOW |
| P14 | **AU-AIR** | 32,823 frames | object bbox + flight data | person + vehicles | Bbox | ~2 GB | Not specified | LOW |
| P15 | **Unicamp-UAV** | 6,500 | 58,555 | person | Bbox | ~5 GB | Public | MEDIUM |
| P16 | **VTSaR** | RGB+thermal pairs | 20K real + 55K synth | person | Bbox | ~10 GB | Not specified | LOW |

### 2C. Vehicle Datasets (19 datasets)

| # | Dataset | Images | Annotations | Classes | Ann. Type | Size | License | Priority |
|---|---------|--------|-------------|---------|-----------|------|---------|----------|
| V1 | **DOTA v2.0** | 11,268 | 1.79M | 18 (vehicles, ships, planes, etc.) | OBB | ~20 GB | Academic only | HIGH |
| V2 | **VisDrone-DET** | 10,209 | ~540K | car, van, bus, truck + 6 others | Bbox | (shared P1) | Research only | HIGH |
| V3 | **xView** | 1,127 | 601,726 | 60 (many vehicle sub-types) | Bbox | ~20 GB | CC BY-NC-SA 4.0 | HIGH |
| V4 | **UAVDT** | 80,000 frames | 835,879 | car, truck, bus | Bbox + attrs | ~10 GB | CC BY 4.0 | HIGH |
| V5 | **SODA-A** | 2,513 | 872,069 | 9 (small/large-vehicle, ship, plane, etc.) | OBB | ~15 GB | Not specified | HIGH |
| V6 | **VEDAI** | 1,210 | 3,640 | 9 (car, truck, van, boat, plane, etc.) | OBB | ~2 GB | Academic | MEDIUM |
| V7 | **COWC** | 6 scenes | 32,716 | 1 (car) | Point (center) | ~5 GB | AGPL-3.0 (code) | MEDIUM |
| V8 | **DroneVehicle** | 56,878 (RGB+IR) | 953,087 | car, truck, bus, van, freight | OBB | ~50 GB | Academic | MEDIUM |
| V9 | **DIOR** | 23,463 | 192,472 | 20 (vehicle, ship, airplane, etc.) | Bbox (VOC) | ~10 GB | Academic | MEDIUM |
| V10 | **iSAID** | 2,806 | 655,451 | 15 (small/large-vehicle, ship, etc.) | Instance seg | ~15 GB | Academic | MEDIUM |
| V11 | **CARPK** | 1,448 | 89,777 | 1 (car) | Bbox | ~2 GB | EULA | LOW |
| V12 | **NWPU VHR-10** | 800 | 3,775 | 10 (vehicle, ship, airplane, etc.) | Bbox+mask | ~1 GB | Public | LOW |
| V13 | **DLR 3K Munich** | 20 | 3,472 | car, truck | OBB | ~1 GB | Public | LOW |
| V14 | **EAGLE (DLR)** | 8,820 tiles | 215,986 | small-veh, large-veh | OBB | ~10 GB | On request | LOW |
| V15 | **VME** | 4,000+ tiles | 100K+ | vehicle | OBB+Bbox | ~5 GB | CC BY 4.0 | MEDIUM |
| V16 | **FAIR1M** | 15,000+ | 1M+ | 37 (vehicle, ship, airplane sub-types) | OBB | large | Restricted | LOW |
| V17 | **HRSC2016** | 1,070 | 2,976 | ship (3-level hierarchy) | OBB+Bbox | ~2 GB | Available | LOW |
| V18 | **ITCVD** | 173 | 29,088 | vehicle | Bbox+point | ~1 GB | Public | LOW |
| V19 | **Stanford Drone** | ~20K targets | tracking | ped, bike, car, bus, etc. | Bbox tracks | (shared P13) | Academic | LOW |

---

## 3. Download Plan

### 3.1 Directory Structure

```
data/
  raw/
    animal/
      izembek/            # A1
      eikelboom/           # A2
      waid/                # A3
      delplanque/          # A4
      weinstein_birds/     # A5
      aed_elephants/       # A6
      savmap_kuzikus/      # A7
      noaa_seals_2019/     # A8
      birdsai/             # A9
      qian_penguins/       # A10
      hayes_seabirds/      # A11
      seabirds_westafrica/ # A12
      big_bird/            # A13
    person/
      visdrone/            # P1 (also V2)
      dronecrowd/          # P2
      tinyperson/          # P3
      dlr_acd/             # P4
      heridal/             # P5
      seadronessee/        # P6
      sard/                # P7
      nomad/               # P8
      hit_uav/             # P9
      mobdrone/             # P10
      unicamp_uav/         # P15
    vehicle/
      dota_v2/             # V1
      xview/               # V3
      uavdt/               # V4
      soda_a/              # V5
      vedai/               # V6
      cowc/                # V7
      dronevehicle/        # V8
      dior/                # V9
      isaid/               # V10
      vme/                 # V15
  processed/               # After harmonization to unified COCO format
  splits/                  # Train/val/test splits
```

### 3.2 Priority Tiers

#### Tier 1 -- Core (~200 GB, download first)

| ID | Dataset | How to Download | Size |
|----|---------|----------------|------|
| A1 | Izembek | `azcopy` from LILA: `https://lila.science/datasets/izembek-lagoon-waterfowl/` | 124 GB |
| A2 | Eikelboom | `wget` from 4TU: `https://data.4tu.nl/articles/dataset/.../12713903/1` | 5.7 GB |
| A3 | WAID | `git clone https://github.com/xiaohuicui/WAID.git` | 1.5 GB |
| A5 | Weinstein Birds | `wget https://zenodo.org/record/5033174` (multiple zip files) | 29 GB |
| A6 | AED | `wget https://zenodo.org/record/3234780` | 16.3 GB |
| P1/V2 | VisDrone | **Manual:** register at aiskyeye.com, download from Google Drive links | ~10 GB |
| P3 | TinyPerson | `git clone https://github.com/ucas-vg/TinyBenchmark` (follow download links) | ~2 GB |
| P5 | HERIDAL | `kaggle datasets download -d imadeddinelassakeur/heridal` | ~1 GB |
| V1 | DOTA v2.0 | **Manual:** register at `captain-whu.github.io/DOTA/`, download splits | ~20 GB |
| V4 | UAVDT | `wget https://zenodo.org/records/14575517` | ~10 GB |

#### Tier 2 -- Extended (~150 GB)

| ID | Dataset | How to Download | Size |
|----|---------|----------------|------|
| A4 | Delplanque | `wget` from ULiege Dataverse: `https://dataverse.uliege.be/file.xhtml?fileId=11098` | 12 GB |
| A10 | Qian Penguins | `wget https://zenodo.org/record/7702635` | 300 MB |
| A11 | Hayes Seabirds | Globus transfer from Duke: `https://research.repository.duke.edu/concern/datasets/kp78gh20s` | 20.5 GB |
| A12 | Seabirds WA | HTTP from LILA: `https://lila.science/datasets/aerial-seabirds-west-africa/` | 2.2 GB |
| A9 | BIRDSAI | HTTP from LILA: `https://lila.science/datasets/conservationdrones` | 3.7 GB |
| P2 | DroneCrowd | `git clone https://github.com/VisDrone/DroneCrowd` (Google Drive links) | ~8 GB |
| P4 | DLR-ACD | **Manual:** register at DLR public datasets page | ~5 GB |
| P6 | SeaDronesSee | **Manual:** register at `https://seadronessee.cs.uni-tuebingen.de/` | ~10 GB |
| P9 | HIT-UAV | `git clone https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset` or Kaggle | ~2 GB |
| P15 | Unicamp-UAV | Download from ISPRS journal supplementary | ~5 GB |
| V3 | xView | **Manual:** register at `xviewdataset.org`, download train images + labels | ~20 GB |
| V5 | SODA-A | Download from `https://shaunyuan22.github.io/SODA/` or Kaggle | ~15 GB |
| V15 | VME | `wget https://zenodo.org/records/14185684` | ~5 GB |

#### Tier 3 -- Large / Specialized (~1 TB+, selective sampling)

| ID | Dataset | How to Download | Size |
|----|---------|----------------|------|
| A8 | NOAA Seals 2019 | `azcopy` from LILA: `https://lila.science/datasets/noaa-arctic-seals-2019/` | ~1 TB |
| A14 | MMLA-Mpala | `huggingface-cli download imageomics/mmla_mpala` | 490 GB |
| A13 | Big Bird | HTTP from LILA: `https://lila.science/datasets/big-bird` | 45 GB |
| V8 | DroneVehicle | Follow links from `https://github.com/VisDrone/DroneVehicle` | ~50 GB |
| P10 | MOBDrone | `wget https://zenodo.org/records/5996890` | 243 GB |
| A15 | BuckTales | Download from `https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.JCZ9WK` | 80 GB |

### 3.3 Manual Download Instructions

| Dataset | Steps |
|---------|-------|
| **VisDrone (P1/V2)** | 1. Go to `http://www.aiskyeye.com` 2. Register with institutional email 3. Download VisDrone-DET 2019 train/val/test from Google Drive links in [GitHub README](https://github.com/VisDrone/VisDrone-Dataset) |
| **DOTA v2.0 (V1)** | 1. Go to `https://captain-whu.github.io/DOTA/dataset.html` 2. Register 3. Download train/val/test images and labels (OBB format) |
| **xView (V3)** | 1. Go to `https://xviewdataset.org/` 2. Register 3. Download train images (tif) + labels (geojson) |
| **DLR-ACD (P4)** | 1. Go to [DLR public datasets](https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/dlr-acd) 2. Fill request form 3. Download 33 images + point annotations |
| **SeaDronesSee (P6)** | 1. Go to `https://seadronessee.cs.uni-tuebingen.de/` 2. Register 3. Download ODv2 split |
| **VEDAI (V6)** | Direct download from `https://downloads.greyc.fr/vedai/` |
| **DroneVehicle (V8)** | Follow Google Drive / Baidu links from `https://github.com/VisDrone/DroneVehicle` |
| **DIOR (V9)** | Download from [IEEE DataPort](https://ieee-dataport.org/documents/dior) or [Hugging Face mirror](https://huggingface.co/datasets) |
| **iSAID (V10)** | Download from Google Drive via `https://captain-whu.github.io/iSAID/dataset.html` |
| **SARD (P7)** | Download from [IEEE DataPort](https://ieee-dataport.org/documents/search-and-rescue-image-dataset-person-detection-sard) |
| **COWC (V7)** | FTP download from `gdo152.llnl.gov/cowc/` or GitHub: `https://github.com/LLNL/cowc` |
| **Stanford Drone (P13/V19)** | Download from `https://cvgl.stanford.edu/projects/uav_data/` |
| **EAGLE (V14)** | Request from DLR: `https://www.dlr.de/en/eoc/.../public-datasets/eagle` |

---

## 4. Data Harmonization

### 4.1 Unified Format

All datasets converted to **COCO-style JSON** with three superclasses:

```json
{
  "categories": [
    {"id": 1, "name": "animal", "supercategory": "object"},
    {"id": 2, "name": "person", "supercategory": "object"},
    {"id": 3, "name": "vehicle", "supercategory": "object"}
  ]
}
```

Fine-grained labels preserved as `attributes.original_class` in each annotation.

### 4.2 Annotation Conversion

| Source Format | Datasets | Conversion |
|---------------|----------|------------|
| COCO JSON | A4, P10 | Remap category IDs |
| YOLO TXT | A3, A14, A19 | yolo2coco |
| Pascal VOC XML | V9, P7 | voc2coco |
| Point (CSV/JSON) | A1, A6, A10, A12, P2, P4, V7 | Point -> pseudo-box (square, radius r) |
| OBB (8-point) | V1, V5, V6, V8 | OBB -> axis-aligned HBB |
| Instance seg | V10 | Mask -> bbox |
| MOT tracks | A9, P13 | Sample frames, extract per-frame bbox |
| Custom CSV | A2, A5, A11, etc. | Per-dataset parser |

### 4.3 Class Mapping

```yaml
animal:
  - all wildlife species (geese, elephants, zebra, giraffe, buffalo, seals, birds, penguins, etc.)
  - BIRDSAI "animal" class

person:
  - VisDrone "pedestrian" and "person"
  - all person classes from HERIDAL, SARD, NOMAD, SeaDronesSee, DLR-ACD, etc.

vehicle:
  - VisDrone: car, van, bus, truck, motor, bicycle, awning-tricycle, tricycle
  - DOTA: small-vehicle, large-vehicle, ship, plane, helicopter
  - xView: all vehicle sub-types (car, truck, bus, etc.)
  - UAVDT: car, truck, bus
  - all vehicle classes from other datasets

excluded:
  - infrastructure (storage-tank, baseball-diamond, bridge, harbor, etc.)
  - non-object items (lifebuoy, surfboard, wood from maritime datasets)
```

### 4.4 Tiling (POLO-style)

- **Tile size:** 640x640 pixels
- **Overlap:** 10% (64 pixels)
- **Empty tiles:** keep 5% as negatives, discard 95%
- **Edge handling:** annotations within overlap/2 of edge included in both tiles
- **Large images:** tiled; smaller images padded to 640x640

---

## 5. Training Plan

### 5.1 Two Parallel Tracks

| | POLO (point-based) | YOLOv8 (bbox-based) |
|---|---|---|
| Architecture | Modified YOLOv8 (Gaussian heatmap head) | Standard YOLOv8n/s/m |
| Output | Point detections | Bounding boxes |
| Loss | MSE on heatmaps | CIoU + cls + dfl |
| Can use point-labeled data | Yes (native) | Yes (via pseudo-boxes) |
| Post-processing | DoR threshold + NMS | Confidence + NMS |

### 5.2 Training Config

```yaml
model: yolov8m
imgsz: 640
batch: 32
epochs: 300
patience: 50
optimizer: SGD
lr0: 0.01
lrf: 0.01
mosaic: 1.0
mixup: 0.0
classes: 3  # animal, person, vehicle
```

### 5.3 Splits

- Use existing train/val/test splits where available
- For datasets without splits: 80% / 5% / 15% (following POLO)
- Hold out 1 full dataset per category for out-of-distribution testing

### 5.4 Evaluation

- **MAE** (Mean Absolute Error) per image -- counting accuracy
- **F1** at distance threshold -- localization accuracy
- **mAP@0.5** -- standard detection metric (bbox track only)
- Per-class and per-dataset breakdown

---

## 6. Implementation Steps

| Step | Script | Description |
|------|--------|-------------|
| 1 | `scripts/download_tier1.sh` | Automated downloads for Tier 1 datasets |
| 2 | `doc/download_manual_instructions.md` | Manual download guide (this section 3.3) |
| 3 | `scripts/convert_to_coco.py` | Per-dataset converters to unified COCO format |
| 4 | `scripts/generate_pseudoboxes.py` | Point -> pseudo-box for point-labeled datasets |
| 5 | `scripts/tile_images.py` | POLO-style 640x640 tiling with overlap |
| 6 | `scripts/merge_datasets.py` | Merge all converted COCO JSONs into unified dataset |
| 7 | `scripts/create_splits.py` | Create train/val/test splits |
| 8 | `scripts/dataset_stats.py` | Dataset statistics and visualizations |
| 9 | `scripts/train_polo.py` | POLO-style point-based training |
| 10 | `scripts/train_yolov8.py` | Standard YOLOv8 bbox-based baseline |
| 11 | `configs/aerial_megadetector.yaml` | Training configuration |
| 12 | `scripts/evaluate.py` | Per-dataset and overall evaluation metrics |

---

## 7. Supporting Reference Documents

Detailed per-dataset descriptions (URLs, citations, format details, BibTeX) are in:

- `doc/aerial_wildlife_datasets_reference.md` -- 26 animal datasets
- `doc/aerial_person_crowd_detection_datasets.md` -- 23 person/crowd datasets
- `doc/aerial_vehicle_detection_datasets.md` -- 22 vehicle datasets

---

## 8. Key Resources

- **POLO paper:** https://arxiv.org/abs/2410.11741
- **POLO code:** (check EPFL/ECEO GitHub when available)
- **MegaDetector:** https://github.com/microsoft/CameraTraps
- **LILA Science:** https://lila.science/datasets/
- **Dan Morris drone-wildlife-datasets:** https://github.com/agentmorris/drone-wildlife-datasets
- **Ultralytics YOLOv8:** https://github.com/ultralytics/ultralytics
