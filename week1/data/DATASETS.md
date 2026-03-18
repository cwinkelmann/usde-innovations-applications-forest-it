# Datasets — Week 1

All datasets are downloaded automatically by `download_data.py` (or by the
first cell in each practical). Run once before Day 1:

```bash
python week1/data/download_data.py --sample   # ~500 MB, 50 images per source
python week1/data/download_data.py --full     # ~5 GB, 500 images per source
python week1/data/download_data.py --n-images 20  # exact count per source
```

---

## Dataset Overview

| Dataset | Source | Annotation | License | Used in |
|---------|--------|------------|---------|---------|
| [HerdNet General Dataset](https://huggingface.co/datasets/karisu/General_Dataset) | HuggingFace | Point annotations (CSV) | ULiège Open Data | P1 (aerial), HerdNet notebook |
| [Iguanas From Above](https://figshare.com/articles/dataset/25196306) | FigShare | Point counts + expert labels | CC-BY 4.0 | P1, P2, case study |
| [Snapshot Serengeti](https://lila.science/datasets/snapshot-serengeti) | LILA BC | Species labels + COCO bboxes | CDLA Permissive | P1, P3, P4 |
| [Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps) | LILA BC | COCO bboxes + species CSV | CDLA Permissive | P3, P5, P6 |
| [Eikelboom 2019](https://huggingface.co/datasets/karisu/Eikelboom2019) | HuggingFace | YOLO bboxes (3 species) | CC-BY 4.0 | P1, P3 (SAHI) |
| [HerdNet pretrained weights](https://huggingface.co/karisu/HerdNet) | HuggingFace | `.pth` checkpoint | — | HerdNet notebook |

### Generale good sources for Datasets
Tree Crown Deliniateion - canpoyRS https://github.com/hugobaudchon/CanopyRS https://huggingface.co/CanopyRS 
https://lila.science/otherdatasets/


---

## Practical-to-Dataset Mapping

| Practical | Dataset | What students use |
|-----------|---------|-------------------|
| P1 — Visual datasets | Serengeti, Eikelboom, HerdNet General | Image labels, bboxes, points |
| P2 — Annotation tools | Iguanas From Above (teaching subset) | Label Studio annotation demo |
| P3 — MegaDetector | Serengeti, Caltech | MegaDetector batch inference |
| P4 — Detection exploration | Serengeti detections from P3 | Detections CSV + crops |
| P5 — Classifier | Caltech crops | timm classifier on animal crops |
| P6 — Evaluation | Caltech (ground truth boxes) | Labelled reference CSV |
| P7 — Segmentation | Iguanas From Above (teaching subset) | SAM segmentation demo |
| P8 — Wrap-up | All of the above | Summary + export |

---

## Dataset Details

### HerdNet General Dataset

Pre-tiled aerial images of African mammals with **point annotations** — one
`(x, y)` pixel coordinate per animal, no bounding boxes.

- **Format:** CSV with columns `images, x, y, labels`
- **Tile size:** 512 x 512 px JPEG
- **Species:** buffalo, elephant, kob, topi, warthog, waterbuck
- **Citation:** Delplanque et al. (2023), ISPRS J. Photogrammetry & Remote Sensing

### Iguanas From Above

Case study dataset: 57,838 drone tiles (1000 x 1000 px) from the Galapagos
Islands annotated by 13,000+ citizen scientists and expert biologists.

- **FigShare (open access):** aggregated counts, presence/absence, expert quality labels
- **Raw tiles (on request):** contact C. Winkelmann for a ~500-tile teaching subset
- **Citation:** Varela-Jaramillo et al. (2025), Scientific Reports 15, 11282

### Snapshot Serengeti

7.1 million camera trap images from Serengeti, 40+ mammalian species. ~76%
of frames are empty — ideal for demonstrating MegaDetector's value. ~150,000
images have bounding box annotations (separate LILA download).

- **Our subset:** 50-500 images filtered for non-empty frames
- **Format:** COCO JSON (`serengeti_meta.json`) with species labels + bboxes
- **Bbox source:** `snapshot-serengeti-bounding-boxes-20190903.json.zip`

### Caltech Camera Traps

243,000 camera trap images from 140 California locations, 21 species. ~27%
have bounding box annotations. Geographically distinct from Serengeti —
tests whether MegaDetector generalises across continents.

- **Our subset:** 50-500 images sampled across species (round-robin, seed 42)
- **Format:** CSV with columns `crop, true_label, bbox_x, bbox_y, bbox_w, bbox_h`
- **Bbox source:** `caltech_bboxes_20200316.json`

### Eikelboom 2019

Aerial drone survey images from Kenya with bounding boxes for three large
mammal species (elephant, zebra, giraffe). High-resolution full-frame
captures — ideal for demonstrating tiled inference with SAHI.

- **Splits:** train (393), val (56), test (112) images
- **Format:** YOLO `.txt` label files per image
- **Original:** [4TU.ResearchData](https://data.4tu.nl/articles/_/12713903/1)
- **Citation:** Eikelboom et al. (2019), Methods in Ecology and Evolution

---

## Where Data Lands After Download

```
week1/data/
  general_dataset/
    test_sample/          <- aerial JPEG tiles
    test_sample.csv       <- point annotations: images, x, y, labels
  camera_trap/
    serengeti_subset/     <- Serengeti JPEG images
    serengeti_meta.json   <- COCO JSON (species labels + bboxes)
    caltech_subset/       <- Caltech JPEG images
  camera_trap_labels.csv  <- Caltech labels + bboxes
  eikelboom/
    train/ val/ test/     <- aerial drone images + YOLO .txt labels
models/
  general_2022/           <- HerdNet pretrained weights (.pth + config.yaml)
```

---

## Additional Datasets (Reference)

Not used in practicals, but useful for students who want to explore further.

### Ready-made small datasets

| Dataset | Images | Species | Bboxes | Format | License |
|---------|--------|---------|--------|--------|---------|
| [Ultralytics African Wildlife](https://docs.ultralytics.com/datasets/detect/african-wildlife/) | 1,504 | 4 | Yes | YOLO | Permissive |
| [Roboflow GRI Camera Trap v1](https://universe.roboflow.com/gri-public/camera-trap-data-v1) | 860 | 15 | Yes | COCO/YOLO/VOC | CC-BY 4.0 |

### Large-scale LILA datasets

| Dataset | Images | Species | Bboxes |
|---------|--------|---------|--------|
| [Snapshot Serengeti (full)](https://lila.science/datasets/snapshot-serengeti) | 78,000 | 61 | Yes (manual) |
| [WCS Camera Traps](https://lila.science/datasets/wcs) | 300,000 | ~675 | Yes (manual) |
| [Nkhotakota Camera Traps](https://lila.science/datasets/nkhotakota) | 33,813 | 46 | Yes (manual) |

### Tools for custom subsetting

- **pycocotools** — filter by category, extract matching images
- **FiftyOne** (Voxel51) — visual curation, import COCO, filter, export
- **MegaDetector utilities** — `coco_to_yolo.py`, species-filtered LILA downloads
- **Roboflow** — no-code format conversion with augmentation

---

## Notes

- **HerdNet weights** are in `karisu/HerdNet` on HuggingFace (separate repo from the dataset)
- **Raw iguana orthomosaics** are large (~10 GB per island) — students work on pre-tiled JPEGs
- **Week 2 (SAR)** datasets are handled by N. Voss & A. Bosu — see `week2/data/`
