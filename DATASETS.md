## Overview of datasets


/Users/christian/work/hnee/usde-innovations-applications-forest-it/week1/data/DATASETS.md



/Users/christian/work/hnee/usde-innovations-applications-forest-it/week2/data/DATASETS.md


# African wildlife camera trap datasets for deep learning tutorials

**The Ultralytics African Wildlife dataset is the fastest path to a working tutorial**: 1,504 images with bounding boxes across four species in YOLO format, ready to train out of the box. For richer species diversity and COCO JSON annotations, Snapshot Serengeti's bounding-box subset (~78,000 annotated images, 61 African species) is the gold standard—easily filtered to a balanced ~1,000-image sample using standard tools. Several Roboflow and Hugging Face options fill the gap between these two extremes, and a mature ecosystem of subsetting tools (pycocotools, FiftyOne, MegaDetector utilities) makes custom subset creation straightforward from any LILA-hosted dataset.

---

## Ready-made small datasets that work immediately

Three datasets match the ~1k-image, multi-class, bounding-box requirement with zero preprocessing:

**Ultralytics African Wildlife** is the single best turnkey option. It contains **1,504 images** (1,052 train / 225 val / 227 test) across four classes—buffalo, elephant, rhino, and zebra—with per-image YOLO-format bounding box annotations. Download is a single ~100 MB zip from GitHub (`african-wildlife.zip`), or auto-fetched via the Ultralytics CLI with `yolo train data=african-wildlife.yaml`. The dataset integrates directly with YOLOv8/YOLO11 training pipelines and includes official tutorial documentation. License is permissive (created by Bianca Ferreira, adapted by the Ultralytics team).

**Roboflow GRI Camera Trap Data v1** offers broader species coverage: **860 images across 15 African wildlife classes** including elephant, lion (male/female), leopard, hyena, impala, warthog, waterbuck, wild dog, honey badger, jackal, and sable. Every image has bounding box annotations exportable from Roboflow Universe in COCO JSON, YOLOv5/v8, Pascal VOC, or TFRecord format—no scripting required. Licensed under **CC BY 4.0**. URL: `universe.roboflow.com/gri-public/camera-trap-data-v1`.

**Roboflow African-Wildlife** provides **1,463 images** with bounding boxes across 4 classes (buffalo, elephant, rhino, zebra) in the same multi-format export system. Also CC BY 4.0. This dataset appears to share the same base images as the Ultralytics version but is available through Roboflow's export pipeline, making it ideal if you need COCO JSON rather than YOLO format.

On Kaggle, the **African Wildlife dataset by Bianca Ferreira** (~1,504 images, 4 classes) is widely used in PyTorch tutorials but provides **classification labels only—no bounding boxes**. It suits species classification tasks but not object detection.

---

## Large-scale LILA datasets with manual bounding boxes

For users who want richer taxonomic diversity or a custom-curated subset, three LILA-hosted datasets provide **manually drawn bounding boxes** on African wildlife in COCO Camera Traps JSON format. All use the permissive **Community Data License Agreement (CDLA)** and are hosted on GCP, AWS, and Azure with per-image HTTP download support.

**Snapshot Serengeti** is the flagship. Its full corpus spans **7.1 million images** across 61 Tanzanian species (wildebeest, zebra, Thomson's gazelle, lion, elephant, giraffe, buffalo, leopard, cheetah, hyena, and more). A separate bounding-box annotation file covers **~150,000 boxes on ~78,000 images**—large enough to create richly balanced subsets. The bbox JSON is a compact download (`SnapshotSerengetiBboxes_20190903.json.zip`), and pre-defined train/val splits are available. The combined species-label metadata (also COCO JSON + CSV) maps categories to images, making programmatic filtering trivial. Images live in cloud buckets accessible via `gsutil`, `aws s3`, or `AzCopy`, and each image has an individual HTTP URL for selective download.

**WCS Camera Traps** spans 12 countries (including several African nations) with **~375,000 bounding boxes on ~300,000 images** and **~675 species**. Bounding box annotations come in two variants: one with full species-level classes, another with coarse animal/person/vehicle labels. African species include elephant, lion, gerenuk, and East African oryx. Filtering by species name or GPS coordinates isolates the African subset. Metadata and splits are downloadable from GCP.

**Nkhotakota Camera Traps** is a pure-African dataset from Malawi's Nkhotakota Wildlife Reserve: **321,562 images**, 46 taxa, with **33,813 images carrying manual bounding boxes**. Species skew toward smaller antelope (common duiker, red duiker, klipspringer, Sharpe's grysbok), making it complementary to Serengeti's large-mammal focus.

Two additional LILA datasets have bounding boxes with African content: **Leopard ID 2022** (6,795 images of African leopards in Botswana, COCO JSON with bounding boxes and individual IDs) and **Hyena ID 2022** (spotted hyenas, similar format). These are designed for re-identification but their bounding boxes work for detection tutorials.

---

## Datasets with species labels only (no bounding boxes)

Several large African datasets provide image-level classification labels in COCO Camera Traps JSON but lack manual bounding boxes. These suit classification tutorials or can gain detection boxes via MegaDetector:

| Dataset | Location | Images | Species | Format |
|---|---|---|---|---|
| Snapshot Safari 2024 Expansion | 15 African sites | 4,029,374 | 151 | COCO JSON + CSV |
| Snapshot Enonkishu | Kenya | 28,544 | 39 | COCO JSON + CSV |
| Snapshot Kruger | South Africa | 10,072 | 46 | COCO JSON + CSV |
| Snapshot Karoo | South Africa | 38,074 | 38 | COCO JSON + CSV |
| Biome Health Project Maasai Mara | Kenya | 37,075 | 100 | COCO JSON |
| Desert Lion Conservation | Namibia | 65,959 | 46 | COCO JSON |

LILA publishes **pre-computed MegaDetector results** for all hosted datasets, providing class-agnostic animal/person/vehicle bounding boxes across **>1.1 million detections**. This effectively gives every species-label-only dataset detection boxes, though without per-species box labels.

---

## Datasets that are not African or not ideal

**ENA24** (~10,000 images, 23 classes) has excellent per-image bounding boxes in COCO JSON but contains **entirely North American species**—American black bear, white-tailed deer, coyote, raccoon. It is a useful format reference but wrong for African wildlife.

**iWildCam 2020–2022** derives from WCS Camera Traps and covers global locations including Africa (~260,000 images, ~675 species). Species labels are in COCO JSON, but bounding boxes come only from automated MegaDetector output (class-agnostic). Subsetting African content requires GPS-coordinate filtering. The 2018–2019 editions use entirely North American data and are irrelevant.

**Wildlife Insights** hosts millions of camera trap images from hundreds of projects, many in Africa. However, it requires account creation, per-project download requests, and delivers annotations in a proprietary CSV schema—not COCO JSON or YOLO. Per-project Creative Commons licensing adds complexity. AI-generated bounding box coordinates appear in the CSV but are not human-verified. This platform is valuable for research but impractical for a quick tutorial dataset.

---

## Tools and scripts for custom subsetting

A robust ecosystem exists for creating balanced ~1,000-image subsets from any LILA dataset:

**pycocotools** (the standard COCO API) works directly with COCO Camera Traps JSON since the format is a strict superset of COCO. Filter by category with `getCatIds(catNms=['elephant','zebra','lion'])`, retrieve matching image IDs with `getImgIds(catIds=...)`, extract annotations with `getAnnIds()`, and write a filtered JSON for your subset.

**FiftyOne** (Voxel51) provides the most versatile visual curation workflow. Import a COCO detection dataset, filter by label (`dataset.match(F("ground_truth.label").is_in([...]))`), take a random sample (`view.take(1000, seed=42)`), inspect results in the GUI, and export to COCO JSON or YOLOv5 format in one call. This is the recommended tool for iterative, visual dataset curation.

**MegaDetector repository** (`github.com/agentmorris/MegaDetector`) contains purpose-built camera trap utilities: `coco_to_yolo.py` for format conversion, `cct_json_utils.py` for COCO Camera Traps manipulation, `generate_crops_from_cct.py` for bounding-box crops, and `get_lila_species_counts.py` for species-filtered downloads across any LILA dataset.

**kwcoco** offers a CLI-based approach: `kwcoco subset` extracts a subset and writes a new annotation file in a single command. **Roboflow** handles format conversion with no code—upload or select a dataset and export in COCO, YOLO, VOC, or TFRecord format with built-in augmentation and train/val/test splitting.

On Hugging Face, `society-ethics/lila_camera_traps` mirrors LILA data with programmatic access via the `datasets` library, and the **IDLE-OO-Camera-Traps** dataset includes Jupyter notebooks (`lilabc_CT.ipynb`, `lilabc_test-filter.ipynb`) demonstrating exactly how to build small balanced subsets from LILA sources.

---

## Comprehensive comparison table

| Dataset | African? | Images | Species | Bounding boxes? | Native format | License | ~1k subset ease |
|---|---|---|---|---|---|---|---|
| **Ultralytics African Wildlife** | ✅ 100% | 1,504 | 4 | ✅ Every image | YOLO | Permissive | Already done |
| **Roboflow GRI Camera Trap v1** | ✅ 100% | 860 | 15 | ✅ Every image | COCO/YOLO/VOC | CC BY 4.0 | Already done |
| **Roboflow African-Wildlife** | ✅ 100% | 1,463 | 4 | ✅ Every image | COCO/YOLO/VOC | CC BY 4.0 | Already done |
| **Snapshot Serengeti (bbox)** | ✅ 100% | 78,000 (bbox) | 61 | ✅ Manual | COCO JSON | CDLA | Easy (filter + download) |
| **WCS Camera Traps (bbox)** | Partial | 300,000 (bbox) | ~675 | ✅ Manual | COCO JSON | CDLA | Medium (GPS/species filter) |
| **Nkhotakota Camera Traps** | ✅ 100% | 33,813 (bbox) | 46 | ✅ Manual | COCO JSON | CDLA | Easy |
| **Leopard ID 2022** | ✅ 100% | 6,795 | 1 (430 individuals) | ✅ Manual | COCO JSON | CDLA | Easy but single species |
| **Snapshot Safari 2024** | ✅ 100% | 4,029,374 | 151 | ❌ Labels only | COCO JSON | CDLA | Easy (labels only) |
| **Kaggle African Wildlife** | ✅ 100% | 1,504 | 4 | ❌ Labels only | ImageFolder | Not specified | Already done (no bbox) |
| **iWildCam 2022** | Partial | ~260,000 | ~675 | ⚠️ MegaDetector only | COCO JSON | CDLA | Medium |
| **Wildlife Insights** | Yes (select) | Millions | 800+ | ⚠️ AI-generated CSV | CSV | Per-project CC | Hard |
| **ENA24** | ❌ N. America | ~10,000 | 23 | ✅ Every image | COCO JSON | CDLA | N/A |

---

## Conclusion: recommended paths for a tutorial

The optimal choice depends on how much setup time is acceptable. For **zero-effort startup**, the Ultralytics African Wildlife dataset delivers 1,504 YOLO-annotated images across four iconic species with a single download command—ideal for a YOLO-based detection tutorial. For **COCO JSON format**, the same base data is available through Roboflow with one-click export, or the GRI Camera Trap dataset adds 15 species for richer classification.

For a **pedagogically richer tutorial** that teaches data curation alongside model training, building a custom ~1,000-image subset from Snapshot Serengeti's bounding-box annotations is the strongest approach. The workflow—download the compact bbox JSON, filter to 8–10 target species using pycocotools, balance the sample, download individual images via HTTP, and optionally convert to YOLO using the MegaDetector repo's `coco_to_yolo.py`—teaches real-world data engineering skills and yields a dataset with far greater taxonomic diversity than any pre-built option. FiftyOne makes this workflow visual and interactive.

The combination of a ready-made small dataset (Ultralytics or Roboflow) for quick prototyping and a LILA subset pipeline for advanced exercises would cover both classification and detection in a single tutorial course.