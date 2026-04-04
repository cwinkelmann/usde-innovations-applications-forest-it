# Aerial Wildlife Detection — Course Overview
Course Materials for the Innovations and Applications of Forest IT (FIT) Module, HNEE

A practical course on automated wildlife monitoring using drone imagery, camera traps, and deep learning.

---

## Intro

Wildlife ecologists are drowning in data. A camera trap network generates millions of images per year.
A single drone survey can produce tens of thousands of images. 
Manual review is too slow, too expensive, and too inconsistent to scale.

This course is about closing that gap with AI — not by building models from scratch,
but by learning to use, evaluate, and critically apply the tools that already exist.

By the end of the course you will be able to:

- Run find if an image contains an animal, person or vehicle on camera trap images
- interpret **MegaDetector** confidence scores, and
  extract animal crops for downstream classification
- Apply a **pre-trained image classifier** to identify species of animals
  from images, and evaluate its performance honestly
- Understand how **point-based detectors** like HerdNet count animals in dense aerial
  imagery
- Know how to train a model like YOLO on a custom dataset, and how to use tools like SAHI for large image inference

The course is deliberately **not** that much about Deep Learning topics like backpropagation, loss functions, or training
from scratch. If you are curious about that check out this bootcamp: [practical-computer-vision](https://github.com/andandandand/practical-computer-vision)
The goal is tool fluency and conceptual literacy — understanding what
these models do, how to run them, and how to evaluate their outputs for real ecological
applications. If you are curios, check out this course: https://github.com/andandandand/practical-computer-vision 


## Prerequisites
In order to get started quickly please come prepared with some preparations made. This will help you focus on the wildlife detection part.

### Software environment

**Recommended IDE:** [PyCharm Pro](https://www.jetbrains.com/pycharm/download/) (free for students) or VSCode.

**Get the repository:**
```bash
git clone https://github.com/cwinkelmann/usde-innovations-applications-forest-it.git
cd usde-innovations-applications-forest-it
```
Or download the ZIP from GitHub and extract it.

---

## Installation

This course uses **two separate Python environments** to keep dependencies clean.
Start with the lightweight `fit-megadetector` environment and install the others when needed.

You need ~20 GB free disk space for both environments and datasets.

Choose **Option A (conda)** or **Option B (virtualenv)** below.
---

### Option A — Conda (recommended, especially on Windows)


Conda handles binary dependencies like GDAL automatically, which is why it is the default.
Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if you do not have it yet.
When you install conda, don't put it into a folder with a space, like "Conda Folder", after installing, start the conda prompt. You might need to install git as well: https://git-scm.com/install/windows

#### 1 — `fit-megadetector` (Practicals P00)
This is the hello world environment. You will not really need it. Only if you want the pure megadetector package.
```bash
conda env create -f environment-megadetector.yml
conda activate fit-megadetector
pip install -e ".[megadetector,dev]"
```

Verify:
```bash
python -c "import megadetector; print('fit-megadetector OK')"
python -c "import wildlife_detection; print('fit-megadetector OK')"

```

#### 2 — `fit-training` (Practicals P01–P08, YOLO + SAHI + classifiers)

```bash
conda env create -f environment-training.yml
conda activate fit-training
```

Verify:
```bash
python -c "import ultralytics, sahi, timm; print('fit-training OK')"
```


To update or remove an environment:
```bash
conda env update -f environment-megadetector.yml --prune  # update
conda deactivate && conda env remove -n fit-megadetector -y  # remove
```

---

### Option B — virtualenv (no Conda required)

Use this if you prefer a plain Python virtual environment, are working in a Linux/macOS environment
where Conda is not set up, or are running on Google Colab.

> **Note:** The `fit-herdnet` environment requires GDAL. On Linux/macOS you can install it
> via your system package manager (`brew install gdal` / `apt install gdal-bin libgdal-dev`).
> On Windows, GDAL is easiest via Conda (Option A) — the virtualenv path is not recommended for P06+.

#### 1 — `fit-megadetector` equivalent
Like above: this is only the hellor world environment. All realy practives use fit-training.
```bash
python -m venv .venv-megadetector
python3 -m venv --clear .venv-megadetector
pip install -e ".[megadetector,dev]"
```

#### 2 — `fit-training` equivalent

```bash
python -m venv .venv-training
python3 -m .venv-training/bin/activate
pip install -e ".[training,dev,labelstudio]"
```



#### Google Colab

Each notebook can be opened in Colab via the badge in the practicals table below.
The first cell in each notebook installs all required dependencies, you only have to uncomment them.
Colab itself works fine, but since notebooks are not synced and cannot start external applications, using data from other notebooks or starting label-studio does not work.

```bash
# Run this in Colab before anything else
!git clone  https://github.com/cwinkelmann/usde-innovations-applications-forest-it.git fit-course
%cd fit-course
!pip install -e ".[megadetector,dev]"   # or [training,dev] / [herdnet,dev]
```




### First Hello-World run check

Open [practical_00_megadetector_legacy.ipynb](week1/practicals/practical_00_megadetector_legacy.ipynb)
in JupyterLab and run it top to bottom. If MegaDetector downloads a checkpoint and produces detections,
your setup is working.

```bash
jupyter lab
```

**Troubleshooting — Label Studio won't start:**
```bash
label-studio start              # default port 8080
label-studio start --port 8081  # if 8080 is busy
```
On Windows, run in a standard terminal, not WSL.

---

### Optional accounts

* [Hugging Face](https://huggingface.co/) — access to datasets and model weights
* [Weights & Biases](https://wandb.ai/) — experiment tracking during training (P07)

---

## What This Repository Is

This repo contains the practical exercises
for the FIT module. It is designed so that students, guest lecturers, and teaching
assistants can follow the full arc of the course — from lecture inputs to hands-on

---
## Course Overview

The module is split into two thematic weeks:

### Week 1 — AI & UAV Wildlife Image Classification (Mar 30 – Apr 2)
How do we use drones and AI to detect, classify, and count animals in the wild?

Starting from the real-world problem of wildlife population monitoring, students
work through the complete pipeline:

- Why ecology needs AI: the data bottleneck in biodiversity monitoring
- Camera trapping and the MegaDetector workflow
- Image classification with pre-trained models
- UAV survey design and drone imagery fundamentals
- Introduction to segmentation — the bridge into Week 2

### Week 1 Practicals

| #    | Notebook | Description                                                                                                                                                                                                                       | Colab |
|------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
| P00  | [practical_00_megadetector_legacy.ipynb](week1/practicals/practical_00_megadetector_legacy.ipynb) | Run MegaDetector v5 (legacy API) on Snapshot Serengeti camera trap images. Visualise detections, crop animals, and compare confidence thresholds.                                                                                 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwinkelmann/usde-innovations-applications-forest-it/blob/course_draft/week1/practicals/practical_00_megadetector_legacy.ipynb) |
| P01  | [practical_01_visual_wildlife_datasets.ipynb](week1/practicals/practical_01_visual_wildlife_datasets.ipynb) | Explore camera trap (Serengeti) and aerial (Eikelboom) datasets side by side. Compare four annotation formats: bounding boxes, points, masks, and class labels. Shows how to tile images and save tiles with annotations to disk. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwinkelmann/usde-innovations-applications-forest-it/blob/course_draft/week1/practicals/practical_01_visual_wildlife_datasets.ipynb) |
| P02  | [practical_02_megadetector_ultralytics.ipynb](week1/practicals/practical_02_megadetector_ultralytics.ipynb) | Run MegaDetector v1000 via the Ultralytics API with SAHI tiled inference. Investigate how confidence scores behave across detection thresholds.                                                                                   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwinkelmann/usde-innovations-applications-forest-it/blob/course_draft/week1/practicals/practical_02_megadetector_ultralytics.ipynb) |
| P03  | [practical_03_hnee_camera_traps.ipynb](week1/practicals/practical_03_hnee_camera_traps.ipynb) | Apply MegaDetector to real HNEE camera trap data. Explore detections, crop animals, and run SpeciesNet species classification with country-level geofencing (Germany). Output feeds into P07.                                     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwinkelmann/usde-innovations-applications-forest-it/blob/course_draft/week1/practicals/practical_03_hnee_camera_traps.ipynb) |
| P05  | [practical_05_species_classification.ipynb](week1/practicals/practical_05_species_classification.ipynb) | Classify animal crops with SpeciesNet on Serengeti images.                                                                                                                                                                        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwinkelmann/usde-innovations-applications-forest-it/blob/course_draft/week1/practicals/practical_05_species_classification.ipynb) |
| P05b | [practical_05b_megadetector_confidence_analysis.ipynb](week1/practicals/practical_05b_megadetector_confidence_analysis.ipynb) | Investigate MegaDetector v1000 confidence scores on Serengeti: compare detections against ground-truth labels to understand precision, recall, and threshold trade-offs.                                                          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwinkelmann/usde-innovations-applications-forest-it/blob/course_draft/week1/practicals/practical_05b_megadetector_confidence_analysis.ipynb) |
| P05c | [practical_05c_speciesnet_evaluation.ipynb](week1/practicals/practical_05c_speciesnet_evaluation.ipynb) | Evaluate SpeciesNet predictions against Serengeti ground-truth species labels. Build a confusion matrix and measure per-species accuracy.                                                                                         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwinkelmann/usde-innovations-applications-forest-it/blob/course_draft/week1/practicals/practical_05c_speciesnet_evaluation.ipynb) |
| P06a | [HerdNet demo notebook](https://github.com/cwinkelmann/HerdNet/blob/main/notebooks/demo-training-testing-herdnet_local.ipynb) | Run HerdNet point-based detection on aerial wildlife imagery.                                                                                                                                                                     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwinkelmann/HerdNet/blob/main/notebooks/demo-training-testing-herdnet_local.ipynb) |
| P06b | [practical_06_aerial_animal_detection_domain_shift.ipynb](week1/practicals/practical_06_aerial_animal_detection_domain_shift.ipynb) | Quantify domain shift: run MegaDetector (camera trap model) vs HerdNet (aerial model) on Eikelboom drone images. Visualise confidence distributions and F1 gaps.                                                                  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwinkelmann/usde-innovations-applications-forest-it/blob/course_draft/week1/practicals/practical_06_aerial_animal_detection_domain_shift.ipynb) |
| P07  | [practical_07_annotation_tools.ipynb](week1/practicals/practical_07_annotation_tools.ipynb) | Upload HNEE or Serengeti images to Label Studio with SpeciesNet species labels as pre-annotations. Filter by species, review and correct bounding boxes, export corrected annotations as COCO JSON.                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwinkelmann/usde-innovations-applications-forest-it/blob/course_draft/week1/practicals/practical_07_annotation_tools.ipynb) |
| P08  | [practical_08_run_training.ipynb](week1/practicals/practical_08_run_training.ipynb) | Train object detectors on labelled date from practical 07.                                                                                                                                                                        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwinkelmann/usde-innovations-applications-forest-it/blob/course_draft/week1/practicals/practical_08_run_training.ipynb) |
| P09  | [practical_09_wrapup.ipynb](week1/practicals/practical_09_wrapup.ipynb) | Export and compare results across all Week 1 practicals. Structured reflection and preview of Week 2 radar remote sensing topics.                                                                                                 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwinkelmann/usde-innovations-applications-forest-it/blob/course_draft/week1/practicals/practical_09_wrapup.ipynb) |


---

## Repository Structure
```
README.md                          ← This file — course overview + installation instructions
Course_layout.md                   ← Master course schedule and pedagogical structure
environment-training.yml           ← Conda environment for all Week 1 practicals

week1/
  lectures/                        ← Slide decks and lecture notes (Week 1)
  practicals/                      ← Jupyter notebooks (one per practical block)
  data/                            ← Sample datasets (downloaded on first run)

week2/
  lectures/                        ← Slide decks and lecture notes (Week 2)
  practicals/                      ← Jupyter notebooks
  data/

weights/                           ← Pre-trained model weights (not in git)
  md_v1000.0.0-larch.pt            ← MegaDetector v1000 larch (YOLO11L, 25 MB)
  md_v1000.0.0-spruce.pt           ← MegaDetector v1000 spruce (YOLOv5s, 13 MB, edge)

scripts/
  training/
    train_combined_yolo11.py       ← Fine-tune YOLO / RT-DETR on any dataset
    phased_finetune.py             ← 3-phase progressive unfreezing
    prepare_combined_dataset.py    ← Tile & merge aerial datasets → YOLO 640px
    eval_eikelboom.py              ← Evaluate on Eikelboom test set

src/wildlife_detection/            ← Python package (pip install -e .)
  tiling/
    boxes.py                       ← Tile images with YOLO bounding-box annotations
    points.py                      ← Tile images with point annotations (HerdNet)
    utils.py                       ← Shared tiling helpers
  training/
    train_yolo_combined.py         ← YOLO/RT-DETR training logic
    phased_finetune.py             ← Phased fine-tuning implementation
    prepare_combined_dataset.py    ← Dataset preparation pipeline
    eval_eikelboom.py              ← Eikelboom evaluation logic
  label_studio.py                  ← Label Studio upload / export helpers
  utils/                           ← Config loading, W&B helpers
```




## Datasets

Each week maintains its own dataset inventory with download instructions,
format descriptions, and practical-to-dataset mappings.

| Week | Topics | Dataset inventory |
|------|--------|-------------------|
| **Week 1** | AI & UAV Wildlife Image Classification | [week1/data/DATASETS.md](./week1/data/DATASETS.md) |
| **Week 2** | Radar Remote Sensing & Galamsey Detection | `week2/data/DATASETS.md` (TODO — N. Voss & A. Bosu) |

All Week 1 datasets are downloaded automatically by
[`week1/data/download_data.py`](scripts/data/download_data.py).

---

## Ideas for Examination Projects, Master Theses, and Further Work

- **Object deduplication via photogrammetry** — Animals detected across overlapping drone
  tiles are often counted multiple times. Use structure-from-motion or homography to
  de-duplicate detections across frames and produce accurate census counts.

- **Hyperparameter optimisation with Optuna** — Replace manual learning-rate and
  augmentation tuning with a principled Bayesian search. Compare Optuna sweeps against
  the fixed hyperparameters used in this course.

- **Species distribution modelling with iWildCam + Landsat 8** — Can the prevalence of
  a species in camera trap images be predicted from satellite spectral data (NDVI, land
  cover, water proximity)? The [iWildCam dataset](https://github.com/visipedia/iwildcam_comp)
  provides geo-tagged detections that could be cross-referenced with Landsat 8 bands.

- **Varroa mite detection in honeybee hives** — Train a close-range detector on macro
  images of bees to identify Varroa destructor infestations — a high-impact use case for
  precision apiculture.

- **Thermal deer detection with BAMBI** — The [BAMBI dataset](https://github.com/bambi-eco/Dataset)
  provides thermal UAV imagery of deer in agricultural fields. The accompanying
  [QGIS plugin](https://github.com/bambi-eco/Bambi-QGIS) enables geospatial annotation
  and export — an end-to-end pipeline from drone survey to population map.

- **Open-ended detector benchmark** — Pick a wildlife dataset from Roboflow or Kaggle,
  establish a YOLOv8 baseline, run a hyperparameter sweep, and report which changes
  moved the needle most.

- Train a aerial megadetector model using aerial dataset


---

# Reference Material

The sections below are a compendium of background material, literature, and tools
covered in the lectures. You do not need to read this to get started — it is here
for reference during and after the course.

<details>
<summary><strong>Table of Contents</strong></summary>

1. [Who is this for?](#who-is-this-for)
2. [Why Wildlife Detection?](#why-wildlife-detection)
3. [Sensor Modalities & Remote Sensing Platforms](#sensor-modalities--remote-sensing-platforms)
4. [Datasets & Data Sources](#datasets--data-sources)
5. [Data Labelling](#data-labelling)
6. [Data Splits, Leakage & Spatial Autocorrelation](#data-splits-leakage--spatial-autocorrelation)
7. [Classical Approaches — Rear-Seat Observers & Jolly II](#classical-approaches--rear-seat-observers--jolly-ii)
8. [Generic Object Detection Models](#generic-object-detection-models)
9. [Camera Trap Pipelines](#camera-trap-pipelines)
   - [Megadetector](#megadetector)
   - [SpeciesNet](#speciesnet)
   - [DeepFaune](#deepfaune)
   - [Camera Trap Management Tools](#camera-trap-management-tools)
10. [Aerial-Specific Detection Models](#aerial-specific-detection-models)
    - [HerdNet with DLA Backbone](#herdnet-with-dla-backbone)
    - [DeepForest](#deepforest)
    - [WildlifeMapper](#wildlifemapper)
11. [Density Maps & Counting as Regression](#density-maps--counting-as-regression)
12. [Landscape Segmentation](#landscape-segmentation)
13. [Re-Identification](#re-identification)
14. [Two-Stage Pipelines & Human-in-the-Loop](#two-stage-pipelines--human-in-the-loop)
15. [Edge AI & On-Device Inference](#edge-ai--on-device-inference)
16. [Experiment Tracking & Reproducibility](#experiment-tracking--reproducibility)
17. [Case Study — Galápagos Marine Iguana Detection](#case-study--galápagos-marine-iguana-detection)
18. [Conference Highlights — ICTC 2026](#conference-highlights--ictc-2026)
19. [Further Reading & Tools](#further-reading--tools)

</details>

---

## Who is this for?

This course is aimed at practitioners and students with a basic background in Python and machine learning
who want to apply deep learning to wildlife monitoring. A prior survey of the expected background:

- **Classical approaches**: random forest, k-means
- **Model types**: classifier, YOLO, U-Net
- **Experiment tracking**: Weights & Biases or similar
- **Programming**: Python, SSH, debugging
- **Personal motivation**: conservation, ecology, or remote sensing

---

## Why Wildlife Detection?

> *"You can only protect what you can measure."*

Reliable population estimates are fundamental to conservation policy. Traditional methods — rear-seat
aerial observers, mark-recapture, transect walks — are labour-intensive, error-prone and hard to
standardise. Deep learning on remote sensing imagery offers a scalable alternative.

**Applications covered in this course:**
- Camera trap classification
- UAV-based population estimation (aerial detection)

---

## Sensor Modalities & Remote Sensing Platforms

### Visual Spectrum
Standard RGB sensors. Most drone datasets and pre-trained models use RGB.

### Non-Visual Spectra
- **Thermal / Infrared**: useful for detecting body heat, especially in vegetation
- **LiDAR**: structural information, canopy height
- **Multispectral / Hyperspectral**: vegetation indices, habitat classification

### Active vs Passive Sensors
- **Passive**: cameras, microphones — record ambient signals
- **Active**: LiDAR, radar — emit and receive signals

### Platforms by Distance / Altitude
| Platform | Typical Altitude | Notes |
|---|---|---|
| Camera trap / timelapse | Ground-level | High temporal resolution |
| Low-altitude drone | < 120 m | High spatial resolution, short flight time |
| Fixed-wing / manned aircraft | 500–3000 m | Large coverage, rear-seat observers |
| Satellite | > 400 km | Global coverage, lower GSD |

---

## Datasets & Data Sources

- **LILA BC** — largest repository of labelled camera trap data: https://lila.science
- **Roboflow** — annotated datasets across many domains including wildlife
- **Hugging Face** — pretrained models and datasets
- **GBIF** — occurrence data API: https://techdocs.gbif.org/en/openapi/v1/species
- **Xeno-Canto** — bird sound recordings: https://xeno-canto.org
- **Kaggle / DrivenData** — competition datasets with ecology focus: https://github.com/drivendataorg
- **iWildCam Challenge Data**: https://www.kaggle.com/competitions/iwildcam2022-fgvc9/data
- **Wildlife Insights** — camera trap uploads + SpeciesNet predictions: http://wildlifeinsights.org 
- [**Bamforest**](https://zenodo.org/records/8136161) - Crown Data for "Accurate delineation of individual tree crowns in tropical forests from aerial RGB imagery using Mask R-CNN"

### Dataset Formats
- **COCO JSON** — bounding boxes, segmentation masks
- **YOLO TXT** — per-image label files
- **CamtrapDP** — emerging standard for camera trap data (IPT integration)
- **Darwin Core / GBIF** — species occurrence records

---

## Data Labelling

### Tools
- **Roboflow** — hosted annotation with model-assisted labelling
- **Label Studio** — open-source, supports bounding boxes, polygons, points, audio
- **CVAT** — open-source, advanced multi-task annotation
- **Hasty.ai** — AI-assisted labelling (used in iguana dataset)
- **Zooniverse** — citizen science crowdsourcing platform
- **QGIS** - geospatial data labeling

### Annotation Types
- Bounding boxes — standard for detection tasks
- Point annotations — used by HerdNet (lower labelling effort)
- Segmentation masks — pixel-level classification
- Keypoints — for pose estimation or re-identification

### Practical Notes
- Leave out "impossible examples" from training (ambiguous, cut-off, motion-blurred)
- Citizen science labelling can lead to noisy or "stuck" model behaviour — curate carefully
- Consolidate label classes to reduce confusion; simpler taxonomies train better

---

## Data Splits, Leakage & Spatial Autocorrelation

This is one of the most underappreciated problems in wildlife ML.

### The Three-Set Rule
At minimum: **training / validation / test**. The test set must never influence model selection.

### Spatial Autocorrelation
Images taken close together share background, lighting, and animals. Naively random splitting will
produce inflated metrics. Tiles from the same flight or same beach should stay in the same split.

### Split Strategies
- **Site-specific model**: all sites in train and val/test. Risk: overfits to known sites, fails on new ones.
- **Site-aware model**: train on different sites than validation. Better generalisation.

### Data Leakage Scenarios (ordered by severity)
Data Leakage means Information from the train data set is in the test test set.
For example a static camera produced images for train and val. The model will perform worse on different perspectives.
Images produced at the same weather create a bias that weather.

### Stratified Sampling
Use stratified sampling to maintain class distributions across splits, especially for imbalanced datasets.


---

## Generic Object Detection Models

These are the building blocks that specialist wildlife models extend:

### YOLO (You Only Look Once)
- Single-stage detector, fast inference
- Current version: YOLOv8/v11 via Ultralytics
- Widely used in ecology for ease of use and high performance

### Faster R-CNN
- Two-stage detector (region proposal + classification)
- Higher accuracy but slower; baseline for many wildlife studies

### RetinaNet
- Single-stage with Focal Loss to correct for class imbalance
- Strong baseline for imbalanced detection tasks (few animals, many background tiles)

### DETR (Detection Transformer)
- Transformer-based end-to-end detector
- No anchor boxes; slower to train but promising



---

## Camera Trap Pipelines

### Megadetector


**Recommended workflow:**
1. Run Megadetector to get animal/human/vehicle crops
2. Pass crops to a species classifier (e.g. SpeciesNet, DeepFaune, custom model)
3. Filter by confidence — visual confirmation matters for low-confidence detections

**Known limitation:** not well-suited for overhead/nadir drone imagery — trained primarily on
camera-trap perspective.

---

### SpeciesNet


SpeciesNet (Google / Wildlife Insights) combines an object detector with a species-level classifier.
It operates on crops and returns taxonomic predictions down to species.

- Successor to the Wildlife Insights pipeline
- Deployed at scale via Wildlife Insights platform
- Also available as a standalone Python package

---

### DeepFaune


European-focused two-stage pipeline (detection + classification). Uses knowledge distillation from
YOLOv5x → YOLOv8s for fast inference. Classification head based on DINOv2 features.

- Training code is not publicly released
- Benefits from sequence context (burst images from the same camera event)

---

### No-Code and Less Code Wildlife Detection Tools



| Tool                                                                                                                                         | Notes                                                                         |
|----------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| [**Agouti**](https://agouti.eu/)                                                                                                             | European camera trap management platform                                      |
| [**Wildlife Insights**](https://www.wildlifeinsights.org/)                                                                                   | Google-backed; uses SpeciesNet; global scale                                  |
| [**Trapper**](gitlab.com/trapper-project/trapper-setup)                                                                                      | Open-source; CamtrapDP data standard;                                         |
| [**Addax-AI**](https://addaxdatascience.com/addaxai/)                                                                                        | Started as Megadetector frontend; evolved into a model zoo                    |
| [**Animl-R**](github.com/conservationtechlab/animl-r)                                                                                        | R-based pipeline; can also train models                                       |
| [**animl-py**](https://github.com/conservationtechlab/animl-py)                                                                              | Wrapper for detection and classification:                                     |
| [**Zamba / ZambaCloud**](https://www.zambacloud.com/)                                                                                        | No-Code Custom model training with less labelling via stratified sampling     |
| [**Bisque2**](https://bisque2.ece.ucsb.edu/client_service/view?resource=https://bisque2.ece.ucsb.edu/data_service/00-cXbwjYe7z7VUm3bVfDGgmS) | Can do annotation too                                                         |
| [**Dan Morris Camera Trap ML Survey**](https://agentmorris.github.io/camera-trap-ml-survey/)                                                                                                     | comprehensive list of tools collected by on of the initiators of megadetector |

---




## Experiment Tracking & Reproducibility

### Tools
- **Weights & Biases** — hyperparameter sweeps, metric logging, model registry
- **MLflow** — open-source alternative: https://github.com/mlops-ai/mlops

### Reproducibility
A method/algorihm/model is only good, when its results can be reproduced.

- [ ] Is there a clear README with instructions to run the code?
- [ ] Can the software be installed from scratch? (test on clean environment)
- [ ] Are model weights versioned and accessible?
- [ ] Are dataset splits deterministically reproducible?
- [ ] Are random seeds set for training?
- [ ] Is the conda/pip environment pinned?

This often not true in ecology technology
Software like WildlifeMapper, Megadetector, HerdNet have complex or even failing installation routines
- Data availability varies — many published datasets are not publicly released
- Spatial data leakage is rarely reported but frequently present

---

## Case Study — Galápagos Marine Iguana Detection
[Automated Marine Iguana Detection Using Drone Imagery and Deep Learning on the Galápagos Islands](https://doi.org/10.6084/m9.figshare.30719999)
### Background
Marine iguanas (*Amblyrhynchus cristatus*) are endemic to the Galápagos and population estimates
are needed to inform conservation policy. Traditional survey methods are impractical at scale across
five islands.

### Dataset: Iguanas from Above
- Multi-phase aerial survey (2020–2024)
- Five islands: Floreana, Fernandina, Genovesa, Isabela, San Cristóbal
- DJI Mavic 2 Pro drone with Hasselblad camera
- Annotation via Hasty.ai and CVAT with expert biologist consensus labelling
- Citizen science baseline via Zooniverse

### Model
HerdNet with DLA-34 backbone, FIDT density maps.

### Key Findings
- DLA-169 outperformed smaller DLA variants (DLA-34 through DLA-102) slightly
- Augmentation strategy required per-dataset tuning (Fernandina ≠ Floreana)
- Photogrammetric deduplication across overlapping drone frames using DEM projection

### Validation
- Comparison against Zooniverse citizen science volunteer counts

---


## Further Reading & Tools

### Key Papers

#### Core Models

| Paper | Key Finding | DOI |
|---|---|---|
| Delplanque et al. (2023) — HerdNet | Outperformed Faster R-CNN and density baselines; F1 of 73.6% at 3.6 s per 24-megapixel image | [10.1016/j.isprsjprs.2023.01.025](https://doi.org/10.1016/j.isprsjprs.2023.01.025) |
| Liang et al. (2023) — FIDT | Eliminates Gaussian kernel overlap in dense scenes; state-of-the-art across six crowd benchmarks | [10.1109/TMM.2022.3203870](https://doi.org/10.1109/TMM.2022.3203870) |
| Eikelboom et al. (2019) — Aerial object detection | RetinaNet found 90–95% of animals in Kenyan surveys, reducing population estimate standard errors by 31–67% | [10.1111/2041-210X.13277](https://doi.org/10.1111/2041-210X.13277) |
| Beery et al. (2019) — MegaDetector | Eliminated ~80% of empty camera-trap images at 0.93+ AP, generalizing to unseen species without retraining | [10.48550/arXiv.1907.06772](https://doi.org/10.48550/arXiv.1907.06772) |
| Gadot et al. (2024) — SpeciesNet | Adding a detection stage before classification gives ~25% macro-F1 gain, scaling to 2,000+ taxa across 65M+ images | [10.1049/cvi2.12318](https://doi.org/10.1049/cvi2.12318) |
| Rigoudy et al. (2023) — DeepFaune | 26 European species classified at 0.97 validation accuracy with precision and recall >0.90 on independent datasets | [10.1007/s10344-023-01742-7](https://doi.org/10.1007/s10344-023-01742-7) |
| Lv et al. (2024) — RT-DETR | Real-time DETR with hybrid encoder; first end-to-end transformer detector to outperform YOLO at comparable speed | [10.1109/CVPR52733.2024.01462](https://doi.org/10.1109/CVPR52733.2024.01462) |

#### Reviews & Perspectives

| Paper | Key Finding | DOI |
|---|---|---|
| Tuia et al. (2022) — ML for wildlife conservation | Up to 17,000 IUCN species remain Data Deficient; closing the gap requires hybrid ML–ecology workflows across modalities | [10.1038/s41467-022-27980-y](https://doi.org/10.1038/s41467-022-27980-y) |
| Xu et al. (2024) — Aerial/satellite detection review | YOLO, Faster R-CNN, U-Net, and ResNet dominate; eight major challenges remain including small objects and uncertainty estimation | [10.1016/j.jag.2024.103732](https://doi.org/10.1016/j.jag.2024.103732) |
| Corcoran et al. (2021) — Drone detection synthesis | Reliable for large species in open habitats with RGB; cryptic or small species require infrared and multirotor platforms | [10.1111/2041-210X.13581](https://doi.org/10.1111/2041-210X.13581) |

#### Aerial & Drone Detection

| Paper | Key Finding | DOI |
|---|---|---|
| Kellenberger et al. (2018) — Mammals in UAV images | Best-practice CNN training reduced false positives by an order of magnitude and cut manual verification workload by 3× at 90% recall | [10.1016/j.rse.2018.06.028](https://doi.org/10.1016/j.rse.2018.06.028) |
| Hodgson et al. (2018) — Drones vs. humans | Drone counts were 43–96% more accurate than ground counts by experienced human observers on replica colonies of known size | [10.1111/2041-210X.12974](https://doi.org/10.1111/2041-210X.12974) |
| Torney et al. (2019) — DL vs. citizen science | YOLOv3 matched Zooniverse volunteer accuracy on 1.3M wildebeest, replacing a 3–6 week manual counting process | [10.1111/2041-210X.13165](https://doi.org/10.1111/2041-210X.13165) |
| Duporge et al. (2021) — Elephants from satellite | CNN matched human performance (F2: 0.73–0.78) on WorldView-3/4 imagery and generalized to a different country without retraining | [10.1002/rse2.195](https://doi.org/10.1002/rse2.195) |
| Wu et al. (2023) — Satellite mammal monitoring | U-Net detected ~500,000 wildebeest and zebra across thousands of km² from 38–50 cm satellite imagery at F1 = 84.75% | [10.1038/s41467-023-38901-y](https://doi.org/10.1038/s41467-023-38901-y) |
| Delplanque et al. (2024) — Semi-automated aerial survey | Deep learning on oblique imagery reduced human photo-processing by 98% and tripled population estimates for small species vs. rear-seat observers | [10.1016/j.ecoinf.2024.102679](https://doi.org/10.1016/j.ecoinf.2024.102679) |
| May et al. (2025) — Minimising annotation effort | Point-label-only YOLO (POLO) achieved lower counting error than bounding-box YOLOv8, eliminating costly box annotation | [10.1016/j.ecoinf.2025.103387](https://doi.org/10.1016/j.ecoinf.2025.103387) |
| Axford et al. (2024) — Collectively advancing drone detection | Cross-institutional review of challenges and best practices for deep learning on drone wildlife imagery | [10.1016/j.ecoinf.2024.102842](https://doi.org/10.1016/j.ecoinf.2024.102842) |

#### Counting as Regression

| Paper | Key Finding | DOI |
|---|---|---|
| Hoekendijk et al. (2021) — DL regression counting | Image-level count labels only; R² = 0.77 for seals and R² = 0.92 for fish otolith rings, processing 100 images in under one minute | [10.1038/s41598-021-02387-9](https://doi.org/10.1038/s41598-021-02387-9) |

#### Data Leakage & Reproducibility

| Paper | Key Finding | DOI |
|---|---|---|
| Kapoor & Narayanan (2023) — Reproducibility crisis | Leakage found in 294 papers across 17 fields; correcting it eliminated the supposed advantage of complex ML over logistic regression in one case study | [10.1016/j.patter.2023.100804](https://doi.org/10.1016/j.patter.2023.100804) |
| Stock et al. (2023) — Leakage in ecological ML | Nonspatial cross-validation under spatial autocorrelation constitutes leakage, producing systematically inflated performance estimates | [10.1038/s41559-023-02162-1](https://doi.org/10.1038/s41559-023-02162-1) |

#### Galápagos Marine Iguana Case Study

| Paper                                                       | Key Finding | DOI |
|-------------------------------------------------------------|---|---|
| Varela-Jaramillo et al. (2023) — Iguana drone pilot         | Drone counts were 14% closer to mark-resight estimates and 17–35% higher than ground counts, reaching previously inaccessible colonies | [10.1186/s12983-022-00478-5](https://doi.org/10.1186/s12983-022-00478-5) |
| Varela-Jaramillo et al. (2025) — Citizen science counts     | 13,000+ volunteers achieved 91–92% counting accuracy; HDBSCAN aggregation outperformed standard majority-vote methods | [10.1038/s41598-025-08381-9](https://doi.org/10.1038/s41598-025-08381-9) |
| Varela-Jaramillo et al. (2025, submitted) — Lessons learned | Lessons from multi-year drone surveys of Galápagos marine iguanas across five islands | — |
| Winkelmann (2025)                                           | Automated Marine Iguana Detection Using Drone Imagery and Deep Learning on the Galápagos Islands | https://doi.org/10.6084/m9.figshare.30719999 |

### Useful Links
| Resource                        | URL |
|---------------------------------|---|
| Agent Morris Camera Trap Survey | https://agentmorris.github.io/camera-trap-ml-survey/ |
| MegaDetector                    | https://github.com/agentmorris/MegaDetector |
| SpeciesNet                      | https://github.com/google/cameratrapai |
| LILA BC datasets                | https://lila.science |
| PyTorch Wildlife                | https://github.com/microsoft/CameraTraps |
| DrivenData / Zamba              | https://github.com/drivendataorg |
| HerdNet                         | https://github.com/alopezgit/HerdNet |
| Trapper                         | https://gitlab.com/trapper-project/trapper-setup |
| TrapTagger                      | https://wildeyeconservation.org/traptagger/ |
| SurveyScope                     | https://wildeyeconservation.org/surveyscope/ |
| BioCLIP2                        | https://imageomics.github.io/bioclip-2/ |
| GBIF Species API                | https://techdocs.gbif.org/en/openapi/v1/species |
| Xeno-Canto Sound                | https://xeno-canto.org |

### Courses

| Resource                         | URL |
|----------------------------------|---|
| Practical Computer Vision Course | https://github.com/andandandand/practical-computer-vision |
