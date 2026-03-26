# Aerial Wildlife Detection — Course Overview
Course Materials for the Innovations and Applications of Forest IT (FIT) Module, HNEE

> A practical course on automated wildlife monitoring using drone imagery, camera traps,
> and deep learning.

---

## TL;DR

Wildlife ecologists are drowning in data. A single drone survey can produce tens of
thousands of images; a camera trap network generates millions per year. Manual review
is too slow, too expensive, and too inconsistent to scale.

This course is about closing that gap with AI — not by building models from scratch,
but by learning to use, evaluate, and critically apply the tools that already exist.

By the end of the course you will be able to:

- Run **MegaDetector** on camera trap images, interpret its confidence scores, and
  extract animal crops for downstream classification
- Apply a **pre-trained image classifier** (SpeciesNet / DeepFaune) to identify species
  from crops, and evaluate its performance honestly
- Understand how **point-based detectors** like HerdNet count animals in dense aerial
  imagery
- Have an idea when to trust a model's output — and when not to — including the common pitfalls
  of spatial data leakage and overconfident predictions
- Know how to train a model like YOLO on a custom dataset, and how to use tools like SAHI for large image inference

The course is deliberately **not** that much about Deep Learning topics like backpropagation, loss functions, or training
from scratch. The goal is tool fluency and conceptual literacy — understanding what
these models do, how to run them, and how to evaluate their outputs for real ecological
applications.


## Prerequisites
In order to get started quickly please come prepared with some preparations made. This will help you focus on the wildlife detection part.

### Software environment: 

* setup up your python environment like you did in the previous semesters. Conda is preferred, in order to get GDAL and other tools working. 
* Install the three conda conda environments **[Installation Instructions](./INSTALLATION_INSTRUCTIONS.md)**
* Use an IDE like [PyCharm Pro](https://www.jetbrains.com/pycharm/download/) 

* to get access to Datasets and Models, get yourself an account for huggingface: https://huggingface.co/
* to track the performance of trained models get yourself an account for https://wandb.ai/

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

| #   | Notebook | Description | Colab |
|-----|----------|-------------|-------|
| P0  | [practical_0_megadetector_legacy.ipynb](week1/practicals/practical_0_megadetector_legacy.ipynb) | Run MegaDetector v5 on camera trap images, extract crops, compare with v1000 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwinkelmann/usde-innovations-applications-forest-it/blob/course_draft/week1/practicals/practical_0_megadetector_legacy.ipynb) |
| P1  | [practical_01_visual_wildlife_datasets.ipynb](week1/practicals/practical_01_visual_wildlife_datasets.ipynb) | Explore camera trap and aerial datasets with four annotation types | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwinkelmann/usde-innovations-applications-forest-it/blob/main/week1/practicals/practical_01_visual_wildlife_datasets.ipynb) |
| P3b | [practical_3_megadetector_ultralytics.ipynb](week1/practicals/practical_3_megadetector_ultralytics.ipynb) | Run MegaDetector v1000 via ultralytics with SAHI tiled inference | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwinkelmann/usde-innovations-applications-forest-it/blob/main/week1/practicals/practical_3_megadetector_ultralytics.ipynb) |
| P4  | [practical_04_annotation_tools.ipynb](week1/practicals/practical_04_annotation_tools.ipynb) | Upload images to Label Studio with MegaDetector pre-annotations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwinkelmann/usde-innovations-applications-forest-it/blob/main/week1/practicals/practical_04_annotation_tools.ipynb) |
| P5  | [practical_5_species_classification.ipynb](week1/practicals/practical_5_species_classification.ipynb) | Classify animal crops with SpeciesNet and DeepFaune, evaluate with confusion matrix | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwinkelmann/usde-innovations-applications-forest-it/blob/main/week1/practicals/practical_5_species_classification.ipynb) |
| P6  | [practical_06_aeirial_object_detection-herdnet.ipynb](week1/practicals/practical_06_aerial_object_detection-herdnet.ipynb) | Run HerdNet point-based detection on aerial wildlife imagery | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwinkelmann/usde-innovations-applications-forest-it/blob/main/week1/practicals/practical_06_aeirial_object_detection-herdnet.ipynb) |
| P7  | [practical_07_segmentation.ipynb](week1/practicals/practical_07_segmentation.ipynb) | Train a U-Net on a drone tile with a hand-drawn mask | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwinkelmann/usde-innovations-applications-forest-it/blob/main/week1/practicals/practical_07_segmentation.ipynb) |
| P8  | [practical_08_wrapup.ipynb](week1/practicals/practical_08_wrapup.ipynb) | Export results, structured reflection, Week 2 preview | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwinkelmann/usde-innovations-applications-forest-it/blob/main/week1/practicals/practical_08_wrapup.ipynb) |

### Week 2 Practicals

| # | Notebook | Description | Colab |
|---|----------|-------------|-------|
| W2 | [practical_geospatial_segmentation.ipynb](week2/practicals/practical_geospatial_segmentation.ipynb) | Pixel classification on georeferenced rasters with QGIS training points | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwinkelmann/usde-innovations-applications-forest-it/blob/main/week2/practicals/practical_geospatial_segmentation.ipynb) |

---

## Repository Structure
```
INSTALLATION_INSTRUCTIONS.md  ← How to set up your environment (read before Day 1)
Course_layout.md              ← Master course schedule and pedagogical structure
week1/
  lectures/                   ← Slide decks and lecture notes (Week 1)
  practicals/                 ← jupyter notebooks and exercise scripts
  data/                       ← Sample datasets for practicals (or download scripts)
week2/
  lectures/                   ← Slide decks and lecture notes (Week 2)
  practicals/                 ← jupyter notebooks and exercise scripts
  data/
```

---

## Learning Objectives

By the end of the module, students will be able to:

1. Explain how AI tools are applied across the ecology monitoring pipeline
2. Run MegaDetector on camera trap imagery and interpret its outputs
3. Apply a pre-trained image classifier to wildlife crops
4. Understand segmentation conceptually and run a basic SAM demo
5. Work with SAR imagery for land cover change detection
6. Critically assess where AI helps — and where it fails — in field ecology

---

## Practical Environment

All practicals run in Python. Follow the **[Installation Instructions](./INSTALLATION_INSTRUCTIONS.md)** to set up your environment before Day 1. The course uses two conda environments to avoid dependency conflicts — the installation guide explains which environment to use for each practical.

Key packages used across the module:

| Package | Purpose |
|---------|---------|
| `megadetector` | Camera trap animal detection |
| `timm` + `torch` | Pre-trained classification models |
| `segment-anything` | SAM segmentation demos |
| `rasterio` / `GDAL` | Geospatial raster data handling |
| `matplotlib` / `folium` | Visualisation |

A QGIS installation is recommended but not required.

---

## Deliberate Scope

This is **not** a deep learning theory course. We do not cover low level machine learning like:

- Backpropagation or training from scratch
- Loss functions, optimisers, or hyperparameter tuning
- Transformer architecture internals
- GPU cluster workflows

The goal is **tool fluency and conceptual literacy** — understanding what these
models do, how to run them, and how to evaluate their outputs for real ecological
applications.

---

## Contributing & Contact

This repository is actively developed alongside the course. If you are a student
and find an error or want to suggest an improvement, open an issue or contact the
teaching team directly.

**J.-P. Mund** — module lead
**C. Winkelmann** — practical sessions (Week 1)
**N. Voss & A. Bosu** — radar remote sensing (Week 2)


## Course Structure

# Week 1 — AI & UAV Wildlife Image Classification
**Module:** FIT — Innovations and Applications of Forest IT
**Dates:** March 30 – April 2, 2026
**Teaching team:** J.-P. Mund & C. Winkelmann

---

## Learning Objectives

By the end of Week 1, students can:
- Explain why and how AI is applied in wildlife ecology and forest monitoring
- Describe the camera trap workflow and how MegaDetector accelerates it
- Distinguish between detection, classification, and segmentation tasks
- Apply a pre-trained classification model to a small wildlife dataset
- Train a bounding box detection model like YOLO, train a Classification Model like ResNet or a point based detection (e.g. HerdNet) on a custom dataset
- Understand segmentation conceptually and why it matters for remote sensing tasks

---

## Day 1 — Monday, March 30

### 09:30–12:30 | Why AI in Ecology? + UAV Surveys
Lecture
---

### 13:15–16:00 | Data & Preprocessing Practicals

**Setup**
- Run the initial Notebook: [Intro](week1/practicals/practical_0_megadetector_legacy.ipynb)
- Download some datasets
- Install/use pre-configured environment
- Run MegaDetector on provided camera trap images
- Parse JSON output: filter by confidence, extract animal crops
- Visualise detections with bounding boxes

**Getting familiar with camera trapping**

- apply a trained model on camera trap images
- sort into empty / animal / person / vehicle
- crop detections to constant sized

**Annotation tools intro**
- 
- Run a species Classification Model on the crops. Which one? Species Net, DeepFaune
- What MegaDetector *doesn't* do: it detects animals, not species
- The two-stage pipeline: detect → crop → classify
- Overview of classification models used in ecology:
  DeepFaune, Wildlife Insights, iNaturalist CV, custom classifiers
- What training data looks like; ImageNet vs. domain-specific datasets
- 
---

## Day 2 — Tuesday, March 31

### 09:30–12:30 | Camera Traps & MegaDetector (Lecture & Seminar)
Lecture

---

### 13:15–16:00 | Data Processing Practicals

**From Camera Trapping to Aerial Images (40 min)**
- Small object detection
- using slided Inference on Full Images to detect animals
- 

**MegaDetector Deep Dive (60 min)**
- What MegaDetector does: animal / person / vehicle detection
- How to run it: CameraTraps / `megadetector` Python package, JSON output
- Interpreting confidence scores; what to do with low-confidence detections
- Wildlife Insights platform as a managed alternative
- Live demo: run MegaDetector on a small camera trap dataset

> **Deliberate skip:** We do not cover YOLO architecture internals, anchor boxes,
> or mAP computation — students use MegaDetector as a tool, not a research object

**Practical 3 — From Detection to Classification**


**Practical 4 — Exploration**
- Students browse their detections: what worked, what failed?
- Common failure modes: motion blur, partial animals, dense vegetation
- push predictions into label studio to correct them

---

## Day 3 — Wednesday, April 1

### 09:30–12:30 | Image Classification for Wildlife (Lecture & Seminar)
Lecture
---

### 13:15–16:00 | Bonus Day

This Afternoon is free

TODO: prepare some things people might want to do on their own
**Practical 5 — Running a classifier**
- Load a pre-trained EfficientNet / TIMM model via Python
- Run inference on the animal crops from Day 2
- Build a simple results table: image → detected class → confidence

**Practical 6 — Evaluating results (30 min)**
- Quick accuracy check against a small labelled reference set
- Where does it fail? What does that mean for field use?

---

## Day 4 — Thursday, April 2

### 09:30–12:30 | Introduction to Segmentation

**From Boxes to Masks (50 min)**
- Detection vs. Classification vs. Segmentation — visual comparison
- Semantic segmentation: every pixel gets a class (habitat mapping, land cover)
- Instance segmentation: separate individual animals or trees
- Why segmentation matters for Week 2: vegetation mapping, illegal mining detection

**Segmentation in Ecology & Remote Sensing (50 min)**
- Land cover mapping with Sentinel-2 + U-Net style models

> **Bridge moment:** "Segmentation is the tool; next week you'll use it on a
> real deforestation detection problem"

**Break**

**Q&A + Synthesis Discussion (40 min)**
- Students explain back the pipeline: detect → classify → segment
- What would a complete AI ecology monitoring system look like?
- Open questions to carry into the Easter break
- Discuss matures geospatial AI tools like TorchGeo

---

### 13:15–16:00 | Segmentation Practicals + Wrap-up
TODO: practical_07_segmentation.ipynb

**Practical 7 — Intro to semantic segmentation**
- Run a pre-trained segmentation model (e.g. SAM or a simple U-Net)
- Apply to a small land cover / drone image dataset
- Visualise class masks overlaid on imagery

**Practical 8 — Week 1 Wrap-up**
- Students export/save their work from all practicals
- Short reflection: one thing that surprised you, one open question
- Preview of Week 2 topics (Radar RS, Galamsey)

---

---

## Day 5 — Friday, April 3

Free Day because of public holiday

TODO: define some bonus materials for students who want to explore on their own

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

## What We Deliberately Skip

- Backpropagation, loss functions, gradient descent
- YOLO/DETR architecture internals
- Model training from scratch
- Transformer attention math

> The goal is **fluent tool use and conceptual literacy**, not ML research expertise.


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
- UAV-based population estimation (aerial detection)
- Camera trap classification
- Sound / passive acoustic monitoring (bioacoustics)

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
- iWildCam Challenge Data: https://www.kaggle.com/competitions/iwildcam2022-fgvc9/data
- **Wildlife Insights** — camera trap uploads + SpeciesNet predictions: http://wildlifeinsights.org 
- **https://zenodo.org/records/8136161** - Crown Data for "Accurate delineation of individual tree crowns in tropical forests from aerial RGB imagery using Mask R-CNN"

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

### Leakage Scenarios (ordered by severity)
- Huge spatial and temporal gap — e.g. Floreana vs. Santa Cruz, 1 year apart ✅ safe
- Small spatial gap, same date — e.g. same beach, different passes ⚠️ borderline
- Tiny spatial gap, same flight — overlapping tiles from the same image ❌ leakage risk

### Stratified Sampling
Use stratified sampling to maintain class distributions across splits, especially for imbalanced datasets.

---

## Classical Approaches — Rear-Seat Observers & Jolly II

### Aerial Transect Surveys
Trained human observers in manned aircraft scan defined transects and count animals visually.
Well-established but subject to observer fatigue, double-counting, and missed detections.

### Mark-Recapture (Jolly-Seber / Jolly II)
Statistical framework for open populations. Animals are captured, marked, released, and re-encountered
in subsequent sessions. Jolly II extends this to allow births, deaths, and immigration.

**Key parameters:**
- `N` — estimated population size
- `φ` — survival probability
- `p` — recapture probability

### Distance Sampling
Animals are counted along transects with distances recorded. Detection probability modelled as a
function of distance. Widely used in aerial surveys.

### Limitations of Classical Methods
- Labour-intensive and expensive (manned aircraft, trained observers)
- Observer fatigue degrades precision over long surveys
- Hard to standardise across teams and years
- Cannot be easily replicated at scale across remote island terrain

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

### Reproducibility Checklist
- [ ] Can the software be installed from scratch? (test on clean environment)
- [ ] Are model weights versioned and accessible?
- [ ] Are dataset splits deterministically reproducible?
- [ ] Are random seeds set for training?
- [ ] Is the conda/pip environment pinned?

### Known Reproducibility Issues in Wildlife ML
- WildlifeMapper, Megadetector, HerdNet — installation complexity varies
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

## Ideas for examination projects / Master Thesis or other projects
- TODO object deduplication via photogrammetry
- TODO inspect optuna for hyperparameter search.
- TODO: look into iwildcam data, is could the prevalence of species be tied to landsat 8 images? like is the densitiy tied to some spectra? - https://github.com/visipedia/iwildcam_comp
- TODO Varoa Mite detection in honeybee hives using AI
- TODO: https://github.com/bambi-eco/Bambi-QGIS / https://github.com/bambi-eco/Dataset 
- Find an interesting dataset on roboflow or kaggle, run a YOLOv8 baseline, and do a hyperparameter sweep to improve it

- Train a aerial megadetector model using aerial dataset


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