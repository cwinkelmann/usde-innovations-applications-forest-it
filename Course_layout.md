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

### 09:30–12:30 | Why AI in Ecology? + UAV Surveys (Lecture & Seminar)
*Lead: J.-P. Mund — framing lecture*
*Contribution: C. Winkelmann — case study input*

**Block 1 — The Big Picture (45 min)**
- The biodiversity monitoring challenge: why we need automated tools
- What AI can and cannot do in ecology (set honest expectations)
- The sensing modality landscape: drones, camera traps, satellites, acoustics, RADAR
  → Keep this visual and fast-paced; goal is curiosity, not completeness

**Block 2 — TODO move this to Day 2 UAV Surveys in Wildlife Ecology (60 min)**
- How drone surveys work: flight planning, GSD, orthomosaics
- From pixel to population: the counting problem
- Case study: Iguanas From Above (Galápagos marine iguanas)
  → Show real imagery; discuss scale, density, camouflage challenges
- Brief intro to annotation: what humans do that models learn from

**Break**

**Block 3 — Seminar Discussion (45 min)**
- What species/habitats would students want to monitor?
- What would stop a drone survey from working? (weather, access, legal, compute)
- Group brainstorm: which part of the pipeline is the bottleneck?

---

### 13:15–16:00 | Data & Preprocessing Practicals
*Lead: C. Winkelmann*

**Practical 0 - Setup**
- Run the initial Notebook
- Download some datasets
- Install/use pre-configured environment
- Run MegaDetector on provided camera trap images
- Parse JSON output: filter by confidence, extract animal crops
- Visualise detections with bounding boxes

**Practical 1 — Getting familiar with camera trapping (90 min)**

- apply a trained model on camera trap images
- sort into empty / animal / person / vehicle
- crop detections to constant sized

**Practical 2 — Annotation tools intro (45 min)**
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
*Lead: J.-P. Mund & C. Winkelmann*

TODO what is part of the lectures?


---

### 13:15–16:00 | Data Processing Practicals
*Lead: C. Winkelmann*

**Practical 1 — From Camera Trapping to Aerial Images (40 min)**
- Small object detection
- using slided Inference on Full Images to detect animals
- 

**Practical 2 — MegaDetector Deep Dive (60 min)**
- What MegaDetector does: animal / person / vehicle detection
- How to run it: CameraTraps / `megadetector` Python package, JSON output
- Interpreting confidence scores; what to do with low-confidence detections
- Wildlife Insights platform as a managed alternative
- Live demo: run MegaDetector on a small camera trap dataset

> **Deliberate skip:** We do not cover YOLO architecture internals, anchor boxes,
> or mAP computation — students use MegaDetector as a tool, not a research object

**Practical 3 — From Detection to Classification (40 min)**


**Practical 4 — Exploration (30 min)**
- Students browse their detections: what worked, what failed?
- Common failure modes: motion blur, partial animals, dense vegetation
- push predictions into label studio to correct them

---

## Day 3 — Wednesday, April 1

### 09:30–12:30 | Image Classification for Wildlife (Lecture & Seminar)
*Lead: J.-P. Mund & C. Winkelmann*

TODO 

---

### 13:15–16:00 | Classification Practicals
*Lead: C. Winkelmann*

**Practical 5 — Running a classifier (90 min)**
- Load a pre-trained EfficientNet / TIMM model via Python
- Run inference on the animal crops from Day 2
- Build a simple results table: image → detected class → confidence

**Practical 6 — Evaluating results (30 min)**
- Quick accuracy check against a small labelled reference set
- Where does it fail? What does that mean for field use?

---

## Day 4 — Thursday, April 2

### 09:30–12:30 | Introduction to Segmentation (Seminar & Practise)
*Lead: J.-P. Mund & C. Winkelmann*

**Block 1 — From Boxes to Masks (50 min)**
- Detection vs. Classification vs. Segmentation — visual comparison
- Semantic segmentation: every pixel gets a class (habitat mapping, land cover)
- Instance segmentation: separate individual animals or trees
- Why segmentation matters for Week 2: vegetation mapping, illegal mining detection

**Block 2 — Segmentation in Ecology & Remote Sensing (50 min)**
- Land cover mapping with Sentinel-2 + U-Net style models
- Compare U-NET, SegFormer and SAM
- Tree crown delineation from UAV/LiDAR
- Brief preview of Week 2 RADAR use case (Galamsey/Ghana): what they'll apply this to

> **Bridge moment:** "Segmentation is the tool; next week you'll use it on a
> real deforestation detection problem"

**Break**

**Block 3 — Q&A + Synthesis Discussion (40 min)**
- Students explain back the pipeline: detect → classify → segment
- What would a complete AI ecology monitoring system look like?
- Open questions to carry into the Easter break
- Discuss matures geospatial AI tools like TorchGeo

---

### 13:15–16:00 | Segmentation Practicals + Wrap-up
*Lead: C. Winkelmann*

**Practical 7 — Intro to semantic segmentation (75 min)**
- Run a pre-trained segmentation model (e.g. SAM or a simple U-Net)
- Apply to a small land cover / drone image dataset
- Visualise class masks overlaid on imagery

**Practical 8 — Week 1 Wrap-up (30 min)**
- Students export/save their work from all practicals
- Short reflection: one thing that surprised you, one open question
- Preview of Week 2 topics (Radar RS, Galamsey)

---

## Suggested Datasets (Practical Use)

| Day | Dataset | Source |
|-----|---------|--------|
| 1 | Drone wildlife tiles | Iguanas From Above / provided |
| 2 | Camera trap images | LILA BC (Snapshot Serengeti subset) |
| 3 | Animal crops from Day 2 | Generated in Practical 3 |
| 4 | Land cover / UAV image | Sentinel-2 clip or provided drone scene |

---

## Tools & Environment

| Tool | Purpose | Install |
|------|---------|---------|
| MegaDetector v5 | Camera trap detection | `pip install megadetector` |
| CVAT / Label Studio | Annotation demo | Web-based |
| TIMM + PyTorch | Classification inference | `pip install timm` |
| Segment Anything (SAM) | Segmentation intro | `pip install segment-anything` |
| QGIS (optional) | Spatial data visualisation | Desktop install |

---

## What We Deliberately Skip

- Backpropagation, loss functions, gradient descent
- YOLO/DETR architecture internals
- Model training from scratch
- Transformer attention math

> The goal is **fluent tool use and conceptual literacy**, not ML research expertise.