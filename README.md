# Aerial Wildlife Detection — Course Overview
Course Materials with the Course Innovations and Applications for Forest IT

> A practical course on automated wildlife population estimation using drone imagery and deep learning,
> grounded in real-world iguana detection research from the Galápagos Islands.

If you are a student in this course please fill out the form so I can prepare myself better and you get a first impression of the course content: https://docs.google.com/forms/d/1DyRN7uuO4OkgGec36HOZ5IznJmNb4vsA1yPgev4bLHY/edit 
---

## Table of Contents

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
standardise. Deep learning on drone imagery offers a scalable alternative.

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
- **Wildlife Insights** — camera trap uploads + SpeciesNet predictions

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

> Stub — expand with usage examples

Megadetector (Microsoft/CameraTraps) is the most widely used pre-trained model for camera traps.
It detects three classes: **animal**, **human**, **vehicle**. It is not a species classifier.

- Based on Ultralytics YOLO framework
- Latest version: MDv5 (YOLOv5) and MDv6 (YOLOv9/v10 variants)
- Can be used with **SAHI** (Slicing Aided Hyper Inference) for large/high-res images

**Installation:**
```bash
pip install PytorchWildlife
```

**Basic usage (PyTorch Wildlife):**
```python
from PytorchWildlife.models import detection as pw_detection

detector = pw_detection.MegaDetectorV5()
results = detector.single_image_detection("path/to/image.jpg")
```

**Recommended workflow:**
1. Run Megadetector to get animal/human/vehicle crops
2. Pass crops to a species classifier (e.g. SpeciesNet, DeepFaune, custom model)
3. Filter by confidence — visual confirmation matters for low-confidence detections

**Known limitation:** not well-suited for overhead/nadir drone imagery — trained primarily on
camera-trap perspective.

---

### SpeciesNet

> Stub — expand with usage examples

SpeciesNet (Google / Wildlife Insights) combines an object detector with a species-level classifier.
It operates on crops and returns taxonomic predictions down to species.

- Successor to the Wildlife Insights pipeline
- Deployed at scale via Wildlife Insights platform
- Also available as a standalone Python package

---

### DeepFaune

> Stub — expand with usage examples

European-focused two-stage pipeline (detection + classification). Uses knowledge distillation from
YOLOv5x → YOLOv8s for fast inference. Classification head based on DINOv2 features.

- Training code is not publicly released
- Benefits from sequence context (burst images from the same camera event)

---

### Camera Trap Management Tools

From ICTC 2026 workshops:

| Tool | Notes |
|---|---|
| **Agouti** | European camera trap management platform |
| **Wildlife Insights** | Google-backed; uses SpeciesNet; global scale |
| **Trapper** | Open-source; CamtrapDP data standard; `gitlab.com/trapper-project/trapper-setup` |
| **TrapTagger** | WildEye Conservation; includes human-in-the-loop verification |
| **Addax-AI** | Started as Megadetector frontend; evolved into a model zoo |
| **Animl-R** | R-based pipeline; can also train models: `github.com/conservationtechlab/animl-r` |
| **Zamba / ZambaCloud** | Custom model training with less labelling via stratified sampling |
| **Bisque2** | Can do annotation too |

**Agent Morris Camera Trap ML Survey** — comprehensive list of tools:
https://agentmorris.github.io/camera-trap-ml-survey/

---

## Aerial-Specific Detection Models

### HerdNet with DLA Backbone

> Stub — expand with training guide

Point-based object detector developed for dense herds in aerial imagery (Delplanque et al., 2023).
Uses **Focal Inverse Distance Transform (FIDT)** density maps instead of Gaussian kernels.

**Why FIDT over Gaussian density maps:**
- Gaussian kernels merge in crowded colonies → incorrect counts in dense aggregations
- FIDT maintains a distinct peak per individual → proximity-invariant precision

**Architecture:**
- Backbone: Deep Layer Aggregation (DLA-34 through DLA-169)
- IDA (Iterative Deep Aggregation) progressively re-aggregates shallow features across multiple stages
- Outperforms UNet-style skip connections for small object detection

**Installation & usage:**
```bash
# TODO: add HerdNet installation steps and inference example
```

**Training tips from iguana detection experiments:**
- Augmentation strategy is dataset-specific — do not copy settings across sites
- Hyperparameter sweeps via Weights & Biases recommended
- Use tiled inference for full orthomosaics (GeoTIFF support)

---

### DeepForest

> Stub — expand with usage examples

Based on RetinaNet with ResNet backbone. Originally developed for tree crown detection; extended to
bird and wildlife detection.

- Supports tiled inference on large images
- Available as a Python package: `pip install deepforest`

---

### WildlifeMapper

> Stub

Based on Segment Anything (SAM) and MedSAM. Used in active studies for transferability testing.

---

## Density Maps & Counting as Regression

Replacing the classification head with a regression head gives a direct count output.

- **Gaussian density maps**: convolve point annotations with a Gaussian kernel → sum = count
- **FIDT**: focal inverse distance transform — maintains per-individual peaks in dense scenes
- **CSRNet / DM-Count**: crowd-counting architectures applicable to wildlife

**Why counting as regression is useful:**
- Reduces annotation effort (points rather than bounding boxes)
- Handles severe occlusion and overlap in dense colonies
- Naturally handles "blobs" of animals in aerial perspective

---

## Landscape Segmentation

Pixel-based classification of habitat / substrate types.

### Use cases
- Identifying iguana roosting habitat (rocky substrate, black lava, shoreline zones)
- Filtering detection ROIs to likely animal zones
- Generating habitat maps for occupancy modelling

### Approaches
- **U-Net**: encoder-decoder with skip connections; standard baseline
- **Segment Anything (SAM)**: zero-shot segmentation with prompt inputs
- **SatDINO / DOFA**: geospatial foundation models (remote sensing spectra)

---

## Re-Identification

Matching individuals across images using visual features — an alternative to physical tags.

### Approaches
- **Metric learning / Triplet training**: learn an embedding space where same-individual images cluster
  (see BearID project — Ed Miller, ICTC 2026)
- **ReID by tracking**: spatial-temporal consistency across overlapping drone frames
- **Correspondence tracking via photogrammetry**: project detections to world coordinates using DEM,
  deduplicate across overlapping images

### Tools & Libraries
- **PyTorch Wildlife** includes ReID modules
- **dlib** (founder: Ed Miller / BearID) — metric learning utilities
- **OpenMV**: https://openmv.io/collections/cameras — for embedded ReID use cases

---

## Two-Stage Pipelines & Human-in-the-Loop

A recurring theme across camera traps, drones, and whale detection at ICTC 2026:

1. **Stage 1 — Detector**: broad animal/object detector (Megadetector, HerdNet, etc.)
2. **Stage 2 — Classifier**: species or behaviour classifier on crops

**Why two stages matter:**
- Separates the generalisation problem (is there an animal?) from the recognition problem (which species?)
- Allows different confidence thresholds per stage
- Facilitates human review at the classifier stage without reviewing all images

**Human-in-the-loop design principles (from ICTC talks):**
- Do not send a ranger for 90% confidence — visual confirmation is important (Sentinel / Conservation X Labs)
- "Captains don't like false alarms" — whale detection systems use live human verification before alerts
- BioCLIP2 and Prompt-CAM offer interpretability to support human review

---

## Edge AI & On-Device Inference

From ICTC 2026 Edge AI workshop:

| Hardware | Notes |
|---|---|
| **Google Coral** | Creates problems — does not support many newer ops |
| **Arm Ethos U55** | Efficient NPU for microcontrollers |
| **AI HAT+ for Raspberry Pi** | Ed Miller (BearID) demo at ICTC |
| **DJI Manifold 3** | Onboard compute for DJI Matrice platforms — complex PSDK integration |
| **Quectel** | Cellular + satellite connectivity for remote camera traps |

**Compression strategies for remote deployment:**
- Autoencoders: compress images to latent representations on-device, decode in cloud (Sentinel system)
- HEF (Hailo Executable Format): much smaller than ONNX for Hailo NPU deployment

**Connectivity:**
- StarLink (Robin/Sparrow system) for remote connectivity
- "Swarm" satellite network (now defunct — acquired/shut down by SpaceX)

---

## Experiment Tracking & Reproducibility

### Tools
- **Weights & Biases** — hyperparameter sweeps, metric logging, model registry
- **MLflow** — open-source alternative: https://github.com/mlops-ai/mlops
- **DrivenData cookiecutter templates** — standardised project structure

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
HerdNet with DLA-169 backbone, FIDT density maps.

### Key Findings
- DLA-169 outperformed smaller DLA variants (DLA-34 through DLA-102)
- Augmentation strategy required per-dataset tuning (Fernandina ≠ Floreana)
- Photogrammetric deduplication across overlapping drone frames using DEM projection

### Validation
- Dynamic occupancy modelling (Bayesian / JAGS)
- Comparison against Zooniverse citizen science volunteer counts

---

## Conference Highlights — ICTC 2026

Notes from the International Conservation Technology Conference, Lima, 16–20 February 2026.

### Key Themes
- **Two-stage detection + human verification** repeated across camera traps, drones, whales
- **Data standards matter**: CamtrapDP, Darwin Core, IPT for interoperability
- **Confidence calibration**: 90% confidence alone is not enough to trigger field action
- **Edge AI**: hardware is maturing but software stack (PSDK, Coral) still has friction

### Notable Tools & Resources
- **PyTorch Wildlife** (Microsoft) — unified Python API for Megadetector + ReID
- **BioCLIP2** — biology-focused CLIP model: https://imageomics.github.io/bioclip-2/
- **Prompt-CAM** — interpretable Vision Transformers for species classification
- **Zamba / ZambaCloud** — low-label custom model training
- **RareSpot** (Zhang et al.) — rare species detection
- **RAPID** (Andras) — fast wildlife detection pipeline
- **FieldKit** — field data collection: https://www.dropbox.com/scl/fi/xmq6zhyqib0h0xtqctdz8/FieldKit_ICTC_Feb2026.pdf
- **IUCN Red List Dashboard**: https://red-list-dashboard.vercel.app/
- **EarthData / VEDA** (NASA): https://www.earthdata.nasa.gov/data/tools/veda

### Sentinel System (Conservation X Labs — Dante Wasmuht)
SD-card attachment for standard camera traps with on-device animal detection and satellite uplink.
Uses autoencoders for image compression. Floreana rat/cat project context.

### BearID (Ed Miller)
Individual re-identification of bears using metric learning / triplet training.
- Uses dlib foundation
- Arm Ethos U55, AI HAT+ for Raspberry Pi deployment
- Trapper / Trapper Keeper Fly Away Kit for field deployment

### Sparrow System (Rahul Dodhia)
Modular sensor units with "click boards". Robin variant connects via StarLink.
Integrates with PyTorch Wildlife.

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

| Paper | Key Finding | DOI |
|---|---|---|
| Varela-Jaramillo et al. (2023) — Iguana drone pilot | Drone counts were 14% closer to mark-resight estimates and 17–35% higher than ground counts, reaching previously inaccessible colonies | [10.1186/s12983-022-00478-5](https://doi.org/10.1186/s12983-022-00478-5) |
| Varela-Jaramillo et al. (2025) — Citizen science counts | 13,000+ volunteers achieved 91–92% counting accuracy; HDBSCAN aggregation outperformed standard majority-vote methods | [10.1038/s41598-025-08381-9](https://doi.org/10.1038/s41598-025-08381-9) |
| Varela-Jaramillo et al. (2025, submitted) — Lessons learned | Lessons from multi-year drone surveys of Galápagos marine iguanas across five islands | — |

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
| Marimo Notebooks                | https://molab.marimo.io |