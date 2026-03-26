# Aerial / Drone Wildlife Detection Datasets -- Comprehensive Reference

> Compiled 2026-03-26 for the FIT module (AI & UAV Wildlife Monitoring).
> Primary source: Dan Morris's curated list at <http://lila.science/aerialdata>
> ([GitHub repo](https://github.com/agentmorris/drone-wildlife-datasets)).

---

## Table of Contents

1. [Izembek Lagoon Waterfowl](#1-izembek-lagoon-waterfowl)
2. [Eikelboom Savanna Aerial Wildlife](#2-eikelboom-savanna-aerial-wildlife)
3. [WAID -- Wildlife Aerial Images from Drone](#3-waid--wildlife-aerial-images-from-drone)
4. [SAVMAP / Kuzikus (Kellenberger et al.)](#4-savmap--kuzikus-kellenberger-et-al)
5. [KABR -- Kenyan Animal Behavior Recognition](#5-kabr--kenyan-animal-behavior-recognition)
6. [NOAA Arctic Seals (2016)](#6-noaa-arctic-seals-2016)
7. [NOAA Arctic Seals (2019)](#7-noaa-arctic-seals-2019)
8. [African Wildlife (Ultralytics)](#8-african-wildlife-ultralytics)
9. [Aerial Elephant Dataset (AED)](#9-aerial-elephant-dataset-aed)
10. [Qian Penguin Counting](#10-qian-penguin-counting)
11. [Hayes Seabird Colonies (Penguins + Albatrosses)](#11-hayes-seabird-colonies-penguins--albatrosses)
12. [Penguin Watch / VGG Counting in the Wild](#12-penguin-watch--vgg-counting-in-the-wild)
13. [Iguanas From Above (Galapagos)](#13-iguanas-from-above-galapagos)
14. [BIRDSAI / Conservation Drones (Thermal)](#14-birdsai--conservation-drones-thermal)
15. [Snapshot Serengeti (Camera Trap)](#15-snapshot-serengeti-camera-trap)
16. [Delplanque African Mammals](#16-delplanque-african-mammals)
17. [Weinstein Global Bird Detection](#17-weinstein-global-bird-detection)
18. [Aerial Seabirds West Africa](#18-aerial-seabirds-west-africa)
19. [Big Bird -- Global Drone Bird Dataset](#19-big-bird--global-drone-bird-dataset)
20. [MMLA-Mpala (Kenya Ungulates)](#20-mmla-mpala-kenya-ungulates)
21. [MMLA-OPC (Ol Pejeta Conservancy)](#21-mmla-opc-ol-pejeta-conservancy)
22. [BuckTales (Blackbuck Antelopes)](#22-bucktales-blackbuck-antelopes)
23. [AWIR -- Aerial Wildlife Image Repository](#23-awir--aerial-wildlife-image-repository)
24. [Right Whale Recognition (NOAA/Kaggle)](#24-right-whale-recognition-noaakaggle)
25. [WildTrack Footprint Dataset](#25-wildtrack-footprint-dataset)
26. [Multi-species Wildlife in South African Savanna](#26-multi-species-wildlife-in-south-african-savanna)

---

## 1. Izembek Lagoon Waterfowl

| Field | Detail |
|---|---|
| **Full name** | Aerial Photo Imagery from Fall Waterfowl Surveys, Izembek Lagoon, Alaska, 2017-2019 |
| **Acronym** | -- (often called "Izembek Lagoon Waterfowl" or "USGS Geese") |
| **Citation** | Weiser EL, Flint PL, Marks DK, Shults BS, Wilson HM, Thompson SJ, Fischer JB (2022). Aerial photo imagery from fall waterfowl surveys, Izembek Lagoon, Alaska, 2017-2019. U.S. Geological Survey data release. https://doi.org/10.5066/P9UHP1LE |
| **Also used in** | POLO paper (Gigumay et al., 2024, arXiv:2410.11741) |
| **URL (original)** | https://alaska.usgs.gov/products/data.php?dataid=484 (1.82 TB) |
| **URL (LILA curated)** | https://lila.science/datasets/izembek-lagoon-waterfowl/ (124 GB) |
| **Images** | 110,667 RGB (original); 9,267 (LILA subset, 8688x5792 px) |
| **Annotations** | 631,349 points (original); 521,270 pseudo-bboxes on LILA (uniform-size boxes centered on point annotations) |
| **Classes** | 5: Brant goose (424,790), Canada goose (47,561), Gull (5,631), Emperor goose (2,013), Other (5,631) |
| **Annotation type** | Point (original) / Pseudo-bbox (LILA) |
| **Resolution / GSD** | 8688x5792 px; ~50 px typical bird size |
| **Vehicle** | Fixed-wing plane |
| **Format** | COCO Camera Traps JSON (LILA); CSV + CountThings JSON (original) |
| **License** | Unspecified, but public domain implied (USGS source) |
| **Freely downloadable** | Yes (GCP / AWS / Azure via LILA; HTTP via USGS) |

---

## 2. Eikelboom Savanna Aerial Wildlife

| Field | Detail |
|---|---|
| **Full name** | Improving the precision and accuracy of animal population estimates with aerial image object detection |
| **Acronym** | -- (commonly "Eikelboom savanna") |
| **Citation** | Eikelboom JA, Wind J, van de Ven E, Kenana LM, Schroder B, de Knegt HJ, van Langevelde F, Prins HH (2019). Improving the precision and accuracy of animal population estimates with aerial image object detection. *Methods in Ecology and Evolution*, 10(11):1875-87. |
| **URL** | https://data.4tu.nl/articles/dataset/Improving_the_precision_and_accuracy_of_animal_population_estimates_with_aerial_image_object_detection/12713903/1 |
| **Images** | 561 RGB images |
| **Annotations** | 4,305 bounding boxes |
| **Classes** | 3: elephant, zebra, giraffe |
| **Annotation type** | Bounding box |
| **Resolution / GSD** | ~50 px typical animal size |
| **Vehicle** | Fixed-wing plane |
| **Format** | CSV metadata |
| **Size** | 5.7 GB |
| **License** | CC0 (public domain) |
| **Freely downloadable** | Yes (HTTP from 4TU) |

---

## 3. WAID -- Wildlife Aerial Images from Drone

| Field | Detail |
|---|---|
| **Full name** | Wildlife Aerial Images from Drone |
| **Acronym** | WAID |
| **Citation** | Mou C, Liu T, Zhu C, Cui X (2023). WAID: A large-scale dataset for wildlife detection with drones. *Applied Sciences*, 13(18):10397. |
| **URL** | https://github.com/xiaohuicui/WAID/tree/main/WAID |
| **Images** | 14,366 images (typically 640x640 px) |
| **Annotations** | Bounding boxes (count not specified in source) |
| **Classes** | 6: sheep, cattle, seal, camel, kiang (Tibetan wild ass), zebra |
| **Annotation type** | Bounding box |
| **Resolution / GSD** | 640x640 px tiles; ~166 px typical animal size |
| **Vehicle** | Drone (UAV) |
| **Format** | YOLO format |
| **Size** | 1.5 GB |
| **License** | Unspecified |
| **Freely downloadable** | Yes (directly from GitHub) |

---

## 4. SAVMAP / Kuzikus (Kellenberger et al.)

| Field | Detail |
|---|---|
| **Full name** | SAVMAP -- Near real-time ultrahigh-resolution imaging from UAVs for sustainable land use management and biodiversity conservation in semi-arid savanna |
| **Acronym** | SAVMAP |
| **Citation** | Reinhard F, Parkan M, Produit T, Betschart S, Bacchilega B, Hauptfleisch ML, Meier P, Joost S, Tuia D. Near real-time ultrahigh-resolution imaging from unmanned aerial vehicles for sustainable land use management and biodiversity conservation in semi-arid savanna (SAVMAP). Zenodo. |
| **Related work** | Kellenberger B, Marcos D, Tuia D (2018). Detecting mammals in UAV images: Best practices to address a substantially imbalanced dataset with deep learning. *Remote Sensing of Environment*, 216:139-153. |
| **URL (Zenodo)** | https://zenodo.org/records/1204408 |
| **URL (Hugging Face)** | https://huggingface.co/datasets/fadel841/savmap (cropped version: 2000x2000, 3545 negative + 379 positive samples) |
| **Images** | 659 images (4000x3000 px) on Zenodo |
| **Annotations** | ~7,500 polygon annotations (approximating boxes), O(thousands) unique |
| **Classes** | 1: animal (single class) |
| **Annotation type** | Polygon (approximating bbox) |
| **Resolution / GSD** | 4000x3000 px; <100 px typical animal size |
| **Vehicle** | UAV (SenseFly drones) |
| **Format** | GeoJSON (Zenodo); YOLO (Hugging Face) |
| **Size** | 3 GB |
| **License** | AFL-3.0 (Academic Free License) |
| **Freely downloadable** | Yes |
| **Notes** | Kuzikus Wildlife Reserve, Namibia. AIDE platform was developed using this dataset. |

---

## 5. KABR -- Kenyan Animal Behavior Recognition

| Field | Detail |
|---|---|
| **Full name** | KABR: In-Situ Dataset for Kenyan Animal Behavior Recognition from Drone Videos |
| **Acronym** | KABR |
| **Citation** | Kholiavchenko M, Kline J, Kukushkin M, Brookes O, Stevens S, Duporge I, Sheets A, Babu RR, Banerji N, Campolongo E, Thompson M, Van Tiel N, Miliko J, Bessa E, Mirmehdi M, Schmid T, Berger-Wolf T, Rubenstein DI, Burghardt T, Stewart CV (2024). KABR: In-Situ Dataset for Kenyan Animal Behavior Recognition from Drone Videos. *WACV 2024 Workshop (CV4Smalls)*. |
| **URL (website)** | https://kabrdata.xyz/ |
| **URL (Hugging Face)** | https://huggingface.co/datasets/imageomics/KABR |
| **URL (raw videos)** | https://huggingface.co/datasets/imageomics/KABR-raw-videos |
| **Images/Video** | 10+ hours of extracted drone video clips; 5472x3078 px (5.4K); altitude 10-50 m |
| **Annotations** | Behavior labels per clip (not spatial detection annotations) |
| **Classes** | 8 behavior classes (7 behaviors + occluded); species: Grevy's zebra, plains zebra, giraffe |
| **Annotation type** | Temporal behavior labels (not bbox/point) |
| **Resolution / GSD** | 5472x3078 px (DJI Mavic 2S) |
| **Vehicle** | Drone (DJI Mavic 2S) |
| **Format** | Video clips + CSV behavior labels |
| **License** | Not explicitly stated |
| **Freely downloadable** | Yes (Google Drive, Hugging Face) |
| **Notes** | This is a behavior recognition dataset, not a detection dataset. Collected at Mpala Research Centre, Kenya. |

---

## 6. NOAA Arctic Seals (2016)

| Field | Detail |
|---|---|
| **Full name** | NOAA Arctic Seals |
| **Acronym** | -- |
| **Citation** | Alaska Fisheries Science Center. A Dataset for Machine Learning Algorithm Development. NOAA AFSC. |
| **URL** | https://lila.science/datasets/arcticseals |
| **Images** | ~1,000,000 thermal/RGB image pairs |
| **Annotations** | ~7,000 seal locations |
| **Classes** | Seal species (ringed seal, bearded seal, unknown seal, + pups) |
| **Annotation type** | Bounding box |
| **Resolution / GSD** | Variable (aerial survey imagery) |
| **Vehicle** | Fixed-wing plane |
| **Format** | CSV |
| **License** | CDLA-permissive (Community Data License Agreement) |
| **Freely downloadable** | Yes (via LILA -- GCP/AWS/Azure) |
| **Notes** | Extremely imbalanced (vast majority of images are empty). Good for multimodal fusion research (thermal + RGB). Chukchi Sea, Alaska. |

---

## 7. NOAA Arctic Seals (2019)

| Field | Detail |
|---|---|
| **Full name** | NOAA Arctic Seals 2019 |
| **Acronym** | -- |
| **Citation** | Alaska Fisheries Science Center (2021). A Dataset for Machine Learning Algorithm Development. |
| **URL** | https://lila.science/datasets/noaa-arctic-seals-2019/ |
| **Images** | 44,185 RGB + IR image pairs |
| **Annotations** | 14,311 bounding boxes (~14k on color, ~14k on thermal) |
| **Classes** | 6: ringed_seal, ringed_pup, bearded_seal, bearded_pup, unknown_seal, unknown_pup |
| **Annotation type** | Bounding box |
| **Resolution / GSD** | Variable; ~55 px typical animal size |
| **Vehicle** | Fixed-wing plane |
| **Format** | CSV |
| **Size** | ~1 TB |
| **License** | CDLA-permissive |
| **Freely downloadable** | Yes (via LILA -- azcopy) |

---

## 8. African Wildlife (Ultralytics)

| Field | Detail |
|---|---|
| **Full name** | African Wildlife Dataset |
| **Acronym** | -- |
| **Citation** | Distributed via Ultralytics; original source unclear (likely Roboflow community) |
| **URL** | https://docs.ultralytics.com/datasets/detect/african-wildlife/ |
| **Images** | 1,504 total (1,052 train / 225 val / 227 test) |
| **Annotations** | Bounding boxes (count unspecified) |
| **Classes** | 4: buffalo, elephant, rhino, zebra |
| **Annotation type** | Bounding box |
| **Resolution / GSD** | Variable; ground-level and some elevated perspectives (NOT strictly aerial/drone) |
| **Vehicle** | Mixed (camera trap-like, not aerial) |
| **Format** | YOLO format (african-wildlife.yaml) |
| **Size** | ~100 MB |
| **License** | AGPL-3.0 (Ultralytics license) |
| **Freely downloadable** | Yes (auto-downloads via Ultralytics API) |
| **Notes** | This is NOT an aerial/drone dataset -- it is ground-level wildlife imagery. Included here because it is commonly referenced for African wildlife detection benchmarks. |

---

## 9. Aerial Elephant Dataset (AED)

| Field | Detail |
|---|---|
| **Full name** | The Aerial Elephant Dataset |
| **Acronym** | AED |
| **Citation** | Naude J, Joubert D (2019). The Aerial Elephant Dataset: A New Public Benchmark for Aerial Object Detection. In *Proceedings of the IEEE/CVF CVPR Workshops* (pp. 48-55). |
| **URL** | https://zenodo.org/record/3234780 |
| **Images** | 2,074 RGB images |
| **Annotations** | 15,581 point annotations |
| **Classes** | 1: elephant |
| **Annotation type** | Point |
| **Resolution / GSD** | Variable; 2.4-13 cm GSD; ~75 px typical animal size |
| **Vehicle** | Drone |
| **Format** | CSV |
| **Size** | 16.3 GB |
| **License** | CC0 (public domain) |
| **Freely downloadable** | Yes (HTTP from Zenodo) |
| **Notes** | African bush elephants in southern and central Africa. |

---

## 10. Qian Penguin Counting

| Field | Detail |
|---|---|
| **Full name** | Counting animals in aerial images with a density map estimation model |
| **Acronym** | -- (commonly "Qian penguins") |
| **Citation** | Qian Y, Humphries G, Trathan P, Lowther A, Donovan C (2023). Counting animals in aerial images with a density map estimation model [Data set]. Zenodo. |
| **URL** | https://zenodo.org/record/7702635 |
| **Images** | 753 RGB orthorectified images |
| **Annotations** | 137,365 point annotations |
| **Classes** | 1: brush-tailed penguins |
| **Annotation type** | Point |
| **Resolution / GSD** | ~30 px typical animal size |
| **Vehicle** | Fixed-wing plane |
| **Format** | JSON (LabelBox standard) |
| **Size** | 300 MB |
| **License** | CC0 (public domain) |
| **Freely downloadable** | Yes (HTTP from Zenodo) |

---

## 11. Hayes Seabird Colonies (Penguins + Albatrosses)

| Field | Detail |
|---|---|
| **Full name** | Data from: Drones and deep learning produce accurate and efficient monitoring of large-scale seabird colonies |
| **Acronym** | -- (commonly "Hayes seabirds") |
| **Citation** | Hayes MC, Gray PC, Harris G, Sedgwick WC, Crawford VD, Chazal N, Crofts S, Johnston DW (2020). Data from: Drones and deep learning produce accurate and efficient monitoring of large-scale seabird colonies. Duke Research Repository. doi:10.7924/r4dn45v9g |
| **URL** | https://research.repository.duke.edu/concern/datasets/kp78gh20s?locale=en |
| **Images** | 3,947 RGB images |
| **Annotations** | 44,966 bounding boxes |
| **Classes** | 2: black-browed albatross, southern rockhopper penguin |
| **Annotation type** | Bounding box |
| **Resolution / GSD** | ~300 px typical animal size |
| **Vehicle** | Drone |
| **Format** | CSV |
| **Size** | 20.5 GB |
| **License** | CC0 (public domain) |
| **Freely downloadable** | Yes (via Globus) |

---

## 12. Penguin Watch / VGG Counting in the Wild

| Field | Detail |
|---|---|
| **Full name** | Counting in the Wild -- Penguin Dataset |
| **Acronym** | -- (commonly "VGG Penguins" or "Penguin Watch") |
| **Citation** | Arteta C, Lempitsky V, Zisserman A (2016). Counting in the Wild. *ECCV 2016*. |
| **URL** | https://www.robots.ox.ac.uk/~vgg/data/penguins/ |
| **Images** | ~82,000 images (fixed time-lapse cameras, NOT aerial/drone) |
| **Annotations** | Crowd-sourced dot (point) annotations via Zooniverse |
| **Classes** | Penguin species (multiple, from 40+ camera locations) |
| **Annotation type** | Point (crowd-sourced dots) |
| **Resolution / GSD** | Variable (fixed camera, not GSD in traditional sense) |
| **Vehicle** | Fixed camera (NOT aerial) |
| **Format** | Custom (X-Y coordinates per volunteer) |
| **License** | CC BY 4.0 |
| **Freely downloadable** | Yes (HTTP from VGG Oxford) |
| **Notes** | Camera-trap-style dataset from Antarctica. Not aerial but highly relevant for crowd counting methods and penguin population monitoring. |

---

## 13. Iguanas From Above (Galapagos)

| Field | Detail |
|---|---|
| **Full name** | Iguanas from Above |
| **Acronym** | IFA |
| **Citation** | Varela-Jaramillo A, Winkelmann C, et al. (2025). Citizen scientists reliably count endangered Galapagos marine iguanas from drone images. *Scientific Reports*. ; Also: Varela-Jaramillo A et al. (2022). A pilot study to estimate the population size of endangered Galapagos marine iguanas using drones. *Frontiers in Zoology*. |
| **URL (Zooniverse)** | https://www.zooniverse.org/projects/andreavarela89/iguanas-from-above |
| **URL (data)** | Not yet publicly released as a standalone ML dataset (data held by Leipzig University) |
| **Images** | 57,838 drone images across 7 islands |
| **Annotations** | 1,375,201 citizen-science classifications (point-based); Gold Standard expert set of 4,345 images |
| **Classes** | 1: marine iguana (Amblyrhynchus cristatus) |
| **Annotation type** | Point (citizen science dot annotations) |
| **Resolution / GSD** | High-resolution drone imagery (DJI); variable altitude |
| **Vehicle** | Drone |
| **Format** | Zooniverse export format |
| **License** | Not publicly specified |
| **Freely downloadable** | Partially -- Zooniverse classifications are accessible; raw imagery requires collaboration with Leipzig University |
| **Notes** | Key dataset for the FIT course (Winkelmann 2025 thesis). Combines citizen science + ML approaches. |

---

## 14. BIRDSAI / Conservation Drones (Thermal)

| Field | Detail |
|---|---|
| **Full name** | Benchmarking IR Dataset for Surveillance with Aerial Intelligence |
| **Acronym** | BIRDSAI |
| **Citation** | Bondi E, Jain R, Aggrawal P, Anand S, Hannaford R, Kapoor A, Piavis J, Shah S, Joppa L, Dilkina B, Tambe M. BIRDSAI: A Dataset for Detection and Tracking in Aerial Thermal Infrared Videos. |
| **URL** | https://lila.science/datasets/conservationdrones |
| **Images** | 61,994 real frames (48 videos) + 100,000 synthetic frames (124 videos from AirSim) |
| **Annotations** | 166,221 bounding boxes (with tracking IDs) |
| **Classes** | Detection: animal vs. human. Species: human, elephant, lion, giraffe, dog, crocodile, hippo, zebra, rhino |
| **Annotation type** | Bounding box + tracking ID (MOT format) |
| **Resolution / GSD** | Thermal infrared (TIR); ~35 px typical animal size |
| **Vehicle** | Drone |
| **Format** | CSV (MOT standard) |
| **Size** | 3.7 GB |
| **License** | CDLA-permissive |
| **Freely downloadable** | Yes (LILA -- HTTP / azcopy) |
| **Notes** | Southern Africa; nighttime TIR. Includes synthetic data. Anti-poaching surveillance use case. |

---

## 15. Snapshot Serengeti (Camera Trap)

| Field | Detail |
|---|---|
| **Full name** | Snapshot Serengeti |
| **Acronym** | SS |
| **Citation** | Swanson AB, Kosmala M, Lintott CJ, Simpson RJ, Smith A, Packer C (2015). Snapshot Serengeti, high-frequency annotated camera trap images of 40 mammalian species in an African savanna. *Scientific Data*, 2:150026. |
| **URL** | https://lila.science/datasets/snapshot-serengeti/ |
| **Images** | ~7.1 million images (2.65M sequences), seasons 1-11 |
| **Annotations** | Species-level labels for all sequences; ~150,000 bounding boxes on ~78,000 images |
| **Classes** | 61 categories (species-level); most common: wildebeest, zebra, Thomson's gazelle |
| **Annotation type** | Image-level species label + partial bounding box coverage |
| **Resolution / GSD** | Camera trap resolution (NOT aerial) |
| **Vehicle** | Camera trap (NOT aerial) |
| **Format** | COCO Camera Traps JSON |
| **License** | CDLA-permissive (Community Data License Agreement) |
| **Freely downloadable** | Yes (LILA -- per-season zipfiles, GCP/AWS/Azure) |
| **Notes** | Not an aerial dataset but one of the most important wildlife classification datasets. ~76% of images labeled as empty. Serengeti National Park, Tanzania. Useful for species classifier pre-training. |

---

## 16. Delplanque African Mammals

| Field | Detail |
|---|---|
| **Full name** | Multispecies detection and identification of African mammals in aerial imagery |
| **Acronym** | -- (commonly "Delplanque mammals") |
| **Citation** | Delplanque A, Foucher S, Lejeune P, Linchant J, Theau J (2022). Multispecies detection and identification of African mammals in aerial imagery using convolutional neural networks. *Remote Sensing in Ecology and Conservation*, 8(2):166-79. |
| **URL** | https://dataverse.uliege.be/file.xhtml?fileId=11098&version=1.0 |
| **Images** | 1,297 images (6000x4000 px each) |
| **Annotations** | 10,239 bounding boxes |
| **Classes** | 6: alcelaphinae, buffalo, kob, warthog, waterbuck, elephant |
| **Annotation type** | Bounding box |
| **Resolution / GSD** | 6000x4000 px; ~47 px typical animal size |
| **Vehicle** | Fixed-wing plane |
| **Format** | COCO JSON |
| **Size** | 12 GB |
| **License** | CC-BY-NC-SA 4.0 |
| **Freely downloadable** | Yes (Liege University Dataverse) |
| **Notes** | HerdNet was trained and evaluated on this dataset. |

---

## 17. Weinstein Global Bird Detection

| Field | Detail |
|---|---|
| **Full name** | A global model of bird detection in high resolution airborne images using computer vision |
| **Acronym** | -- (commonly "Weinstein birds" or "DeepForest bird data") |
| **Citation** | Weinstein BG, Garner L, Saccomanno VR, Steinkraus A, Ortega A, Brush K, Yenni G, McKellar AE, Converse R, Lippitt CD, Wegmann A (2022). A general deep learning model for bird detection in high-resolution airborne imagery. *Ecological Applications*, e2694. |
| **URL** | https://zenodo.org/record/5033174 |
| **Images** | 23,765 RGB images from 13 ecosystems |
| **Annotations** | 386,638 bounding boxes |
| **Classes** | 1: bird (single class, multi-ecosystem) |
| **Annotation type** | Bounding box |
| **Resolution / GSD** | Variable; ~35 px typical bird size |
| **Vehicle** | Variable (drone, plane, etc.) |
| **Format** | CSV |
| **Size** | 29 GB |
| **License** | CC BY 4.0 |
| **Freely downloadable** | Yes (HTTP from Zenodo) |
| **Notes** | Basis for the DeepForest bird detection pre-trained model. |

---

## 18. Aerial Seabirds West Africa

| Field | Detail |
|---|---|
| **Full name** | Aerial Seabirds West Africa |
| **Acronym** | -- |
| **Citation** | Kellenberger B, Veen T, Folmer E, Tuia D (2021). 21,000 birds in 4.5 h: efficient large-scale seabird detection with machine learning. *Remote Sensing in Ecology and Conservation*. |
| **URL** | https://lila.science/datasets/aerial-seabirds-west-africa/ |
| **Images** | Single aerial orthomosaic RGB image(s) |
| **Annotations** | 21,516 point annotations |
| **Classes** | 6: great white pelican, royal tern, caspian tern, slender-billed gull, gray-headed gull, great cormorant |
| **Annotation type** | Point |
| **Resolution / GSD** | ~30 px typical animal size |
| **Vehicle** | Fixed-wing plane |
| **Format** | CSV |
| **Size** | 2.2 GB |
| **License** | CDLA-permissive |
| **Freely downloadable** | Yes (LILA -- HTTP / azcopy) |

---

## 19. Big Bird -- Global Drone Bird Dataset

| Field | Detail |
|---|---|
| **Full name** | Big Bird: A global dataset of birds in drone imagery annotated to species level |
| **Acronym** | Big Bird |
| **Citation** | Wilson JP, Amano T, Bregnballe T, Corregidor-Castro A, Francis R, Gallego-Garcia D, Hodgson JC, Jones LR, Luque-Fernandez CR, Marchowski D, McEvoy J (2026). Big Bird: A global dataset of birds in drone imagery annotated to species level. *Remote Sensing in Ecology and Conservation*. |
| **URL (UQ eSpace)** | https://espace.library.uq.edu.au/view/UQ:27809f1 |
| **URL (LILA)** | https://lila.science/datasets/big-bird |
| **Images** | 23,865 images total; 4,824 with bounding box annotations |
| **Annotations** | 49,490 bird annotations (species-level) on the bbox subset |
| **Classes** | 100 bird species |
| **Annotation type** | Bounding box (subset) + species label |
| **Resolution / GSD** | Variable (drone imagery) |
| **Vehicle** | Drone |
| **Format** | LabelMe format |
| **Size** | ~45 GB |
| **License** | "Permitted reuse with acknowledgement" (UQ license) |
| **Freely downloadable** | Yes (UQ eSpace + LILA) |
| **Notes** | Largest species-level annotated drone bird dataset as of 2026. |

---

## 20. MMLA-Mpala (Kenya Ungulates)

| Field | Detail |
|---|---|
| **Full name** | MMLA Mpala Dataset |
| **Acronym** | MMLA-Mpala |
| **Citation** | Kline J, Kholiavchenko M, Zhong A, Ramirez M, Stevens S, Van Tiel N, Campolongo E, Thompson M, Ramesh Babu R, Banerji N, Sheets A, Magersupp M, Balasubramaniam S, Duporge I, Miliko J, Rosser N, Stewart CV, Berger-Wolf T, Rubenstein DI (2025). MMLA Mpala Dataset. |
| **URL** | https://huggingface.co/datasets/imageomics/mmla_mpala |
| **Images** | 130,102 images |
| **Annotations** | ~617,000 bounding boxes |
| **Classes** | 2: zebra, giraffe |
| **Annotation type** | Bounding box |
| **Resolution / GSD** | ~1138 px typical animal size (low-altitude) |
| **Vehicle** | Drone |
| **Format** | YOLO format |
| **Size** | 490 GB |
| **License** | CC0 1.0 (public domain) |
| **Freely downloadable** | Yes (Hugging Face) |
| **Notes** | Mpala Research Center, Kenya. Very large dataset. |

---

## 21. MMLA-OPC (Ol Pejeta Conservancy)

| Field | Detail |
|---|---|
| **Full name** | MMLA Ol Pejeta Conservancy (OPC) Dataset |
| **Acronym** | MMLA-OPC |
| **Citation** | Kline J, Nguyen Ngoc D, Duncan H, Rondeau Saint-Jean C, Maalouf G, Juma B, Kilwaya A, Vuyiya B, Irungu M, Njoroge W, Mutisya S, Guerin D, Costelloe B, Pastucha E, Hermansen J, Kjeld J, Watson M, Richardson T, Schultz Lundquist UP (2025). MMLA Ol Pejeta Conservancy (OPC) Dataset. |
| **URL** | https://huggingface.co/datasets/imageomics/mmla_opc |
| **Images** | 29,268 RGB frames |
| **Annotations** | ~163,000 bounding boxes |
| **Classes** | 1: zebra |
| **Annotation type** | Bounding box |
| **Vehicle** | Drone |
| **Format** | YOLO format |
| **Size** | 64 GB |
| **License** | CC0 1.0 (public domain) |
| **Freely downloadable** | Yes (Hugging Face) |
| **Notes** | Ol Pejeta Conservancy, Kenya. |

---

## 22. BuckTales (Blackbuck Antelopes)

| Field | Detail |
|---|---|
| **Full name** | BuckTales: A multi-UAV dataset for multi-object tracking and re-identification of wild antelopes |
| **Acronym** | BuckTales |
| **Citation** | Naik H, Yang J, Das D, Crofoot MC, Rathore A, Sridhar VH (2024). BuckTales: A multi-UAV dataset for multi-object tracking and re-identification of wild antelopes. *Advances in Neural Information Processing Systems*, 37:81992-82009. |
| **URL** | https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.JCZ9WK |
| **Images** | 320 images (detection set); 12 video sequences (tracking set) |
| **Annotations** | 18,400 boxes (detection); 1.2M boxes (tracking); 730 individual IDs (re-ID) |
| **Classes** | 6: drone, bird, unknown, shadow, female blackbuck, male blackbuck |
| **Annotation type** | Bounding box + tracking ID + individual re-ID |
| **Resolution / GSD** | ~40 px typical animal size |
| **Vehicle** | Multi-drone |
| **Format** | COCO + YOLO (detection); MOT (tracking) |
| **Size** | 80 GB |
| **License** | CC BY-SA 4.0 |
| **Freely downloadable** | Yes (Edmond / Max Planck repository) |
| **Notes** | Three-in-one dataset: detection, tracking, and re-identification. NeurIPS 2024. |

---

## 23. AWIR -- Aerial Wildlife Image Repository

| Field | Detail |
|---|---|
| **Full name** | Aerial Wildlife Image Repository |
| **Acronym** | AWIR |
| **Citation** | Boopalan SK et al. (2024). Aerial Wildlife Image Repository for animal monitoring with drones in the age of artificial intelligence. *Database* (Oxford), baae070. doi:10.1093/database/baae070 |
| **URL** | https://projectportal.gri.msstate.edu/awir/ |
| **Images** | 1,325 visible and thermal drone images |
| **Annotations** | 6,587 animal objects |
| **Classes** | 13 species (3 mammal, 10 bird); predominantly large birds and mammals |
| **Annotation type** | Bounding box + polygon |
| **Resolution / GSD** | Variable (drone imagery, visible + thermal) |
| **Vehicle** | Drone |
| **Format** | Platform-native (upload/download via web interface) |
| **License** | Open access (publicly available for minimum 2 years from 2024 publication) |
| **Freely downloadable** | Yes (via AWIR web portal) |
| **Notes** | North America focused. Dynamic repository -- users can upload and annotate new images. Mississippi State University. |

---

## 24. Right Whale Recognition (NOAA/Kaggle)

| Field | Detail |
|---|---|
| **Full name** | NOAA Right Whale Recognition |
| **Acronym** | -- |
| **Citation** | NOAA Fisheries / Kaggle competition |
| **URL** | https://www.kaggle.com/c/noaa-right-whale-recognition |
| **Images** | 11,468 RGB images |
| **Annotations** | 4,544 individual IDs (re-identification task) |
| **Classes** | 1: right whale (individual-level labels) |
| **Annotation type** | Individual ID (re-identification, not detection boxes) |
| **Resolution / GSD** | ~1500 px typical animal size |
| **Vehicle** | Helicopter |
| **Format** | CSV |
| **Size** | 10 GB |
| **License** | Unspecified (public domain implied, NOAA source) |
| **Freely downloadable** | Yes (Kaggle) |

---

## 25. WildTrack Footprint Dataset

| Field | Detail |
|---|---|
| **Full name** | WildTrack -- Footprint Identification Technology |
| **Acronym** | WildTrack/FIT |
| **Citation** | WildTrack (wildtrack.org) |
| **URL** | https://www.wildtrack.org/our-work/fit-technology |
| **Images** | 1,928 footprint images |
| **Annotations** | Individual ID labels |
| **Classes** | 11 species, 7-13 known individuals per species |
| **Annotation type** | Individual ID (footprint-based, not aerial detection) |
| **Resolution / GSD** | N/A (ground-level footprint photographs) |
| **Vehicle** | Ground camera |
| **License** | Not publicly specified |
| **Freely downloadable** | By request |
| **Notes** | Non-invasive wildlife monitoring via footprint tracking. NOT an aerial detection dataset but relevant to wildlife monitoring curriculum. The EPFL WILDTRACK multi-camera pedestrian dataset (Chavdarova et al., CVPR 2018) is a different, unrelated dataset. |

---

## 26. Multi-species Wildlife in South African Savanna

| Field | Detail |
|---|---|
| **Full name** | Evaluating machine learning models for multi-species wildlife detection and identification on remote sensed nadir imagery in South African savanna |
| **Acronym** | -- |
| **Citation** | Allin P, Seydou F, Frans R, Davies A, Leslie A (2026). Evaluating machine learning models for multi-species wildlife detection and identification on remote sensed nadir imagery in South African savanna. *Wildlife Biology*, e01523. |
| **URL** | https://datadryad.org/dataset/doi:10.5061/dryad.9ghx3ffvc |
| **Images** | TBD (very large dataset) |
| **Annotations** | TBD |
| **Classes** | TBD (multi-species) |
| **Annotation type** | Bounding box |
| **Vehicle** | UAV |
| **Format** | COCO + Label Studio format |
| **Size** | ~450 GB |
| **License** | "Public domain" (per Dryad, exact terms TBC) |
| **Freely downloadable** | Yes (Dryad) |
| **Notes** | Very recent (2026). Detailed metadata still being catalogued. |

---

## Summary Comparison Table

| # | Dataset | Images | Annotations | Classes | Ann. Type | Vehicle | License | Free? |
|---|---------|--------|-------------|---------|-----------|---------|---------|-------|
| 1 | Izembek Lagoon | 9,267 (LILA) / 110,667 (full) | 521K bbox / 631K pts | 5 | Point/Bbox | Plane | Public domain | Yes |
| 2 | Eikelboom Savanna | 561 | 4,305 | 3 | Bbox | Plane | CC0 | Yes |
| 3 | WAID | 14,366 | Boxes (unspec.) | 6 | Bbox | Drone | Unspecified | Yes |
| 4 | SAVMAP/Kuzikus | 659 | ~7,500 | 1 | Polygon | UAV | AFL-3.0 | Yes |
| 5 | KABR | 10+ hrs video | Behavior labels | 8 behaviors, 3 spp | Temporal | Drone | Unspecified | Yes |
| 6 | NOAA Seals 2016 | ~1M pairs | ~7,000 | 6 | Bbox | Plane | CDLA-perm. | Yes |
| 7 | NOAA Seals 2019 | 44,185 pairs | 14,311 | 6 | Bbox | Plane | CDLA-perm. | Yes |
| 8 | African Wildlife | 1,504 | Boxes (unspec.) | 4 | Bbox | Ground | AGPL-3.0 | Yes |
| 9 | AED (Elephants) | 2,074 | 15,581 | 1 | Point | Drone | CC0 | Yes |
| 10 | Qian Penguins | 753 | 137,365 | 1 | Point | Plane | CC0 | Yes |
| 11 | Hayes Seabirds | 3,947 | 44,966 | 2 | Bbox | Drone | CC0 | Yes |
| 12 | VGG Penguins | ~82,000 | Crowd dots | Multiple | Point | Fixed cam | CC BY 4.0 | Yes |
| 13 | Iguanas From Above | 57,838 | 1.375M classif. | 1 | Point | Drone | Not specified | Partial |
| 14 | BIRDSAI | 62K + 100K synth | 166,221 | 9 species | Bbox+Track | Drone | CDLA-perm. | Yes |
| 15 | Snapshot Serengeti | 7.1M | 150K bbox; all seq. labeled | 61 | Label+Bbox | Cam trap | CDLA-perm. | Yes |
| 16 | Delplanque Mammals | 1,297 | 10,239 | 6 | Bbox | Plane | CC-BY-NC-SA | Yes |
| 17 | Weinstein Birds | 23,765 | 386,638 | 1 | Bbox | Variable | CC BY 4.0 | Yes |
| 18 | Seabirds W. Africa | Orthomosaic | 21,516 | 6 | Point | Plane | CDLA-perm. | Yes |
| 19 | Big Bird | 23,865 (4,824 w/box) | 49,490 | 100 spp | Bbox+Label | Drone | Reuse w/ack. | Yes |
| 20 | MMLA-Mpala | 130,102 | ~617,000 | 2 | Bbox | Drone | CC0 | Yes |
| 21 | MMLA-OPC | 29,268 | ~163,000 | 1 | Bbox | Drone | CC0 | Yes |
| 22 | BuckTales | 320 (det.) + video | 18.4K (det.); 1.2M (track) | 6 | Bbox+Track+ReID | Multi-drone | CC BY-SA 4.0 | Yes |
| 23 | AWIR | 1,325 | 6,587 | 13 | Bbox+Polygon | Drone | Open access | Yes |
| 24 | Right Whale | 11,468 | 4,544 IDs | 1 | Individual ID | Helicopter | Public domain | Yes |
| 25 | WildTrack FIT | 1,928 | Individual IDs | 11 spp | Individual ID | Ground | By request | Partial |
| 26 | SA Savanna (2026) | TBD (large) | TBD | Multi-spp | Bbox | UAV | Public domain | Yes |

---

## Key Curated Resources

- **Dan Morris's drone-wildlife-datasets list**: https://github.com/agentmorris/drone-wildlife-datasets (canonical, regularly updated)
  - Permalink: http://lila.science/aerialdata
- **LILA BC (Labeled Information Library of Alexandria)**: https://lila.science/datasets/
- **LILA other datasets list**: https://lila.science/otherdatasets/
- **Kellenberger CV for Wildlife Aerial Imagery page**: https://bkellenb.github.io/cv-for-wildlife-aerial-imagery/
- **Wildlife Datasets (re-ID focused)**: https://wildlifedatasets.github.io/wildlife-datasets/datasets/

---

## Notes for Course Use

1. **Best for detection practicals (Week 1)**: Eikelboom (#2), AED (#9), Delplanque (#16), WAID (#3), and Izembek (#1) are small enough to download and use in a classroom setting.
2. **Best for classification**: Snapshot Serengeti (#15) for species classification; Big Bird (#19) for bird species.
3. **Best for point-based detection (HerdNet/POLO)**: Izembek (#1), AED (#9), Qian Penguins (#10), Seabirds West Africa (#18).
4. **Best for thermal/multi-modal**: BIRDSAI (#14), NOAA Arctic Seals (#6, #7).
5. **Largest bbox-annotated aerial datasets**: MMLA-Mpala (#20, 617K boxes), Izembek (#1, 521K pseudo-boxes), Weinstein Birds (#17, 387K boxes).
6. **Course-specific (Iguanas From Above)**: Dataset #13 is directly relevant to the course via Winkelmann (2025). Not fully public yet.


#### TODO
EW-IL22 (Izembek Lagoon Waterfowl)

LILA BC: https://lila.science/datasets/izembek-lagoon-waterfowl/
ScienceBase (original USGS release): https://www.sciencebase.gov/catalog/item/62438853d34e21f8275ffd67

JE-TL19 (Eikelboom, Laikipia-Samburu)

4TU.ResearchData: https://doi.org/10.4121/uuid:ba99a206-3e5a-4673-b830-b5c866445b8c

BK-L23 (Koger, Laikipia)

Data on EDMOND (Max Planck): https://doi.org/10.17617/3.EMRZGH
Code + worked examples: https://github.com/benkoger/overhead-video-worked-examples
Zenodo (code archive): https://doi.org/10.5281/zenodo.7622940