# Aerial / Drone / Satellite Vehicle Detection Datasets

Compiled 2026-03-26. Covers major publicly available datasets for vehicle detection
in aerial, drone, and satellite imagery.

---

## Summary Table

| # | Dataset | Year | Images | Instances | Classes | Annotation | GSD / Resolution | License | Free DL |
|---|---------|------|--------|-----------|---------|------------|-----------------|---------|---------|
| 1 | DOTA v1.0 | 2018 | 2,806 | 188,282 | 15 | OBB (8-point) | 800--20,000 px; mixed GSD | Academic only | Yes |
| 2 | DOTA v1.5 | 2019 | 2,806 | 403,318 | 16 | OBB (8-point) | same as v1.0 | Academic only | Yes |
| 3 | DOTA v2.0 | 2021 | 11,268 | 1,793,658 | 18 | OBB (8-point) | same as v1.0 | Academic only | Yes |
| 4 | VisDrone-DET 2019 | 2019 | 10,209 | ~540,000 | 10 | HBB | 1920x1080 typ. | Academic only | Yes |
| 5 | VEDAI | 2016 | 1,210 | 3,640 | 9 | OBB | 12.5 cm (1024) / 25 cm (512) | Academic only | Yes |
| 6 | COWC | 2016 | 6 large scenes | 32,716 cars | 1 (car) | Point (center) | 15 cm | AGPL-3.0 (code) | Yes |
| 7 | CARPK | 2017 | 1,448 | ~89,777 | 1 (car) | HBB | Drone ~40 m altitude | EULA required | Yes* |
| 8 | PUCPR+ | 2017 | 125 | 16,456 | 1 (car) | HBB | 720x1280 px; fixed cam | EULA required | Yes* |
| 9 | UAVDT | 2018 | 80,000 (frames) | 835,879 | 3 (car, truck, bus) | HBB + attributes | 1080x540 px | CC BY 4.0 | Yes |
| 10 | DLR 3K Munich | 2015 | 20 | 3,472 | 2 (car, truck) | OBB | 13 cm | Public | Yes |
| 11 | EAGLE (DLR) | 2020 | 318 (8,820 tiles) | 215,986 | 2 (small-veh, large-veh) | OBB (4-point + orient.) | 5--45 cm | On request (DLR) | Yes* |
| 12 | xView | 2018 | 1,127 | 601,726 (all) | 60 (many vehicle) | HBB | 30 cm (WorldView-3) | CC BY-NC-SA 4.0 | Yes |
| 13 | DIOR | 2020 | 23,463 | 192,472 | 20 | HBB (VOC) | 800x800 px; 0.5--30 m | Academic only | Yes |
| 14 | NWPU VHR-10 | 2016 | 800 | 3,775 | 10 | HBB + masks | 0.08--2 m | Publicly available | Yes |
| 15 | iSAID | 2019 | 2,806 | 655,451 | 15 | Instance segmentation | same images as DOTA v1.0 | Academic only | Yes |
| 16 | FAIR1M | 2021 | 15,000+ | 1,000,000+ | 37 (5 super-cat) | OBB (rotated) | 0.3--0.8 m (GF / GE) | Restricted / challenge | Partial |
| 17 | HRSC2016 | 2016 | 1,070 | 2,976 | Ships (3-level) | OBB + HBB + seg | 0.4--2 m | Available on request | Yes |
| 18 | SODA-A | 2022 | 2,513 | 872,069 | 9 | OBB (oriented) | ~4761x2777 px avg | Not specified | Yes |
| 19 | DroneVehicle | 2022 | 56,878 (28,439 pairs) | 953,087 | 5 (car,truck,bus,van,freight) | OBB | Drone 80--120 m alt. | Academic (VisDrone) | Yes |
| 20 | VME | 2025 | 4,000+ tiles | 100,000+ | 1 (vehicle) | OBB + HBB | 30--50 cm (Maxar) | CC BY 4.0 | Yes |
| 21 | ITCVD | 2018 | 173 | 29,088 | 1 (vehicle) | HBB + point | 10 cm (nadir) | Public (DANS) | Yes |
| 22 | Stanford Drone | 2016 | 20,000+ targets | ~20,000 agents | 6 (ped,bike,skate,car,bus,golf) | BBox tracks | Top-view, campus | Academic | Yes |

**Legend:** HBB = Horizontal Bounding Box, OBB = Oriented Bounding Box, GSD = Ground Sampling Distance.
*Yes\** = Free but requires EULA form or registration.

---

## Detailed Dataset Descriptions

### 1. DOTA (Dataset for Object deTection in Aerial images)

- **Versions:** v1.0 (CVPR 2018), v1.5 (2019), v2.0 (2021)
- **Paper:** Xia et al., "DOTA: A Large-scale Dataset for Object Detection in Aerial Images", CVPR 2018
- **arXiv:** https://arxiv.org/abs/1711.10398
- **URL:** https://captain-whu.github.io/DOTA/dataset.html
- **Images:** v1.0: 2,806 | v1.5: 2,806 (same) | v2.0: 11,268
- **Instances:** v1.0: 188,282 | v1.5: 403,318 (includes tiny <10px) | v2.0: 1,793,658
- **Classes (vehicle-relevant):** small-vehicle, large-vehicle, ship, harbor, helicopter, plane
- **All v1.0 classes (15):** plane, ship, storage-tank, baseball-diamond, tennis-court, basketball-court, ground-track-field, harbor, bridge, large-vehicle, small-vehicle, helicopter, roundabout, soccer-ball-field, swimming-pool
- **v1.5 adds:** container-crane (16 total)
- **v2.0 adds:** container-crane, airport, helipad (18 total)
- **Annotation:** Oriented bounding box (arbitrary quadrilateral, 8 d.o.f.)
- **Image size:** 800x800 to 20,000x20,000 px
- **Sources:** Google Earth, GF-2 satellite, JL-1 satellite, CycloMedia aerial
- **License:** Academic use only; commercial use prohibited
- **Download:** Free via official website (registration required)

---

### 2. VisDrone-DET 2019

- **Paper:** Du et al., "VisDrone-DET2019: The Vision Meets Drone Object Detection in Image Challenge Results", ICCV-W 2019
- **URL:** https://github.com/VisDrone/VisDrone-Dataset
- **Website:** http://www.aiskyeye.com
- **Images:** 10,209 total (train: 6,471 | val: 548 | test-dev: 1,610 | test-challenge: 1,580)
- **Instances:** ~540,000 bounding boxes
- **Classes (10):** pedestrian, person, car, van, bus, truck, motor, bicycle, awning-tricycle, tricycle
- **Vehicle classes (4):** car, van, bus, truck
- **Annotation:** Horizontal bounding box + occlusion/truncation attributes
- **Resolution:** Varies; captured by various drone-mounted cameras across 14 Chinese cities
- **License:** Academic use only
- **Download:** Free via GitHub / aiskyeye.com (registration)

---

### 3. VEDAI (Vehicle Detection in Aerial Imagery)

- **Paper:** Razakarivony & Jurie, "Vehicle Detection in Aerial Imagery: A small target detection benchmark", JVCIR 2016
- **URL:** https://downloads.greyc.fr/vedai/
- **Images:** 1,210 images in two resolutions (512x512 and 1024x1024), both color and infrared
- **Instances:** 3,640 vehicle instances
- **Classes (9):** boat, car, camping-car, plane, pick-up, tractor, truck, van, other
- **Annotation:** Oriented bounding box
- **GSD:** 12.5 cm/px (1024 version), 25 cm/px (512 version)
- **Source:** Utah AGRC aerial imagery
- **License:** Academic and non-commercial use only
- **Download:** Freely available at https://downloads.greyc.fr/vedai/

---

### 4. COWC (Cars Overhead With Context)

- **Paper:** Mundhenk et al., "A Large Contextual Dataset for Classification, Detection and Counting of Cars with Deep Learning", ECCV 2016
- **URL:** https://gdo152.llnl.gov/cowc/
- **GitHub:** https://github.com/LLNL/cowc
- **Images:** 6 large-area scenes (Toronto, Selwyn NZ, Potsdam, Vaihingen, Columbus, Utah)
- **Instances:** 32,716 unique annotated cars + 58,247 negative examples
- **Classes:** 1 (car); COWC-M variant adds 4 sub-classes (sedan, pickup, other, unknown)
- **Annotation:** Center point per car (not bounding box)
- **GSD:** 15 cm/px (all EO)
- **License:** AGPL-3.0 (code); data freely available via FTP
- **Download:** Free via FTP from LLNL

---

### 5. CARPK (Car Parking Lot Dataset)

- **Paper:** Hsieh et al., "Drone-based Object Counting by Spatially Regularized Regional Proposal Network", ICCV 2017
- **URL:** https://lafi.github.io/LPN/
- **Images:** 1,448 (train: 989, test: 459)
- **Instances:** ~89,777 cars across 4 parking lots
- **Classes:** 1 (car)
- **Annotation:** Horizontal bounding box (Pascal VOC format)
- **Source:** DJI Phantom 3 Professional drone, ~40 m altitude
- **License:** EULA required before download
- **Download:** Free after EULA submission; password emailed

---

### 6. PUCPR+ (Pontifical Catholic University of Parana+)

- **Paper:** Hsieh et al., "Drone-based Object Counting by Spatially Regularized Regional Proposal Network", ICCV 2017
- **URL:** https://lafi.github.io/LPN/
- **Images:** 125 (train: 100, test: 25)
- **Instances:** 16,456 cars
- **Classes:** 1 (car)
- **Annotation:** Horizontal bounding box
- **Resolution:** 720x1280 px; fixed camera from 10th floor building
- **License:** EULA required before download
- **Download:** Free after EULA submission; password emailed

---

### 7. UAVDT (UAV Detection and Tracking)

- **Paper:** Du et al., "The Unmanned Aerial Vehicle Benchmark: Object Detection and Tracking", ECCV 2018
- **URL:** https://zenodo.org/records/14575517
- **Also:** https://datasetninja.com/uavdt
- **Images:** ~80,000 representative frames from 100 video sequences (>10 hours)
- **Instances:** 835,879 labeled objects
- **Classes (3):** car, truck, bus (+ "other")
- **Annotation:** HBB + 14 attribute types (weather, altitude, camera view, occlusion, etc.)
- **Resolution:** 1080x540 px JPEG, 30 fps
- **Source:** UAV platform over urban areas
- **License:** CC BY 4.0
- **Download:** Freely available via Zenodo, Kaggle, dataset-ninja

---

### 8. DLR 3K Munich Vehicle Dataset

- **Paper:** Liu & Mattyus, "Fast Multiclass Vehicle Detection on Aerial Images", IEEE GRSL 2015
- **URL:** https://www.dlr.de/eoc/en/desktopdefault.aspx/tabid-5431/9230_read-42467/
- **Images:** 20 aerial images (10 train, 10 test), 5616x3744 px each
- **Instances:** 3,472 (3,418 cars + 54 trucks)
- **Classes (2):** car, truck
- **Annotation:** Oriented (rotatable) bounding box
- **GSD:** ~13 cm (captured at 1000 m altitude)
- **Source:** DLR 3K camera system over Munich, Germany
- **License:** Publicly available for research
- **Download:** Free from DLR website

---

### 9. EAGLE (DLR)

- **Paper:** Azimi et al., "EAGLE: Large-scale Vehicle Detection Dataset in Real-World Scenarios using Aerial Imagery", ICPR 2020
- **arXiv:** https://arxiv.org/abs/2007.06124
- **URL:** https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/eagle
- **Images:** 318 large images -> 8,820 tiles of 936x936 px
- **Instances:** 215,986 annotated vehicles
- **Classes (2):** small-vehicle (car, van, SUV, ambulance, police), large-vehicle (truck, bus, minibus, fire engine, construction)
- **Annotation:** Oriented bounding box (4 corner points + orientation angle 0-360 deg)
- **GSD:** 5--45 cm (multiple sensors, altitudes, campaigns 2006--2019)
- **Tasks:** 3 tasks: HBB detection, rotated BB detection, oriented BB detection
- **License:** Available on request from DLR
- **Download:** Via DLR public datasets page (may require request form)

---

### 10. xView

- **Paper:** Lam et al., "xView: Objects in Context in Overhead Imagery", 2018
- **arXiv:** https://arxiv.org/abs/1802.07856
- **URL:** https://xviewdataset.org/
- **Challenge:** https://www.diu.mil/ai-xview-challenge
- **Images:** 1,127 (train: 846, val: 281, test: held out)
- **Instances:** 601,726 across all 60 classes
- **Classes (60):** Including many vehicle types: small-car, bus, pickup-truck, utility-truck, cargo-truck, truck-w/box, truck-w/flatbed, truck-w/liquid, dump-truck, crane-truck, railway-vehicle, passenger-vehicle, motorboat, sailboat, tugboat, barge, fishing-vessel, ferry, yacht, container-ship, oil-tanker, engineering-vehicle, tower-crane, excavator, etc.
- **Annotation:** Horizontal bounding box
- **GSD:** 0.3 m (WorldView-3 satellite)
- **Format:** RGB and 8-band multispectral
- **License:** CC BY-NC-SA 4.0
- **Download:** Free via xviewdataset.org (registration required)

---

### 11. DIOR (Detection in Optical Remote sensing)

- **Paper:** Li et al., "Object Detection in Optical Remote Sensing Images: A Survey and A New Benchmark", ISPRS J. 2020
- **arXiv:** https://arxiv.org/abs/1909.00133
- **URL:** https://ieee-dataport.org/documents/dior ; also on Hugging Face, Mendeley Data
- **Images:** 23,463 (800x800 px)
- **Instances:** 192,472
- **Classes (20):** airplane, airport, baseball-field, basketball-court, bridge, chimney, dam, expressway-service-area, expressway-toll-station, golf-course, ground-track-field, harbor, overpass, ship, stadium, storage-tank, tennis-court, train-station, **vehicle**, windmill
- **Vehicle classes:** 1 explicit "vehicle" class
- **Annotation:** Horizontal bounding box (Pascal VOC XML)
- **GSD:** Varies (0.5--30 m); Google Earth imagery
- **License:** Academic use (IEEE DataPort requires subscription for some access)
- **Download:** Available via IEEE DataPort, Mendeley Data, Hugging Face

---

### 12. NWPU VHR-10

- **Paper:** Cheng et al., "Multi-class geospatial object detection and geographic image classification based on collection of part detectors", ISPRS J. 2014
- **URL:** https://www.kaggle.com/datasets/larbisck/nwpu-vhr-10
- **Also:** Google Drive, TorchGeo
- **Images:** 800 (650 positive + 150 negative)
- **Instances:** 3,775 total (477 vehicles, 757 airplanes, 302 ships, etc.)
- **Classes (10):** airplane, ship, storage-tank, baseball-diamond, tennis-court, basketball-court, ground-track-field, harbor, bridge, **vehicle**
- **Annotation:** HBB + instance segmentation masks (COCO format available)
- **GSD:** 0.5--2 m (Google Earth) and 0.08 m (Vaihingen)
- **License:** Publicly available (referenced as MIT in TorchGeo)
- **Download:** Free via Kaggle, Google Drive, TorchGeo

---

### 13. iSAID (Instance Segmentation in Aerial Images Dataset)

- **Paper:** Zamir et al., "iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images", CVPR-W 2019
- **arXiv:** https://arxiv.org/abs/1905.12886
- **URL:** https://captain-whu.github.io/iSAID/dataset.html
- **GitHub:** https://github.com/CAPTAIN-WHU/iSAID_Devkit
- **Images:** 2,806 (same imagery as DOTA v1.0)
- **Instances:** 655,451
- **Classes (15):** small-vehicle, large-vehicle, plane, ship, helicopter, harbor, bridge, storage-tank, baseball-diamond, tennis-court, basketball-court, ground-track-field, soccer-ball-field, roundabout, swimming-pool
- **Vehicle classes:** small-vehicle, large-vehicle
- **Annotation:** Pixel-level instance segmentation masks
- **Resolution:** Same as DOTA (800x800 to 20,000x20,000 px)
- **License:** Academic use only; commercial use prohibited
- **Download:** Free via Google Drive or Baidu Drive

---

### 14. FAIR1M

- **Paper:** Sun et al., "FAIR1M: A Benchmark Dataset for Fine-grained Object Recognition in High-Resolution Remote Sensing Imagery", ISPRS J. 2022
- **arXiv:** https://arxiv.org/abs/2103.05569
- **URL:** https://gaofen-challenge.com/benchmark
- **Also:** Hugging Face (partial)
- **Images:** 15,000+
- **Instances:** 1,000,000+
- **Classes (37 fine-grained in 5 super-categories):**
  - Ships (9): motorboat, fishing-boat, tugboat, engineering-ship, liquid-cargo-ship, dry-cargo-ship, warship, passenger-ship, other-ship
  - Vehicles (10): small-car, bus, cargo-truck, dump-truck, van, trailer, tractor, truck-tractor, excavator, other-vehicle
  - Airplanes (11): Boeing-737, Boeing-747, Boeing-777, Boeing-787, Airbus-A220, Airbus-A321, Airbus-A330, Airbus-A350, C919, ARJ21, other-airplane
  - Courts (4): baseball-field, basketball-court, football-field, tennis-court
  - Roads (3): intersection, bridge, roundabout
- **Annotation:** Oriented (rotated) bounding box (Pascal VOC XML)
- **GSD:** 0.3--0.8 m (Gaogen / GF satellites + Google Earth)
- **License:** Restricted; linked to Gaofen Challenge
- **Download:** Partially available (train/part1 public; full access via challenge registration)

---

### 15. HRSC2016 (High-Resolution Ship Collection)

- **Paper:** Liu et al., "A High Resolution Optical Satellite Image Dataset for Ship Recognition and Some New Baselines", ICPRAM 2017
- **URL:** https://www.kaggle.com/datasets/guofeng/hrsc2016
- **Also:** IEEE DataPort, GitHub (HRSC2016-MS variant)
- **Images:** 1,070
- **Instances:** 2,976 ships
- **Classes:** 3-level hierarchy: ship -> ship-category -> ship-type (warship, merchant, civilian, aircraft-carrier, etc.)
- **Annotation:** HBB + OBB + pixel-level segmentation (test set)
- **Image size:** 300x300 to 1500x900 px
- **GSD:** 0.4--2 m
- **License:** Available (IEEE DataPort subscription or Kaggle)
- **Download:** Free via Kaggle; IEEE DataPort requires subscription

---

### 16. SODA-A (Small Object Detection dAtaset -- Aerial)

- **Paper:** Cheng et al., "Towards Large-Scale Small Object Detection: Survey and Benchmarks", TPAMI 2023
- **arXiv:** https://arxiv.org/abs/2207.14096
- **URL:** https://shaunyuan22.github.io/SODA/
- **Also:** Kaggle, Hugging Face
- **Images:** 2,513 (avg resolution ~4761x2777 px)
- **Instances:** 872,069
- **Classes (9):** airplane, helicopter, small-vehicle, large-vehicle, ship, container, storage-tank, swimming-pool, windmill
- **Vehicle classes:** small-vehicle, large-vehicle
- **Annotation:** Oriented bounding box
- **Source:** Google Earth imagery
- **License:** Not explicitly specified; available on multiple platforms
- **Download:** Free via official site, Kaggle, Hugging Face

---

### 17. DroneVehicle

- **Paper:** Sun et al., "Drone-based RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware Learning", IEEE T-CSVT 2022
- **arXiv:** https://arxiv.org/abs/2003.02437
- **GitHub:** https://github.com/VisDrone/DroneVehicle
- **Images:** 56,878 total (28,439 RGB-Infrared pairs)
- **Instances:** 953,087 (across both modalities)
- **Classes (5):** car, truck, bus, van, freight-car
- **Annotation:** Oriented bounding box
- **Source:** Drone at 80--120 m altitude; vertical and oblique views (15/30/45 deg); day/night/dark-night
- **License:** Academic use (VisDrone project)
- **Download:** Free via GitHub download links

---

### 18. VME (Vehicles in the Middle East)

- **Paper:** Alemadi et al., "VME: A Satellite Imagery Dataset and Benchmark for Detecting Vehicles in the Middle East and Beyond", Scientific Data 2025
- **URL:** https://zenodo.org/records/14185684
- **GitHub:** https://github.com/nalemadi/VME_CDSI_dataset_benchmark
- **Images:** 4,000+ tiles (512x512 px)
- **Instances:** 100,000+ vehicles
- **Classes:** 1 (vehicle)
- **Annotation:** OBB (YOLO format) + HBB (MS-COCO JSON)
- **GSD:** 30--50 cm (Maxar satellite)
- **Coverage:** 54 cities across 12 Middle Eastern countries
- **License:** CC BY 4.0
- **Download:** Free via Zenodo
- **Note:** Accompanied by CDSI benchmark combining VME + xView + DOTA-v2.0 + VEDAI + DIOR + FAIR1M-2.0

---

### 19. ITCVD (ITC Vehicle Detection)

- **Paper:** Yang et al., "Vehicle Detection in Aerial Images", IEEE GRSL 2018
- **arXiv:** https://arxiv.org/abs/1801.07339
- **URL:** https://research.utwente.nl/en/datasets/itcvd-dataset
- **Also:** https://phys-techsciences.datastations.nl/dataset.xhtml?persistentId=doi:10.17026/dans-xnc-h2fu
- **Images:** 173 (train: 135, test: 38)
- **Instances:** 29,088 (train: 23,543, test: 5,545)
- **Classes:** 1 (vehicle)
- **Annotation:** HBB + center-point annotations (for counting)
- **GSD:** 10 cm (nadir); airplane platform at ~330 m, Enschede, Netherlands
- **License:** Publicly available via DANS
- **Download:** Free

---

### 20. Stanford Drone Dataset (SDD)

- **Paper:** Robicquet et al., "Learning Social Etiquette: Human Trajectory Prediction In Crowded Scenes", ECCV 2016
- **URL:** https://cvgl.stanford.edu/projects/uav_data/
- **Images/Video:** 8 unique scenes on Stanford campus, ~20,000 tracked agents
- **Instances:** ~20,000 targets (11,200 pedestrians, 6,400 bicyclists, 1,300 cars, 300 skateboarders, 200 golf carts, 100 buses)
- **Classes (6):** pedestrian, bicyclist, skateboarder, car, bus, golf-cart
- **Annotation:** Bounding box tracks (trajectory data)
- **Source:** Top-down drone video
- **License:** Academic use
- **Download:** Free via Stanford CVGL

---

## Additional / Niche Datasets

| Dataset | Year | Focus | Key Stats | URL |
|---------|------|-------|-----------|-----|
| **DLR-MVDA** | 2015 | Multi-class vehicle, Munich | 20 images, 3,472 vehicles, OBB, 13cm GSD | DLR public datasets page |
| **COWC-M** | 2018 | 4-class car variant of COWC | Sedan/pickup/other/unknown | https://github.com/LLNL/cowc |
| **VehSat** | 2020 | Satellite vehicle detection | Large-scale | IEEE Xplore |
| **SDM-Car** | 2024 | Dim moving vehicles in sat video | 99 videos (Luojia 3-01 sat) | arXiv 2412.18214 |
| **VAID** | 2020 | Aerial vehicle classification | Vehicle sub-types | ResearchGate |
| **AU-AIR** | 2020 | Multi-modal drone traffic | 32,823 frames, multi-sensor | https://bozcani.github.io/auairdataset |
| **VETRA** | 2024 | Vehicle tracking, aerial (DLR) | Tracking benchmark | DLR elib |

---

## Key Citations

```bibtex
@inproceedings{xia2018dota,
  title={DOTA: A large-scale dataset for object detection in aerial images},
  author={Xia, Gui-Song and Bai, Xiang and Ding, Jian and others},
  booktitle={CVPR},
  year={2018}
}

@inproceedings{du2019visdrone,
  title={VisDrone-DET2019: The vision meets drone object detection in image challenge results},
  author={Du, Dawei and Zhu, Pengfei and Wen, Longyin and others},
  booktitle={ICCV Workshops},
  year={2019}
}

@article{razakarivony2016vedai,
  title={Vehicle detection in aerial imagery: A small target detection benchmark},
  author={Razakarivony, S{\'e}bastien and Jurie, Fr{\'e}d{\'e}ric},
  journal={Journal of Visual Communication and Image Representation},
  year={2016}
}

@inproceedings{mundhenk2016cowc,
  title={A large contextual dataset for classification, detection and counting of cars with deep learning},
  author={Mundhenk, T Nathan and Konjevod, Goran and Sakla, Wesam A and Boakye, Kofi},
  booktitle={ECCV},
  year={2016}
}

@inproceedings{hsieh2017carpk,
  title={Drone-based object counting by spatially regularized regional proposal network},
  author={Hsieh, Meng-Ru and Lin, Yen-Liang and Hsu, Winston H},
  booktitle={ICCV},
  year={2017}
}

@inproceedings{du2018uavdt,
  title={The unmanned aerial vehicle benchmark: Object detection and tracking},
  author={Du, Dawei and Qi, Yuankai and Yu, Hongyang and others},
  booktitle={ECCV},
  year={2018}
}

@article{liu2015dlr3k,
  title={Fast multiclass vehicle detection on aerial images},
  author={Liu, Kang and Mattyus, Gellert},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2015}
}

@inproceedings{azimi2020eagle,
  title={EAGLE: Large-scale vehicle detection dataset in real-world scenarios using aerial imagery},
  author={Azimi, Seyed Majid and Bahmanyar, Reza and Henry, Christopher and Kurz, Franz},
  booktitle={ICPR},
  year={2020}
}

@article{lam2018xview,
  title={xView: Objects in context in overhead imagery},
  author={Lam, Darius and Kuzma, Richard and McGee, Kevin and others},
  journal={arXiv preprint arXiv:1802.07856},
  year={2018}
}

@article{li2020dior,
  title={Object detection in optical remote sensing images: A survey and a new benchmark},
  author={Li, Ke and Wan, Gang and Cheng, Gong and Meng, Liqiu and Han, Junwei},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  year={2020}
}

@article{cheng2014vhr10,
  title={Multi-class geospatial object detection and geographic image classification},
  author={Cheng, Gong and Han, Junwei and Zhou, Peicheng and Guo, Lei},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  year={2014}
}

@inproceedings{zamir2019isaid,
  title={iSAID: A large-scale dataset for instance segmentation in aerial images},
  author={Zamir, Syed Waqas and Arora, Amanpreet and Gupta, Akshita and others},
  booktitle={CVPR Workshops},
  year={2019}
}

@article{sun2022fair1m,
  title={FAIR1M: A benchmark dataset for fine-grained object recognition in high-resolution remote sensing imagery},
  author={Sun, Xian and Wang, Peijin and Yan, Zhiyuan and others},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  year={2022}
}

@inproceedings{liu2017hrsc2016,
  title={A high resolution optical satellite image dataset for ship recognition and some new baselines},
  author={Liu, Zikun and Yuan, Liu and Weng, Lubin and Yang, Yiping},
  booktitle={ICPRAM},
  year={2017}
}

@article{cheng2023soda,
  title={Towards large-scale small object detection: Survey and benchmarks},
  author={Cheng, Gong and Yuan, Xiang and Wang, Xiwen and others},
  journal={IEEE TPAMI},
  year={2023}
}

@article{sun2022dronevehicle,
  title={Drone-based RGB-infrared cross-modality vehicle detection via uncertainty-aware learning},
  author={Sun, Yiming and Cao, Bing and Zhu, Pengfei and Hu, Qinghua},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2022}
}

@article{alemadi2025vme,
  title={VME: A satellite imagery dataset and benchmark for detecting vehicles in the Middle East and beyond},
  author={Alemadi, Nayef and others},
  journal={Scientific Data},
  year={2025}
}

@article{yang2018itcvd,
  title={Vehicle detection in aerial images},
  author={Yang, Michael Ying and Liao, Wentong and Li, Xinbo and Rosenhahn, Bodo},
  journal={Photogrammetric Engineering and Remote Sensing},
  year={2018}
}

@inproceedings{robicquet2016sdd,
  title={Learning social etiquette: Human trajectory prediction in crowded scenes},
  author={Robicquet, Alexandre and Sadeghian, Amir and Alahi, Alexandre and Savarese, Silvio},
  booktitle={ECCV},
  year={2016}
}
```

---

## Notes for Practitioners

1. **For vehicle detection training:** DOTA v2.0, VisDrone, xView, and EAGLE offer the largest
   labeled vehicle instance counts. DroneVehicle adds infrared modality.

2. **For oriented detection:** DOTA, VEDAI, EAGLE, FAIR1M, SODA-A, and DroneVehicle provide
   OBB annotations. Others (VisDrone, xView, DIOR) use axis-aligned HBB only.

3. **For counting / density estimation:** COWC (point annotations), CARPK, and PUCPR+ are
   specifically designed for vehicle counting tasks.

4. **For fine-grained classification:** FAIR1M (37 sub-classes) and xView (60 classes with
   many vehicle sub-types) offer the most detailed taxonomies.

5. **For ship-specific detection:** HRSC2016, DOTA (ship class), and FAIR1M (9 ship types)
   are the primary benchmarks.

6. **Cross-domain / multi-modal:** DroneVehicle (RGB+IR), VEDAI (color+IR), xView (RGB+multispectral).

7. **Most accessible for quick experiments:** NWPU VHR-10 (small, easy download), UAVDT (CC BY 4.0,
   Zenodo), VME (CC BY 4.0, Zenodo), DLR 3K Munich (small, public).