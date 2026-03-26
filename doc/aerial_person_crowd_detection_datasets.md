# Aerial/Drone Person and Crowd Detection Datasets

Comprehensive survey of publicly available datasets for person detection, pedestrian detection,
crowd counting, and human activity recognition from aerial/drone imagery.

**Compiled:** 2026-03-26

---

## Summary Table

| # | Dataset | Year | Images/Frames | Annotations | Classes (person-related) | Annotation Type | Resolution | Altitude | License | Free Download |
|---|---------|------|---------------|-------------|--------------------------|-----------------|------------|----------|---------|---------------|
| 1 | VisDrone-DET | 2019 | 10,209 images | 2.6M+ bbox | pedestrian, person + 8 others | BBox | ~2000x1500 | Various | Research only | Yes |
| 2 | DroneCrowd | 2021 | 33,600 frames (112 clips) | 4.8M head annotations, 20,800 trajectories | person (head) | Point, density map, bbox | 1920x1080 | Various | Research only | Yes |
| 3 | TinyPerson | 2020 | 1,610 images | 72,651 bbox | person (sea_person, earth_person) | BBox | Large images, objects <20px | Long distance | Not specified | Yes |
| 4 | Stanford Drone Dataset (SDD) | 2016 | ~20,000+ frames (60 videos, 8 scenes) | Tracking annotations | pedestrian, skateboarder, bicyclist, cart, car, bus | BBox (tracking) | ~1400x900 | Bird's-eye view (~fixed) | CC BY-NC-SA 3.0 | Yes |
| 5 | UAVDT | 2018 | 77,819 images (~80K frames) | 840K+ bbox | car, truck, bus (NO person) | BBox | 1080x540 | Various | Not specified | Yes |
| 6 | UAV-Human | 2021 | 67,428 video sequences | Multi-task annotations | person (action, pose, re-id, attributes) | BBox, skeleton, attributes | Various | Various | Research only | Yes |
| 7 | Okutama-Action | 2017 | 77,365 frames (43 sequences) | Multi-action labels per person | person (12 action classes) | BBox + action labels | 3840x2160 (4K) | 10-45m | Not specified | Yes |
| 8 | AU-AIR | 2020 | 32,823 frames (8 clips) | Object annotations + flight data | person + car, van, truck, motorbike, bike, bus, trailer | BBox | 1920x1080 | Max 30m | Not specified | Yes |
| 9 | CARPK | 2017 | 1,444 images | 89,777 bbox | car only (NO pedestrian) | BBox | 720x1280 | ~40m | Not specified | Yes |
| 10 | PUCPR+ | 2017 | 125 images | 16,456 bbox | car only (NO pedestrian) | BBox | 720x1280 | Fixed | Not specified | Yes |
| 11 | ERA | 2020 | 2,864 videos | Video-level event labels | 25 event categories (incl. crowd-related) | Video-level label (not bbox) | 640x640 | Various | Not specified | Yes |
| 12 | DLR-ACD | 2019 | 33 large aerial images | 226,291 point annotations | person | Point | GSD 4.5-15 cm/px | Helicopter altitude | Research (DLR terms) | Yes (registration) |
| 13 | HERIDAL | 2019 | ~500 full images + 68,750 patches | Person annotations | person | BBox | 4000x3000 (full) / 800x800 (patches) | Helicopter/UAV | CC BY 3.0 | Yes |
| 14 | SARD | 2021 | 1,981 images | Person bbox | person (various poses) | BBox (PascalVOC) | Video frame resolution | UAV altitude | Publicly available | Yes (IEEE DataPort) |
| 15 | NOMAD | 2024 | 42,825 frames | Person bbox with visibility levels | person (10 visibility levels) | BBox + visibility label | 5.4K video | Multiple aerial distances | Not specified | Yes |
| 16 | SeaDronesSee | 2022 | 14,127+ images (ODv2) | Person + object bbox | person, boat, lifebuoy, surfboard, wood | BBox | Full HD | 10-60m+ | Not specified | Yes |
| 17 | HIT-UAV | 2023 | 2,898 infrared images | 24,899 bbox | person, car, bicycle, other vehicle | BBox (standard + oriented) | Infrared thermal | 60-130m | CC BY 4.0 | Yes |
| 18 | MOBDrone | 2022 | 126,170 frames (66 clips) | 180K+ bbox (113K+ person) | person, boat, lifebuoy, surfboard, wood | BBox (COCO format) | Full HD | 10-60m | CC BY 4.0 | Yes (Zenodo) |
| 19 | VTSaR | 2025 | RGB + thermal paired images | 19,956 real + 54,749 synthetic instances | person | BBox | Not specified | 50-200m | Not specified | Yes (Baidu Pan) |
| 20 | Unicamp-UAV | 2025 | 6,500 images | 58,555 instances | person | BBox | UAV imagery | ~30m | Publicly available | Yes |
| 21 | UCF-QNRF | 2018 | 1,535 images | 1.25M head points | person (head) | Point, density map | Various (high-res) | Ground-level (NOT aerial) | Research only | Yes |
| 22 | ShanghaiTech | 2016 | 1,198 images (Part A+B) | 330,165 head points | person (head) | Point, density map | Various | Ground-level (NOT aerial) | Research only | Yes |
| 23 | DOTA v2.0 | 2021 | 11,268 images | 1.79M oriented bbox | 18 classes (NO person class) | Oriented BBox | ~4000x4000 | Aerial/satellite | IEEE subscription | Conditional |

---

## Detailed Dataset Descriptions

### 1. VisDrone-DET (VisDrone 2019 - Detection)

| Field | Details |
|-------|---------|
| **Full Name** | VisDrone - Vision Meets Drones: A Challenge |
| **Paper** | Zhu et al., "Detection and Tracking Meet Drones Challenge," TPAMI 2021 |
| **URL** | https://github.com/VisDrone/VisDrone-Dataset |
| **Alt URL** | http://www.aiskyeye.com |
| **Images** | 10,209 images (train: 6,471 / val: 548 / test-dev: 1,610 / test-challenge: 1,580) |
| **Annotations** | 2.6M+ bounding boxes |
| **Classes** | 10: pedestrian, person, car, van, bus, truck, motor, bicycle, awning-tricycle, tricycle |
| **Person-specific** | "pedestrian" = standing/walking; "person" = other poses |
| **Annotation Type** | Bounding box + attributes (occlusion, truncation) |
| **Resolution** | Variable, typically ~2000x1500 |
| **Altitude** | Various (collected across 14 Chinese cities) |
| **Platform** | Various drone platforms |
| **License** | Research use only (academic registration required) |
| **Free Download** | Yes (GitHub + institutional registration at aiskyeye.com) |
| **Notes** | Also has video detection, SOT, MOT, and crowd counting sub-datasets. Standard benchmark for drone-based detection. |

---

### 2. DroneCrowd

| Field | Details |
|-------|---------|
| **Full Name** | DroneCrowd: Detection, Tracking, and Counting Meets Drones in Crowds |
| **Paper** | Wen et al., "Detection, Tracking, and Counting Meets Drones in Crowds: A Benchmark," CVPR 2021 |
| **URL** | https://github.com/VisDrone/DroneCrowd |
| **Frames** | 33,600 frames from 112 video clips (70 different scenarios) |
| **Annotations** | 4.8M head annotations, 20,800 people trajectories |
| **Classes** | person (head point annotations) |
| **Annotation Type** | Point annotations, density maps, tracking trajectories |
| **Resolution** | 1920x1080 |
| **Altitude** | Various (arbitrary flight altitude) |
| **License** | Research use only |
| **Free Download** | Yes (GitHub) |
| **Notes** | Largest drone-based crowd counting dataset. Supports density estimation, localization, and tracking. Associated STANet method. |

---

### 3. TinyPerson

| Field | Details |
|-------|---------|
| **Full Name** | TinyPerson: Scale Match for Tiny Person Detection |
| **Paper** | Yu et al., "Scale Match for Tiny Person Detection," WACV 2020 |
| **URL** | https://github.com/ucas-vg/TinyBenchmark |
| **Images** | 1,610 images |
| **Annotations** | 72,651 bounding boxes |
| **Classes** | 1: person (sub-categories: sea_person, earth_person) |
| **Annotation Type** | Bounding box |
| **Resolution** | Large images; all person objects are smaller than 20 pixels |
| **Altitude** | Long-distance surveillance (not traditional low-altitude drone) |
| **License** | Not explicitly specified |
| **Free Download** | Yes (GitHub) |
| **Notes** | Ignores: crowds, ambiguous regions, water reflections, unrecognizable objects. Benchmark challenges at ICCV 2019 and ECCV 2020. |

---

### 4. Stanford Drone Dataset (SDD)

| Field | Details |
|-------|---------|
| **Full Name** | Stanford Drone Dataset |
| **Paper** | Robicquet et al., "Learning Social Etiquette: Human Trajectory Prediction In Crowded Scenes," ECCV 2016 |
| **URL** | https://cvgl.stanford.edu/projects/uav_data/ |
| **Alt URLs** | Kaggle, Academic Torrents |
| **Videos** | 60 videos across 8 campus scenes (bookstore, coupa, deathCircle, gates, hyang, little, nexus, quad) |
| **Annotations** | Bounding box tracking annotations at 30 FPS |
| **Classes** | 6: pedestrian, skateboarder, bicyclist, cart, car, bus |
| **Annotation Type** | Bounding box (tracking format) with occlusion/lost flags |
| **Resolution** | ~1400x900 per frame |
| **Altitude** | Fixed bird's-eye view (campus buildings) |
| **License** | CC BY-NC-SA 3.0 |
| **Free Download** | Yes |
| **Notes** | Widely used for trajectory prediction, not just detection. Campus environment only. |

---

### 5. UAVDT (Unmanned Aerial Vehicle Detection and Tracking)

| Field | Details |
|-------|---------|
| **Full Name** | The Unmanned Aerial Vehicle Benchmark: Object Detection and Tracking |
| **Paper** | Du et al., "The Unmanned Aerial Vehicle Benchmark: Object Detection and Tracking," ECCV 2018 |
| **URL** | https://sites.google.com/view/grli-uavdt |
| **Images** | 77,819 images from 100 video sequences |
| **Annotations** | 840K+ bounding boxes across 2,700+ vehicles |
| **Classes** | 4: car, truck, bus, vehicle |
| **Annotation Type** | Bounding box + attributes (vehicle category, occlusion) |
| **Resolution** | 1080x540 |
| **Altitude** | Various |
| **License** | Not explicitly specified |
| **Free Download** | Yes (Google Drive) |
| **Notes** | **Does NOT contain person/pedestrian annotations.** Vehicle-focused only. Included here for completeness as it is commonly referenced. |

---

### 6. UAV-Human

| Field | Details |
|-------|---------|
| **Full Name** | UAV-Human: A Large Benchmark for Human Behavior Understanding with Unmanned Aerial Vehicles |
| **Paper** | Li et al., "UAV-Human: A Large Benchmark for Human Behavior Understanding With Unmanned Aerial Vehicles," CVPR 2021 |
| **URL** | https://github.com/sutdcv/UAV-Human |
| **Alt URL** | https://huggingface.co/datasets/hibana2077/UAV-Human |
| **Data** | 67,428 video sequences (119 subjects) |
| **Annotations** | 22,476 frames (pose), 41,290 frames (re-id, 1,144 identities), 22,263 frames (attributes) |
| **Classes** | person (155 action categories, 12 body attributes) |
| **Annotation Type** | BBox, skeleton pose, person attributes, re-identification labels |
| **Resolution** | Various |
| **Modalities** | RGB, fisheye, night-vision, infrared (IR), depth maps |
| **License** | Research use only |
| **Free Download** | Yes |
| **Notes** | Multi-modal, multi-task. Richest annotations for human behavior from UAV. |

---

### 7. Okutama-Action

| Field | Details |
|-------|---------|
| **Full Name** | Okutama-Action: An Aerial View Video Dataset for Concurrent Human Action Detection |
| **Paper** | Barekatain et al., "Okutama-Action: An Aerial View Video Dataset for Concurrent Human Action Detection," CVPR Workshop 2017 |
| **URL** | http://okutama-action.org/ |
| **GitHub** | https://github.com/miquelmarti/Okutama-Action |
| **Frames** | 77,365 frames from 43 minute-long sequences |
| **Annotations** | Multi-action bounding box labels per person per frame |
| **Classes** | 12 action classes (person with action label) |
| **Annotation Type** | Bounding box + multi-action labels |
| **Resolution** | 3840x2160 (4K); also available downsampled at 1280x720 |
| **Altitude** | 10-45 meters |
| **Camera Angle** | 45 or 90 degrees |
| **License** | Not explicitly specified (academic use implied) |
| **Free Download** | Yes (Dropbox / AWS) |
| **Notes** | Multi-labeled actors (concurrent actions). Challenging: dynamic action transitions, scale changes. |

---

### 8. AU-AIR

| Field | Details |
|-------|---------|
| **Full Name** | AU-AIR: A Multi-modal Unmanned Aerial Vehicle Dataset for Low Altitude Traffic Surveillance |
| **Paper** | Bozcan & Kayacan, "AU-AIR: A Multi-modal Unmanned Aerial Vehicle Dataset for Low Altitude Traffic Surveillance," 2020 |
| **URL** | https://bozcani.github.io/auairdataset |
| **GitHub** | https://github.com/sunw71/auairdataset |
| **Images** | 32,823 frames from 8 video clips (~2 hours total) |
| **Annotations** | Bounding boxes + flight telemetry data |
| **Classes** | 8: person, car, van, truck, motorbike, bike, bus, trailer |
| **Annotation Type** | Bounding box + flight metadata (GPS, altitude, IMU, velocity) |
| **Resolution** | 1920x1080 at 30 FPS |
| **Altitude** | Max 30 meters |
| **Platform** | Parrot Bebop 2 |
| **License** | Not explicitly specified |
| **Free Download** | Yes (Google Drive: images 2.2 GB, annotations 3.9 MB) |
| **Notes** | Unique multi-modal aspect: visual + flight sensor data. Traffic surveillance focus. |

---

### 9. CARPK

| Field | Details |
|-------|---------|
| **Full Name** | Car Parking Lot Dataset |
| **Paper** | Hsieh et al., "Drone-based Object Counting by Spatially Regularized Regional Proposal Networks," ICCV 2017 |
| **URL** | https://lafi.github.io/LPN/ |
| **Images** | 1,444 images |
| **Annotations** | 89,777 bounding boxes |
| **Classes** | 1: car |
| **Annotation Type** | Bounding box |
| **Resolution** | 720x1280 |
| **Altitude** | ~40 meters |
| **Platform** | DJI Phantom 3 Professional |
| **License** | Not explicitly specified |
| **Free Download** | Yes (~2 GB combined with PUCPR+) |
| **Notes** | **Car counting only -- NO pedestrian annotations.** Included because it is frequently referenced in aerial counting literature. 4 distinct parking lots. |

---

### 10. PUCPR+

| Field | Details |
|-------|---------|
| **Full Name** | PUCPR+ (Pontifical Catholic University of Parana - Plus) |
| **Paper** | Hsieh et al., "Drone-based Object Counting by Spatially Regularized Regional Proposal Networks," ICCV 2017 |
| **URL** | https://lafi.github.io/LPN/ |
| **Images** | 125 images |
| **Annotations** | 16,456 bounding boxes |
| **Classes** | 1: car |
| **Annotation Type** | Bounding box |
| **Resolution** | 720x1280 |
| **Altitude** | Fixed camera position |
| **License** | Not explicitly specified |
| **Free Download** | Yes (bundled with CARPK) |
| **Notes** | **Car counting only -- NO pedestrian annotations.** Single parking lot with 331 spaces. Subset of original PUCPR dataset. |

---

### 11. ERA (Event Recognition in Aerial Videos)

| Field | Details |
|-------|---------|
| **Full Name** | ERA: A Dataset and Deep Learning Benchmark for Event Recognition in Aerial Videos |
| **Paper** | Mou et al., "ERA: A Dataset and Deep Learning Benchmark for Event Recognition in Aerial Videos," 2020 |
| **URL** | https://lcmou.github.io/ERA_Dataset/ |
| **Videos** | 2,864 video clips (5 seconds each, 24 FPS) |
| **Annotations** | Video-level event class labels |
| **Classes** | 25 event categories grouped into: Security, Disaster, Traffic, Productive activity, Social activity, Sport, Non-event |
| **Annotation Type** | Video-level classification label (NOT bounding box) |
| **Resolution** | 640x640 |
| **Source** | YouTube aerial videos |
| **License** | Not explicitly specified |
| **Free Download** | Yes |
| **Notes** | Event recognition, not object detection. No person-level bounding boxes. Relevant for crowd activity / event understanding context. |

---

### 12. DLR-ACD (DLR Aerial Crowd Dataset)

| Field | Details |
|-------|---------|
| **Full Name** | DLR's Aerial Crowd Dataset |
| **Paper** | Bahmanyar et al., "MRCNet: Crowd Counting and Density Map Estimation in Aerial and Ground Imagery," BMVC Workshop 2019 |
| **URL** | https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/dlr-acd |
| **Images** | 33 large aerial images from 16 flight campaigns |
| **Annotations** | 226,291 point annotations (285 to 24,368 per image) |
| **Classes** | 1: person |
| **Annotation Type** | Point annotation (for counting and density estimation) |
| **Resolution** | GSD 4.5-15 cm/pixel (very high spatial resolution) |
| **Altitude** | Helicopter-mounted DSLR cameras |
| **Scenes** | Mass events: sport events, city centers, open-air fairs, festivals |
| **License** | DLR research terms (registration required) |
| **Free Download** | Yes (with registration at DLR website) |
| **Notes** | One of the highest quality aerial crowd counting datasets. Very high resolution imagery from manned helicopter. |

---

### 13. HERIDAL

| Field | Details |
|-------|---------|
| **Full Name** | HEliRescue Image Dataset for ALgoritm development |
| **Paper** | Bozic-Stulic et al., "A Deep Learning Approach in Aerial Imagery for Supporting Land Search and Rescue Missions," 2019 |
| **URL (official)** | http://ipsar.fesb.unist.hr/HERIDAL%20database.html |
| **Alt URLs** | Kaggle: https://www.kaggle.com/datasets/imadeddinelassakeur/heridal ; Roboflow: https://universe.roboflow.com/drone-internship/heridal-human-detection |
| **Images** | ~500 full-size images + 68,750 image patches (29,050 positive, 39,700 negative) |
| **Annotations** | Person bounding boxes |
| **Classes** | 1: person |
| **Annotation Type** | Bounding box |
| **Resolution** | 4000x3000 (full images), 800x800 (patches) |
| **Altitude** | UAV / unmanned helicopter |
| **Terrain** | Mediterranean/sub-Mediterranean wilderness: mountains, forests, coasts |
| **Camera** | Canon Powershot S110 |
| **License** | CC BY 3.0 Unported |
| **Free Download** | Yes (multiple sources) |
| **Notes** | Designed for Search and Rescue. Non-urban terrain only. Created by University of Split. |

---

### 14. SARD (Search And Rescue Dataset)

| Field | Details |
|-------|---------|
| **Full Name** | Search And Rescue image Dataset for Person Detection |
| **Paper** | Sambolek & Ivasic-Kos, "Automatic Person Detection in Search and Rescue Operations Using Deep CNN Detectors," IEEE Access, 2021 |
| **URL** | https://ieee-dataport.org/documents/search-and-rescue-image-dataset-person-detection-sard |
| **Alt URL** | https://universe.roboflow.com/dataset-ay6sw/sard-peykp |
| **Images** | 1,981 images (extracted from video frames) |
| **Annotations** | Person bounding boxes |
| **Classes** | 1: person (various poses: running, walking, standing, sitting, lying) |
| **Annotation Type** | Bounding box (PascalVOC format) |
| **Resolution** | Video frame resolution |
| **Altitude** | UAV altitude |
| **Terrain** | Macadam roads, quarries, low/high grass, forest shade |
| **License** | Publicly available (IEEE DataPort) |
| **Free Download** | Yes |
| **Notes** | Actors simulate exhausted/injured persons. Realistic SAR scenario simulation. |

---

### 15. NOMAD

| Field | Details |
|-------|---------|
| **Full Name** | NOMAD: A Natural, Occluded, Multi-scale Aerial Dataset for Emergency Response Scenarios |
| **Paper** | Bernal et al., "NOMAD: A Natural, Occluded, Multi-scale Aerial Dataset, for Emergency Response Scenarios," WACV 2024 |
| **URL** | https://github.com/ArtRuss/NOMAD |
| **Frames** | 42,825 frames from 5.4K resolution videos |
| **Actors** | 100 different actors |
| **Annotations** | Bounding boxes with 10 visibility levels (based on % body visible) |
| **Classes** | 1: person |
| **Annotation Type** | Bounding box + occlusion/visibility label |
| **Resolution** | 5.4K video |
| **Altitude** | Five different aerial distances |
| **Demographics** | Ages 18-78, multi-ethnic |
| **Seasons** | Cross-seasonal (summer to winter) |
| **License** | Not explicitly specified |
| **Free Download** | Yes |
| **Notes** | Most comprehensive SAR dataset for occluded persons. WACV 2024 paper + WACV 2025 follow-up. |

---

### 16. SeaDronesSee

| Field | Details |
|-------|---------|
| **Full Name** | SeaDronesSee: A Maritime Benchmark for Detecting Humans in Open Water |
| **Paper** | Varga et al., "SeaDronesSee: A Maritime Benchmark for Detecting Humans in Open Water," WACV 2022 |
| **URL** | https://seadronessee.cs.uni-tuebingen.de/ |
| **GitHub** | https://github.com/Ben93kie/SeaDronesSee |
| **Images (ODv2)** | Train: 8,930 / Val: 1,547 / Test: 3,750 |
| **Annotations** | Bounding boxes + rich metadata (altitude, GPS, gimbal angles) |
| **Classes** | person, boat, lifebuoy, surfboard, wood |
| **Annotation Type** | Bounding box + flight metadata |
| **Resolution** | Full HD |
| **Altitude** | Various (open water scenarios) |
| **License** | Not explicitly specified |
| **Free Download** | Yes |
| **Tracks** | Object Detection (v1, v2), Single-Object Tracking, Multi-Object Tracking, Multi-Spectral OD |
| **Notes** | Maritime SAR focus. Active challenge series at WACV (2023-2025). Leaderboards available. |

---

### 17. HIT-UAV

| Field | Details |
|-------|---------|
| **Full Name** | HIT-UAV: A High-Altitude Infrared Thermal Dataset for UAV-based Object Detection |
| **Paper** | Suo et al., "HIT-UAV: A high-altitude infrared thermal dataset for Unmanned Aerial Vehicle-based object detection," Scientific Data, 2023 |
| **URL** | https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset |
| **Alt URL** | https://www.kaggle.com/datasets/pandrii000/hituav-a-highaltitude-infrared-thermal-dataset |
| **Images** | 2,898 infrared thermal images (from 43,470 video frames) |
| **Split** | Train: 2,029 / Test: 579 / Val: 290 |
| **Annotations** | 24,899 bounding boxes (standard + oriented) |
| **Classes** | 5: Person, Car, Bicycle, OtherVehicle, DontCare |
| **Annotation Type** | Bounding box (standard + oriented) in VOC XML, COCO JSON |
| **Resolution** | Infrared thermal |
| **Altitude** | 60-130 meters |
| **Camera Perspective** | 30-90 degrees |
| **Scenes** | Schools, parking lots, roads, playgrounds |
| **License** | CC BY 4.0 |
| **Free Download** | Yes |
| **Notes** | Infrared/thermal modality. Includes oriented bounding boxes for overlapping instances. |

---

### 18. MOBDrone

| Field | Details |
|-------|---------|
| **Full Name** | MOBDrone: A Drone Video Dataset for Man OverBoard Rescue |
| **Paper** | Ciampi et al., "MOBDrone: A Drone Video Dataset for Man OverBoard Rescue," ICIAP 2022 |
| **URL** | https://aimh.isti.cnr.it/dataset/mobdrone/ |
| **Zenodo** | https://zenodo.org/records/5996890 |
| **GitHub** | https://github.com/ciampluca/MOBDrone_eval |
| **Frames** | 126,170 frames from 66 video clips |
| **Annotations** | 180K+ bounding boxes (113K+ for person) |
| **Classes** | 5: person, boat, lifebuoy, surfboard, wood |
| **Annotation Type** | Bounding box (MS COCO format) |
| **Resolution** | Full HD |
| **Altitude** | 10-60 meters |
| **License** | CC BY 4.0 |
| **Free Download** | Yes (Zenodo; videos 5.5 GB, images 243 GB) |
| **Notes** | Marine environment. People in water simulating rescue scenarios. Largest maritime person detection dataset. |

---

### 19. VTSaR

| Field | Details |
|-------|---------|
| **Full Name** | VTSaR: Visible-Thermal Search and Rescue Dataset |
| **Paper** | Zhang et al., "Transformer-Based Person Detection in Paired RGB-T Aerial Images With VTSaR Dataset," IEEE JSTARS, 2025 |
| **URL** | https://github.com/zxq309/VTSaR |
| **Download** | Baidu Pan: https://pan.baidu.com/s/12P4tUg03KSYQYlA3E1iD0Q?pwd=zs5z |
| **Annotations** | 19,956 real-world instances + 54,749 synthetic instances |
| **Variants** | UA-VTSaR (unaligned), A-VTSaR (aligned), AS-VTSaR (aligned synthetic) |
| **Classes** | 1: person |
| **Annotation Type** | Bounding box |
| **Modalities** | Paired visible (RGB) + infrared thermal |
| **Altitude** | 50-200 meters |
| **Capture Angles** | 45, 60, 75 degrees |
| **Scenes** | Neighborhood, suburbs, shore, maritime, industrial zone, wild area |
| **License** | Not explicitly specified |
| **Free Download** | Yes (Baidu Pan) |
| **Notes** | Newest SAR dataset (2025). Multi-modal RGB-T. Includes synthetic data for training augmentation. |

---

### 20. Unicamp-UAV

| Field | Details |
|-------|---------|
| **Full Name** | Unicamp-UAV: An Open Dataset for Human Detection in UAV Imagery |
| **Paper** | Published in ISPRS Journal of Photogrammetry and Remote Sensing, 2025 |
| **URL** | Check ISPRS journal publication |
| **Images** | 6,500 images |
| **Annotations** | 58,555 person instances |
| **Classes** | 1: person |
| **Annotation Type** | Bounding box |
| **Resolution** | UAV imagery |
| **Altitude** | ~30 meters |
| **Platform** | DJI Phantom 4 |
| **Setting** | Urban areas, daytime |
| **License** | Publicly available |
| **Free Download** | Yes |
| **Notes** | Very recent (2025). Evaluated with YOLOv7/v8/v9/v10/v11. AP50 > 70% with best models. |

---

### 21. UCF-QNRF (for reference -- NOT aerial)

| Field | Details |
|-------|---------|
| **Full Name** | UCF-QNRF: A Large Crowd Counting Data Set |
| **Paper** | Idrees et al., "Composition Loss for Counting, Density Map Estimation and Localization in Dense Crowds," ECCV 2018 |
| **URL** | https://www.crcv.ucf.edu/data/ucf-qnrf/ |
| **Images** | 1,535 images (train: 1,201 / test: 334) |
| **Annotations** | 1.25M head point annotations |
| **Classes** | 1: person (head) |
| **Annotation Type** | Point annotation, density map |
| **Resolution** | Various (high resolution) |
| **Viewpoint** | **Ground-level (NOT aerial/drone)** |
| **License** | Research use only |
| **Free Download** | Yes |
| **Notes** | Included for completeness as it is a major crowd counting benchmark, but it is NOT an aerial/drone dataset. Web-collected images from diverse crowd scenarios. |

---

### 22. ShanghaiTech (for reference -- NOT aerial)

| Field | Details |
|-------|---------|
| **Full Name** | ShanghaiTech Crowd Counting Dataset |
| **Paper** | Zhang et al., "Single-Image Crowd Counting via Multi-Column Convolutional Neural Network," CVPR 2016 |
| **URL** | https://github.com/desenzhou/ShanghaiTechDataset |
| **Images** | Part A: 482 images (train: 300 / test: 182); Part B: 716 images (train: 400 / test: 316) |
| **Annotations** | 330,165 head point annotations total |
| **Classes** | 1: person (head) |
| **Annotation Type** | Point annotation, density map |
| **Viewpoint** | **Ground-level (NOT aerial/drone)** |
| **License** | Research use only |
| **Free Download** | Yes |
| **Notes** | Included for completeness. Major crowd counting benchmark but NOT aerial. Part A: dense crowds from web; Part B: Shanghai street scenes. |

---

### 23. DOTA v2.0 (for reference -- NO person class)

| Field | Details |
|-------|---------|
| **Full Name** | DOTA: A Large-scale Dataset for Object Detection in Aerial Images |
| **Paper** | Xia et al., "DOTA: A Large-Scale Dataset for Object Detection in Aerial Images," CVPR 2018; Ding et al., TPAMI 2021 |
| **URL** | https://captain-whu.github.io/DOTA/ |
| **Images** | v2.0: 11,268 images |
| **Annotations** | 1,793,658 oriented bounding boxes |
| **Classes** | 18: plane, ship, storage tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge, large vehicle, small vehicle, helicopter, roundabout, soccer ball field, swimming pool, container crane, airport, helipad |
| **Person-related** | **NONE -- DOTA does NOT contain person/pedestrian classes** |
| **Annotation Type** | Oriented bounding box |
| **Resolution** | ~4000x4000 |
| **License** | IEEE DataPort subscription for some versions |
| **Free Download** | Conditional |
| **Notes** | Included for reference only. Despite being a major aerial detection benchmark, it has NO person/pedestrian class. |

---

## Additional Datasets Discovered

### D-PTUAC (Drone-Person Tracking in Uniform Appearance Crowd)

| Field | Details |
|-------|---------|
| **Paper** | Published in Scientific Data, 2023 |
| **Data** | 138 sequences, 121K+ frames |
| **Annotations** | Bounding boxes + attributes |
| **Classes** | person (tracking in uniform crowds) |
| **Annotation Type** | Bounding box (tracking) |
| **Free Download** | Yes |

### POP (Partially Occluded Person - Infrared)

| Field | Details |
|-------|---------|
| **Data** | 8,768 labeled thermal images |
| **Annotations** | Person bounding boxes |
| **Classes** | person (partially occluded, infrared) |
| **Annotation Type** | Bounding box |
| **Modality** | Infrared thermal |
| **Free Download** | Yes |

---

## Key Findings and Recommendations

### Datasets with Direct Person/Pedestrian Detection (BBox):
1. **VisDrone-DET** -- largest and most established aerial person detection benchmark
2. **AU-AIR** -- person class with flight telemetry
3. **Okutama-Action** -- person with action labels, 4K
4. **HERIDAL** -- SAR-focused, wilderness terrain
5. **SARD** -- SAR-focused, various terrain
6. **NOMAD** -- SAR-focused with occlusion labels
7. **SeaDronesSee** -- maritime person detection
8. **MOBDrone** -- maritime person in water
9. **HIT-UAV** -- infrared thermal person detection
10. **VTSaR** -- paired RGB-thermal person detection
11. **Unicamp-UAV** -- urban person detection (2025)
12. **TinyPerson** -- extreme small-object person detection

### Datasets with Person Crowd Counting (Point/Density):
1. **DroneCrowd** -- largest aerial crowd counting dataset
2. **DLR-ACD** -- highest resolution aerial crowd images

### Datasets that do NOT contain person annotations (common misconceptions):
- **DOTA** (v1.0, v1.5, v2.0) -- no person class at all
- **UAVDT** -- vehicles only
- **CARPK / PUCPR+** -- cars only
- **UCF-QNRF** -- not aerial (ground-level)
- **ShanghaiTech** -- not aerial (ground-level)
- **ERA** -- video-level event labels only, no person bounding boxes

### Open Licenses (CC BY or more permissive):
- HERIDAL (CC BY 3.0)
- HIT-UAV (CC BY 4.0)
- MOBDrone (CC BY 4.0)
- Stanford Drone Dataset (CC BY-NC-SA 3.0)

---

## Sources

- [VisDrone GitHub](https://github.com/VisDrone/VisDrone-Dataset)
- [DroneCrowd GitHub](https://github.com/VisDrone/DroneCrowd)
- [TinyPerson / TinyBenchmark](https://github.com/ucas-vg/TinyBenchmark)
- [Stanford Drone Dataset](https://cvgl.stanford.edu/projects/uav_data/)
- [UAVDT Benchmark](https://sites.google.com/view/grli-uavdt)
- [UAV-Human GitHub](https://github.com/sutdcv/UAV-Human)
- [Okutama-Action](http://okutama-action.org/)
- [AU-AIR Dataset](https://bozcani.github.io/auairdataset)
- [CARPK / PUCPR+](https://lafi.github.io/LPN/)
- [ERA Dataset](https://lcmou.github.io/ERA_Dataset/)
- [DLR-ACD](https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/dlr-acd)
- [HERIDAL](http://ipsar.fesb.unist.hr/HERIDAL%20database.html)
- [SARD on IEEE DataPort](https://ieee-dataport.org/documents/search-and-rescue-image-dataset-person-detection-sard)
- [NOMAD GitHub](https://github.com/ArtRuss/NOMAD)
- [SeaDronesSee](https://seadronessee.cs.uni-tuebingen.de/)
- [HIT-UAV GitHub](https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset)
- [MOBDrone on Zenodo](https://zenodo.org/records/5996890)
- [VTSaR GitHub](https://github.com/zxq309/VTSaR)
- [DOTA Official](https://captain-whu.github.io/DOTA/)
- [UCF-QNRF](https://www.crcv.ucf.edu/data/ucf-qnrf/)
- [ShanghaiTech GitHub](https://github.com/desenzhou/ShanghaiTechDataset)
- [Aerial Person Detection Survey (2024)](https://spj.science.org/doi/10.34133/remotesensing.0474)
- [VisDrone Ultralytics Docs](https://docs.ultralytics.com/datasets/detect/visdrone/)