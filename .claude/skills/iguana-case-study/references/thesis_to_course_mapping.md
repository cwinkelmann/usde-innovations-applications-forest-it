# Thesis-to-Course Mapping

How the Winkelmann (2025) thesis maps onto Week 1 practicals and lectures.

## Thesis → Practical Mapping

| Thesis Topic | FIT Practical | What Students Do |
|---|---|---|
| Orthomosaic → tiles → annotations (Sec 1.30) | P1 — Drone imagery | Explore aerial tiles, understand CSV point annotations |
| Annotation tools, CVAT/Hasty.ai (Sec 2.3.0) | P2 — Annotation tools | Try point/box/polygon annotation in Label Studio |
| MegaDetector baseline (camera traps) | P3 — MegaDetector | Run MD on camera trap images, not iguana data |
| Detection results, false positives (Sec 3.0.0) | P4 — Detection exploration | Browse detections, discuss failure modes |
| Two-stage pipeline (Sec 4.1) | P5 — Classifier | Run timm classifier on animal crops |
| F1, precision, recall, matching threshold (Sec 2.2.4) | P6 — Evaluation | Compute metrics against reference labels |
| Landscape segmentation for habitat (Sec 4.3) | P7 — Segmentation | SAM demo on land cover scene |
| Reflection on pipeline limitations (Sec 4.1) | P8 — Wrap-up | Export work, reflect on what failed |

## Thesis → Lecture Mapping

| Thesis Chapter | Lecture Slot | Key Message |
|---|---|---|
| 1.01 Conservation context | Day 1 AM — Why AI in Ecology | Biodiversity monitoring bottleneck |
| 1.02 Marine iguana biology | Day 1 AM — Case study intro | Ideal study species for drones |
| 1.03 Current monitoring methods | Day 1 AM — Classical vs AI | Drone counts 17-35% higher than ground |
| 1.04 Computer vision challenges | Day 1 PM — after P1 | Camouflage, density, image quality |
| 2.1.4 HerdNet architecture | Day 2 AM — Detection concepts | Point-based vs bounding box; FIDT maps |
| 2.2.3 HEAD vs BODY annotations | Day 2 AM — Annotation design | Annotation protocol choices matter |
| 2.2.2 Training curve / data saturation | Day 3 AM — How much data? | Quality > quantity |
| 2.2.4 Hyperparameter optimization | Day 3 AM — Model tuning | Inference parameters matter as much as training |
| 2.3.0 HITL workflow | Day 3 PM — Human-in-the-loop | 330 annotations/hour; model-assisted review |
| 3.Z5 Final results | Day 4 AM — Evaluation | F1 = 0.85-0.93 across islands |
| 4.1 Discussion / limitations | Day 4 AM — Honest assessment | Cross-island fails; no true ground truth |
| 4.3 Outlook | Day 4 PM — Wrap-up | Future: automated flights, onboard inference |

## Quantitative Results Quick Reference

Use these exact numbers when generating course material:

| Metric | Value | Source |
|---|---|---|
| Floreana F1 (optimized) | 0.90 | Results Sec 3.Z5 |
| Fernandina F1 (optimized) | 0.80 | Results Sec 3.Z5 |
| Body-center vs head F1 gain | +0.17 | Results Sec 3_HEAD_vs_Body |
| Training curve plateau (Floreana) | ~952 annotations | Results Sec 3_Training_Curve |
| Training curve plateau (Fernandina) | ~3,096 annotations | Results Sec 3_Training_Curve |
| HITL annotation speed | 330 annotations/hour | Results Sec 3_Human_In_The_Loop |
| HITL total effort (Fernandina) | 35 hours for 11,500 predictions | Results Sec 3_Human_In_The_Loop |
| Expert consensus dataset | 4 experts, 496 images, 1,014 iguanas | Results Sec 3_1_results_human |
| Expert overcounting | 3-8% | Results Sec 3_1_results_human |
| Best DLA backbone | DLA-34 (not larger variants) | Results Sec 3_Exp_3_hyperparameter |
| Optimal matching threshold | 75 pixels | Results Sec 3_Exp_3_hyperparameter |
| Optimal LMDS kernel | 5x5 | Results Sec 3_Exp_3_hyperparameter |
| Genovesa F1 (worst case) | 0.43 | Discussion Sec 4.1 |
| Annotation variability | ~30% | Discussion Sec 4.1 |
| Drone survey images total | >180,000 | Methods Sec 2.1.2 |
| Survey phases | 6 (2020-2024) | Methods Sec 2.1.2 |
| Islands surveyed | 5 (Floreana, Fernandina, Genovesa, Isabela, San Cristóbal) | Methods Sec 2.1.0 |
| GSD (DroneDeploy) | 0.93 cm/pixel | Methods Sec 2.1.2 |
| Camera | DJI Mavic 2 Pro (Hasselblad, 20 MP) | Methods Sec 2.1.2 |

## Augmentation Strategy (for exercises)

Beneficial augmentations (Floreana optimized):
- RandomRotate90: p=0.66
- HorizontalFlip: p=0.42
- Perspective: p=0.30
- ImageCompression: p=0.20
- Blur, MotionBlur, RandomScale, RandomShadow: lower probabilities

Not beneficial:
- HSV color augmentation (volcanic rock texture > color information)
