# Thesis as Case Study -- Miesner 2025 Reference Data

## Purpose

This reference provides all results, parameters, and findings from the Miesner 2025 master's thesis on aerial detection of marine iguanas. All agents in the practical-cv-wildlife skill use this data to ground exercises, examples, and discussions in real research rather than fabricated numbers.

---

## Thesis Overview

| Attribute | Value |
|-----------|-------|
| Title | Automated Counting of Marine Iguanas from Drone Imagery using Deep Learning |
| Author | Thomas J. Miesner |
| Year | 2025 |
| Institution | HNEE (Hochschule fuer nachhaltige Entwicklung Eberswalde) |
| Subject | Applied aerial wildlife detection with deep learning |
| Species | Marine iguanas (Amblyrhynchus cristatus) |
| Location | Galapagos Islands (Floreana and Fernandina) |

---

## Species: Marine Iguanas

| Attribute | Value |
|-----------|-------|
| Scientific name | Amblyrhynchus cristatus |
| IUCN status | Vulnerable |
| Body length (adult) | 30-40 cm (nose to tail base) |
| Body width (dorsal view) | ~10-15 cm |
| Typical appearance from above | Dark (black/grey), roughly oval/elongated |
| Habitat | Rocky coastlines, lava fields |
| Behavior | Basking in dense aggregations, thermoregulation |
| Key challenge | Dense colonies where individuals overlap and blend with lava rock substrate |

---

## Drone and Survey Setup

### Drone Specifications

| Parameter | Value |
|-----------|-------|
| Drone model | DJI Mavic 2 Pro |
| Camera sensor | 1-inch CMOS (13.2 mm x 8.8 mm) |
| Focal length | 10.3 mm |
| Image resolution | 5472 x 3648 pixels (20 MP) |
| Pixel pitch | 0.00241 mm/px |

### Flight Parameters

| Parameter | Floreana | Fernandina |
|-----------|----------|------------|
| Altitude (AGL) | Primarily 40 m | Mixed (40 m and 60 m) |
| Front overlap | 70% | 70% |
| Side overlap | 50% | 50% |
| View angle | Nadir (straight down) | Nadir (straight down) |

### GSD (Computed)

| Altitude | GSD | Iguana body in pixels (~35 cm) |
|----------|-----|-------------------------------|
| 40 m | ~0.94 cm/px | ~37 px |
| 60 m | ~1.41 cm/px | ~25 px |

---

## Model Architecture

### HerdNet

| Component | Choice |
|-----------|--------|
| Architecture | HerdNet (adapted for wildlife counting) |
| Backbone | DLA-34 (Deep Layer Aggregation, 34 layers) |
| Alternative backbone tested | DINOv2 (ViT-based) |
| Backbone comparison | DLA-34 outperformed DINOv2 on this dataset |
| Output | FIDT (Focal Inverse Distance Transform) maps |
| Detection method | Point-based detection (not bounding boxes) |
| Training annotations | Point annotations (body-center and head positions) |

### FIDT Maps

FIDT (Focal Inverse Distance Transform) maps encode point annotations as continuous density fields:
- Each annotated point generates a peaked response that falls off with distance
- The model learns to predict these maps from the input image
- Peaks in the predicted map correspond to detected individuals
- Count = number of detected peaks above a confidence threshold

### Why Point-Based Over Box-Based

The thesis chose point-based detection over bounding-box detection (e.g., YOLO) because:
1. Marine iguanas in aerial imagery are small (25-37 px)
2. Colonies are densely packed with overlapping individuals
3. Bounding boxes overlap heavily, making NMS unreliable
4. Point annotation is faster than drawing bounding boxes
5. FIDT maps handle density naturally without NMS

---

## Key Results

### Detection Performance (F1 Score)

| Metric | Floreana | Fernandina |
|--------|----------|------------|
| F1 score | **0.934** | **0.843** |
| Precision | [from thesis] | [from thesis] |
| Recall | [from thesis] | [from thesis] |

### Performance Difference Explanation

Floreana (F1=0.934) outperforms Fernandina (F1=0.843) because:
- Floreana surveys were primarily at 40 m (higher resolution)
- Fernandina had mixed altitudes (40 m and 60 m)
- Fernandina has more complex substrate (fresher lava with greater color/texture variation)
- Fernandina colonies may be denser

---

## Annotation Findings

### Body-Center vs Head Annotations

| Annotation Type | F1 Improvement |
|----------------|----------------|
| Body-center annotations | Baseline |
| Head annotations | **-0.10 F1** (worse) |

**Explanation:** At 40-60 m altitude, iguana heads are only 4-5 pixels wide. Body centers are more visually distinct (larger target, ~15-20 px area) and produce more consistent annotations between annotators.

### Inter-Annotator Agreement

Point annotations introduce annotator variability:
- Body-center placement varies by several pixels between annotators
- Head position is more ambiguous from nadir view
- FIDT maps are somewhat tolerant of annotation noise (the peaked response has width)

### Learning Curves

| Annotations | Performance |
|-------------|------------|
| 500 | Training begins to converge |
| 1000 | Reasonable detection quality |
| 2000-2500 | **Performance plateaus** |
| >2500 | Diminishing returns |

**Teaching implication:** Students should understand that more data does not always help. The plateau at 2000-2500 annotations suggests the model has learned the core visual patterns and additional data provides redundant information.

---

## Orthomosaic Software Comparison

| Software | F1 Score (relative) | Notes |
|----------|-------------------|-------|
| Pix4D | Baseline (higher) | Better seam line placement, more consistent stitching |
| DroneDeploy | **~0.07 lower** | Faster processing, lower quality orthomosaics |

**Explanation:** Orthomosaic quality directly affects detection performance. Poor seam lines can cut through iguanas, creating artifacts that confuse the model. Pix4D's better stitching preserves animal integrity at seam boundaries.

---

## Cross-Island Generalization

| Training | Testing | Result |
|----------|---------|--------|
| Floreana | Floreana | F1 = 0.934 |
| Fernandina | Fernandina | F1 = 0.843 |
| Floreana | Fernandina | **Significantly degraded** |
| Fernandina | Floreana | **Significantly degraded** |

**Key finding:** Cross-island training fails. Models must be trained per island (or per substrate type). This is because:
- Different lava substrates (age, color, texture)
- Different colony densities
- Different altitude mixtures
- Potential color variation in iguana populations

**Teaching implication:** Domain shift is real even within the same species on the same archipelago. Transfer learning is not a magic bullet.

---

## Human-in-the-Loop (HITL) Finding

| Metric | Value |
|--------|-------|
| Human undercounting rate | **22-30%** |
| HITL workflow | Model predictions reviewed by human annotator |
| HITL benefit | Catches systematically missed individuals |

**Explanation:** Human counters consistently undercount marine iguanas in dense colonies. The model identifies individuals that humans miss, particularly:
- Small/juvenile iguanas
- Partially occluded individuals
- Iguanas at the edges of dense clusters

**Teaching implication:** AI is not replacing human ecologists -- it is augmenting them. The HITL workflow produces better counts than either humans alone or the model alone.

---

## Usage in Exercises

### GSD Calculation Exercise
Use: Mavic 2 Pro specs, 40 m and 60 m altitudes, compute GSD and compare to iguana body size.

### Survey Planning Exercise
Use: 70/50 overlap values, compute trigger distance and strip spacing for a 500x300 m area.

### Detection Comparison Exercise
Use: F1 scores to motivate point-based vs box-based detection discussion.

### Counting Accuracy Exercise
Use: HITL finding (22-30% undercounting) to create model-vs-human comparison exercise.

### Domain Shift Exercise
Use: Cross-island generalization failure to discuss domain adaptation challenges.

### Annotation Strategy Exercise
Use: Body-center vs head annotation results to discuss annotation design decisions.

### Learning Curve Exercise
Use: 2000-2500 annotation plateau to discuss data efficiency and diminishing returns.
