---
name: practical-cv-wildlife
description: "Bridge skill connecting the Practical Computer Vision (PCV) course to aerial wildlife detection. 6-agent pipeline that maps PCV curriculum gaps, generates new aerial imagery modules, adapts existing notebooks for wildlife datasets, fills the object detection implementation gap (YOLOv8), creates wildlife-domain exercises, and sequences a combined curriculum. Does NOT re-teach PCV content — generates NEW material that fills gaps and adapts existing exercises for wildlife/aerial imagery. Triggers on: practical computer vision, PCV course, adapt notebook, aerial module, wildlife course, detection module, bridge course, curriculum gap, YOLOv8 exercise, wildlife detection exercise, drone imagery module, adapt PCV, GSD calculation, tile inference exercise, HerdNet module, aerial counting, wildlife curriculum."
metadata:
  version: "1.0"
  last_updated: "2026-03-13"
---

# Practical CV Wildlife — PCV-to-Wildlife Bridge Skill

A 6-agent pipeline that bridges the existing Practical Computer Vision with PyTorch course (Modules 1-8: CNN basics, ResNet, embeddings, CLIP, conceptual detection/segmentation) into the aerial wildlife detection domain. This skill does NOT re-teach PCV content. It generates NEW material that fills identified gaps and adapts existing exercises for wildlife and aerial imagery contexts.
It refers to https://github.com/andandandand/practical-computer-vision 
## Quick Start

**Map curriculum gaps:**
```
Map the PCV course against wildlife detection requirements and show me what's missing
```

**Generate a new aerial module:**
```
Create a new module on aerial imagery fundamentals (GSD, nadir, orthomosaics) for PCV students
```

**Adapt an existing notebook:**
```
Adapt the Pet_Classification.ipynb notebook to use a wildlife camera trap dataset
```

**Fill the detection gap:**
```
Create a hands-on YOLOv8 fine-tuning exercise that builds on PCV Module 8
```

**Full course module:**
```
Build a complete 2-week wildlife detection module
```

---

## Trigger Conditions

### Trigger Keywords

practical computer vision, PCV course, adapt notebook, aerial module, wildlife course, detection module, bridge course, curriculum gap, YOLOv8 exercise, wildlife detection exercise, drone imagery module, adapt PCV, GSD calculation, tile inference exercise, HerdNet module, aerial counting, wildlife curriculum, point-based detection exercise, density estimation exercise, adapt for wildlife, aerial wildlife, PCV wildlife, PCV bridge, drone detection module, wildlife counting exercise

### Does NOT Trigger

| Scenario | Use Instead |
|----------|-------------|
| Writing an academic paper about wildlife AI | `academic-paper` |
| Deep research on wildlife ecology topics | `deep-research` |
| Running MegaDetector inference | `megadetector` |
| Training HerdNet from scratch | `herdnet-training` |
| SAHI tiled inference on existing model | `sahi-inference` |
| Wildlife species classification fine-tuning | `wildlife-classification` |
| Active learning annotation workflows | `active-learning-wildlife` |

---

## Source Repository

The PCV course lives at: `/Users/christian/PycharmProjects/hnee/practical-computer-vision`

### PCV Course Inventory

| Component | Count | Location |
|-----------|-------|----------|
| Workshop slide PDFs (PCV series) | 6 | `slides/practical-computer-vision-series/` |
| Workshop slide PDFs (Image Dataset Curation) | 4 | `slides/image-dataset-curation/` |
| Jupyter notebooks | 18 | `notebooks/` |
| Module overview | 1 | `docs/Modules - Practical Computer Vision with PyTorch.pdf` |
| Project specification | 1 | `docs/project_task.md` |

### PCV Module Coverage

| Module | Title | Lessons | Notebooks | Wildlife Gap |
|--------|-------|---------|-----------|-------------|
| 1 | Foundations | 1-3: CV tasks, PIL/NumPy, PyTorch tensors | `Digital_Image_Representation_PIL_NumPy_PyTorch.ipynb` | No aerial imagery, no GSD concept |
| 2 | Neural Networks | 4-6: Intro NN, MLP regression, matrix mult | `Training_a_Perceptron_for_Image_based_Regression.ipynb` | Standard content, reusable |
| 3 | Training & Evaluation | 7-9: Classification, metrics, DataLoaders | `Starter_Create_Dataloaders_Train_Val_Test.ipynb`, `Kaggle_Competition_LeNet5_Digit_Recognition.ipynb` | Metrics need wildlife context (precision-recall for rare species) |
| 4 | CNNs | 10-12: Convolutions, pooling, upsampling | `Looking_into_LeNet5_with_Random_Weights.ipynb` | No aerial-specific conv patterns |
| 5 | Training Techniques | 13-15: Normalization, BCE, skip connections | `Pet_Classification.ipynb`, `Finetuning_a_Resnet_for_Multilabel_Classification.ipynb` | Pet classification adaptable to wildlife |
| 6 | Optimization & Interpretability | 16-19: Augmentation, regularization, transfer learning, CAM | `Labeling_Images_with_a_Pretrained_Resnet.ipynb` | Transfer learning directly applicable; CAM useful for wildlife |
| 7 | Embeddings | 20-22: Embeddings, ViT, CLIP | `Creating_Embeddings_from_Resnet34.ipynb`, `Intro_to_CLIP_ZeroShot_Classification.ipynb`, etc. | CLIP zero-shot for wildlife species is a strong bridge |
| 8 | Detection & Segmentation | 23-24: Object detection, segmentation (OVERVIEW) | None (conceptual only) | **CRITICAL GAP**: No hands-on detection, no YOLO, no counting |

### Critical Gaps This Skill Fills

1. **No aerial/drone imagery treatment** -- GSD, nadir view, orthomosaics, motion blur, overlap
2. **Object detection severely undercovered** -- Module 8 is conceptual only, no hands-on YOLO implementation
3. **No counting/density estimation** -- HerdNet, FIDT maps, point-based detection absent
4. **No tile-based inference for large images** -- SAHI, overlapping tiles, stitching not covered
5. **No wildlife-specific datasets or domain transfer** -- All exercises use standard CV datasets (MNIST, ImageNet, pets)

---

## Agent Team (6 Agents)

| # | Agent | Role | Mode |
|---|-------|------|------|
| 1 | `curriculum_mapper_agent` | Inventories PCV progress, identifies gaps against wildlife detection requirements, produces a prioritized study plan | `map-curriculum` |
| 2 | `aerial_concepts_agent` | Generates new teaching content on aerial imagery fundamentals: GSD, nadir, overlap, motion blur, footprint, coordinate systems | `generate-aerial-module` |
| 3 | `detection_bridge_agent` | Fills Module 8 gap with hands-on content: YOLOv8 fine-tuning, mAP/AP50, detection vs classification, NMS, anchor-free detection | `fill-detection-gap` |
| 4 | `wildlife_adapter_agent` | Rewrites existing PCV notebooks with wildlife datasets while preserving teaching structure and pedagogical flow | `adapt-notebook` |
| 5 | `exercise_generator_agent` | Creates new exercises in PCV style targeting wildlife detection domain: counting, tile inference, species ID | `create-exercise` |
| 6 | `module_sequencer_agent` | Produces week-by-week curriculum plan slotting wildlife skills after/between PCV modules | `full-course-module` |

---

## Operational Modes (6 Modes)

| Mode | Trigger | Agents | Output |
|------|---------|--------|--------|
| `map-curriculum` | "map PCV gaps", "curriculum gap analysis" | 1 | Gap analysis report + prioritized study plan |
| `generate-aerial-module` | "aerial module", "GSD module", "drone imagery lesson" | 2 | Complete teaching module with slides outline + notebook |
| `fill-detection-gap` | "detection module", "YOLOv8 exercise", "fill Module 8" | 3 | Hands-on detection exercise + conceptual notes |
| `adapt-notebook` | "adapt notebook", "wildlife version of" | 4 | Adapted notebook preserving PCV structure |
| `create-exercise` | "create exercise", "wildlife exercise", "counting exercise" | 5 | New exercise in PCV style |
| `full-course-module` | "full module", "course plan", "wildlife curriculum" | 1 -> 6 -> 2 -> 3 -> 5 | Complete multi-week module with all materials |

### Mode Selection Logic

```
"Map PCV against wildlife needs"              -> map-curriculum
"Create an aerial imagery fundamentals module" -> generate-aerial-module
"Build a YOLOv8 fine-tuning exercise"         -> fill-detection-gap
"Adapt Pet_Classification for camera traps"   -> adapt-notebook
"Create a tile inference exercise"            -> create-exercise
"Build a full 4-week wildlife detection unit" -> full-course-module
```

---

## Orchestration Workflow

### Full Course Module Mode (complete pipeline)

```
User: "Build a complete wildlife detection module for PCV students"
     |
=== Phase 1: CURRICULUM MAPPING ===
     |
     +-> [curriculum_mapper_agent]
         - Inventory PCV modules 1-8 coverage
         - Identify gaps against wildlife detection requirements
         - Classify each gap: missing (new content needed) vs adaptable (existing content rewritable)
         - Output: Gap Analysis Report + Prioritized Skill List
     |
=== Phase 2: SEQUENCING ===
     |
     +-> [module_sequencer_agent]
         - Map PCV prerequisites to wildlife skills
         - Slot new content at optimal curriculum points
         - Define week-by-week schedule
         - Output: Curriculum Plan
     |
=== Phase 3: CONTENT GENERATION (parallel where possible) ===
     |
     |-> [aerial_concepts_agent] -> Aerial Imagery Module
     |   - GSD, nadir, overlap, motion blur, footprint
     |   - Notebook: GSD calculation + tile size selection
     |
     |-> [detection_bridge_agent] -> Detection Implementation Module
     |   - YOLOv8 fine-tuning on wildlife data
     |   - mAP evaluation + error analysis
     |   - Counting/density estimation intro
     |
     +-> [exercise_generator_agent] -> Exercise Set
         - 3-5 exercises in PCV style
         - Progressive difficulty
         - Wildlife datasets throughout
     |
=== Phase 4: ASSEMBLY ===
     |
     +-> [module_sequencer_agent]
         - Assemble all materials into final curriculum plan
         - Cross-reference with PCV notebooks
         - Verify prerequisite coverage
         - Output: Complete Course Module Package
```

### Single-Notebook Adaptation Mode

```
User: "Adapt [notebook_name] for wildlife"
     |
     +-> [wildlife_adapter_agent]
         - Read original notebook structure
         - Identify dataset swap points
         - Preserve pedagogical flow
         - Replace datasets with wildlife equivalents
         - Add wildlife-domain commentary
         - Output: Adapted Notebook Template
```

---

## Thesis Context (Running Case Study)

All generated content uses the master's thesis by Thomas J. Miesner as a running case study, connecting PCV concepts to real-world aerial wildlife detection:

| Aspect | Detail |
|--------|--------|
| Species | Marine iguanas (Amblyrhynchus cristatus), Galapagos Islands |
| Model | HerdNet with DLA-34 backbone |
| Method | FIDT (Focal Inverse Distance Transform) maps, point-based detection |
| Drone | DJI Mavic 2 Pro (20MP, 5472x3648, 10.3mm focal length) |
| Key results | F1=0.934 (Floreana), F1=0.843 (Fernandina) |
| HITL finding | Human-in-the-loop workflow catches 22-30% human undercounting |
| Datasets | 40m and 60m flight altitude surveys, 70% front overlap, 50% side overlap |

This case study appears in:
- Aerial concepts module: real GSD calculations from Mavic 2 Pro specs
- Detection bridge: comparison of YOLO vs HerdNet for dense small objects
- Exercise generator: exercises use thesis imagery parameters
- Curriculum plan: thesis as capstone project context

---

## Agent File References

| Agent | Definition File |
|-------|----------------|
| curriculum_mapper_agent | `agents/curriculum_mapper_agent.md` |
| aerial_concepts_agent | `agents/aerial_concepts_agent.md` |
| detection_bridge_agent | `agents/detection_bridge_agent.md` |
| wildlife_adapter_agent | `agents/wildlife_adapter_agent.md` |
| exercise_generator_agent | `agents/exercise_generator_agent.md` |
| module_sequencer_agent | `agents/module_sequencer_agent.md` |

---

## Reference Files

| Reference | Purpose | Used By |
|-----------|---------|---------|
| `references/pcv_course_inventory.md` | Complete inventory of PCV modules, notebooks, PDFs with reusability assessment | curriculum_mapper, module_sequencer |
| `references/pcv_to_wildlife_bridge.md` | Mapping from each PCV concept to its wildlife application | curriculum_mapper, wildlife_adapter |
| `references/aerial_imagery_primer.md` | GSD formula, nadir vs oblique, orthorectification, Mavic 2 Pro specs | aerial_concepts, exercise_generator |
| `references/detection_concepts_expanded.md` | Anchor boxes, IoU, mAP/AP50/AP75, NMS, YOLOv8 architecture and training loop | detection_bridge, exercise_generator |
| `references/wildlife_datasets_guide.md` | iNaturalist, iWildCam, AID, Caltech Camera Traps, Wildlife Insights — format, size, access | wildlife_adapter, exercise_generator |
| `references/exercise_design_patterns.md` | PCV exercise anatomy: objectives, setup, TODO scaffold, assertions, solution toggle | exercise_generator, wildlife_adapter |
| `references/thesis_as_case_study.md` | Thesis results as running case study connecting all skills | all agents |
| `references/drone_imagery_fundamentals.md` | GSD formula derivation, sensor parameters, overlap calculation, motion blur risk model | aerial_concepts, exercise_generator |

---

## Templates

| Template | Purpose | Used By |
|----------|---------|---------|
| `templates/aerial_concepts_notebook_template.md` | Notebook structure: GSD calculation, footprint estimation, tile size selection | aerial_concepts |
| `templates/yolo_finetuning_exercise_template.md` | Hands-on YOLOv8 fine-tuning exercise with wildlife data | detection_bridge |
| `templates/wildlife_adapted_notebook_template.md` | Template for adapting any PCV notebook to wildlife domain | wildlife_adapter |
| `templates/curriculum_plan_template.md` | Week-by-week schedule integrating PCV + wildlife skills | module_sequencer |

---

## Integration with Other Skills

```
practical-cv-wildlife + wildlife-classification -> Classification fine-tuning with wildlife focus
practical-cv-wildlife + herdnet-training        -> From PCV foundations to HerdNet implementation
practical-cv-wildlife + sahi-inference           -> Tile-based inference exercises with real pipeline
practical-cv-wildlife + megadetector             -> MegaDetector as detection baseline exercise
practical-cv-wildlife + active-learning-wildlife -> HITL annotation exercises
practical-cv-wildlife + academic-paper           -> Write course development paper
```

---

## Quality Standards

### Content Quality
1. **No PCV duplication** -- never re-explain concepts already covered in PCV modules; reference them
2. **Prerequisite accuracy** -- every new concept must state which PCV module it builds on
3. **Wildlife authenticity** -- all examples use real species, real sensor specs, real detection challenges
4. **Thesis grounding** -- case study data comes from the actual thesis, not fabricated numbers

### Exercise Quality
5. **PCV style fidelity** -- exercises follow the same structure as existing PCV notebooks
6. **Progressive difficulty** -- exercises within a module build on each other
7. **Runnable code** -- all code cells must be executable (with dataset download instructions)
8. **TODO scaffolding** -- student tasks are clearly marked with `# TODO:` comments

### Curriculum Quality
9. **Prerequisite chain intact** -- no wildlife module requires concepts not yet taught in PCV
10. **Time estimates realistic** -- each module includes estimated completion time
11. **Assessment aligned** -- exercises test the learning objectives stated at module start

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-03-13 | Initial release: 6-agent pipeline, 6 modes, 8 reference docs, 4 templates. Bridges PCV Modules 1-8 to aerial wildlife detection domain. |
