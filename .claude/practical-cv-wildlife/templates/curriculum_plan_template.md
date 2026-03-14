# Curriculum Plan Template — PCV + Wildlife Detection

## Course Title
[e.g., "From Computer Vision Fundamentals to Aerial Wildlife Detection"]

## Duration
[e.g., 4 weeks / 8 sessions]

## Prerequisites
- PCV Modules 1-6 completed (CNN basics through transfer learning)
- Python, PyTorch basics
- Access to GPU (Colab or local)

---

## Week-by-Week Schedule

### Week 1: Aerial Imagery Fundamentals + Detection Bridge

**Session 1: Aerial Imagery for Computer Vision** (builds on PCV Module 1)
- Learning objectives:
  - [ ] Calculate GSD from sensor specs and flight altitude
  - [ ] Understand nadir vs oblique imaging
  - [ ] Compute image footprint and survey overlap
- Materials: `aerial_concepts_notebook_template.md`
- PCV prerequisites: Module 1 (image representation)
- Estimated time: 2 hours

**Session 2: From Classification to Detection** (fills PCV Module 8 gap)
- Learning objectives:
  - [ ] Understand anchor-based vs anchor-free detection
  - [ ] Train YOLOv8 on a wildlife dataset
  - [ ] Evaluate with mAP, AP50
- Materials: `yolo_finetuning_exercise_template.md`
- PCV prerequisites: Module 5 (training techniques), Module 8 (conceptual detection)
- Estimated time: 3 hours

---

### Week 2: Wildlife Classification + Transfer Learning

**Session 3: Wildlife Species Classification** (adapts PCV Module 5-6)
- Learning objectives:
  - [ ] Fine-tune a pretrained model for wildlife species
  - [ ] Apply wildlife-appropriate augmentation (nadir vs camera trap)
  - [ ] Evaluate with per-class F1 and confusion matrix
- Materials: Adapted `Pet_Classification.ipynb` with wildlife data
- PCV prerequisites: Module 5 (fine-tuning), Module 6 (transfer learning, CAM)
- Estimated time: 2.5 hours

**Session 4: Embeddings for Wildlife** (extends PCV Module 7)
- Learning objectives:
  - [ ] Generate species embeddings with DINOv2/CLIP
  - [ ] Visualize species similarity with t-SNE
  - [ ] Zero-shot classification for unknown species
- Materials: Adapted `Intro_to_CLIP_ZeroShot_Classification.ipynb`
- PCV prerequisites: Module 7 (embeddings, ViT, CLIP)
- Estimated time: 2 hours

---

### Week 3: Counting + Density Estimation

**Session 5: MegaDetector + Detection Pipeline**
- Learning objectives:
  - [ ] Run MegaDetector on camera trap images
  - [ ] Build detect-then-classify pipeline
  - [ ] Understand confidence thresholds for wildlife surveys
- Materials: MegaDetector skill templates
- PCV prerequisites: Module 8 (detection concepts)
- Estimated time: 2 hours

**Session 6: Point-Based Detection with HerdNet**
- Learning objectives:
  - [ ] Understand FIDT maps and point-based counting
  - [ ] Configure HerdNet for a new dataset
  - [ ] Evaluate counting accuracy vs detection accuracy
- Materials: HerdNet skill templates
- PCV prerequisites: Sessions 1-2 (aerial, detection)
- Estimated time: 3 hours

---

### Week 4: Large Image Inference + Capstone

**Session 7: Tile-Based Inference for Orthomosaics**
- Learning objectives:
  - [ ] Implement overlapping tile inference
  - [ ] Understand Hann window stitching
  - [ ] Handle edge effects and double-counting
- Materials: SAHI / HerdNetStitcher exercises
- PCV prerequisites: Session 6 (HerdNet)
- Estimated time: 2.5 hours

**Session 8: Capstone — Iguanas From Above**
- Learning objectives:
  - [ ] Apply full pipeline to marine iguana drone imagery
  - [ ] Compare detection vs classification vs counting approaches
  - [ ] Interpret results against thesis benchmarks (F1=0.934)
- Materials: Thesis case study data
- PCV prerequisites: All previous sessions
- Estimated time: 3 hours

---

## Assessment

### Formative (ongoing)
- Exercise completion with automated checks (assertion cells)
- GSD calculation worksheet
- Model comparison table (YOLO vs HerdNet vs MegaDetector)

### Summative (final)
- Capstone report: apply one pipeline to a new wildlife dataset
- Include: data preparation, model selection rationale, evaluation metrics, error analysis
- Compare results to published baselines

---

## Running Case Study: Marine Iguanas (Miesner 2025)

| Aspect | Detail |
|--------|--------|
| Species | Marine iguanas, Galapagos Islands |
| Drone | DJI Mavic 2 Pro, 40m and 60m altitude |
| Model | HerdNet + DLA-34, FIDT maps |
| Best F1 | 0.934 (Floreana), 0.843 (Fernandina) |
| Key insight | Body-center annotations > head (+0.10 F1) |
| HITL finding | Catches 22-30% human undercounting |

This case study appears in Sessions 1 (GSD), 6 (HerdNet), 7 (stitching), and 8 (capstone).

---

## Adaptation Notes

- **Shorter course (2 weeks):** Use Sessions 1, 2, 5, 8 only
- **Classification focus:** Replace Sessions 5-7 with more depth on Sessions 3-4
- **Detection focus:** Replace Sessions 3-4 with more depth on Sessions 5-7
- **No GPU access:** Use Colab; reduce batch sizes; skip HerdNet training (inference only)
