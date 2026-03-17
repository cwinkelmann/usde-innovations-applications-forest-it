---
name: active-learning-wildlife
description: >
  Active learning workflows for wildlife detection using HILDA. Covers the full
  human-in-the-loop (HITL) cycle: model predicts, uncertain samples are selected
  via query strategies (uncertainty sampling, embedding clustering, RGB contrast),
  exported to annotation tools (CVAT, Label Studio), expert reviews and corrects
  annotations, corrected labels are downloaded, and the model is retrained.
  Keywords: active learning, HILDA, CVAT, Label Studio, annotation, human-in-the-loop,
  HITL, uncertainty sampling, embedding clustering, RGB contrast, annotation workflow,
  expert review, sample selection, annotation budget, oracle, query strategy.
version: "1.0"
---

# Active Learning for Wildlife Detection — HILDA Skill

## Purpose

This skill guides users through active learning workflows for wildlife detection,
centered on the **HILDA** framework (v0.3.0). The core workflow minimizes human
annotation effort by intelligently selecting which images an expert should review:

```
Train → Predict → Select uncertain samples → Export to CVAT/Label Studio →
Expert corrects → Download annotations → Retrain → Repeat
```

## Agents

| Agent | File | Role |
|-------|------|------|
| **config_agent** | `agents/config_agent.md` | Determines user state (has model? data? annotations?) and routes to the appropriate workflow |
| **workflow_agent** | `agents/workflow_agent.md` | Orchestrates the full active learning loop from initialization through repeated rounds |
| **sampling_strategy_agent** | `agents/sampling_strategy_agent.md` | Explains and implements the three HILDA sampling strategies (RGB Contrast, Embedding Clustering, Logit Uncertainty) |
| **annotation_tool_agent** | `agents/annotation_tool_agent.md` | Handles export to and import from CVAT and Label Studio, including pre-annotation formatting |
| **exercise_designer_agent** | `agents/exercise_designer_agent.md` | Creates exercises and assignments covering active learning concepts at multiple difficulty levels |

## Modes

### 1. `setup-workflow` (default)
Configure and launch an active learning loop for a specific wildlife detection project.
The config_agent assesses the user's current state, then the workflow_agent walks through
the loop step by step.

### 2. `explain-concept`
Explain an active learning concept (query strategies, annotation budgets, cold start,
stopping criteria, learning curves) without running any code. Uses the sampling_strategy_agent
and references for clear explanations.

### 3. `select-strategy`
Help the user choose the best sampling strategy for their situation. The sampling_strategy_agent
compares RGB Contrast, Embedding Clustering, and Logit Uncertainty based on the user's
available resources (model, compute, annotation budget).

### 4. `create-exercise`
Generate exercises, assignments, or exam questions about active learning for wildlife ecology
courses. The exercise_designer_agent creates tasks at basic through advanced levels.

### 5. `full-course-module`
Build a complete teaching module on active learning for wildlife detection, including lecture
notes, exercises, and practical lab instructions using HILDA and CVAT/Label Studio.

## Does NOT Trigger

| If the user wants... | Use this skill instead |
|---|---|
| Train/fine-tune a detection model (code) | herdnet-training or wildlife-classification |
| Run MegaDetector inference | megadetector |
| Tiled inference on large images | sahi-inference |
| Course material from thesis/defence slides | iguana-case-study |
| Annotation tool setup only (no active learning loop) | iguana-case-study (create-practical) |

---

## Key Facts: Active Learning for Wildlife

### The Core Loop
Active learning is an iterative machine learning paradigm where the model actively selects
which data points should be labeled next. Instead of randomly choosing images for annotation,
the model identifies the samples it would learn the most from.

**Standard loop:**
1. Train model on current labeled dataset
2. Run predictions on unlabeled pool
3. Apply query strategy to select most informative samples
4. Export selected samples (with pre-annotations) to annotation tool
5. Expert (oracle) reviews, corrects, and validates annotations
6. Download corrected annotations
7. Merge with training set and retrain
8. Evaluate on held-out validation set
9. Repeat until budget exhausted or performance plateaus

### Budget Awareness
- The goal is to **minimize the number of human annotations** while **maximizing model improvement**
- Each annotation round has a cost: annotator time, project management overhead, compute for retraining
- Diminishing returns typically set in after 3-5 active learning rounds
- A well-designed active learning workflow can achieve the same model performance with 30-60% fewer annotations than random sampling

### HILDA Sampling Strategies (v0.3.0)

**RGB Contrast** — No model needed
- Selects visually diverse images based on color histogram analysis
- Best for cold start when no trained model exists
- Simple, fast, requires no GPU

**Embedding Clustering** — Needs feature extractor
- Clusters image embeddings (DINOv2, CLIP, ResNet) and selects representatives from each cluster
- Ensures diversity across scenes, habitats, lighting conditions
- Best for ensuring coverage of the visual domain

**Logit Uncertainty** — Needs trained model
- Selects images where the model is least confident (highest entropy, smallest margin, lowest max probability)
- Directly targets model weaknesses
- Best for targeted improvement after initial training

### Annotation Tools

**CVAT** (Computer Vision Annotation Tool)
- Open-source, self-hosted (Docker)
- Strong support for bounding boxes, polygons, points
- SDK for automation: `pip install cvat-sdk`
- Well-suited for detection tasks

**Label Studio**
- Open-source, more flexible annotation types
- Supports mixed annotation (points + boxes + polygons + text)
- SDK for automation: `pip install label-studio-sdk`
- Better for multi-modal labeling workflows

### Export Formats
- **COCO JSON**: Most detection models (HerdNet, Faster R-CNN, DETR)
- **YOLO txt**: YOLOv8, MegaDetector fine-tuning
- **Pascal VOC XML**: Legacy models
- **CSV (x, y, labels)**: HerdNet/animaloc point-based models

### Thesis Finding
The HITL (human-in-the-loop) workflow catches **22-30% human undercounting** in iguana surveys.
Expert annotators consistently miss animals in dense colonies; model predictions with expert
review find significantly more individuals than expert-only annotation.

## Integration with Other Skills

| Skill | Integration Point |
|-------|-------------------|
| `herdnet-training` | Retrain HerdNet after each active learning round with expanded annotations |
| `megadetector` | Use MegaDetector for initial predictions (cold start pre-annotations) |
| `wildlife-classification` | Retrain species classifier on actively-selected samples |
| `sahi-inference` | Use SAHI tiled inference for predictions on high-resolution drone imagery |

## Failure Paths

### Cold Start (No Model Yet)
- **Problem**: Logit Uncertainty requires a trained model, but there is none
- **Solution**: Use RGB Contrast or random sampling for the first 1-2 rounds, then switch to uncertainty sampling once a model is trained

### Annotation Budget Exhausted
- **Problem**: No more annotation budget, but model still underperforming
- **Solution**: Evaluate if the problem is data quantity or quality. Consider pseudo-labeling high-confidence predictions. Assess whether the task is feasible with available data.

### Annotator Disagreement
- **Problem**: Multiple annotators label the same images differently
- **Solution**: Measure inter-annotator agreement (Cohen's kappa, IoU overlap). Use consensus or adjudication for disagreements. Add clear annotation guidelines.

### Model Not Improving Despite New Annotations
- **Problem**: Learning curve has plateaued
- **Solution**: Check for label noise in annotations. Verify the model architecture is appropriate. Consider augmentation strategies. The problem may require a fundamentally different approach (e.g., switching from detection to density estimation).

## Source Reference

HILDA framework source: `/Users/christian/PycharmProjects/hnee/active_learning`
