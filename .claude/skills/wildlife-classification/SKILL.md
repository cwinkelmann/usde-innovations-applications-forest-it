---
name: wildlife-classification
description: "Wildlife image classification skill with 7-agent pipeline. Fine-tuning image classification models for wildlife species using timm, DeepFaune (DINOv2 ViT-L), and SpeciesNet (EfficientNetV2-M). Central concern: catastrophic forgetting mitigation via layer freezing, discriminative learning rates, and gradual unfreezing. Supports code generation, concept explanation, model evaluation, exercise creation, full course modules, and model comparison. Triggers on: timm, wildlife classification, species classification, DeepFaune, SpeciesNet, fine-tune, fine-tuning, catastrophic forgetting, layer freezing, discriminative learning rate, transfer learning, camera trap classification, wildlife AI, animal classification, DINOv2 fine-tune, iguana classification."
metadata:
  version: "1.0"
  last_updated: "2026-03-13"
---

# Wildlife Classification — Fine-Tuning Image Classifiers for Wildlife Species

A specialized skill for fine-tuning pretrained image classification models on wildlife species data. Covers three model families (timm/DINOv2, DeepFaune, SpeciesNet), with catastrophic forgetting as the central concern. Draws on real code from the Iguanas From Above project and the DeepFaune camera trap classifier.

## Quick Start

**Minimal command:**
```
Fine-tune a DINOv2 model for iguana classification from drone imagery
```

```
Compare timm, DeepFaune, and SpeciesNet for my camera trap project
```

```
Explain catastrophic forgetting and how to prevent it when fine-tuning a ViT
```

**Execution flow:**
1. Configuration -- understand user's species, data, hardware, and goal
2. Dataset preparation -- ImageFolder structure, splits, augmentation
3. Model selection -- timm vs DeepFaune vs SpeciesNet decision
4. Fine-tuning strategy -- freezing, discriminative LRs, gradual unfreezing
5. Training code generation -- complete runnable script
6. Evaluation -- per-class metrics, confusion matrix, calibration
7. (Optional) Exercise design -- scaffolded notebook for teaching

---

## Trigger Conditions

### Trigger Keywords

**Primary**: timm, wildlife classification, species classification, DeepFaune, SpeciesNet, fine-tune, fine-tuning, catastrophic forgetting, layer freezing, discriminative learning rate, transfer learning

**Secondary**: camera trap classification, wildlife AI, animal classification, DINOv2 fine-tune, iguana classification, wildlife model, species identification, camera trap AI, drone classification, wildlife deep learning, MegaDetector classifier, EfficientNet wildlife

### Does NOT Trigger

| Scenario | Use Instead |
|----------|-------------|
| Object detection / counting from drone imagery (not classification) | `megadetector` or custom detection skill |
| General deep research on AI in ecology | `deep-research` |
| Writing a paper about wildlife AI methods | `academic-paper` |
| MegaDetector detection only (no species classification) | `megadetector` |

---

## Agent Team (7 Agents)

| # | Agent | Role | Phase |
|---|-------|------|-------|
| 1 | `config_agent` | Reads user intent, selects model family, determines execution mode, validates hardware constraints | Phase 0 |
| 2 | `dataset_prep_agent` | ImageFolder structure, GroupShuffleSplit by site, class balance analysis, augmentation pipeline | Phase 1 |
| 3 | `model_selection_agent` | Comparative analysis of timm vs DeepFaune vs SpeciesNet for user's specific context | Phase 2 |
| 4 | `fine_tuning_strategy_agent` | Layer freezing, discriminative LRs, gradual unfreezing, knowledge distillation | Phase 3 |
| 5 | `training_code_agent` | Writes complete training scripts using timm APIs (create_model, create_optimizer_v2) | Phase 4 |
| 6 | `evaluation_agent` | Per-class P/R/F1, confusion matrix, calibration curves, threshold sweep | Phase 5 |
| 7 | `exercise_designer_agent` | Scaffolded notebook exercises with TODO markers for teaching contexts | Phase 6 |

---

## Orchestration Flowchart

```
User Request
    |
    v
[config_agent] --> Determine mode + model family + hardware
    |
    +-- mode: generate-code -----> [dataset_prep_agent] -> [model_selection_agent] -> [fine_tuning_strategy_agent] -> [training_code_agent] -> [evaluation_agent]
    |
    +-- mode: explain-concept ---> [fine_tuning_strategy_agent] (standalone explanation)
    |
    +-- mode: evaluate-model ----> [evaluation_agent] (standalone evaluation)
    |
    +-- mode: create-exercise ---> [exercise_designer_agent] (uses templates)
    |
    +-- mode: full-course-module -> ALL agents in sequence (Phase 0-6)
    |
    +-- mode: compare-models ----> [model_selection_agent] (standalone comparison)
```

---

## Execution Modes

### 1. `generate-code` (default)
Produce a complete, runnable fine-tuning script. Runs Phases 0-5.

**Output:** Python training script + evaluation script + shell launcher

### 2. `explain-concept`
Explain a concept (catastrophic forgetting, discriminative LRs, etc.) with code snippets and diagrams.

**Output:** Markdown explanation with annotated code blocks

### 3. `evaluate-model`
Given a trained model checkpoint, produce evaluation metrics.

**Output:** Per-class metrics table + confusion matrix code + calibration analysis

### 4. `create-exercise`
Design a scaffolded exercise notebook for teaching fine-tuning.

**Output:** Markdown notebook template with TODO markers and solution hints

### 5. `full-course-module`
Complete teaching module: explanation + exercise + solution + evaluation rubric.

**Output:** Multiple files covering theory, practice, and assessment

### 6. `compare-models`
Side-by-side comparison of timm, DeepFaune, and SpeciesNet for the user's use case.

**Output:** Decision table with rationale

---

## Model Family Quick Reference

### timm (PyTorch Image Models)
- **Backbone options:** Any timm model; DINOv2 ViT-B/L recommended for ecology
- **Key model IDs:** `vit_base_patch14_dinov2.lvd142m`, `vit_large_patch14_dinov2.lvd142m`
- **Input:** Configurable; 518x518 for DINOv2, 512x512 for CNNs
- **Training:** Full script with `create_model`, `create_optimizer_v2`, `create_scheduler_v2`
- **Fine-tuning:** Full control -- freeze any layer, discriminative LRs, gradual unfreezing
- **License:** Apache 2.0

### DeepFaune
- **Backbone:** `vit_large_patch14_dinov2.lvd142m` (loaded via timm)
- **Input:** 182x182px, BICUBIC resize, ImageNet normalization
- **Weights:** `deepfaune-vit_large_patch14_dinov2.lvd142m.v3.pt` (1.1GB)
- **Weight format:** `{'args': {...}, 'state_dict': ...}`
- **Pipeline:** YOLOv8s @ 960px detection -> ViT-L @ 182x182 classification
- **Species:** 34 European species only; backbone-only transfer for non-European
- **License:** CeCILL + CC BY-NC-SA 4.0 (non-commercial only)
- **Training code:** NOT provided -- must write custom fine-tuning

### SpeciesNet
- **Architecture:** EfficientNetV2-M + MegaDetector ensemble
- **CLI:** `python -m speciesnet.scripts.run_model --folders "images/" --predictions_json "out.json" --country GBR`
- **Variants:** v4.0.2a (always-crop) vs v4.0.2b (full-image)
- **Species:** 2000+ species with geographic filtering
- **Fine-tuning:** NOT supported -- inference only
- **Install:** `pip install speciesnet`

---

## Catastrophic Forgetting Strategies

| Strategy | When to Use | Backbone LR | Head LR | Complexity |
|----------|-------------|-------------|---------|------------|
| Freeze backbone, train head only | <200 images/class, quick baseline | 0 | 1e-4 | Low |
| Discriminative LRs | 200-1000 images/class (recommended default) | 1e-6 | 1e-4 | Medium |
| Gradual unfreezing (ULMFiT-style) | >200 images/class, fine-grained differences | Scheduled | 1e-4 | High |
| Knowledge distillation | Critical to preserve original capabilities | N/A (separate loss) | 1e-4 | Advanced |

---

## Key Reference Files

| File | Purpose |
|------|---------|
| `references/timm_fine_tuning_guide.md` | create_model, create_optimizer_v2, layer groups, checkpoints |
| `references/iguana_training_case_study.md` | Walk-through of actual iguana training code |
| `references/deepfaune_architecture.md` | Two-stage pipeline, weight loading, backbone transfer |
| `references/speciesnet_guide.md` | Ensemble architecture, CLI, geographic filtering |
| `references/catastrophic_forgetting.md` | Four strategies with concrete LR ratios |
| `references/classification_augmentation.md` | timm create_transform, nadir-specific advice |
| `references/class_imbalance_strategies.md` | WeightedRandomSampler, focal loss, class weights |
| `references/model_evaluation_standards.md` | Per-class vs macro F1, calibration, threshold sweep |
| `references/dataset_split_rules.md` | GroupShuffleSplit by site, data leakage for tiled data |
| `references/pytorch_conventions.md` | Device idiom, torch.load, Path objects |

---

## Failure Paths

### User has no labeled data
1. Recommend SpeciesNet for zero-shot inference
2. Suggest MegaDetector + manual labeling workflow
3. Point to Wildlife Insights or iNaturalist for pre-labeled data

### User's species not in DeepFaune's 34 classes
1. DeepFaune backbone transfer only (freeze classifier, replace head)
2. Recommend timm DINOv2 direct fine-tuning instead
3. If camera trap data: suggest SpeciesNet first as baseline

### Hardware too limited for ViT-L
1. Downgrade to ViT-B (`vit_base_patch14_dinov2.lvd142m`)
2. Reduce input resolution (518 -> 384 -> 224)
3. Enable AMP (`--amp` flag)
4. Reduce batch size (ViT-L: 10, ViT-B: 20 on single GPU)

### Very small dataset (<50 images per class)
1. Freeze entire backbone -- head-only training
2. Heavy augmentation (timm create_transform with auto-augment)
3. Consider few-shot learning or prototype networks
4. Use SpeciesNet as pseudo-labeler to bootstrap more data

### User wants to fine-tune SpeciesNet
1. Explain: SpeciesNet does NOT support fine-tuning
2. Redirect to timm + DINOv2 or DeepFaune backbone
3. Suggest SpeciesNet as evaluation baseline only

---

## Quality Criteria

A generated training script is NOT complete until:
- [ ] Model creation uses `timm.create_model` with `pretrained=True`
- [ ] Optimizer uses `timm.optim.create_optimizer_v2` with proper param groups
- [ ] Learning rates follow the discriminative LR pattern (backbone << head)
- [ ] Data loading uses ImageFolder or equivalent with proper train/val split
- [ ] ImageNet normalization is applied: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- [ ] Evaluation reports per-class precision, recall, and F1
- [ ] Script handles device selection (CUDA > MPS > CPU)
- [ ] Checkpointing saves best model by validation metric
- [ ] No hardcoded absolute paths (all paths via argparse or config)
- [ ] AMP (automatic mixed precision) is enabled by default
