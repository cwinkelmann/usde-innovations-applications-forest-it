# Model Selection Rationale Template

Use this template for the `model_selection_agent` output. The agent fills in the decision table, scoring matrix, and recommendation narrative based on the user's Configuration Record.

---

## Model Selection Rationale

**Project:** [Project name from Configuration Record]
**Date:** [YYYY-MM-DD]
**Species / Target:** [Species or classification target]
**Dataset size:** [N] images across [C] classes

---

### Hardware Constraints

| Resource | Available | Notes |
|----------|-----------|-------|
| GPU | [Model, VRAM] or CPU-only | |
| RAM | [N] GB | |
| Disk | [N] GB free | For weights + checkpoints |
| Training budget | [N] hours | |

---

### Candidate Models

| Dimension | timm DINOv2 ViT-B | timm DINOv2 ViT-L | DeepFaune ViT-L | SpeciesNet EfficientNetV2-M |
|-----------|-------------------|-------------------|-----------------|----------------------------|
| **Parameters** | 86M | 304M | 304M (pretrained on 34 spp.) | 54M (ensemble, no fine-tune) |
| **Input size** | 518x518 | 518x518 | 182x182 | 384x384 |
| **GPU memory (bs=1)** | ~3 GB | ~8 GB | ~8 GB | ~2 GB |
| **Recommended batch size** | 20 (24 GB) / 6 (8 GB) | 6 (24 GB) / 2 (8 GB) | 10 (24 GB) / 3 (8 GB) | 32 (24 GB) / 16 (8 GB) |
| **Pretrained domain** | ImageNet-22k (general) | ImageNet-22k (general) | 34 European wildlife spp. | Camera traps (global) |
| **Fine-tuning support** | Full (discriminative LRs) | Full (discriminative LRs) | Full (backbone transfer) | None (inference only) |
| **Min dataset for fine-tuning** | ~200 images | ~500 images | ~100 images (domain-matched) | N/A |
| **Catastrophic forgetting risk** | Medium | High (more params) | Low (domain-matched) | N/A |
| **License** | Apache 2.0 | Apache 2.0 | CeCILL + CC BY-NC-SA 4.0 | Apache 2.0 |

---

### 5-Dimension Scoring Matrix

Score each dimension 1--5. Higher is better for the user's specific scenario.

| Dimension | Weight | timm ViT-B | timm ViT-L | DeepFaune ViT-L | SpeciesNet | Justification |
|-----------|--------|-----------|-----------|-----------------|------------|---------------|
| **Accuracy potential** | [0.0--1.0] | [1--5] | [1--5] | [1--5] | [1--5] | [Why this score] |
| **Hardware fit** | [0.0--1.0] | [1--5] | [1--5] | [1--5] | [1--5] | [Why this score] |
| **Data efficiency** | [0.0--1.0] | [1--5] | [1--5] | [1--5] | [1--5] | [Why this score] |
| **Forgetting resistance** | [0.0--1.0] | [1--5] | [1--5] | [1--5] | [1--5] | [Why this score] |
| **Deployment simplicity** | [0.0--1.0] | [1--5] | [1--5] | [1--5] | [1--5] | [Why this score] |
| **Weighted total** | 1.0 | **[X.X]** | **[X.X]** | **[X.X]** | **[X.X]** | |

> **Weights must sum to 1.0.** Adjust weights based on user priorities stated in the Configuration Record.

---

### Decision Rules Applied

Check all that apply and note which rule determined the recommendation:

- [ ] **Dataset < 100 images and target species overlap DeepFaune's 34 classes** -- Use SpeciesNet or DeepFaune inference without fine-tuning
- [ ] **Dataset < 200 images and species are European mammals/birds** -- Use DeepFaune backbone transfer (domain-matched features reduce data requirements)
- [ ] **Dataset 200--1000 images, GPU >= 8 GB** -- Use timm DINOv2 ViT-B with frozen backbone warmup
- [ ] **Dataset > 1000 images, GPU >= 16 GB** -- Consider timm DINOv2 ViT-L for maximum accuracy
- [ ] **No GPU available** -- Use SpeciesNet inference or timm with CPU (small batch, long training)
- [ ] **Non-commercial license acceptable** -- DeepFaune backbone transfer is viable
- [ ] **Commercial deployment required** -- Exclude DeepFaune (CC BY-NC-SA 4.0)
- [ ] **Camera trap images with geographic metadata** -- Consider SpeciesNet first (geographic filtering improves accuracy)
- [ ] **Drone/UAV nadir imagery** -- Prefer timm DINOv2 (trained on diverse viewpoints); DeepFaune less suitable (trained on camera trap perspectives)
- [ ] **Need inference only, no training** -- Use SpeciesNet CLI: `python -m speciesnet.scripts.run_model`

---

### Recommendation

**Primary recommendation:** [Model name]

**Rationale (2--3 sentences):**
[Explain why this model best fits the user's scenario. Reference the scoring matrix, decision rules, and any specific constraints from the Configuration Record.]

**Forgetting mitigation strategy:** [Freeze backbone / Discriminative LRs / Gradual unfreezing / Knowledge distillation]

**Recommended hyperparameters:**

| Parameter | Value | Source |
|-----------|-------|--------|
| Backbone LR | [e.g., 1e-6] | [Rule or reference] |
| Head LR | [e.g., 1e-4] | [Rule or reference] |
| Input size | [e.g., 518x518] | [Model default] |
| Batch size | [e.g., 20] | [Based on GPU VRAM] |
| Warmup epochs | [e.g., 5] | [Head-only phase] |
| Total epochs | [e.g., 100] | [Dataset size heuristic] |
| Weight decay | [e.g., 0.05] | [Standard for AdamW] |
| Scheduler | [e.g., CosineAnnealingLR] | |
| AMP | [Yes/No] | [CUDA only] |

---

### Alternative Recommendation

**Fallback model:** [Model name]

**When to switch:** [Describe the condition under which the user should abandon the primary recommendation and switch to this alternative -- e.g., "If validation accuracy plateaus below 70% after 20 epochs with the primary model, switch to [alternative] because..."]

---

### Template Selection

Based on this recommendation, use the following template to generate the training script:

| Recommendation | Template file | Key modifications |
|----------------|--------------|-------------------|
| timm DINOv2 (any size) | `timm_finetune_template.py` | Set `--model`, `--input-size`, `--backbone-lr`, `--head-lr` |
| DeepFaune backbone transfer | `deepfaune_transfer_template.py` | Set `--deepfaune-weights`, `--num-classes`, `--input-size` |
| SpeciesNet inference only | N/A (CLI command) | `python -m speciesnet.scripts.run_model --folders [path]` |

---

### Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Catastrophic forgetting | [Low/Medium/High] | [Specific strategy chosen] |
| Overfitting (small dataset) | [Low/Medium/High] | [Augmentation, early stopping, dropout] |
| Class imbalance | [Low/Medium/High] | [WeightedRandomSampler, focal loss, oversampling] |
| Data leakage (spatial autocorrelation) | [Low/Medium/High] | [GroupShuffleSplit by site/transect] |
| GPU OOM during training | [Low/Medium/High] | [Reduce batch size, gradient accumulation, AMP] |
