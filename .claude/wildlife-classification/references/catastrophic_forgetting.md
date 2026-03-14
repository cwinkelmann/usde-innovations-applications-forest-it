# Catastrophic Forgetting in Wildlife Classification

Comprehensive reference on catastrophic forgetting -- what it is, why it matters for wildlife classification, and four concrete mitigation strategies with specific learning rate ratios and implementation patterns.

---

## What Is Catastrophic Forgetting?

When a neural network trained on task A is subsequently trained on task B, it tends to lose its ability to perform task A. The weights update to minimize loss on the new data, overwriting the representations that encoded the original knowledge. This phenomenon is called catastrophic forgetting (also: catastrophic interference).

**In the wildlife classification context:** A DINOv2 model pretrained on ImageNet (14M images, 1000 classes) has learned powerful visual features -- edges, textures, shapes, object parts, spatial relationships. When fine-tuned on a small wildlife dataset (e.g., 500 images, 5 species), naive training destroys these general features. The model overfits to the small dataset and loses the ability to extract robust visual representations.

**Why this matters more for wildlife than typical CV tasks:**
1. Wildlife datasets are typically small (hundreds, not millions of images)
2. The domain shift is significant (drone nadir, camera trap, underwater, thermal)
3. Class boundaries are subtle (congeneric species differ in minor features)
4. Deployment environments vary (lighting, season, weather, vegetation)

---

## Diagnosing Catastrophic Forgetting

**Symptoms during training:**
- Validation loss decreases initially, then increases while training loss continues to decrease (classic overfitting, often driven by forgetting)
- Early layers' activations become noisy or degenerate
- The model becomes overconfident on training classes, underconfident on others

**Symptoms during evaluation:**
- Training accuracy >> validation accuracy (large generalization gap)
- Model fails on images that are slightly different from training distribution
- Features extracted by the backbone are less discriminative than pretrained features

**Diagnostic experiment:**
```python
# Compare backbone feature quality before and after fine-tuning
# Using linear probing on a held-out dataset

# 1. Extract features from pretrained backbone
pretrained_model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True)
pretrained_features = extract_features(pretrained_model, held_out_dataset)

# 2. Extract features from fine-tuned backbone
finetuned_features = extract_features(finetuned_model, held_out_dataset)

# 3. Train a linear probe on each set of features
pretrained_score = linear_probe_accuracy(pretrained_features)
finetuned_score = linear_probe_accuracy(finetuned_features)

# If finetuned_score << pretrained_score on the held-out task,
# catastrophic forgetting has occurred
```

---

## Strategy 1: Freeze Backbone, Train Head Only

**The simplest and safest approach.** Zero risk of forgetting because backbone weights do not change.

### When to Use
- Dataset size: <200 images per class
- First experiment with a new dataset
- When pretrained features are already good (similar domain)

### Implementation

```python
import timm

model = timm.create_model(
    'vit_base_patch14_dinov2.lvd142m',
    pretrained=True,
    num_classes=5,
)

# Freeze all backbone parameters
for name, param in model.named_parameters():
    if 'head' not in name:
        param.requires_grad = False

# Only the head is trainable
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=0.05,
)
```

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Head LR | 1e-4 to 1e-3 | Standard for linear head training |
| Backbone LR | 0 (frozen) | No gradient computation |
| Weight decay | 0.05 | Standard for AdamW |
| Epochs | 20-50 | Converges quickly with frozen backbone |
| Batch size | Large as GPU allows | No backbone gradient -> lower memory |

### Pros and Cons
- **Pro:** Zero forgetting risk
- **Pro:** Fast training (small number of trainable params)
- **Pro:** Low memory (no backbone gradients)
- **Con:** Accuracy ceiling -- backbone cannot adapt to domain-specific features
- **Con:** May underperform if pretrained features are poor for the target domain

---

## Strategy 2: Discriminative Learning Rates (Recommended Default)

**The backbone trains at 1/100th the learning rate of the head.** This allows gentle adaptation without catastrophic loss. This is the recommended default for most wildlife classification tasks.

### When to Use
- Dataset size: 200-1000 images per class
- Production models where both accuracy and robustness matter
- Standard approach -- try this first after frozen-backbone baseline

### Implementation

```python
import timm
import torch

model = timm.create_model(
    'vit_base_patch14_dinov2.lvd142m',
    pretrained=True,
    num_classes=5,
)

# Separate parameters
backbone_params = []
head_params = []
for name, param in model.named_parameters():
    if 'head' in name:
        head_params.append(param)
    else:
        backbone_params.append(param)

# Discriminative LRs: backbone = head / 100
optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': 1e-6, 'weight_decay': 0.05},
    {'params': head_params, 'lr': 1e-4, 'weight_decay': 0.05},
])
```

### Concrete LR Ratios

| Backbone Type | Backbone LR | Head LR | Ratio | Source |
|---------------|-------------|---------|-------|--------|
| DINOv2 ViT-B | 1e-6 | 1e-4 | 1:100 | Iguana training |
| DINOv2 ViT-L | 1e-6 | 1e-4 | 1:100 | Iguana training |
| ResNet-50 | 1e-5 | 1e-3 | 1:100 | Iguana training |
| EfficientNet-B0 | 1e-5 | 1e-3 | 1:100 | Iguana training |
| ConvNeXt-T | 1e-5 | 1e-3 | 1:100 | Iguana training |
| DLA-34 | 1e-5 | 1e-3 | 1:100 | Iguana training |

**Why 1:100?**
- Too small a ratio (1:10) allows too much backbone change -> forgetting
- Too large a ratio (1:1000) effectively freezes the backbone -> limited adaptation
- 1:100 has been empirically validated across the iguana training experiments

**Why ViTs use lower absolute LRs than CNNs:**
- ViT attention weights are more sensitive to gradient updates
- DINOv2 self-supervised features are particularly fragile
- The iguana training script uses 1e-6 for DINOv2 vs 1e-5 for CNNs

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Head LR | 1e-4 (ViT) / 1e-3 (CNN) | Higher end of typical fine-tuning |
| Backbone LR | Head LR / 100 | The 1:100 ratio |
| Weight decay | 0.05 | Standard for AdamW |
| Epochs | 50-100 | More epochs needed for backbone to adapt |
| Scheduler | Cosine annealing | With 5-epoch warmup |

---

## Strategy 3: Gradual Unfreezing (ULMFiT-style)

**Unfreeze layers progressively from top to bottom.** Higher layers (closer to the classification head) adapt first, then lower layers are gradually unfrozen with decreasing learning rates.

### When to Use
- Dataset size: >200 images per class
- Maximum accuracy is the goal
- Willing to invest 2-3x training time
- Significant domain shift (e.g., underwater, thermal)

### Implementation

```python
import timm
import torch

model = timm.create_model(
    'vit_base_patch14_dinov2.lvd142m',
    pretrained=True,
    num_classes=5,
)

# ViT-B has 12 blocks: blocks.0 (lowest) through blocks.11 (highest)

# Phase 1: Head only (5 epochs)
for param in model.parameters():
    param.requires_grad = False
for param in model.head.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,
    weight_decay=0.05,
)
train(model, optimizer, epochs=5)

# Phase 2: Unfreeze last 2 blocks + norm + head (15 epochs)
for param in model.blocks[10].parameters():
    param.requires_grad = True
for param in model.blocks[11].parameters():
    param.requires_grad = True
for param in model.norm.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW([
    {'params': model.blocks[10].parameters(), 'lr': 1e-5},
    {'params': model.blocks[11].parameters(), 'lr': 5e-5},
    {'params': model.norm.parameters(), 'lr': 5e-5},
    {'params': model.head.parameters(), 'lr': 1e-4},
], weight_decay=0.05)
train(model, optimizer, epochs=15)

# Phase 3: Unfreeze all (20 epochs, very low LR)
for param in model.parameters():
    param.requires_grad = True

# Layer-wise LR decay: lower blocks get lower LRs
num_blocks = len(model.blocks)
param_groups = []
for i, block in enumerate(model.blocks):
    lr = 1e-6 * (2 ** (i / num_blocks))  # Exponential decay from bottom to top
    param_groups.append({'params': block.parameters(), 'lr': lr})
param_groups.append({'params': model.head.parameters(), 'lr': 1e-4})
param_groups.append({'params': model.norm.parameters(), 'lr': 1e-5})

optimizer = torch.optim.AdamW(param_groups, weight_decay=0.05)
train(model, optimizer, epochs=20)
```

### Hyperparameters

| Parameter | Phase 1 | Phase 2 | Phase 3 |
|-----------|---------|---------|---------|
| Frozen layers | All except head | All except last 2 blocks + head | None |
| Head LR | 1e-3 | 1e-4 | 1e-4 |
| Top blocks LR | - | 5e-5 | Exponential decay |
| Bottom blocks LR | - | - | 1e-6 to 5e-6 |
| Epochs | 5 | 15 | 20 |

---

## Strategy 4: Knowledge Distillation

**Train the student model to match both ground-truth labels AND the teacher model's output.** The teacher is a frozen copy of the pretrained model.

### When to Use
- Critical to preserve original model capabilities alongside new ones
- Multi-task scenarios (original classes + new classes)
- Advanced users comfortable with custom loss functions

### Implementation

```python
import timm
import torch
import torch.nn.functional as F

# Teacher: frozen pretrained model
teacher = timm.create_model(
    'vit_base_patch14_dinov2.lvd142m',
    pretrained=True,
    num_classes=num_classes,
)
teacher.eval()
for param in teacher.parameters():
    param.requires_grad = False

# Student: trainable model (same architecture)
student = timm.create_model(
    'vit_base_patch14_dinov2.lvd142m',
    pretrained=True,
    num_classes=num_classes,
)

def distillation_loss(student_logits, teacher_logits, targets, alpha=0.5, temperature=3.0):
    """
    Combined loss: (1-alpha) * hard_loss + alpha * soft_loss

    alpha: weight of distillation loss (0 = only CE, 1 = only KD)
    temperature: softening temperature (higher = softer distributions)
    """
    # Hard loss: standard cross-entropy with ground truth
    hard_loss = F.cross_entropy(student_logits, targets)

    # Soft loss: KL divergence between softened student and teacher distributions
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
    soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)

    return (1 - alpha) * hard_loss + alpha * soft_loss


# Training loop:
for images, targets in train_loader:
    images, targets = images.to(device), targets.to(device)

    with torch.no_grad():
        teacher_logits = teacher(images)

    student_logits = student(images)
    loss = distillation_loss(student_logits, teacher_logits, targets, alpha=0.5, temperature=3.0)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Hyperparameters

| Parameter | Recommended | Range |
|-----------|------------|-------|
| alpha | 0.5 | 0.3-0.7 |
| temperature | 3.0 | 1.0-10.0 |
| Student LR | Same as discriminative LR strategy | - |

**Note on alpha:**
- alpha=0 reduces to standard fine-tuning (no distillation)
- alpha=1 reduces to pure distillation (no ground-truth supervision)
- alpha=0.5 is a balanced starting point

---

## Strategy Selection Summary

```
                    Forgetting Risk
                    Low ←────────→ High

Frozen backbone    ████████████████████   <200 img/class
Discriminative LR  ██████████████         200-1000 img/class
Gradual unfreezing ████████               >200 img/class, max accuracy
Distillation       ████████████████       preserve original capabilities

                    Accuracy Ceiling
                    Low ←────────→ High

Frozen backbone    ████
Discriminative LR  ██████████████
Gradual unfreezing ████████████████████
Distillation       ██████████████████
```
