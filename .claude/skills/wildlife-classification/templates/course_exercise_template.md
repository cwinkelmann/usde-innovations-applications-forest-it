# Course Exercise Template

Use this structure for all exercises produced by the `exercise_designer_agent`. Each exercise follows a consistent format with progressive scaffolding.

---

## Exercise [N]: [Title]

**Estimated time:** [N] minutes
**Prerequisites:** [List prior exercises or concepts needed]
**Difficulty:** [Beginner / Intermediate / Advanced]

---

### Learning Objectives

After completing this exercise, you will be able to:

1. [Specific, measurable objective -- uses action verbs: implement, compare, evaluate, explain]
2. [Specific, measurable objective]
3. [Optional third objective]

---

### Background

[2-3 paragraphs providing ecological context and technical motivation. Connect the coding task to a real wildlife conservation scenario.]

[Example: "You are part of a research team monitoring marine iguanas (*Amblyrhynchus cristatus*) on the Galapagos Islands. Your team has collected 500 drone images and manually labeled each 518x518 tile as either 'iguana present' or 'background.' Your goal is to train a classifier that can automate this labeling process for the remaining 50,000 unlabeled tiles."]

[Technical context: "Transfer learning with a pretrained DINOv2 backbone allows you to leverage features learned from millions of images, even though your dataset has only hundreds of labeled examples. The key challenge is avoiding catastrophic forgetting..."]

---

### Setup

Run this cell first. Do not modify.

```python
# ============================================================
# Setup -- do not modify this cell
# ============================================================
import torch
import torch.nn as nn
import numpy as np
import timm
from pathlib import Path

# Device selection
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")

# Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 5  # Adjust for your dataset
INPUT_SIZE = 518

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
```

---

### Task [N.1]: [First subtask title]

[Clear description of what the student needs to do. 2-3 sentences.]

```python
# ============================================================
# Task [N.1]: [Brief instruction]
# ============================================================

# TODO: [Clear, specific instruction for what to implement]
# Hint: Use timm.create_model() with pretrained=True
# Expected output: a model object with [N] classes

model = None  # Replace with your implementation

# --- Your code here ---


# --- End your code ---
```

#### Verification

```python
# Verification -- do not modify
assert model is not None, "Model not created. Check your create_model call."
total_params = sum(p.numel() for p in model.parameters())
print(f"Model created with {total_params:,} parameters")
print(f"Output classes: {model.head.out_features if hasattr(model, 'head') else 'unknown'}")
print("Task [N.1] complete!")
```

---

### Task [N.2]: [Second subtask title]

[Clear description of what the student needs to do.]

```python
# ============================================================
# Task [N.2]: [Brief instruction]
# ============================================================

# TODO: [Specific instruction]
# Hint: [First hint -- conceptual direction]
# Expected: [What the output should look like]

# --- Your code here ---


# --- End your code ---
```

#### Verification

```python
# Verification -- do not modify
# [Assertions and print statements that check correctness]
print("Task [N.2] complete!")
```

---

### Task [N.3]: [Third subtask title]

[If needed -- some exercises may have 2-4 subtasks]

```python
# ============================================================
# Task [N.3]: [Brief instruction]
# ============================================================

# TODO: [Specific instruction]

# --- Your code here ---


# --- End your code ---
```

---

### Hints

If you get stuck, expand the hints below (try each hint before moving to the next):

<details>
<summary>Hint 1: [Conceptual hint title]</summary>

[Conceptual guidance -- what approach to take, not exact code]

Example: "To freeze the backbone, iterate over `model.named_parameters()` and set `requires_grad = False` for any parameter whose name does not contain 'head'."
</details>

<details>
<summary>Hint 2: [More specific hint title]</summary>

[More specific guidance -- maybe a code pattern without exact values]

```python
# Pattern:
for name, param in model.named_parameters():
    if 'head' not in name:
        param.requires_grad = ???
```
</details>

<details>
<summary>Hint 3: [Near-solution hint]</summary>

[Almost-complete solution with one key value missing]

```python
for name, param in model.named_parameters():
    if 'head' not in name:
        param.requires_grad = False
# Check: how many parameters are now trainable?
```
</details>

---

### Solution

<details>
<summary>Click to reveal the complete solution</summary>

```python
# ============================================================
# Complete solution for Exercise [N]
# ============================================================

# Task [N.1]
model = timm.create_model(
    'vit_base_patch14_dinov2.lvd142m',
    pretrained=True,
    num_classes=NUM_CLASSES,
)
model = model.to(DEVICE)

# Task [N.2]
for name, param in model.named_parameters():
    if 'head' not in name:
        param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

# Task [N.3]
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=0.05,
)
```
</details>

---

### Reflection Questions

Answer these questions after completing the exercise (no code needed):

1. **Ecological question:** [Question connecting the code to wildlife conservation]
   - Example: "Why is it important that our train/val split separates different survey sites rather than randomly splitting individual tiles?"

2. **Technical question:** [Question about trade-offs or design decisions]
   - Example: "What would happen if we used a learning rate of 1e-3 for the backbone instead of 1e-6? How would this affect the model's general feature extraction ability?"

3. **Application question:** [Question about applying this to the student's own work]
   - Example: "If you were to apply this approach to your own species, what would you change about the augmentation pipeline and why?"

---

### Extension Challenge (Optional)

For students who finish early:

[An open-ended extension that goes beyond the exercise, e.g.:]
- "Modify the training loop to use a cosine annealing scheduler and compare the learning curves."
- "Implement WeightedRandomSampler to handle the class imbalance in your dataset."
- "Add W&B logging and track the per-class accuracy over training."
