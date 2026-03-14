# Class Imbalance Strategies for Wildlife Classification

Wildlife datasets are almost always imbalanced -- common species dominate, rare species are underrepresented, and "background/empty" images may outnumber all species combined. This reference covers detection, measurement, and mitigation of class imbalance.

---

## Measuring Imbalance

```python
from collections import Counter
import numpy as np

def measure_imbalance(dataset):
    """Compute imbalance metrics for an ImageFolder dataset."""
    labels = [label for _, label in dataset.samples]
    counts = Counter(labels)

    total = sum(counts.values())
    max_count = max(counts.values())
    min_count = min(counts.values())

    # Imbalance ratio
    ir = max_count / max(min_count, 1)

    # Effective number of classes (Shannon entropy-based)
    probs = np.array([c / total for c in counts.values()])
    entropy = -np.sum(probs * np.log(probs))
    effective_classes = np.exp(entropy)

    print(f"Total samples: {total}")
    print(f"Classes: {len(counts)}")
    print(f"Imbalance ratio: {ir:.1f}:1")
    print(f"Effective classes: {effective_classes:.1f} / {len(counts)}")

    for cls_idx in sorted(counts.keys()):
        cls_name = dataset.classes[cls_idx]
        count = counts[cls_idx]
        pct = 100 * count / total
        print(f"  {cls_name}: {count} ({pct:.1f}%)")

    return counts, ir
```

### Imbalance Severity Thresholds

| Imbalance Ratio | Severity | Impact | Action Required |
|-----------------|----------|--------|-----------------|
| < 2:1 | Mild | Negligible | None |
| 2:1 - 5:1 | Moderate | Minority class recall drops | WeightedRandomSampler |
| 5:1 - 10:1 | Severe | Model biased toward majority | Sampler + class weights in loss |
| 10:1 - 50:1 | Extreme | Minority class nearly ignored | Sampler + focal loss + augmentation |
| > 50:1 | Critical | Consider merging or excluding | Restructure the classification task |

---

## Strategy 1: WeightedRandomSampler (Recommended Default)

Oversample minority classes so each class appears equally often per epoch.

```python
from torch.utils.data import WeightedRandomSampler
import numpy as np

def create_weighted_sampler(dataset):
    """Create a WeightedRandomSampler for class-balanced training."""
    labels = [label for _, label in dataset.samples]
    class_counts = np.bincount(labels)

    # Weight per class = 1 / count
    class_weights = 1.0 / class_counts
    # Weight per sample = weight of its class
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),    # Sample same number as dataset size
        replacement=True,             # Must be True for oversampling
    )

    return sampler

# Usage:
sampler = create_weighted_sampler(train_dataset)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler,         # Replaces shuffle=True
    # shuffle=False,         # Cannot use both sampler and shuffle
    num_workers=4,
    pin_memory=True,
)
```

**Important:** When using a sampler, do NOT set `shuffle=True` in the DataLoader -- they are mutually exclusive.

**When to use:** Imbalance ratio > 3:1. This is the safest default.

---

## Strategy 2: Class-Weighted Cross-Entropy Loss

Penalizes misclassification of rare species more heavily.

```python
import torch
import torch.nn as nn
import numpy as np

def compute_class_weights(dataset, method='inverse'):
    """
    Compute class weights for weighted cross-entropy.

    Methods:
    - 'inverse': weight = total / (num_classes * count_per_class)
    - 'sqrt_inverse': weight = sqrt(max_count / count_per_class)
    - 'effective': effective number of samples (Cui et al., 2019)
    """
    labels = [label for _, label in dataset.samples]
    class_counts = np.bincount(labels)
    num_classes = len(class_counts)
    total = sum(class_counts)

    if method == 'inverse':
        weights = total / (num_classes * class_counts)
    elif method == 'sqrt_inverse':
        max_count = max(class_counts)
        weights = np.sqrt(max_count / class_counts)
    elif method == 'effective':
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes
    else:
        raise ValueError(f"Unknown method: {method}")

    return torch.FloatTensor(weights)

# Usage:
class_weights = compute_class_weights(train_dataset, method='effective')
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
```

**When to use:** Moderate imbalance (3:1 to 10:1). Can combine with WeightedRandomSampler.

---

## Strategy 3: Focal Loss

Down-weights well-classified examples, forces model to focus on hard samples.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma: focusing parameter (higher = more focus on hard examples)
        gamma=0 reduces to standard cross-entropy
        gamma=2 is the recommended default (Lin et al., 2017)

    alpha: class weight (optional, can combine with class imbalance weights)
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Can be a tensor of per-class weights
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t = probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# Usage:
class_weights = compute_class_weights(train_dataset, method='effective')
criterion = FocalLoss(gamma=2.0, alpha=class_weights.to(device))
```

**When to use:** Extreme imbalance (>10:1) or when model is confident on majority class.

---

## Strategy 4: Oversampling with Augmentation

Copy minority class samples and apply heavy augmentation to create diversity.

```python
from torchvision.datasets import ImageFolder
from torch.utils.data import ConcatDataset

# Create extra copies of minority class with different augmentations
minority_dataset = ImageFolder('data/train/rare_species/', transform=heavy_augment)
combined = ConcatDataset([full_train_dataset, minority_dataset, minority_dataset])
```

**When to use:** <50 images in minority class. Risk: overfitting if augmentation is not diverse enough.

---

## Decision Matrix

| Imbalance Ratio | Images/Minority Class | Recommended Strategy |
|-----------------|----------------------|----------------------|
| < 3:1 | > 200 | No action needed |
| 3:1 - 10:1 | > 100 | WeightedRandomSampler |
| 3:1 - 10:1 | 50-100 | Sampler + class weights in loss |
| > 10:1 | > 100 | Sampler + Focal Loss |
| > 10:1 | < 50 | Oversampling + augmentation + Focal Loss |

---

## Iguana Training Example

The iguana training codebase addresses imbalance through the `positive_ratio` parameter in `IguanaPresenceDataset`:

```python
dataset = IguanaPresenceDataset(
    positive_ratio=0.5,  # 50% of crops contain iguanas, 50% background
)
```

This is a form of balanced sampling built directly into the dataset -- equivalent to WeightedRandomSampler but implemented at the dataset level.

---

## Evaluation with Imbalanced Data

Standard accuracy is misleading with imbalanced data. Use these metrics instead:

| Metric | What It Shows | When to Use |
|--------|--------------|-------------|
| Balanced accuracy | Mean per-class accuracy | Always with imbalanced data |
| Macro F1 | Mean of per-class F1 | Standard multi-class metric |
| Per-class recall | Individual class detection rate | Identify failing classes |

**Critical rule:** Never report only accuracy on imbalanced datasets. A model predicting the majority class achieves high accuracy but zero recall on minority classes.

---

## Common Mistakes

1. **Ignoring imbalance** -- model learns to always predict majority class
2. **Using accuracy as metric** -- 95% accuracy means nothing if 95% of data is one class; use macro F1
3. **Combining sampler with shuffle=True** -- DataLoader raises error; sampler replaces shuffle
4. **Over-correcting** -- too-high class weights cause model to overpredict minority class
