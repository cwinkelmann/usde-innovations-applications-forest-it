# Model Evaluation Standards for Wildlife Classification

Standards and best practices for evaluating wildlife image classification models. Covers metric selection, reporting requirements, calibration assessment, and threshold optimization.

---

## Metric Hierarchy

### Mandatory Metrics (always report)

| Metric | Formula | Purpose |
|--------|---------|---------|
| Per-class Precision | TP / (TP + FP) | How many predictions for class X are correct |
| Per-class Recall | TP / (TP + FN) | What fraction of true class X is detected |
| Per-class F1 | 2 * P * R / (P + R) | Harmonic mean of P and R |
| Macro F1 | Mean of per-class F1 | Balanced multi-class performance |
| Balanced Accuracy | Mean of per-class recall | Unbiased by class frequency |
| Confusion Matrix | N x N count table | Which classes are confused |

### Optional but Recommended

| Metric | When to Use |
|--------|------------|
| Weighted F1 | When class importance is proportional to frequency |
| Cohen's Kappa | When comparing against random or human baselines |
| Matthews Correlation Coefficient (MCC) | Binary classification with imbalance |
| Expected Calibration Error (ECE) | When confidence scores are used downstream |
| Average Precision (AP) per class | When threshold selection matters |

### Never Use Alone

| Metric | Why It Is Misleading |
|--------|---------------------|
| Overall Accuracy | Dominated by majority class in imbalanced datasets |
| Top-5 Accuracy | Irrelevant for wildlife tasks with fewer than 10 classes |
| Loss value | Not comparable across models or datasets |

**Default for wildlife ecology: Use macro F1** -- rare species matter as much as common ones.

---

## Per-Class vs Macro vs Weighted Metrics

```python
from sklearn.metrics import precision_recall_fscore_support, classification_report

# Per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, average=None, zero_division=0,
)

# Macro average (treats all classes equally)
macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average='macro', zero_division=0,
)

# Weighted average (weights by class support)
weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average='weighted', zero_division=0,
)

# Full classification report
report = classification_report(
    y_true, y_pred,
    target_names=class_names,
    digits=3,
    zero_division=0,
)
print(report)
```

**When to use which average:**
- **Macro** (default for wildlife): Treats every species equally. Preferred when all species matter equally regardless of frequency.
- **Weighted**: Accounts for class frequency. Use when common species are more important.
- **Per-class**: Always report alongside averages to identify specific failure modes.

---

## Confusion Matrix Best Practices

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred, class_names, normalize=True, output_path=None):
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        cm_display = cm
        fmt = 'd'
        title = 'Confusion Matrix (Counts)'

    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) * 0.8)))
    sns.heatmap(
        cm_display, annot=True, fmt=fmt, cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, square=True,
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
```

### What to Look For

1. **Off-diagonal clusters** -- systematic confusion between species pairs
2. **Low diagonal values** -- classes the model struggles with
3. **Asymmetric confusion** -- species A confused as B but not vice versa
4. **Background column dominance** -- model misclassifying animals as background

---

## Confidence Calibration

A model that outputs 80% confidence should be correct 80% of the time.

### Expected Calibration Error (ECE)

```python
def compute_ece(y_true, y_probs, n_bins=15):
    """
    Compute Expected Calibration Error.
    Lower is better. ECE < 0.05 is well-calibrated.
    """
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += (in_bin.sum() / len(y_true)) * abs(avg_accuracy - avg_confidence)

    return ece
```

### Reliability Diagram

```python
from sklearn.calibration import calibration_curve

def plot_reliability_diagram(y_true, y_probs, n_bins=10, output_path=None):
    confidences = np.max(y_probs, axis=1)
    correct = (np.argmax(y_probs, axis=1) == y_true).astype(float)

    fraction_of_positives, mean_predicted_value = calibration_curve(
        correct, confidences, n_bins=n_bins, strategy='uniform'
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax1.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
    ax1.set_xlabel('Mean predicted confidence')
    ax1.set_ylabel('Fraction correct')
    ax1.set_title('Reliability Diagram')
    ax1.legend()

    ax2.hist(confidences, bins=n_bins, range=(0, 1), edgecolor='black')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.set_title('Confidence Distribution')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
```

### Temperature Scaling (Post-Hoc Calibration)

```python
def find_optimal_temperature(logits, labels, lr=0.01, max_iter=1000):
    """Find temperature that minimizes NLL on validation set."""
    temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

    def eval():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(eval)
    return temperature.item()
```

---

## Threshold Sweep for Species Detection

For binary or one-vs-rest evaluation:

```python
from sklearn.metrics import precision_recall_curve

def find_optimal_threshold(y_true_binary, class_probs):
    """Find threshold that maximizes F1 for a single class."""
    precision, recall, thresholds = precision_recall_curve(y_true_binary, class_probs)
    f1_scores = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[min(best_idx, len(thresholds) - 1)]
    return best_threshold, f1_scores[best_idx]
```

---

## Complete Evaluation Pipeline

```python
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

def evaluate_model(model, dataloader, class_names, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Per-class report
    report = classification_report(all_labels, all_preds,
                                   target_names=class_names, digits=3)
    print(report)

    # Balanced accuracy
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    print(f"Balanced accuracy: {bal_acc:.3f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    return report, cm, all_preds, all_labels, all_probs
```

---

## Quality Checklist

- [ ] Per-class P/R/F1 reported (not just accuracy)
- [ ] Macro F1 reported as primary metric
- [ ] Balanced accuracy reported alongside standard accuracy
- [ ] Confusion matrix generated and inspected
- [ ] Confidence distribution checked (ECE if confidence used downstream)
- [ ] Results compared to baseline (random, majority class, SpeciesNet)
- [ ] Evaluation performed on val/test set, NOT training set
