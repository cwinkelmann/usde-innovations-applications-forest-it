#!/usr/bin/env python3
"""
Wildlife Classification -- Evaluation Report Template
Loads a trained checkpoint and produces a comprehensive evaluation report:
per-class metrics, confusion matrix, calibration analysis, threshold sweep.

Usage:
    python evaluation_template.py \
        --checkpoint checkpoints/best_model.pth \
        --data-dir path/to/imagefolder/test \
        --model-name vit_base_patch14_dinov2.lvd142m \
        --output-dir evaluation_results/
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)

import timm
from timm.data import resolve_data_config, create_transform

# ============================================================================
# Device
# ============================================================================

def get_device(preferred=None):
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ============================================================================
# Model loading
# ============================================================================

def load_model(checkpoint_path, model_name, num_classes, device):
    """Load model from checkpoint."""
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)

    return model.to(device).eval(), ckpt


# ============================================================================
# Inference
# ============================================================================

@torch.no_grad()
def run_inference(model, loader, device):
    """Run inference and collect predictions, labels, and probabilities."""
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ============================================================================
# Metrics
# ============================================================================

def compute_ece(y_true, y_probs, n_bins=15):
    """Compute Expected Calibration Error."""
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


def generate_report(preds, labels, probs, class_names, output_dir):
    """Generate comprehensive evaluation report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 70)
    lines.append("WILDLIFE CLASSIFICATION EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Summary
    accuracy = np.mean(preds == labels)
    bal_acc = balanced_accuracy_score(labels, preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    ece = compute_ece(labels, probs)

    lines.append("SUMMARY")
    lines.append("-" * 70)
    lines.append(f"  Total samples:       {len(labels)}")
    lines.append(f"  Classes:             {len(class_names)}")
    lines.append(f"  Overall accuracy:    {accuracy:.4f}")
    lines.append(f"  Balanced accuracy:   {bal_acc:.4f}")
    lines.append(f"  Macro precision:     {precision:.4f}")
    lines.append(f"  Macro recall:        {recall:.4f}")
    lines.append(f"  Macro F1:            {f1:.4f}")
    lines.append(f"  ECE:                 {ece:.4f}")
    lines.append("")

    # Per-class report
    lines.append("PER-CLASS METRICS")
    lines.append("-" * 70)
    report = classification_report(labels, preds, target_names=class_names, digits=3, zero_division=0)
    lines.append(report)

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)

    lines.append("CONFUSION MATRIX (COUNTS)")
    lines.append("-" * 70)
    header = "".ljust(20) + "".join(n[:10].ljust(12) for n in class_names)
    lines.append(header)
    for i, row in enumerate(cm):
        line = class_names[i][:18].ljust(20) + "".join(str(v).ljust(12) for v in row)
        lines.append(line)
    lines.append("")

    lines.append("CONFUSION MATRIX (NORMALIZED)")
    lines.append("-" * 70)
    lines.append(header)
    for i, row in enumerate(cm_normalized):
        line = class_names[i][:18].ljust(20) + "".join(f"{v:.3f}".ljust(12) for v in row)
        lines.append(line)
    lines.append("")

    # Confidence analysis
    confidences = np.max(probs, axis=1)
    correct_mask = preds == labels

    lines.append("CONFIDENCE ANALYSIS")
    lines.append("-" * 70)
    lines.append(f"  Mean confidence (all):       {confidences.mean():.3f}")
    lines.append(f"  Mean confidence (correct):   {confidences[correct_mask].mean():.3f}")
    if (~correct_mask).any():
        lines.append(f"  Mean confidence (incorrect): {confidences[~correct_mask].mean():.3f}")
    lines.append(f"  Median confidence:           {np.median(confidences):.3f}")
    lines.append(f"  Min confidence:              {confidences.min():.3f}")
    lines.append("")

    # Top confusion pairs
    lines.append("TOP CONFUSION PAIRS")
    lines.append("-" * 70)
    confusions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusions.append((class_names[i], class_names[j], cm[i, j], cm_normalized[i, j]))
    confusions.sort(key=lambda x: -x[2])
    for true_cls, pred_cls, count, rate in confusions[:10]:
        lines.append(f"  {true_cls} -> {pred_cls}: {count} ({rate:.1%})")
    lines.append("")

    # Write report
    report_text = "\n".join(lines)
    print(report_text)

    report_path = output_dir / 'evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"\nReport saved to: {report_path}")

    # Save metrics as JSON
    metrics = {
        'accuracy': float(accuracy),
        'balanced_accuracy': float(bal_acc),
        'macro_precision': float(precision),
        'macro_recall': float(recall),
        'macro_f1': float(f1),
        'ece': float(ece),
        'total_samples': int(len(labels)),
        'num_classes': len(class_names),
        'class_names': class_names,
    }
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    # Save confusion matrix as numpy
    np.save(output_dir / 'confusion_matrix.npy', cm)

    return metrics


# ============================================================================
# Main
# ============================================================================

def main(args):
    device = get_device(args.device)

    # Load checkpoint metadata
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    class_names = ckpt.get('class_names', None)
    num_classes = ckpt.get('num_classes', args.num_classes)

    # Build model
    model, _ = load_model(args.checkpoint, args.model_name, num_classes, device)

    # Build transform
    input_size = ckpt.get('input_size', 518)
    val_transform = transforms.Compose([
        transforms.Resize(
            (input_size, input_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load test data
    test_ds = ImageFolder(args.data_dir, transform=val_transform)
    if class_names is None:
        class_names = test_ds.classes

    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    # Run inference
    preds, labels, probs = run_inference(model, test_loader, device)

    # Generate report
    generate_report(preds, labels, probs, class_names, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate wildlife classifier')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True, help='Test set ImageFolder path')
    parser.add_argument('--model-name', type=str, default='vit_base_patch14_dinov2.lvd142m')
    parser.add_argument('--num-classes', type=int, default=None, help='Override if not in checkpoint')
    parser.add_argument('--output-dir', type=str, default='evaluation_results/')
    parser.add_argument('--device', type=str, default=None)
    main(parser.parse_args())
