#!/usr/bin/env python
"""
Diagnose training runs and suggest hyperparameter fixes.

Reads results.csv from ultralytics training runs (or fetches from WandB) and
detects common training pathologies:

1. **Early F1/mAP dip** — pretrained features disrupted before model recovers.
   Fix: freeze backbone layers, increase warmup, lower backbone LR.

2. **Overfitting** — val metrics plateau/decrease while train loss keeps dropping.
   Fix: early stopping, more augmentation, reduce epochs.

3. **Underfitting** — val metrics plateau early and never improve.
   Fix: unfreeze more layers, increase LR, train longer.

4. **LR too high** — loss spikes or oscillates.
   Fix: reduce lr0, increase warmup.

Usage:
    # Analyze a local training run
    python scripts/training/diagnose_training.py \
        --results_dir ./output/md1000-larch-50ep/

    # Analyze from WandB
    python scripts/training/diagnose_training.py \
        --wandb_project wildlife-detection \
        --wandb_run md1000-larch-50ep

    # Analyze and generate a fixed training command
    python scripts/training/diagnose_training.py \
        --results_dir ./output/md1000-larch-50ep/ \
        --suggest_cmd
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ─── Data loading ────────────────────────────────────────────────────────────

def load_from_csv(results_dir: str) -> dict:
    """Load metrics from ultralytics results.csv."""
    csv_path = Path(results_dir) / "results.csv"
    if not csv_path.exists():
        # Try looking inside runs/detect/output/<name>/
        for candidate in Path(results_dir).rglob("results.csv"):
            csv_path = candidate
            break
        else:
            raise FileNotFoundError(f"No results.csv found in {results_dir}")

    metrics = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                key = key.strip()
                if key not in metrics:
                    metrics[key] = []
                try:
                    metrics[key].append(float(val))
                except (ValueError, TypeError):
                    metrics[key].append(None)

    # Compute F1 if not present
    if "metrics/F1(B)" not in metrics and "metrics/precision(B)" in metrics:
        metrics["metrics/F1(B)"] = []
        for p, r in zip(metrics["metrics/precision(B)"], metrics["metrics/recall(B)"]):
            if p is not None and r is not None:
                f1 = 2 * p * r / max(p + r, 1e-8)
                metrics["metrics/F1(B)"].append(f1)
            else:
                metrics["metrics/F1(B)"].append(None)

    # Also load args.yaml if present for model info
    args_path = Path(csv_path).parent / "args.yaml"
    train_args = {}
    if args_path.exists():
        import yaml
        with open(args_path) as f:
            train_args = yaml.safe_load(f)

    return metrics, train_args


def load_from_wandb(project: str, run_name: str) -> dict:
    """Load metrics from WandB."""
    import wandb
    api = wandb.Api()
    runs = api.runs(project, filters={"display_name": run_name})
    run = next(iter(runs), None)
    if run is None:
        raise ValueError(f"Run '{run_name}' not found in project '{project}'")

    history = run.history(samples=500)
    metrics = {}
    for col in history.columns:
        vals = history[col].tolist()
        metrics[col] = [v if not (isinstance(v, float) and np.isnan(v)) else None for v in vals]

    return metrics, dict(run.config)


# ─── Diagnosis ───────────────────────────────────────────────────────────────

@dataclass
class Diagnosis:
    name: str
    severity: str  # "info", "warning", "critical"
    description: str
    evidence: str
    fix: str
    hyperparams: dict = field(default_factory=dict)


def _clean(series):
    """Remove None values and return as numpy array."""
    return np.array([v for v in series if v is not None])


def diagnose_early_dip(metrics: dict) -> list[Diagnosis]:
    """Detect early F1/mAP dip (pretrained features disrupted)."""
    diagnoses = []

    for metric_name in ["metrics/F1(B)", "metrics/mAP50(B)", "metrics/precision(B)"]:
        vals = metrics.get(metric_name)
        if not vals or len(vals) < 5:
            continue

        clean = _clean(vals)
        if len(clean) < 5:
            continue

        # Check if metric drops in first 5 epochs then recovers
        initial = clean[0]
        first_5_min = clean[:5].min()
        first_5_min_idx = clean[:5].argmin()
        recovery = clean[min(10, len(clean) - 1)]

        # Dip = drops >10% from initial, then recovers to near or above initial
        dip_pct = (initial - first_5_min) / max(initial, 1e-8) * 100

        if dip_pct > 10 and first_5_min_idx > 0 and recovery > first_5_min * 1.1:
            severity = "critical" if dip_pct > 30 else "warning"
            diagnoses.append(Diagnosis(
                name=f"early_dip_{metric_name.split('/')[-1]}",
                severity=severity,
                description=f"Early training dip in {metric_name.split('/')[-1]}",
                evidence=(
                    f"Epoch 0: {initial:.4f} → min at epoch {first_5_min_idx}: "
                    f"{first_5_min:.4f} ({dip_pct:+.1f}%) → epoch 10: {recovery:.4f}"
                ),
                fix=(
                    "Freeze backbone layers for the first 5-10 epochs to prevent "
                    "disrupting pretrained features. Increase warmup to 5 epochs. "
                    "Consider lowering lr0."
                ),
                hyperparams={"freeze": 10, "warmup_epochs": 5},
            ))

    return diagnoses


def diagnose_overfitting(metrics: dict) -> list[Diagnosis]:
    """Detect overfitting: val metrics decrease while train loss drops."""
    diagnoses = []

    train_loss = metrics.get("train/box_loss")
    val_map = metrics.get("metrics/mAP50(B)")

    if not train_loss or not val_map or len(train_loss) < 10:
        return diagnoses

    clean_loss = _clean(train_loss)
    clean_map = _clean(val_map)
    n = min(len(clean_loss), len(clean_map))

    if n < 10:
        return diagnoses

    # Check last 30% of training
    split = int(n * 0.7)
    late_loss = clean_loss[split:]
    late_map = clean_map[split:]

    # Train loss still decreasing?
    loss_decreasing = late_loss[-1] < late_loss[0] * 0.95
    # Val mAP decreasing or flat?
    best_map_epoch = clean_map.argmax()
    map_from_best = clean_map[-1] / max(clean_map.max(), 1e-8)

    if loss_decreasing and map_from_best < 0.97 and best_map_epoch < n * 0.8:
        gap = clean_map.max() - clean_map[-1]
        diagnoses.append(Diagnosis(
            name="overfitting",
            severity="warning",
            description="Validation mAP peaked early and declined while training loss kept decreasing",
            evidence=(
                f"Best mAP@0.5={clean_map.max():.4f} at epoch {best_map_epoch}, "
                f"final={clean_map[-1]:.4f} (gap={gap:.4f}). "
                f"Train loss still decreasing: {late_loss[0]:.4f} → {late_loss[-1]:.4f}"
            ),
            fix=(
                "Train fewer epochs or use patience-based early stopping. "
                "Increase augmentation (mosaic, mixup). "
                "Add weight decay or dropout."
            ),
            hyperparams={"patience": 10, "mosaic": 1.0, "mixup": 0.1},
        ))

    return diagnoses


def diagnose_lr_issues(metrics: dict) -> list[Diagnosis]:
    """Detect learning rate problems: spikes, oscillation, NaN."""
    diagnoses = []

    for loss_key in ["train/box_loss", "train/cls_loss"]:
        vals = metrics.get(loss_key)
        if not vals or len(vals) < 5:
            continue

        clean = _clean(vals)
        if len(clean) < 5:
            continue

        # Check for NaN
        if any(v is None for v in vals):
            nan_epoch = next(i for i, v in enumerate(vals) if v is None)
            diagnoses.append(Diagnosis(
                name=f"nan_loss_{loss_key.split('/')[-1]}",
                severity="critical",
                description=f"NaN loss detected in {loss_key}",
                evidence=f"NaN at epoch {nan_epoch}",
                fix="Reduce lr0 by 10x. For RT-DETR, set amp=False.",
                hyperparams={"lr0": 0.0001},
            ))
            continue

        # Check for spikes (>2x rolling average)
        window = 3
        if len(clean) > window + 2:
            for i in range(window, len(clean)):
                rolling_avg = clean[i - window:i].mean()
                if clean[i] > rolling_avg * 2:
                    diagnoses.append(Diagnosis(
                        name=f"loss_spike_{loss_key.split('/')[-1]}",
                        severity="warning",
                        description=f"Loss spike in {loss_key}",
                        evidence=f"Epoch {i}: {clean[i]:.4f} vs rolling avg {rolling_avg:.4f}",
                        fix="Reduce lr0 or increase warmup. Check for corrupted images.",
                        hyperparams={"lr0": 0.0005, "warmup_epochs": 5},
                    ))
                    break  # Report only first spike

    return diagnoses


def diagnose_underfitting(metrics: dict) -> list[Diagnosis]:
    """Detect underfitting: metrics plateau early."""
    diagnoses = []

    val_map = metrics.get("metrics/mAP50(B)")
    if not val_map or len(val_map) < 15:
        return diagnoses

    clean = _clean(val_map)
    if len(clean) < 15:
        return diagnoses

    # Check if mAP plateaus in second half (< 2% improvement)
    mid = len(clean) // 2
    first_half_max = clean[:mid].max()
    second_half_max = clean[mid:].max()
    improvement = (second_half_max - first_half_max) / max(first_half_max, 1e-8) * 100

    # Also check if final mAP is low
    if second_half_max < 0.5 and improvement < 5:
        diagnoses.append(Diagnosis(
            name="underfitting",
            severity="warning",
            description="Model appears to underfit — low mAP with minimal improvement in second half",
            evidence=(
                f"mAP@0.5 first half max: {first_half_max:.4f}, "
                f"second half max: {second_half_max:.4f} "
                f"(+{improvement:.1f}% improvement)"
            ),
            fix=(
                "Increase lr0. Unfreeze more layers (reduce freeze). "
                "Train longer. Check dataset quality."
            ),
            hyperparams={"lr0": 0.005, "freeze": 0, "epochs": 100},
        ))

    return diagnoses


def run_all_diagnoses(metrics: dict) -> list[Diagnosis]:
    """Run all diagnostic checks."""
    all_diag = []
    all_diag.extend(diagnose_early_dip(metrics))
    all_diag.extend(diagnose_overfitting(metrics))
    all_diag.extend(diagnose_lr_issues(metrics))
    all_diag.extend(diagnose_underfitting(metrics))
    return all_diag


# ─── Reporting ───────────────────────────────────────────────────────────────

def print_summary(metrics: dict, train_args: dict):
    """Print training run summary."""
    print(f"\n{'=' * 65}")
    print(f" Training Run Summary")
    print(f"{'=' * 65}")

    if train_args:
        model = train_args.get("model", "unknown")
        epochs = train_args.get("epochs", "?")
        batch = train_args.get("batch", "?")
        imgsz = train_args.get("imgsz", "?")
        lr0 = train_args.get("lr0", "?")
        freeze = train_args.get("freeze", "none")
        warmup = train_args.get("warmup_epochs", "?")
        print(f"  Model: {model}")
        print(f"  Epochs: {epochs}, Batch: {batch}, ImgSz: {imgsz}")
        print(f"  LR: {lr0}, Freeze: {freeze}, Warmup: {warmup}")

    # Final metrics
    for key in ["metrics/precision(B)", "metrics/recall(B)", "metrics/F1(B)",
                 "metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
        vals = metrics.get(key)
        if vals:
            clean = _clean(vals)
            if len(clean) > 0:
                label = key.split("/")[-1].replace("(B)", "")
                print(f"  {label:>15}: final={clean[-1]:.4f}, "
                      f"best={clean.max():.4f} (epoch {clean.argmax()})")


def print_diagnoses(diagnoses: list[Diagnosis]):
    """Print diagnosis results."""
    if not diagnoses:
        print("\n  No issues detected. Training looks healthy.")
        return

    severity_icons = {"info": "ℹ", "warning": "⚠", "critical": "✘"}
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    sorted_diag = sorted(diagnoses, key=lambda d: severity_order.get(d.severity, 3))

    print(f"\n{'─' * 65}")
    print(f" Diagnoses ({len(diagnoses)} issues found)")
    print(f"{'─' * 65}")

    for d in sorted_diag:
        icon = severity_icons.get(d.severity, "?")
        print(f"\n  {icon} [{d.severity.upper()}] {d.description}")
        print(f"    Evidence: {d.evidence}")
        print(f"    Fix: {d.fix}")
        if d.hyperparams:
            print(f"    Suggested params: {d.hyperparams}")


def suggest_command(train_args: dict, diagnoses: list[Diagnosis]):
    """Generate a suggested training command with fixes applied."""
    if not diagnoses:
        return

    # Merge all suggested hyperparams
    fixes = {}
    for d in diagnoses:
        fixes.update(d.hyperparams)

    # Build command
    model = train_args.get("model", "MD1000-larch")
    data = train_args.get("data", "./week1/data/eikelboom_yolo_tiled/dataset.yaml")
    epochs = fixes.get("epochs", train_args.get("epochs", 50))
    batch = train_args.get("batch", 16)
    imgsz = train_args.get("imgsz", 640)
    freeze = fixes.get("freeze", train_args.get("freeze"))
    warmup = fixes.get("warmup_epochs")
    lr0 = fixes.get("lr0")
    patience = fixes.get("patience")

    print(f"\n{'─' * 65}")
    print(f" Suggested training command")
    print(f"{'─' * 65}")
    cmd = (
        f"python scripts/training/train_megadetector.py \\\n"
        f"  --model {model} \\\n"
        f"  --data {data} \\\n"
        f"  --epochs {epochs} --batch {batch} --imgsz {imgsz}"
    )
    if freeze is not None:
        cmd += f" \\\n  --freeze {freeze}"
    if lr0 is not None:
        cmd += f" \\\n  --lr0 {lr0}"
    if warmup is not None:
        cmd += f" \\\n  --warmup_epochs {warmup}"
    if patience is not None:
        cmd += f" \\\n  --patience {patience}"

    cmd += " \\\n  --log wandb"
    print(f"\n  {cmd}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Diagnose training runs and suggest fixes")
    p.add_argument("--results_dir", default=None,
                   help="Path to ultralytics training output (contains results.csv)")
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_run", default=None)
    p.add_argument("--suggest_cmd", action="store_true",
                   help="Generate a suggested fix command")
    return p.parse_args()


def main():
    args = parse_args()

    if args.results_dir:
        metrics, train_args = load_from_csv(args.results_dir)
        print(f"Loaded from: {args.results_dir}")
    elif args.wandb_project and args.wandb_run:
        metrics, train_args = load_from_wandb(args.wandb_project, args.wandb_run)
        print(f"Loaded from WandB: {args.wandb_project}/{args.wandb_run}")
    else:
        print("Error: provide --results_dir or --wandb_project + --wandb_run")
        sys.exit(1)

    print_summary(metrics, train_args)
    diagnoses = run_all_diagnoses(metrics)
    print_diagnoses(diagnoses)

    if args.suggest_cmd:
        suggest_command(train_args, diagnoses)


if __name__ == "__main__":
    main()
