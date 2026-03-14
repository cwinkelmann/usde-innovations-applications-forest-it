#!/usr/bin/env python3
"""
Wildlife Species Classification -- timm Fine-Tuning Template
Fine-tune any timm model with discriminative learning rates.
Mirrors patterns from iguana_train.py and run_training_iguana.sh.

Default: DINOv2 ViT-B @ 518x518 with discriminative LRs (backbone=1e-6, head=1e-4).

Usage:
    python training_script_template.py \
        --data-dir path/to/imagefolder \
        --train-split train \
        --val-split val \
        --model vit_base_patch14_dinov2.lvd142m \
        --num-classes 5 \
        --epochs 100 \
        --output ./output \
        --experiment my_species_experiment \
        --amp

Directory structure expected:
    data-dir/
        train-split/
            class_a/
            class_b/
        val-split/
            class_a/
            class_b/
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, balanced_accuracy_score

import timm
from timm.data import resolve_data_config

# ============================================================================
# Device selection
# ============================================================================

def get_device(preferred=None):
    """Select best available device: CUDA > MPS > CPU."""
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ============================================================================
# Model creation
# ============================================================================

def create_model(args):
    """Create a timm model with pretrained weights and custom head."""
    model = timm.create_model(
        args.model,
        pretrained=True,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
    )
    return model


# ============================================================================
# Parameter groups for discriminative learning rates
# ============================================================================

def create_param_groups(model, backbone_lr, head_lr, weight_decay):
    """
    Separate model parameters into backbone and head groups.

    The head gets head_lr; all other parameters get backbone_lr.
    For frozen-backbone training, set backbone_lr=0 and filter requires_grad.
    """
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'head' in name or 'classifier' in name or 'fc' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': weight_decay},
        {'params': head_params, 'lr': head_lr, 'weight_decay': weight_decay},
    ]

    n_backbone = sum(p.numel() for p in backbone_params)
    n_head = sum(p.numel() for p in head_params)
    print(f"Parameter groups: backbone={n_backbone:,} (lr={backbone_lr}), "
          f"head={n_head:,} (lr={head_lr})")

    return param_groups


# ============================================================================
# Data loading
# ============================================================================

def build_transforms(input_size):
    """Build train and validation transforms with ImageNet normalization."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(
            (input_size, input_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(
            (input_size, input_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, val_transform


def build_dataloaders(args, train_transform, val_transform):
    """Create train and validation DataLoaders from ImageFolder."""
    train_dir = os.path.join(args.data_dir, args.train_split)
    val_dir = os.path.join(args.data_dir, args.val_split)

    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform)

    print(f"Train: {len(train_dataset)} images, {len(train_dataset.classes)} classes")
    print(f"Val:   {len(val_dataset)} images, {len(val_dataset.classes)} classes")
    print(f"Classes: {train_dataset.classes}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset.classes


# ============================================================================
# Training loop
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """Train for one epoch. Returns average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(device.type, enabled=(scaler is not None)):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * targets.size(0)
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, amp_enabled=False):
    """Validate the model. Returns loss, accuracy, and per-sample predictions."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device.type, enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, targets)

        total_loss += loss.item() * targets.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())

    metrics = {
        'loss': total_loss / total,
        'accuracy': correct / total,
        'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
    }

    return metrics


# ============================================================================
# Checkpoint management
# ============================================================================

def save_checkpoint(model, optimizer, epoch, metrics, class_names, output_dir, filename='best.pth'):
    """Save model checkpoint with metadata."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'class_names': class_names,
        'num_classes': len(class_names),
    }
    path = os.path.join(output_dir, filename)
    torch.save(state, path)
    print(f"  Saved checkpoint: {path} (val_acc={metrics['accuracy']:.4f})")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Wildlife Classification Fine-Tuning')

    # Dataset
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Root directory containing train/val splits')
    parser.add_argument('--train-split', type=str, default='train',
                        help='Training split directory name')
    parser.add_argument('--val-split', type=str, default='val',
                        help='Validation split directory name')

    # Model
    parser.add_argument('--model', type=str, default='vit_base_patch14_dinov2.lvd142m',
                        help='timm model name')
    parser.add_argument('--num-classes', type=int, required=True,
                        help='Number of species classes')
    parser.add_argument('--input-size', type=int, default=518,
                        help='Input image size (default: 518 for DINOv2)')
    parser.add_argument('--drop', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--drop-path', type=float, default=0.0, help='Drop path rate')

    # Optimizer
    parser.add_argument('--backbone-lr', type=float, default=1e-6,
                        help='Backbone learning rate (default: 1e-6 for DINOv2)')
    parser.add_argument('--head-lr', type=float, default=1e-4,
                        help='Head learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=20)

    # Training
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Enable automatic mixed precision')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda:0, mps, cpu)')
    parser.add_argument('--workers', type=int, default=4)

    # Output
    parser.add_argument('--output', type=str, default='./output',
                        help='Output directory')
    parser.add_argument('--experiment', type=str, default='wildlife_classification',
                        help='Experiment name for organizing outputs')
    parser.add_argument('--checkpoint-hist', type=int, default=1,
                        help='Number of checkpoints to keep')

    args = parser.parse_args()

    # Setup
    device = get_device(args.device)
    print(f"Device: {device}")

    output_dir = os.path.join(args.output, args.experiment)
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Model
    model = create_model(args)
    model = model.to(device)
    print(f"Model: {args.model}, num_classes={args.num_classes}")

    # Data
    train_transform, val_transform = build_transforms(args.input_size)
    train_loader, val_loader, class_names = build_dataloaders(args, train_transform, val_transform)

    # Optimizer with discriminative LRs
    param_groups = create_param_groups(model, args.backbone_lr, args.head_lr, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # AMP
    scaler = torch.amp.GradScaler(device.type) if args.amp and device.type == 'cuda' else None

    # Training loop
    best_accuracy = 0.0
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'Val Loss':>12} "
          f"{'Val Acc':>10} {'Bal Acc':>10} {'Time':>8}")
    print("-" * 75)

    for epoch in range(args.epochs):
        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        val_metrics = validate(model, val_loader, criterion, device, args.amp)

        scheduler.step()

        elapsed = time.time() - start

        print(f"{epoch+1:>6} {train_loss:>12.4f} {train_acc:>10.4f} "
              f"{val_metrics['loss']:>12.4f} {val_metrics['accuracy']:>10.4f} "
              f"{val_metrics['balanced_accuracy']:>10.4f} {elapsed:>7.1f}s")

        if val_metrics['accuracy'] > best_accuracy:
            best_accuracy = val_metrics['accuracy']
            save_checkpoint(
                model, optimizer, epoch, val_metrics, class_names,
                output_dir, filename='best.pth'
            )

    print(f"\nTraining complete. Best val accuracy: {best_accuracy:.4f}")
    print(f"Checkpoint saved to: {output_dir}/best.pth")


if __name__ == '__main__':
    main()
