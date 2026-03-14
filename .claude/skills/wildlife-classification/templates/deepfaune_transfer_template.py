#!/usr/bin/env python3
"""
DeepFaune Backbone Transfer -- Fine-tune for custom species.
Loads DeepFaune's DINOv2 ViT-L weights, replaces classification head,
fine-tunes with frozen backbone warmup then discriminative LRs.

DeepFaune architecture details:
  - Backbone: vit_large_patch14_dinov2.lvd142m (via timm)
  - Input: 182x182, BICUBIC resize, ImageNet normalization
  - Weight format: {'args': {...}, 'state_dict': ...}
  - state_dict keys have 'base_model.' prefix (wrapper class)
  - Original: 34 European species
  - License: CeCILL + CC BY-NC-SA 4.0 (non-commercial only)

Usage:
    python deepfaune_adapter_template.py \
        --data-dir path/to/imagefolder \
        --deepfaune-weights deepfaune-vit_large_patch14_dinov2.lvd142m.v3.pt \
        --num-classes 5 \
        --epochs 30
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report

import timm

# ============================================================================
# Constants (from DeepFaune classifTools.py)
# ============================================================================

DEEPFAUNE_BACKBONE = 'vit_large_patch14_dinov2.lvd142m'
DEEPFAUNE_INPUT_SIZE = 182
DEEPFAUNE_NUM_CLASSES = 34
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================================
# Device selection
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
# DeepFaune weight loading
# ============================================================================

def load_deepfaune_backbone(weights_path, num_new_classes, input_size=None):
    """
    Load DeepFaune weights and replace the classification head.

    DeepFaune wraps the timm model in a Model class with self.base_model,
    so the state_dict keys have a 'base_model.' prefix. We load into the
    wrapper first, then extract the timm model.

    Args:
        weights_path: Path to deepfaune-vit_large_patch14_dinov2.lvd142m.v3.pt
        num_new_classes: Number of target species
        input_size: Override input size (default: 182 from DeepFaune)

    Returns:
        model: timm ViT-L model with new head, backbone weights from DeepFaune
    """
    # Step 1: Create wrapper matching DeepFaune's Model class
    class DeepFauneWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = timm.create_model(
                DEEPFAUNE_BACKBONE,
                pretrained=False,
                num_classes=DEEPFAUNE_NUM_CLASSES,
                dynamic_img_size=True,
            )

    wrapper = DeepFauneWrapper()

    # Step 2: Load DeepFaune weights
    print(f"Loading DeepFaune weights from: {weights_path}")
    params = torch.load(weights_path, map_location='cpu', weights_only=False)

    # Validate weight file format
    assert 'state_dict' in params, "Expected DeepFaune weight format: {'args': {...}, 'state_dict': ...}"
    assert 'args' in params, "Missing 'args' in weight file"

    args_info = params['args']
    print(f"  Backbone: {args_info.get('backbone', 'unknown')}")
    print(f"  Original classes: {args_info.get('num_classes', 'unknown')}")

    wrapper.load_state_dict(params['state_dict'])

    # Step 3: Extract the timm model from the wrapper
    model = wrapper.base_model

    # Step 4: Replace the classification head
    in_features = model.head.in_features  # 1024 for ViT-L
    model.head = nn.Linear(in_features, num_new_classes)
    nn.init.xavier_uniform_(model.head.weight)
    nn.init.zeros_(model.head.bias)

    print(f"  Replaced head: {DEEPFAUNE_NUM_CLASSES} -> {num_new_classes} classes")
    print(f"  Head input features: {in_features}")

    return model


# ============================================================================
# Transforms
# ============================================================================

def build_transforms(input_size):
    """Build transforms matching DeepFaune preprocessing."""
    train_tf = transforms.Compose([
        transforms.Resize(
            (input_size, input_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(
            (input_size, input_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_tf, val_tf


# ============================================================================
# Training phases
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast(device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ============================================================================
# Main
# ============================================================================

def main(args):
    device = get_device(args.device)
    print(f"Device: {device}")

    # Load model
    input_size = args.input_size or DEEPFAUNE_INPUT_SIZE
    model = load_deepfaune_backbone(args.deepfaune_weights, args.num_classes, input_size)
    model = model.to(device)

    # Data
    train_tf, val_tf = build_transforms(input_size)
    train_ds = ImageFolder(str(Path(args.data_dir) / 'train'), transform=train_tf)
    val_ds = ImageFolder(str(Path(args.data_dir) / 'val'), transform=val_tf)
    class_names = train_ds.classes

    print(f"Train: {len(train_ds)} images, Val: {len(val_ds)} images")
    print(f"Classes: {class_names}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    scaler = torch.amp.GradScaler(device.type) if args.amp and device.type == 'cuda' else None

    # === Phase 1: Freeze backbone, train head only ===
    warmup_epochs = min(args.warmup_epochs, args.epochs // 3)
    print(f"\n--- Phase 1: Head-only training ({warmup_epochs} epochs) ---")

    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.head_lr)

    for epoch in range(warmup_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        print(f"  Epoch {epoch+1}/{warmup_epochs} -- "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    # === Phase 2: Unfreeze backbone with discriminative LRs ===
    remaining = args.epochs - warmup_epochs
    print(f"\n--- Phase 2: Full fine-tuning with discriminative LRs ({remaining} epochs) ---")
    print(f"  Backbone LR: {args.backbone_lr}, Head LR: {args.head_lr}")

    for param in model.parameters():
        param.requires_grad = True

    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.backbone_lr},
        {'params': head_params, 'lr': args.head_lr},
    ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)

    for epoch in range(warmup_epochs, args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, preds, labels = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Epoch {epoch+1}/{args.epochs} -- "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'num_classes': args.num_classes,
                'class_names': class_names,
                'backbone': DEEPFAUNE_BACKBONE,
                'input_size': input_size,
            }, output_dir / 'best_deepfaune_transfer.pth')
            print(f"    Saved best model (val_acc={val_acc:.3f})")

    # Final evaluation
    print(f"\n--- Final Evaluation ---")
    print(f"Best val_acc: {best_val_acc:.3f}")
    if len(preds) > 0:
        print(classification_report(labels, preds, target_names=class_names, digits=3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepFaune backbone transfer')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='ImageFolder root (train/ val/ subdirs)')
    parser.add_argument('--deepfaune-weights', type=str, required=True,
                        help='Path to deepfaune-vit_large_patch14_dinov2.lvd142m.v3.pt')
    parser.add_argument('--num-classes', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Epochs for head-only warmup phase')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Batch size (ViT-L needs small batches)')
    parser.add_argument('--backbone-lr', type=float, default=1e-6,
                        help='Backbone learning rate')
    parser.add_argument('--head-lr', type=float, default=1e-4,
                        help='Head learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--input-size', type=int, default=None,
                        help='Override input size (default: 182 from DeepFaune)')
    parser.add_argument('--output-dir', type=str, default='checkpoints/')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--amp', action='store_true', default=False)
    main(parser.parse_args())
