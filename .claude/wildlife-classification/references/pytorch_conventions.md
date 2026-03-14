# PyTorch Conventions for Wildlife Classification

Canonical patterns for device selection, model loading, path handling, AMP, reproducibility, and DataLoader configuration in wildlife classification projects.

---

## Device Selection

Always check CUDA first, then MPS (Apple Silicon), then CPU:

```python
import torch

def get_device(preferred=None):
    """Select best available device."""
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

DEVICE = get_device()
```

**Note on DeepFaune:** DeepFaune deliberately falls back to CPU when MPS is available (see `classifTools.py` `get_device()`), likely due to early MPS compatibility issues. Modern PyTorch 2.4+ has improved MPS support.

---

## Model Loading with `torch.load`

### PyTorch 2.6+ Default Change

Starting with PyTorch 2.6, `torch.load` defaults to `weights_only=True`. This will fail on checkpoints containing non-tensor data (like args dicts or class name lists). Always use `weights_only=False` for wildlife model checkpoints:

```python
# Correct for checkpoints with metadata:
checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)

# Common checkpoint formats:

# Format 1: state_dict only
model.load_state_dict(checkpoint)

# Format 2: dict with state_dict key (timm standard, DeepFaune)
model.load_state_dict(checkpoint['state_dict'])

# Format 3: dict with model_state_dict key (custom training scripts)
model.load_state_dict(checkpoint['model_state_dict'])
```

### DeepFaune Weight Format

DeepFaune uses a non-standard format with a `base_model.` prefix:

```python
# DeepFaune weights: {'args': {...}, 'state_dict': {'base_model.xxx': ...}}
params = torch.load(weight_path, map_location='cpu', weights_only=False)
args = params['args']                      # {'backbone': ..., 'num_classes': 34}
state_dict = params['state_dict']          # Keys have 'base_model.' prefix
```

To load into a plain timm model (without the wrapper class), strip the prefix:

```python
# Strip 'base_model.' prefix
stripped = {k.replace('base_model.', ''): v for k, v in state_dict.items()}
timm_model.load_state_dict(stripped)
```

---

## Path Handling

Use `pathlib.Path` objects, not string concatenation:

```python
from pathlib import Path

data_dir = Path('/path/to/images')
output_dir = Path('/path/to/output')
output_dir.mkdir(parents=True, exist_ok=True)

checkpoint_path = output_dir / 'best_model.pth'
```

**Do NOT** use:
```python
# Bad: fragile, platform-dependent
path = '/path/to/images' + '/' + 'train'

# Good:
path = Path('/path/to/images') / 'train'
```

---

## Import Ordering

Group imports as:
1. Standard library (pathlib, json, os, argparse)
2. Third-party (torch, PIL, pandas, numpy, sklearn)
3. Domain-specific (timm, megadetector, albumentations)

```python
# Standard library
import argparse
import json
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report

# Domain-specific
import timm
from timm.data import create_transform, resolve_data_config
from timm.optim import create_optimizer_v2
```

---

## Checkpoint Saving

Save both model state and training metadata:

```python
def save_checkpoint(model, optimizer, epoch, val_metric, output_path, **extra):
    """Save model checkpoint with metadata."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_metric': val_metric,
    }
    state.update(extra)  # class_names, num_classes, args, etc.
    torch.save(state, output_path)
```

**Always include in checkpoints:**
- `model_state_dict` (or `state_dict`)
- `epoch`
- `val_metric` (the metric used for best-model selection)
- `class_names` (for evaluation without re-reading the dataset)
- `num_classes`

---

## AMP (Automatic Mixed Precision)

Always enable for CUDA training. Reduces memory ~40%, speeds up ~20%:

```python
# Modern PyTorch API (2.0+):
scaler = torch.amp.GradScaler('cuda')

for images, labels in train_loader:
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    optimizer.zero_grad()

    with torch.amp.autocast('cuda'):
        outputs = model(images)
        loss = criterion(outputs, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**timm's NativeScaler wrapper** (used in iguana_train.py):
```python
from timm.utils import NativeScaler

scaler = NativeScaler()
# In training loop:
# scaler(loss, optimizer, parameters=model.parameters())
# This handles backward(), step(), and update() in one call
```

**AMP compatibility:**
- CUDA: Full support (recommended)
- MPS: Limited support (skip scaler, autocast may work with bfloat16)
- CPU: No benefit, skip entirely

---

## Reproducibility

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic mode (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

**Warning:** Full determinism requires `torch.backends.cudnn.deterministic = True` and `benchmark = False`, which reduces training speed by 10-20%. Use for final experiments, not exploratory runs.

---

## DataLoader Best Practices

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,            # or sampler= for imbalanced data
    num_workers=4,           # 4 per GPU is typical
    pin_memory=True,         # faster CPU-to-GPU transfer
    drop_last=True,          # avoid small last batch issues with BatchNorm
    persistent_workers=True, # keep workers alive between epochs
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,           # can be larger (no gradients stored)
    shuffle=False,           # deterministic evaluation
    num_workers=4,
    pin_memory=True,
)
```

**Key rules:**
- `pin_memory=True` only when using CUDA (no effect on CPU/MPS)
- `drop_last=True` for training to avoid BatchNorm instability with tiny last batch
- `persistent_workers=True` saves worker spawn overhead between epochs
- `shuffle=True` and `sampler=` are mutually exclusive

---

## Gradient Accumulation for Large Models

When batch size is limited by GPU memory:

```python
accumulation_steps = 4  # Effective batch size = batch_size * accumulation_steps

for i, (images, labels) in enumerate(train_loader):
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    with torch.amp.autocast('cuda'):
        outputs = model(images)
        loss = criterion(outputs, labels) / accumulation_steps

    scaler.scale(loss).backward()

    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```
