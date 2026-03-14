# timm Fine-Tuning Guide

Reference for using the PyTorch Image Models (timm) library to fine-tune pretrained models for wildlife classification. All API details verified against timm source code and the iguana training codebase.

---

## Model Creation: `timm.create_model`

```python
import timm

model = timm.create_model(
    model_name,           # e.g., 'vit_base_patch14_dinov2.lvd142m'
    pretrained=True,      # Load ImageNet pretrained weights
    num_classes=5,        # Automatically replaces the classification head
    drop_rate=0.0,        # Dropout rate in the classifier head
    drop_path_rate=0.0,   # Stochastic depth rate (for ViTs)
    dynamic_img_size=True, # Allow variable input sizes (useful for ViTs)
)
```

**Key behaviors:**
- When `num_classes` differs from the pretrained model's default, timm automatically creates a new `nn.Linear` head with random initialization.
- The backbone weights are loaded from pretrained; only the head is random.
- `dynamic_img_size=True` is important for ViT models -- it allows input sizes different from the pretrained resolution (e.g., 518 instead of the default 224).

**Model name format:** `timm/<model_name>` or just `<model_name>` -- the `timm/` prefix is optional for `create_model` but used in HuggingFace Hub references.

### Common Wildlife-Relevant Models

| Model ID | Architecture | Params | Default Input | Notes |
|----------|-------------|--------|---------------|-------|
| `vit_base_patch14_dinov2.lvd142m` | ViT-B/14 DINOv2 | 87M | 518x518 | Recommended for wildlife |
| `vit_large_patch14_dinov2.lvd142m` | ViT-L/14 DINOv2 | 304M | 518x518 | Higher accuracy, 3x memory |
| `resnet50.a1_in1k` | ResNet-50 | 26M | 224x224 | Solid CNN baseline |
| `efficientnet_b0.ra_in1k` | EfficientNet-B0 | 5M | 224x224 | Lightweight, fast |
| `convnext_tiny.in12k_ft_in1k` | ConvNeXt-T | 29M | 224x224 | Modern CNN |
| `dla34.in1k` | DLA-34 | 16M | 224x224 | Used in HerdNet |

---

## Optimizer Creation: `timm.optim.create_optimizer_v2`

```python
from timm.optim import create_optimizer_v2

# Simple usage (single LR for all parameters):
optimizer = create_optimizer_v2(
    model_or_params=model,
    opt='adamw',
    lr=1e-4,
    weight_decay=0.05,
)
```

**`create_optimizer_v2` parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_or_params` | Model or params | required | Model or parameter iterable |
| `opt` | str | `'sgd'` | Optimizer name: `'adamw'`, `'sgd'`, `'adam'`, `'lamb'`, `'adafactor'` |
| `lr` | float | `None` | Learning rate |
| `weight_decay` | float | `0.0` | Weight decay |
| `momentum` | float | `0.9` | Momentum (SGD/similar) |
| `filter_bias_and_bn` | bool | `True` | Exclude bias/BN params from weight decay |

**For discriminative LRs, bypass create_optimizer_v2 and use raw PyTorch:**

```python
backbone_params = []
head_params = []
for name, param in model.named_parameters():
    if 'head' in name:
        head_params.append(param)
    else:
        backbone_params.append(param)

optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': 1e-6, 'weight_decay': 0.05},
    {'params': head_params, 'lr': 1e-4, 'weight_decay': 0.05},
])
```

---

## Scheduler Creation: `timm.scheduler.create_scheduler_v2`

```python
from timm.scheduler import create_scheduler_v2, scheduler_kwargs

# Using argparse namespace (as in iguana_train.py):
scheduler, num_epochs = create_scheduler_v2(
    optimizer,
    sched='cosine',       # 'cosine', 'step', 'plateau', 'poly'
    num_epochs=100,
    warmup_epochs=5,
    warmup_lr=1e-7,
    min_lr=1e-7,
    decay_rate=0.1,       # For step scheduler
)

# Per-epoch step:
scheduler.step(epoch)
```

**Recommended scheduler for wildlife fine-tuning:** Cosine annealing with 5-epoch linear warmup. This is the standard for ViT fine-tuning and matches the approach in most DINOv2 downstream papers.

---

## Layer Groups for ViT Models

Understanding ViT layer structure is essential for gradual unfreezing:

```python
# Inspect model structure
for name, param in model.named_parameters():
    print(f"{name:<60} {str(param.shape):<25} requires_grad={param.requires_grad}")
```

**ViT-B (12 blocks) layer hierarchy:**
```
patch_embed.proj.weight          # Patch embedding (low-level)
patch_embed.proj.bias
cls_token                         # Class token
pos_embed                         # Position embeddings
blocks.0.norm1.weight            # Block 0 (lowest level)
blocks.0.norm1.bias
blocks.0.attn.qkv.weight
blocks.0.attn.qkv.bias
blocks.0.attn.proj.weight
blocks.0.attn.proj.bias
blocks.0.norm2.weight
blocks.0.norm2.bias
blocks.0.mlp.fc1.weight
blocks.0.mlp.fc1.bias
blocks.0.mlp.fc2.weight
blocks.0.mlp.fc2.bias
...                               # blocks.1 through blocks.10
blocks.11.mlp.fc2.bias           # Block 11 (highest level before head)
norm.weight                       # Final layer norm
norm.bias
head.weight                       # Classification head
head.bias
```

**Freezing by block:**
```python
# Freeze everything except last N blocks + head
def freeze_except_last_n_blocks(model, n=2):
    total_blocks = len(model.blocks)
    for name, param in model.named_parameters():
        param.requires_grad = False  # Freeze everything first

    # Unfreeze last N blocks
    for i in range(total_blocks - n, total_blocks):
        for param in model.blocks[i].parameters():
            param.requires_grad = True

    # Unfreeze head and final norm
    for param in model.head.parameters():
        param.requires_grad = True
    for param in model.norm.parameters():
        param.requires_grad = True
```

---

## Checkpointing

**Saving (matching iguana_train.py pattern):**
```python
def save_checkpoint(model, optimizer, epoch, metric, output_dir, filename='best.pth.tar'):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'metric': metric,
    }
    path = os.path.join(output_dir, filename)
    torch.save(state, path)
```

**Loading:**
```python
def load_checkpoint(model, checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('metric', 0)
```

**Note on `weights_only`:** Starting with PyTorch 2.6, `torch.load` defaults to `weights_only=True`, which will fail on checkpoints containing non-tensor data like args dicts. Use `weights_only=False` for checkpoints that include metadata (like DeepFaune weights).

---

## Data Configuration: `resolve_data_config`

```python
from timm.data import resolve_data_config

# Get the model's expected data configuration
data_config = resolve_data_config({}, model=model)
print(data_config)
# Output example:
# {'input_size': (3, 518, 518), 'interpolation': 'bicubic',
#  'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
#  'crop_pct': 1.0, 'crop_mode': 'center'}
```

This is useful for ensuring your data preprocessing matches what the model expects. For DINOv2 models, the default input size is 518x518 with ImageNet normalization.

---

## AMP (Automatic Mixed Precision)

```python
from timm.utils import NativeScaler

# timm's NativeScaler wraps torch.cuda.amp.GradScaler
scaler = NativeScaler()

# In training loop:
with torch.cuda.amp.autocast(enabled=True):
    outputs = model(images)
    loss = criterion(outputs, targets)

optimizer.zero_grad()
scaler(loss, optimizer, parameters=model.parameters())
# scaler handles: loss.backward(), scaler.step(optimizer), scaler.update()
```

**When to use AMP:**
- Always for GPU training (reduces memory ~40%, speeds up ~20%)
- Not for CPU training (no benefit, may cause issues)
- Not for MPS on Apple Silicon (limited support as of PyTorch 2.x)
