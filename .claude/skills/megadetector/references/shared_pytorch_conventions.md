# Shared PyTorch Wildlife Conventions

## Device Selection

```python
import torch

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
```

## Model Loading

Always use `weights_only=False` when loading wildlife model checkpoints (they contain custom class definitions):

```python
checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
```

## Path Handling

Use `pathlib.Path` objects, not string concatenation:

```python
from pathlib import Path
data_dir = Path('/path/to/images')
output_file = data_dir / 'results.json'
```

## Imports

Prefer explicit imports over star imports. Group as:
1. Standard library (pathlib, json, os)
2. Third-party (torch, PIL, pandas, numpy)
3. Domain-specific (megadetector, timm, animaloc)
