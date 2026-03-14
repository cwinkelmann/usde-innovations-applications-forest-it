# DeepFaune Architecture Reference

Detailed documentation of the DeepFaune camera trap classification system, derived from the actual source code at `/Users/christian/PycharmProjects/hnee/deepfaune_software/`. This covers the two-stage pipeline, weight format, backbone extraction for transfer learning, and licensing constraints.

---

## Overview

DeepFaune is a two-stage camera trap image classifier developed by CNRS (Centre National de la Recherche Scientifique). It combines a YOLOv8s object detector with a DINOv2 ViT-L classifier to identify 34 European wildlife species from camera trap images.

**Pipeline:** Full image -> YOLOv8s detection (960px) -> Crop best box -> Square padding -> ViT-L classification (182x182)

---

## Stage 1: Detection (YOLOv8s)

From `detectTools.py`:

```python
YOLO_WIDTH = 960        # Image width for detection
YOLO_THRES = 0.6        # Confidence threshold for detection
YOLOHUMAN_THRES = 0.4   # Human detection threshold
YOLOCOUNT_THRES = 0.6   # Counting threshold

yoloweight_path = os.path.join(DFPATH, 'deepfaune-yolov8s_960.pt')
```

**Detection categories:**
- Category 0: Empty (no detection above threshold)
- Category 1: Animal (-> passed to classifier)
- Category 2: Person
- Category 3: Vehicle

**Cropping logic** (from `cropSquareCVtoPIL`):
- Extract the bounding box coordinates (xmin, ymin, xmax, ymax)
- Pad the shorter dimension to make a square crop
- Clip to image boundaries
- Convert from OpenCV BGR to PIL RGB

---

## Stage 2: Classification (DINOv2 ViT-L)

From `classifTools.py`:

### Constants

```python
CROP_SIZE = 182
BACKBONE = "vit_large_patch14_dinov2.lvd142m"
weight_path = os.path.join(DFPATH, 'deepfaune-vit_large_patch14_dinov2.lvd142m.v3.pt')
```

### Preprocessing Transform

```python
from torchvision.transforms import InterpolationMode, transforms
from torch import tensor

self.transforms = transforms.Compose([
    transforms.Resize(
        size=(CROP_SIZE, CROP_SIZE),        # 182x182
        interpolation=InterpolationMode.BICUBIC,
        max_size=None,
        antialias=None,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=tensor([0.4850, 0.4560, 0.4060]),   # ImageNet mean
        std=tensor([0.2290, 0.2240, 0.2250]),     # ImageNet std
    ),
])
```

**Important:** The normalization uses ImageNet values (same as timm default), but note the slight precision difference -- DeepFaune uses 3 decimal places while timm uses `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`. In practice, these are equivalent.

### Model Architecture

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = timm.create_model(
            BACKBONE,                  # 'vit_large_patch14_dinov2.lvd142m'
            pretrained=False,          # Weights loaded separately
            num_classes=34,            # 34 European species
            dynamic_img_size=True,     # Variable input size support
        )
        self.backbone = BACKBONE
        self.nbclasses = 34

    def forward(self, input):
        x = self.base_model(input)
        return x
```

### Weight Loading

```python
def loadWeights(self, path):
    device = get_device()
    params = torch.load(path, map_location=device)
    args = params['args']                      # {'backbone': ..., 'num_classes': 34}
    self.load_state_dict(params['state_dict'])  # Full model state dict
```

**Weight file format:**
```python
{
    'args': {
        'backbone': 'vit_large_patch14_dinov2.lvd142m',
        'num_classes': 34,
        # ... other training args
    },
    'state_dict': {
        'base_model.patch_embed.proj.weight': tensor(...),
        'base_model.patch_embed.proj.bias': tensor(...),
        'base_model.cls_token': tensor(...),
        'base_model.pos_embed': tensor(...),
        'base_model.blocks.0.norm1.weight': tensor(...),
        # ... all ViT-L parameters
        'base_model.head.weight': tensor([34, 1024]),
        'base_model.head.bias': tensor([34]),
    }
}
```

**Note:** The state dict keys have a `base_model.` prefix because the `Model` class wraps `timm.create_model` as `self.base_model`.

### Species Classes (34 European Species)

```python
# English names (from classifTools.py txt_animalclasses['en']):
species = [
    'bison', 'badger', 'ibex', 'beaver', 'red deer', 'chamois', 'cat',
    'goat', 'roe deer', 'dog', 'fallow deer', 'squirrel', 'moose', 'equid',
    'genet', 'wolverine', 'hedgehog', 'lagomorph', 'wolf', 'otter', 'lynx',
    'marmot', 'micromammal', 'mouflon', 'sheep', 'mustelid', 'bird', 'bear',
    'nutria', 'raccoon', 'fox', 'reindeer', 'wild boar', 'cow',
]  # 34 classes
```

### Prediction with Softmax

```python
def predict(self, data, withsoftmax=True):
    self.eval()
    device = get_device()
    self.to(device)
    with torch.no_grad():
        x = data.to(device)
        if withsoftmax:
            output = self.forward(x).softmax(dim=1)
        else:
            output = self.forward(x)
    return np.array(output.tolist())
```

---

## Backbone Transfer for Non-European Species

Since DeepFaune's 34-class head is useless for non-European species (e.g., marine iguanas), the value lies in the DINOv2 ViT-L backbone which has been fine-tuned on wildlife camera trap images. To transfer:

```python
import timm
import torch
import torch.nn as nn

# Step 1: Create the DeepFaune model structure
class DeepFauneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = timm.create_model(
            'vit_large_patch14_dinov2.lvd142m',
            pretrained=False,
            num_classes=34,
            dynamic_img_size=True,
        )

deepfaune = DeepFauneModel()

# Step 2: Load DeepFaune weights
params = torch.load(
    'deepfaune-vit_large_patch14_dinov2.lvd142m.v3.pt',
    map_location='cpu',
    weights_only=False,  # Required -- weight file contains non-tensor 'args' dict
)
deepfaune.load_state_dict(params['state_dict'])

# Step 3: Extract the backbone (timm model inside)
backbone = deepfaune.base_model

# Step 4: Replace the classification head for new species
num_new_classes = 5  # Your number of species
in_features = backbone.head.in_features  # 1024 for ViT-L
backbone.head = nn.Linear(in_features, num_new_classes)

# Step 5: Freeze backbone, train head only
for name, param in backbone.named_parameters():
    if 'head' not in name:
        param.requires_grad = False

# backbone is now ready for fine-tuning with your species
model = backbone
```

**Why transfer from DeepFaune instead of raw DINOv2?**
- DeepFaune's backbone has been fine-tuned on wildlife camera trap images (mostly European fauna)
- It may have learned wildlife-relevant features (body shapes, fur textures, pose patterns) that raw DINOv2 lacks
- This is especially useful for species with visual similarity to European fauna

**When NOT to transfer from DeepFaune:**
- Your images are very different from camera traps (e.g., nadir drone imagery, underwater)
- You need commercial use (CC BY-NC-SA 4.0 license)
- The input resolution difference matters (DeepFaune uses 182x182; you may want 518x518)

---

## Sequence-Level Prediction (from `predictTools.py`)

DeepFaune averages predictions across image sequences (camera trap burst captures):

1. Classify each image in a sequence independently
2. If ANY image in the sequence detects an animal, the sequence is "animal"
3. Average the logits across animal-detected images
4. Apply temperature scaling: T=1.06 for single images, T=1.00 for sequences
5. Apply softmax to get final class probabilities

This is relevant for camera trap deployment but not for drone imagery.

---

## Licensing

- **Code:** CeCILL license (French equivalent of GPL)
- **Weights:** CC BY-NC-SA 4.0 (non-commercial, share-alike)
- **Authors:** Simon Chamaille (CEFE-CNRS), Vincent Miele (Univ. Lyon 1)
- **Implication:** Any model fine-tuned from DeepFaune weights inherits the CC BY-NC-SA 4.0 restriction. This means:
  - Academic/research use: OK
  - Commercial use: NOT permitted
  - Derivative works must use the same license
  - Attribution required

---

## Device Selection

DeepFaune's device selection (from `classifTools.py`):

```python
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("cpu")  # MPS available but deliberately uses CPU
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
```

**Note:** DeepFaune deliberately falls back to CPU when MPS is available, likely due to compatibility issues with the ViT-L model on Apple Silicon at the time of development. Modern PyTorch versions (2.4+) have improved MPS support, so this may be overly conservative.
