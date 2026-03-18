# MegaDetector Fine-Tuning with Ultralytics
## MDv6 RT-DETR (PytorchWildlife) + MDv1000 (Dan Morris)

---

## TL;DR Decision Table

| Model | Ultralytics compatible? | Fine-tune how? | License |
|-------|------------------------|----------------|---------|
| MDV6-rtdetr-c.pt | ✅ Yes | `RTDETR("MDV6-rtdetr-c.pt").train()` | AGPL |
| MDV6-apa-rtdetr-c.pth | ❌ No | lyuwenyu/RT-DETR scripts only | Apache 2.0 |
| MDV6-apa-rtdetr-e.pth | ❌ No | lyuwenyu/RT-DETR scripts only | Apache 2.0 |
| MDv1000-larch (YOLOv11L) | ✅ Yes | `YOLO("md_v1000.0.0-larch.pt").train()` | MIT |
| MDv1000-sorrel (YOLOv11s) | ✅ Yes | `YOLO("md_v1000.0.0-sorrel.pt").train()` | MIT |
| MDv1000-cedar (YOLOv9c) | ⚠️ Partial | needs `yolov9pip`, not plain ultralytics | MIT |
| MDv1000-redwood | ❌ No | torch.hub (YOLOv5 repo) only | AGPL |
| MDv1000-spruce (YOLOv5s) | ❌ No | torch.hub (YOLOv5 repo) only | AGPL |

**Bottom line:** For a clean ultralytics-only environment, your fine-tuning options are
`MDV6-rtdetr-c.pt` (RTDETR class), `larch`, and `sorrel` (both YOLO class).

---

## Environment

One environment works for everything in this guide:

```bash
conda create -n md-finetune python=3.10 -y
conda activate md-finetune

# PyTorch with CUDA 12.4 — must be pip, not conda
pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Ultralytics — handles RTDETR, YOLOv9, YOLOv11
pip install "ultralytics>=8.3.150"  # must be AFTER the #20627 KeyError fix

# Verify
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

> ⚠️ **Pin ultralytics to ≥8.3.150.** Version 8.3.133 introduced a `KeyError` that
> breaks RT-DETR fine-tuning (`model.0.conv.weight` not found). This is now fixed
> but you must be past the regression.

---

## Dataset Format (same for all models)

All three ultralytics-compatible models use standard YOLO txt format:

```
dataset/
  images/
    train/   *.jpg
    val/     *.jpg
  labels/
    train/   *.txt   # one file per image
    val/     *.txt
  data.yaml
```

Each `.txt` label file, one row per detection:
```
# class  x_center  y_center  width  height  (all normalized 0–1)
0  0.512  0.341  0.089  0.124
```

MegaDetector class mapping (0-indexed, same across all versions):
```
0 = animal    1 = person    2 = vehicle
```

`data.yaml`:
```yaml
path: /absolute/path/to/dataset
train: images/train
val:   images/val
nc: 3
names: ["animal", "person", "vehicle"]
```

---

## Part 1 — Fine-tuning MDV6 RT-DETR

### Weight file

```bash
wget "https://zenodo.org/records/15398270/files/MDV6-rtdetr-c.pt?download=1" \
     -O MDV6-rtdetr-c.pt
```

66 MB. AGPL-licensed. Trained by Microsoft/PytorchWildlife team using ultralytics.
Input resolution: 1280×1280.

### Why the AGPL `.pt` works but the Apache `.pth` does not

PytorchWildlife ships two RT-DETR lineages:

- `MDV6-rtdetr-c.pt` — trained with ultralytics, serialized in ultralytics `.pt`
  format (bundles architecture YAML + weights). Loads natively with `RTDETR()`.
- `MDV6-apa-rtdetr-*.pth` — trained with the lyuwenyu/RT-DETR codebase
  (CVPR 2024 paper's Apache-licensed implementation). Plain PyTorch state dict.
  Incompatible with ultralytics entirely.

The 5× size difference (66 MB vs 322 MB) alone confirms they are different
codebases, not just different formats.

### Fine-tuning code

```python
from ultralytics import RTDETR

model = RTDETR("MDV6-rtdetr-c.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=1280,     # RT-DETR MDv6 runs at 1280 — don't downsize
    batch=4,        # transformer encoder is VRAM-hungry; 4–8 on RTX 4080 at 1280px
    device=0,
    lr0=0.0001,     # RT-DETR requires much lower LR than YOLO
    weight_decay=0.0001,
    amp=False,      # CRITICAL — AMP causes NaN losses in RT-DETR bipartite matching
    project="runs/finetune",
    name="mdv6-rtdetr-custom"
)
```

### RT-DETR specific gotchas

**`amp=False` is mandatory.** The ultralytics source code itself warns that AMP
"can lead to NaN outputs and may produce errors during bipartite graph matching."
This is not a maybe — on most datasets it will silently corrupt your training run.
The VRAM cost is real: at 1280px, batch=4 without AMP will use ~14 GB on an RTX 4080.

**Learning rate is critical.** `lr0=0.0001` is the community standard. Using
`lr0=0.001` (fine for YOLO) will destabilize the attention layers within a few
epochs. If you see loss spiking or NaN after epoch 10–20, halve the LR.

**`freeze` parameter has no effect.** RT-DETR's backbone/encoder/decoder are not
cleanly separable the way YOLO layers are. Use `weight_decay` + early stopping
to control overfitting on small datasets instead.

**`model.tune()` does not work** for RT-DETR — it internally calls the YOLO
trainer. Do not use it.

**`deterministic=False`** — PyTorch's `F.grid_sample` rejects deterministic mode.
If you've globally set `torch.use_deterministic_algorithms(True)`, disable it.

**mAP may decrease while training loss decreases.** Watch validation metrics
directly and use `patience=10` to stop early. RT-DETR overfits faster than YOLO.

### Evaluate and export

```python
# Load best checkpoint
model = RTDETR("runs/finetune/mdv6-rtdetr-custom/weights/best.pt")

# Validate
metrics = model.val(data="data.yaml")
print(f"mAP@50:     {metrics.box.map50:.3f}")
print(f"mAP@50:95:  {metrics.box.map:.3f}")

# Export to ONNX for deployment
model.export(format="onnx", imgsz=1280)
```

---

## Part 2 — Fine-tuning MDv1000 (Dan Morris) via Ultralytics

Only `larch` (YOLOv11L) and `sorrel` (YOLOv11s) are natively ultralytics-compatible.
Both are MIT-licensed. `cedar` requires `yolov9pip`. `redwood` and `spruce` require
the old `ultralytics/yolov5` torch.hub path and cannot be fine-tuned via ultralytics.

### Weight files

```bash
# Recommended default for fine-tuning (best accuracy, MIT license)
wget https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/md_v1000.0.0-larch.pt

# Lightweight option (~30 MB, good for CPU/edge)
wget https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/md_v1000.0.0-sorrel.pt
```

| Model | Architecture | Input | Size | naAP vs MDv5a |
|-------|-------------|-------|------|---------------|
| larch | YOLOv11L | 640px | ~120 MB | 0.97 |
| sorrel | YOLOv11s | 960px | ~30 MB | 0.97 |

### Fine-tuning code

```python
from ultralytics import YOLO

# Use larch for best accuracy, sorrel for smaller footprint
model = YOLO("md_v1000.0.0-larch.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,       # larch is a 640px model
    batch=16,
    device=0,
    lr0=0.001,
    freeze=10,       # freeze first 10 backbone layers — works cleanly with YOLO
    half=True,       # FP16 safe for YOLO on RTX 4080
    project="runs/finetune",
    name="mdv1000-larch-custom"
)
```

For sorrel at its native 960px:

```python
model = YOLO("md_v1000.0.0-sorrel.pt")
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=960,
    batch=8,         # larger input = smaller batch
    device=0,
    lr0=0.001,
    freeze=10,
    half=True,
    project="runs/finetune",
    name="mdv1000-sorrel-custom"
)
```

### YOLOv11 fine-tuning notes

YOLOv11 is well-behaved with ultralytics — none of the RT-DETR stability issues apply.

**`freeze=10`** is the right lever for small datasets. It keeps the backbone frozen
and only trains the detection head, preventing the pretrained wildlife features from
being overwritten by a small domain dataset.

**`half=True`** (FP16) is safe and gives ~2× speedup on RTX 4080. Unlike RT-DETR,
YOLO bipartite matching is not used so AMP does not cause NaN issues.

**Dan Morris's own advice** on the MDv1000 models: use the largest model your time
budget allows. On a GPU like the RTX 4080, larch processes well over 1M images/day,
so there is no practical reason to use sorrel unless you need edge deployment.

### Evaluate and run inference

```python
# Load fine-tuned weights
model = YOLO("runs/finetune/mdv1000-larch-custom/weights/best.pt")

# Validate
metrics = model.val(data="data.yaml")
print(f"mAP@50:     {metrics.box.map50:.3f}")
print(f"mAP@50:95:  {metrics.box.map:.3f}")

# Inference
CLASS_NAMES = {0: "animal", 1: "person", 2: "vehicle"}
results = model.predict("image.jpg", conf=0.2, imgsz=640, verbose=False)
for result in results:
    for box in result.boxes:
        cls  = int(box.cls.item())
        conf = box.conf.item()
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"{CLASS_NAMES[cls]}: {conf:.3f}  [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
```

---

## Part 3 — Choosing Between Them

| Consideration | MDV6-rtdetr-c | MDv1000-larch | MDv1000-sorrel |
|--------------|---------------|----------------|----------------|
| License | AGPL | MIT | MIT |
| Loader | `RTDETR()` | `YOLO()` | `YOLO()` |
| Input size | 1280px | 640px | 960px |
| Fine-tune difficulty | High (AMP, LR) | Low | Low |
| VRAM at fine-tune | ~14 GB (batch 4) | ~8 GB (batch 16) | ~10 GB (batch 8) |
| NMS required | No (end-to-end) | Yes | Yes |
| Speed (RTX 4080) | ~5× faster than MDv5 | very fast | fastest |
| Best for | Max recall, no NMS | General fine-tuning | Edge / CPU |

For the course practical, **larch is the correct choice** — MIT license, clean YOLO
API, no stability surprises, and naAP 0.97 is indistinguishable from redwood for
most ecological datasets. RT-DETR is interesting academically but too fragile for
a classroom setting.

For your own production work (iguana detection, wildlife surveys), RT-DETR's
NMS-free inference is genuinely useful when animals are densely packed and NMS
threshold tuning becomes painful. Start with larch, switch to RT-DETR if you're
spending time debugging NMS suppression of adjacent animals.

---

## Common Failure Modes

| Symptom | Model | Likely cause | Fix |
|---------|-------|-------------|-----|
| NaN loss after epoch 10 | RT-DETR | AMP enabled | Set `amp=False` |
| mAP=0 throughout training | RT-DETR | LR too high | Set `lr0=0.0001` |
| `KeyError: model.0.conv.weight` | RT-DETR | ultralytics <8.3.150 | Upgrade ultralytics |
| No GPU detected | Both | PyTorch installed via conda | Reinstall PyTorch via pip |
| Loss converges but val mAP drops | RT-DETR | Overfitting | Add `patience=10`, reduce epochs |
| `YOLO("md_v5a.0.0.pt")` fails | MDv5/redwood | Wrong format | Use `torch.hub.load` instead |

---

## TODO for Claude Code

- [ ] Download larch and MDV6-rtdetr-c weights to `weights/`
- [ ] Create a small test dataset from LILA BC Snapshot Serengeti (100 images)
  to verify fine-tuning pipelines run end-to-end before scaling
- [ ] Write a unified `finetune.py` script with `--model {larch,sorrel,rtdetr}`
  argument that picks the correct loader and training params
- [ ] Benchmark mAP before/after fine-tuning on a domain-specific holdout set
- [ ] Test RT-DETR export to ONNX for deployment comparison