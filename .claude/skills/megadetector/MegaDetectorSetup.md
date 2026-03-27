# MegaDetector Setup & Usage

## Goal
Run all MegaDetector weight versions (MDv5, MDv1000-redwood, MDv6) on Linux with
an RTX 4080 (CUDA 12.4), without the `megadetector` pip package.

---

## Environment

```bash
conda create -n megadetector python=3.10 -y
conda activate megadetector

# PyTorch — must be pip, not conda
pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# YOLOv5 repo — required for MDv5 and MDv1000-redwood via torch.hub
pip install yolov5

# Ultralytics — required for MDv6, MDv1000-larch, MDv1000-sorrel
pip install ultralytics>=8.3.0

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Weights

| Model | Loader | Size | Download |
|-------|--------|------|----------|
| MDv5a | torch.hub (yolov5) | 281 MB | `https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt` |
| MDv5b | torch.hub (yolov5) | 281 MB | `https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5b.0.0.pt` |
| MDv1000-redwood | torch.hub (yolov5) | ~280 MB | `https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/md_v1000.0.0-redwood.pt` |
| MDv1000-larch | ultralytics | ~120 MB | `https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/md_v1000.0.0-larch.pt` |
| MDv1000-sorrel | ultralytics | ~30 MB | `https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/md_v1000.0.0-sorrel.pt` |
| MDv6-yolov9-c | ultralytics | 51.6 MB | `https://zenodo.org/records/15398270/files/MDV6-yolov9-c.pt?download=1` |
| MDv6-yolov10-c | ultralytics | 5.8 MB | `https://zenodo.org/records/15398270/files/MDV6-yolov10-c.pt?download=1` |

```bash
# Download all weights at once
wget https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt
wget https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/md_v1000.0.0-redwood.pt
wget https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/md_v1000.0.0-larch.pt
wget "https://zenodo.org/records/15398270/files/MDV6-yolov9-c.pt?download=1" -O MDV6-yolov9-c.pt
```

---

## Class Labels (all versions)

```python
CLASS_NAMES = {0: "animal", 1: "person", 2: "vehicle"}
```

> Note: the `megadetector` package JSON output uses 1-indexed classes (1=animal,
> 2=person, 3=vehicle). When loading weights directly, you get 0-indexed.

---

## Inference

### MDv5a / MDv1000-redwood (torch.hub loader)

```python
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='md_v1000.0.0-redwood.pt')
model.conf = 0.2
model.iou  = 0.45

results = model('image.jpg', size=1280)

CLASS_NAMES = {0: "animal", 1: "person", 2: "vehicle"}
for _, det in results.pandas().xyxy[0].iterrows():
    print(f"{CLASS_NAMES[int(det['class'])]}: {det['confidence']:.3f} "
          f"[{det['xmin']:.0f},{det['ymin']:.0f},{det['xmax']:.0f},{det['ymax']:.0f}]")
```

### MDv6 / MDv1000-larch / MDv1000-sorrel (ultralytics loader)

```python
from ultralytics import YOLO

model = YOLO("MDV6-yolov9-c.pt")

results = model.predict(
    source="image.jpg",
    imgsz=640,    # use 1280 for *-1280 variants
    conf=0.2,
    max_det=300,
    verbose=False
)

CLASS_NAMES = {0: "animal", 1: "person", 2: "vehicle"}
for result in results:
    for box in result.boxes:
        cls  = int(box.cls.item())
        conf = box.conf.item()
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"{CLASS_NAMES[cls]}: {conf:.3f} [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
```

---

## Key Gotchas

- **MDv5/redwood needs `size=1280`** — they are P6 models; 640px degrades performance
- **MDv6/larch/sorrel use `imgsz=640`** unless the filename ends in `-1280`
- **YOLOv10 variants ignore `iou`** — they use NMS-free end-to-end inference
- **FP16 on RTX 4080:** pass `half=True` to `model.predict()` for ~2× speedup on MDv6
- **`torch.hub` caches** the YOLOv5 repo to `~/.cache/torch/hub/` on first load
- **`YOLO("mdv5a.0.0.pt")` does NOT work** — ultralytics and yolov5 are incompatible weight formats

---

## TODO / Next Steps

- [ ] Download weights to a `weights/` folder
- [ ] Write a unified inference wrapper that picks the right loader based on filename
- [ ] Test on a sample camera trap image from LILA BC
- [ ] Benchmark RTX 4080 throughput for redwood vs larch vs MDv6-yolov9-c
- [ ] Integrate into course practical (Practical 3)