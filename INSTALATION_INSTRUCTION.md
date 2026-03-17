# Installation Instructions

This course uses **two conda environments** to avoid dependency conflicts
between the MegaDetector/YOLO stack and HerdNet. Practicals 1–2 need only
basic Python packages and run in either environment.

---

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Git
- ~10 GB free disk space for both environments + datasets

---

## Which environment for which practical?

| Practical | Environment | Why |
|-----------|-------------|-----|
| P1 — Drone imagery | `fit-megadetector` | Only needs numpy, pandas, matplotlib, PIL |
| P2 — Annotation tools | `fit-megadetector` | Adds Label Studio (included in env) |
| P3 — MegaDetector & SAHI | `fit-megadetector` | megadetector, ultralytics, sahi |
| P4 — Detection exploration | `fit-megadetector` | matplotlib, pandas |
| P5 — Classifier (timm) | `fit-megadetector` | timm, torch |
| P6 — Evaluation | `fit-megadetector` | sklearn |
| P7 — Segmentation (SAM) | `fit-megadetector` | segment-anything, torch |
| P8 — Wrap-up | `fit-megadetector` | — |
| HerdNet notebook | `fit-herdnet` | animaloc, GDAL, rasterio |
| Pipeline scripts (01–06) | `fit-herdnet` | animaloc, geospatial stack |

---

## Why separate environments?

| Conflict | Reason |
|----------|--------|
| MegaDetector vs HerdNet | The `megadetector` package pins specific YOLO versions that conflict with `animaloc` |
| GDAL / rasterio | Must be installed via conda, not pip — pip installs break other packages. Only needed for HerdNet/geospatial work |

---

## 1 — `fit-megadetector` (Practicals 1–8)

This is the main environment for Week 1. It covers everything from data
exploration (P1) through MegaDetector inference, YOLO fine-tuning, SAHI
tiled inference, classification, and segmentation.

```bash
conda create -n fit-megadetector python=3.11 -y
conda activate fit-megadetector

# PyTorch (CPU is fine for practicals; add cuda channel if you have a GPU)
conda install -c pytorch -c conda-forge pytorch torchvision -y

# MegaDetector + detection pipeline
pip install megadetector ultralytics sahi

# Classification & segmentation
pip install timm segment-anything

# Data exploration & annotation
pip install numpy pandas pillow scipy matplotlib tqdm
pip install marimo jupyterlab label-studio
pip install huggingface_hub requests python-dotenv

# This project
pip install -e "."
```

Verify:
```bash
python -c "
from megadetector.detection.run_detector import load_detector
import ultralytics, sahi, timm
print('fit-megadetector OK')
"
```

---

## 2 — `fit-herdnet` (HerdNet training & geospatial pipeline)

Used for the HerdNet notebook and `scripts/06_train_herdnet.py`.
This is the only environment that needs GDAL and the geospatial stack.

```bash
conda create -n fit-herdnet python=3.11 -y
conda activate fit-herdnet

# PyTorch
conda install -c pytorch -c conda-forge pytorch torchvision -y

# Geospatial stack — MUST be installed via conda, not pip
conda install -c conda-forge gdal rasterio geopandas fiona shapely -y

# HerdNet (animaloc) — install from the course fork
git clone https://github.com/cwinkelmann/HerdNet.git
pip install -e ./HerdNet

# Training utilities
pip install albumentations wandb omegaconf

# Utilities
pip install numpy pandas pillow scipy matplotlib tqdm marimo jupyterlab
pip install huggingface_hub python-dotenv

# This project (includes training modules + all optional extras)
pip install -e ".[all]"
```

Verify:
```bash
python -c "
import animaloc
from animaloc.models import HerdNet
from wildlife_detection.training.herdnet import HerdNetDataset
print(f'fit-herdnet OK  (animaloc {animaloc.__version__})')
"
```

### A note on geospatial packages

The `fit-herdnet` environment installs GDAL, rasterio, and geopandas because
HerdNet works with GeoTIFF orthomosaics and needs coordinate transformations.
You do **not** need these packages for Practicals 1–8 — the `fit-megadetector`
environment handles those without any geospatial dependencies.

If you are only running the Week 1 practicals and not the HerdNet pipeline,
you can skip this environment entirely.

---

## Downloading datasets

All Week 1 datasets are downloaded by the script `week1/data/download_data.py`
or automatically by the first cell in Practical 1.

```bash
conda activate fit-megadetector
cd week1/data

# Minimal set for quick testing (~500 MB)
python download_data.py --sample

# Full teaching subsets (~5 GB)
python download_data.py --full
```

See `DATASETS.md` for a full description of each dataset and where it is used.

---

## Installing `wildlife-detection` (this project)

`pip install -e "."` installs the `wildlife_detection` Python package in
editable mode from `pyproject.toml`. This makes the tiling utilities, training
helpers, and config loader importable from any notebook or script.

Optional extras defined in `pyproject.toml`:

```bash
pip install -e ".[yolo]"          # YOLOv8 only
pip install -e ".[segmentation]"  # segmentation-models-pytorch + transformers
pip install -e ".[all]"           # everything
```

The `[all]` extra is only needed in the `fit-herdnet` environment.
For `fit-megadetector`, a plain `pip install -e "."` is sufficient.

---

## Running the tests

From the repo root with any environment active:

```bash
pytest tests/ -v
```

---

## Running a Marimo notebook

```bash
conda activate fit-megadetector
marimo edit week1/practicals/p1_drone_imagery.py    # interactive editing
marimo run  week1/practicals/p1_drone_imagery.py    # read-only app mode
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'pkg_resources'`**
```bash
pip install setuptools==70.0.0
```

**`torch` not found after conda install on Apple Silicon**
Use `conda install -c pytorch -c conda-forge pytorch` rather than `pip install torch` —
the conda build is compiled for arm64.

**MegaDetector weights not downloading**
The model (~600 MB) downloads to `~/.cache/huggingface/` on first use.
Check your internet connection and that `huggingface_hub` is installed.

**Label Studio won't start**
```bash
label-studio start              # default port 8080
label-studio start --port 8081  # if 8080 is busy
```
On Windows, run in a standard terminal, not WSL.
