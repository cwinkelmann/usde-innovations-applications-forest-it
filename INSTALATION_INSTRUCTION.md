# Installation Instructions

This course uses **three separate conda environments** to avoid dependency conflicts
between MegaDetector, HerdNet, and the geospatial pipeline. Install only the one(s)
you need for the practicals you are running.

---

## Why separate environments?

| Conflict | Reason |
|----------|--------|
| MegaDetector vs HerdNet | PytorchWildlife pins specific YOLO versions that conflict with the `animaloc` package |
| GDAL / rasterio | Must be installed via conda, not pip — installing via pip breaks other packages |
| `geo2ml` | Not on PyPI; requires GDAL already present before it can be installed |

---

## Environment overview

| Environment | Used for | Approx. size |
|-------------|----------|--------------|
| `fit-geo` | Tiling, data prep, annotation tools (P1, P2) | ~1 GB |
| `fit-megadetector` | MegaDetector + classification practicals (P3–P6, P7) | ~4 GB |
| `fit-herdnet` | HerdNet training and the HerdNet notebook | ~5 GB |

---

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Git
- ~10 GB free disk space for all three environments

---

## 1 — `fit-geo` (tiling + data prep)

Lightest environment. Used for P1, P2, and running `download_data.py`.

```bash
conda create -n fit-geo python=3.11 -y
conda activate fit-geo

# Geospatial stack — GDAL must come from conda, not pip
conda install -c conda-forge gdal rasterio geopandas fiona shapely -y

# Python packages
pip install numpy pandas pillow scipy matplotlib tqdm marimo jupyterlab
pip install omegaconf python-dotenv huggingface_hub

# geo2ml — not on PyPI; install after GDAL is present in the environment
pip install --no-build-isolation git+https://github.com/mayrajeo/geo2ml.git

# This project (tiling utilities)
git clone <repo-url> && cd usde-innovations-applications-forest-it
pip install -e "."
```

Verify:
```bash
python -c "import rasterio, geopandas, geo2ml, wildlife_detection; print('fit-geo OK')"
```

---

## 2 — `fit-megadetector` (MegaDetector + classification)

Used for P3, P4, P5, P6, P7 (SAM).

```bash
conda create -n fit-megadetector python=3.11 -y
conda activate fit-megadetector

# Geospatial stack
conda install -c conda-forge gdal rasterio geopandas fiona shapely -y

# PyTorch — conda resolves the right build for your platform (CPU / CUDA / Apple Silicon)
conda install -c pytorch -c conda-forge pytorch torchvision -y

# Wildlife tools
pip install PytorchWildlife       # MegaDetector v5 + ReID
pip install timm                  # pre-trained classification models
pip install segment-anything      # SAM

# Download SAM weights (vit_b ~375 MB) and place in week1/data/:
#   https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Utilities
pip install numpy pandas pillow matplotlib scikit-learn tqdm marimo jupyterlab
pip install omegaconf python-dotenv huggingface_hub


# This project
pip install -e "."
```

Verify:
```bash
python -c "
from PytorchWildlife.models import detection as pw_detection
import timm, segment_anything
print('fit-megadetector OK')
"
```

---

## 3 — `fit-herdnet` (HerdNet training)

Used for the HerdNet notebook and `scripts/06_train_herdnet.py`.

```bash
conda create -n fit-herdnet python=3.11 -y
conda activate fit-herdnet

# Geospatial stack
conda install -c conda-forge gdal rasterio geopandas fiona shapely -y

# PyTorch
conda install -c pytorch -c conda-forge pytorch torchvision -y

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

---

## Installing `wildlife-detection` (this project)

`pip install -e ".[all]"` installs the `wildlife_detection` Python package in
editable mode from `pyproject.toml`. This makes the tiling utilities, training
helpers, and config loader importable from any notebook or script.

The only prerequisite is GDAL installed via conda (required by `rasterio`):

```bash
conda install -c conda-forge gdal rasterio geopandas -y
pip install -e ".[all]"
```

Optional extras defined in `pyproject.toml`:

```bash
pip install -e ".[yolo]"          # YOLOv8 only
pip install -e ".[segmentation]"  # segmentation-models-pytorch + transformers
pip install -e ".[all]"           # everything
```

---

## Running the tests

From the repo root with any environment active:

```bash
pytest tests/ -v
```

Expected result: **46 passed, 1 failed** (`test_empty_tile_count_zero` — known
test naming bug, does not affect runtime behaviour).

---

## Running a Marimo notebook

```bash
conda activate fit-megadetector          # use the right env for the practical
marimo edit week1/practicals/p3_megadetector.py   # interactive editing
marimo run  week1/practicals/p3_megadetector.py   # read-only app mode
```

---

## Which environment for which practical?

| Practical | Environment |
|-----------|-------------|
| P1 — Drone imagery | `fit-geo` |
| P2 — Annotation tools | `fit-geo` |
| P3 — MegaDetector | `fit-megadetector` |
| P4 — Detection exploration | `fit-megadetector` |
| P5 — Classifier (timm) | `fit-megadetector` |
| P6 — Evaluation | `fit-megadetector` |
| P7 — Segmentation (SAM) | `fit-megadetector` |
| P8 — Wrap-up | `fit-megadetector` |
| HerdNet notebook | `fit-herdnet` |
| Pipeline scripts (01–06) | `fit-herdnet` |

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
