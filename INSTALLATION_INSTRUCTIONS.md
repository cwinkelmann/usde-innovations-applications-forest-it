# Installation Instructions

It is recommended to use an IDE like VSCode or PyCharm for working with the codebase, but you can also run everything from the command line und jpyter notebooks if you prefer. Get the Pro version of PyCharm from here: https://www.jetbrains.com/help/pycharm/getting-started.html ( it is free as a student ). 

Clone this using your IDE or download the ZIP from GitHub and extract it to your desired location.

This course uses **three conda environments** to keep dependencies clean.
Start with the lightweight `fit-megadetector` environment for the first
practicals and install the others as needed.

---

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- ~10 GB free disk space for both environments + datasets

---

## Why separate environments?

| Environment | Purpose | Key constraint                               |
|-------------|---------|----------------------------------------------|
| `fit-megadetector` | Lightweight exploration + MegaDetector inference (P1–P2) | Only megadetector, no ultralytics            |
| `fit-training` | Full training pipeline (P3–P8) | Adds ultralytics, sahi, timm |
| `fit-herdnet` | HerdNet + geospatial (advanced) | GDAL/rasterio via conda, animaloc from git   |

The split avoids: megadetector pinning old protobuf (breaks wandb/label-studio),
ultralytics fork conflicts between megadetector and animaloc, and GDAL pip
installs breaking other packages.


## Which environment for which practical?

| Practical | Environment | Key packages |
|-----------|-------------|-------------|
| P1 — Visual datasets | `fit-megadetector` | numpy, pandas, matplotlib, PIL, huggingface_hub |
| P2 — Annotation tools | `fit-megadetector` | + Label Studio via Docker |
| P3 — MegaDetector & SAHI | `fit-training` | ultralytics, sahi, megadetector |
| P4 — Detection exploration | `fit-training` | matplotlib, pandas |
| P5 — Classifier (timm) | `fit-training` | timm, torch |
| P6 — Evaluation | `fit-training` | sklearn |
| P7 — Segmentation (SAM) | `fit-training` | segment-anything, torch |
| P8 — Wrap-up | `fit-training` | — |
| HerdNet notebook | `fit-herdnet` | animaloc, GDAL, rasterio |
| Pipeline scripts (01–06) | `fit-herdnet` | animaloc, geospatial stack |

---



---

## 1 — `fit-megadetector` (Practicals 1–2, data exploration)

Lightweight environment for dataset exploration, downloading, and basic
MegaDetector inference via megadetector package. **Do this before Day 1.**

```bash
conda env create -f environment-megadetector.yml
conda activate fit-megadetector
```

```bash
python -c "
import megadetector
print('fit-megadetector OK')
"
```

Verify, run the notebook practical_3_megadetector_legacy.ipynb, and check that the MegaDetector model downloads and runs:

---

## 2 — `fit-training` (Practicals 3–8, full training pipeline)

Adds ultralytics (YOLOv8/v11), SAHI tiled inference, timm classifiers,

```bash
conda env create -f environment-training.yml
conda activate fit-training
```

Verify:
```bash
python -c "
import ultralytics, sahi, timm
print('fit-training OK')
"
```

**Note:** Label Studio has a protobuf conflict with megadetector. Run it
via Docker instead:
```bash
docker run -p 8080:8080 humansignal/label-studio
```

---

## 3 — `fit-herdnet` (HerdNet training & geospatial pipeline)

Used for the HerdNet notebook and `scripts/06_train_herdnet.py`.
This is the only environment that needs GDAL and the geospatial stack.

```bash
conda env create -f environment-herdnet.yml
conda activate fit-herdnet

# HerdNet (animaloc) must be cloned and installed separately
git clone https://github.com/cwinkelmann/HerdNet.git
pip install -e ./HerdNet
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
pip install -e ".[megadetector,dev]"   # fit-megadetector (P1–P2)
pip install -e ".[training,dev]"      # fit-training (P3–P8)
pip install -e ".[herdnet,dev]"       # fit-herdnet (HerdNet pipeline)
```

---

## Running the tests

From the repo root with any environment active:

```bash
pytest tests/ -v
```

---

## Running a Jupyter notebook

```bash
conda activate fit-megadetector
jupyter lab week1/practicals/
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
