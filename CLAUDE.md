# CLAUDE.md — FIT Module: AI & UAV Wildlife Monitoring

Course repository for the **FIT module** at HNEE (Eberswalde University for Sustainable
Development). The repo hosts lecture structure, hands-on practicals, and supporting
Python utilities for a two-week course on AI-assisted wildlife monitoring.

**Key references:**
- `README.md` — course overview, literature, case studies, learning objectives
- `INSTALLATION_INSTRUCTIONS.md` — full day-by-day schedule, learning goals, deliberate skips

- doc/* — research documentation, dataset surveys, training strategy notes
- src/wildlife_detection/ — Python package with tiling utilities, training scripts
- .claude/skill - extensive notes for Claude Code on architecture, design patterns, and coding conventions
- .claude/agents — agent definitions for code generation, refactoring, and test writing (TODO)

---

## Course Timeline

| Week | Dates | Topic |
|------|-------|-------|
| 1 | Mar 30 – Apr 2, 2026 | AI & UAV Wildlife Image Classification |
| 2 | Apr 7 – Apr 10, 2026 | Radar Remote Sensing & Galamsey Detection |

**Module lead:** J.-P. Mund
**Week 1 practicals:** C. Winkelmann
**Week 2 radar:** N. Voss & A. Bosu

---

## Repository Structure

Target layout (see `Course_layout.md` for day-by-day detail):

```
README.md                       ← course overview & literature
Course_layout.md                ← full schedule (source of truth for practicals)
CLAUDE.md                       ← this file
INSTALLATION_INSTRUCTIONS.md

week1/
  lectures/                     ← slide decks and lecture notes
  practicals/                   ← Jupyter notebooks (one per practical block)
    practical_3_megadetector_legacy.ipynb        ← P3a: MegaDetector inference (legacy API)
    practical_3_megadetector_ultralytics.ipynb   ← P3b: MegaDetector via ultralytics + SAHI
    practical_5_species_classification.ipynb     ← P5: TIMM classifier inference on crops
  data/                         ← sample datasets (or download scripts)

week2/
  lectures/
  practicals/
  data/

shared/
  environment.yml               ← conda env for all practicals     [TODO]
  setup_instructions.md         ← day-1 setup guide

src/wildlife_detection/         ← Python utility package (tiling, training, utils)
configs/                        ← YAML configs for pipeline steps
notebooks/                      ← exploratory notebooks
data/                           ← raw/, processed/, splits/ (gitkeep)
tests/                          ← pytest unit tests
```

---

## Practical Format

All practicals are **Jupyter notebooks** (`.ipynb` files).
Each notebook follows this structure:

1. **Context cell** — one-paragraph framing: what problem are we solving today?
2. **Script snippet** — minimal, readable code block students can run as-is
3. **Exercise cell** — a gap or parameter to change, with a guiding question
4. **Reflection cell** — what worked, what failed, what would you do differently?

Each notebook must run top-to-bottom without errors in its designated conda environment.

---

## What's Implemented

| Component | Location | Status |
|-----------|----------|--------|
| GeoTIFF tiling utilities | `src/wildlife_detection/tiling/` | done |
| HerdNet training utilities | `src/wildlife_detection/training/` | done |
| Config loading + W&B helpers | `src/wildlife_detection/utils/` | done |
| Combined dataset preparation | `src/wildlife_detection/training/prepare_combined_dataset.py` | done |
| YOLO/RT-DETR training | `src/wildlife_detection/training/train_yolo_combined.py` | done |
| 3-phase fine-tuning | `src/wildlife_detection/training/phased_finetune.py` | done |
| Eikelboom evaluation | `src/wildlife_detection/training/eval_eikelboom.py` | done |
| Pipeline walkthrough notebook | `notebooks/01_pipeline_walkthrough.ipynb` | done |
| Unit tests | `tests/` | done |
| `Course_layout.md` | root | done |
| Setup instructions | `INSTALLATION_INSTRUCTIONS.md` | partial |

### Training scripts (CLI wrappers)

All training logic lives in `src/wildlife_detection/training/`. Thin CLI wrappers are
in `scripts/training/` — see `scripts/CLAUDE.md` for details.

### Training outputs and artifacts

| Path | Contents |
|------|----------|
| `output/comparison_2ep.json` | Model comparison from `evaluate_detectors.py` (deleted in cleanup, was in git commit `e1cb40e`). Contains per-class P/R/F1 for RT-DETR 2-epoch baseline on Eikelboom val at conf=0.3. |
| `output/dfine_nano_smoke/` | D-FINE nano smoke test (Transformers format, 3 classes: Elephant/Giraffe/Zebra) |
| `output/eikelboom_rtdetr_tiled/` | RT-DETR v2 trained on Eikelboom tiles (Transformers format) |
| `output/mdv6_finetune_test/` | MDV6 RT-DETR fine-tuning test run |
| `weights/md_v1000.0.0-larch.pt` | MegaDetector v1000 larch (YOLO11L, MIT license) |
| `wandb/` | WandB run logs from March 18, 2026 — 4 runs: sorrel (no freeze), sorrel (freeze=10), larch (freeze=10), rtdetr (freeze=10) |

### Research documentation

| Path | Contents |
|------|----------|
| `doc/fine_tuning_yolo11.md` | Dataset analysis and 3-phase training strategy for aerial wildlife YOLO fine-tuning. Covers Eikelboom benchmark, MMLA, Koger, Liège, WAID datasets. |
| `doc/aerial_wildlife_datasets_reference.md` | Comprehensive survey of aerial wildlife detection datasets |

### External datasets (NAS storage)

Training datasets live on NAS at `/data/mnt/storage/Datasets/`. Key datasets:

| Dataset | NAS Path | Format |
|---------|----------|--------|
| Eikelboom 2019 | `ImprovingPrecisionAccuracy_Eikelboom2019data/` | CSV bboxes, 5000×3000 images |
| MMLA Wilds | `mmla_wilds/` | YOLO native, 2720-4096px frames |
| Koger Ungulates | `Quantifying the movement.../kenyan-ungulates/` | COCO JSON, 4096×2160 |
| Koger Geladas | `Quantifying the movement.../geladas/` | COCO JSON |
| Liège Multispecies | `Multispecies detection.../general_dataset/` | CSV + COCO JSON, 6000×4000 |
| Combined (tiled) | `combined_aerial_yolo_640/` | Unified YOLO 640px tiles (built by `prepare_combined_dataset.py`) |

---

## What Still Needs to Be Built

### Week 1 Practicals

| Practical | Notebook | Env | Status |
|-----------|----------|-----|--------|
| P3a — MegaDetector (legacy) | `practical_3_megadetector_legacy.ipynb` | fit-megadetector | done |
| P3b — MegaDetector (ultralytics) | `practical_3_megadetector_ultralytics.ipynb` | fit-training | done |
| P5 — Classifier inference | `practical_5_species_classification.ipynb` | fit-training | done |

### Week 2 Materials

All Week 2 content (SAR/Sentinel-1, Galamsey change detection practicals) is TODO.
Radar lead: N. Voss & A. Bosu.

### Shared Environment

- `shared/environment.yml` — single conda env covering both weeks (TODO)
- `shared/setup_instructions.md` — consolidate from `INSTALLATION_INSTRUCTIONS.md` (TODO)

### Sample Datasets

| Day | Dataset needed | Source |
|-----|---------------|--------|
| 1 | Drone wildlife tiles | Iguanas From Above subset |
| 2 | Camera trap images | LILA BC — Snapshot Serengeti |
| 3 | Animal crops | Generated in P3 |
| 4 | Land cover scene | Sentinel-2 clip or provided drone scene |

---

## Conventions

- All utility scripts accept `--config <yaml>` and optionally `--override key=value`
- W&B logging is **optional** — set `WANDB_PROJECT` and `WANDB_ENTITY` in `.env`;
  never hardcode credentials
- Coordinate systems: all geospatial operations use the native CRS of the GeoTIFF;
  local pixel coordinates in CSVs are always `(col, row)` relative to the tile origin
- Tile filenames encode their origin: `{source_stem}_{col_offset}_{row_offset}.jpg`
- All annotation outputs use **local tile coordinates**, not geographic coordinates
- Points within `overlap / 2` pixels of a tile edge are included in both neighbouring
  tiles — avoids dropping annotations at boundaries
- **Never discard empty tiles** at data prep time — empty-tile sampling is a training
  concern, not a data concern
- **Windows compatibility** — students use Windows laptops; use `os.path`, avoid
  bash-specific syntax, test path handling cross-platform
