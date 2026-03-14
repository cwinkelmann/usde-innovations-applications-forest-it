# CLAUDE.md — FIT Module: AI & UAV Wildlife Monitoring

Course repository for the **FIT module** at HNEE (Eberswalde University for Sustainable
Development). The repo hosts lecture structure, hands-on practicals, and supporting
Python utilities for a two-week course on AI-assisted wildlife monitoring.

**Key references:**
- `README.md` — course overview, literature, case studies, learning objectives
- `Course_layout.md` — full day-by-day schedule, learning goals, deliberate skips

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
INSTALATION_INSTRUCTION.md

week1/
  lectures/                     ← slide decks and lecture notes
  practicals/                   ← Marimo notebooks (one per practical block)
    p1_drone_imagery.py         ← Practical 1: explore aerial wildlife dataset (tiles + annotations)
    p2_annotation_tools.py      ← Practical 2: point / box / polygon annotation demo
    p3_megadetector.py          ← Practical 3: run MegaDetector, parse JSON output, fine tune a model
    p4_detection_exploration.py ← Practical 4: browse detections, failure modes
    p5_classifier.py            ← Practical 5: TIMM classifier inference on crops
    p6_evaluation.py            ← Practical 6: accuracy check against reference set
    p7_segmentation.py          ← Practical 7: SAM / U-Net / Segformer on land cover image
    p8_wrapup.py                ← Practical 8: export work, reflection prompts
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

All practicals are **Marimo notebooks** (`.py` files runnable as interactive apps).
Each notebook follows this structure:

1. **Context cell** — one-paragraph framing: what problem are we solving today?
2. **Script snippet** — minimal, readable code block students can run as-is
3. **Exercise cell** — a gap or parameter to change, with a guiding question
4. **Reflection cell** — what worked, what failed, what would you do differently?

Marimo docs: https://docs.marimo.io
Each notebook must run top-to-bottom without errors on the shared conda environment.

---

## What's Implemented

| Component | Location | Status |
|-----------|----------|--------|
| GeoTIFF tiling utilities | `src/wildlife_detection/tiling/` | done |
| HerdNet training utilities | `src/wildlife_detection/training/` | done |
| Config loading + W&B helpers | `src/wildlife_detection/utils/` | done |
| Pipeline walkthrough notebook | `notebooks/01_pipeline_walkthrough.ipynb` | done |
| Unit tests | `tests/` | done |
| `Course_layout.md` | root | done |
| Setup instructions | `INSTALATION_INSTRUCTION.md` | partial |

---

## What Still Needs to Be Built

### Week 1 Practicals

| Practical | Marimo file | Key tools | Status |
|-----------|------------|-----------|--------|
| P1 — Dataset exploration | `week1/practicals/p1_drone_imagery.py` | `PIL`, `pandas`, `matplotlib` | done |
| P2 — Annotation demo | `week1/practicals/p2_annotation_tools.py` | CVAT/Label Studio tour | TODO |
| P3 — MegaDetector | `week1/practicals/p3_megadetector.py` | `PytorchWildlife` | TODO |
| P4 — Detection exploration | `week1/practicals/p4_detection_exploration.py` | `matplotlib`, `pandas` | TODO |
| P5 — Classifier inference | `week1/practicals/p5_classifier.py` | `timm`, `torch` | TODO |
| P6 — Evaluation | `week1/practicals/p6_evaluation.py` | `sklearn.metrics` | TODO |
| P7 — Segmentation | `week1/practicals/p7_segmentation.py` | `segment-anything` | TODO |
| P8 — Wrap-up | `week1/practicals/p8_wrapup.py` | — | TODO |

### Week 2 Materials

All Week 2 content (SAR/Sentinel-1, Galamsey change detection practicals) is TODO.
Radar lead: N. Voss & A. Bosu.

### Shared Environment

- `shared/environment.yml` — single conda env covering both weeks (TODO)
- `shared/setup_instructions.md` — consolidate from `INSTALATION_INSTRUCTION.md` (TODO)

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
