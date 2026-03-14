# CLAUDE.md — Aerial Wildlife Detection Pipeline

This file describes the project structure, conventions, and implementation plan for the
aerial wildlife detection university course. All pipeline steps are implemented as scripts, not notebooks.

The README.md described the course content. This CLAUDE.md describes the project structure and implementation plan for the codebase.
The Course Layout is referenced in the README


---


## Repository Structure

TODO: build a repository structure diagram which fits the course layout

---

## Conventions

- All scripts accept a config YAML via `--config` and optionally `--override key=value`
- All scripts log to optionally to **Weights & Biases** — set `WANDB_PROJECT` and `WANDB_ENTITY` in `.env`
- Coordinate systems: all geospatial operations use the CRS of the GeoTIFF; local pixel
  coordinates in CSVs are always `(col, row)` relative to the tile origin
- Tile filenames encode their origin: `{source_stem}_{col_offset}_{row_offset}.jpg`
- All annotation outputs use **local tile coordinates**, not geographic coordinates
- Points that fall within `overlap / 2` pixels of a tile edge are included in both
  neighbouring tiles to avoid dropping annotations at boundaries
- Never discard empty tiles at data preparation time — let the training scripts handle
  empty-tile sampling strategy (hard negative mining is a training concern, not a data concern)
- users use windows laptops mostly, so avoid bashisms and ensure all scripts are cross-platform compatible (e.g. use `os.path` in Python, avoid bash commands in bash)
---

## Course Implementation Plan



Set credentials via environment variables — never hardcode:
```bash
export WANDB_API_KEY=your_key
export WANDB_PROJECT=wildlife-detection
export WANDB_ENTITY=your-entity
```

under