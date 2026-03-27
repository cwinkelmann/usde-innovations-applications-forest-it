# Changelog

## 2026-03-17

### INSTALATION_INSTRUCTION.md — rewritten

- Reduced from three environments to **two**: `fit-megadetector` (P1-P8) and `fit-herdnet` (HerdNet only)
- Removed undefined `fit-geo` environment — P1/P2 need no geospatial packages
- Fixed broken `fit-megadetector` section (was missing conda create, package installs, opening code fence)
- Updated verify command: `megadetector` package replaces `PytorchWildlife`
- Added `sahi` and `ultralytics` to `fit-megadetector` (required by P3)
- Moved geospatial stack (GDAL, rasterio, geopandas) explanation to a note under `fit-herdnet`, clarifying it is only needed for HerdNet/pipeline work
- Added "Downloading datasets" section linking to `download_data.py`
- Added Label Studio troubleshooting entry
- Updated practical-to-environment table (all P1-P8 now use `fit-megadetector`)

### DATASETS.md — rewritten

- Removed stale macOS absolute paths at top of file
- Added "Datasets used in this course" section with practical-to-dataset mapping table
- Added "Where data lands after download" directory tree
- Documented all five datasets actually used: HerdNet General Dataset, Snapshot Serengeti subset, Caltech Camera Traps subset, Eikelboom 2019, HerdNet pretrained weights
- Connected dataset descriptions to `download_data.py` commands
- Moved the broad survey of additional African wildlife datasets to a clearly labelled "reference" section

### P1 (`p1_drone_imagery.py`) — bbox format conversions moved out

- Removed bounding box format conversion section (COCO/YOLO/VOC explanation + interactive demo, ~85 lines) — too early in the curriculum
- Kept bbox visualisation (Step 4) with a forward pointer to P3 Step 9

### P3 (`p3_megadetector.py`) — bbox format conversions moved in

- Added bounding box format conversion section (explanation + interactive 3-panel demo) before Step 9, where students need it to understand YOLO label preparation