# HILDA Test Environment Setup

## Test Data Requirements

### Required Test Data Structure
```
tests/data/
├── Isabela/
│   └── ISWF01_22012023_subset/          # 9 Mavic 2 Pro JPGs with GPS
├── raw_images/
│   ├── mavic2pro/Floreana/FLMO04_03022021/  # 50 Mavic 2 Pro images
│   └── matrice4e/Eberswalde/rpark_20250718/ # 51 Matrice 4E images
├── models/
│   ├── dla34/                           # HerdNet DLA-34 (211 MB)
│   ├── convnext/                        # ConvNeXt model (1 GB)
│   └── herdnet_general/                 # Legacy HerdNet (210 MB)
├── correspondence/                      # For deduplication tests
│   ├── cameras_Isa_ISWF01_22012023.xml # Camera parameters
│   ├── dem_Isa_ISWF01_22012023.tif     # DEM file
│   └── detections.csv                  # Test detections
└── geospatial/
    ├── FMO02_full_orthophoto.tif        # Test orthomosaic
    └── Isa_ISVP01_27012023*.csv         # Detection results
```

## Environment Check

Use the automated environment checker:
```bash
python .claude/skills/run_testing/scripts/check_test_env.py
```

### Manual Environment Validation

**Check Conda Environment:**
```bash
conda info | grep "active environment"
```

**Check Test Data Presence:**
```bash
# Quick validation
ls tests/data/Isabela/ISWF01_22012023_subset/*.JPG | wc -l  # Should be 9
ls tests/data/raw_images/mavic2pro/**/*.JPG | wc -l        # Should be 50
ls tests/data/raw_images/matrice4e/**/*.JPG | wc -l        # Should be 51
```

**Check GPU Availability:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

**Check Models:**
```bash
ls tests/models/*/best_model.pth  # Should show 3 model files
```

## Test Data Sources

If test data is missing, source locations:

**Isabela Images (9 images, ~100 MB):**
- Source: `/Volumes/storage/.../Isabela/ISWF01_22012023_subset/`
- Already present in tests/data/

**Mavic 2 Pro Images (50 images, ~700 MB):**
- Source: `/Volumes/storage/.../Floreana/FLMO04_03022021/`
- Copy first 50 images to: `tests/data/raw_images/mavic2pro/Floreana/FLMO04_03022021/`

**Matrice 4E Images (51 images, ~150 MB):**
- Source: `~/Library/CloudStorage/.../Matrice 4e/Eberswalde/rpark_renamed_20250718/`
- Target: `tests/data/raw_images/matrice4e/Eberswalde/rpark_20250718/`

**HerdNet Models (211 MB - 1 GB):**
- Source: `HerdNet/best_models/`
- Already copied to: `tests/models/{dla34,convnext,herdnet_general}/`

## Environment Issues & Solutions

### Common Environment Problems

**"No module named 'active_learning'"**
```bash
pip install -e .
```

**"FileNotFoundError: test data not found"**
```bash
# Run the testing_data skill to set up missing data
# Or manually copy from source locations above
```

**"CUDA out of memory"**
```bash
# Run tests with CPU only
CUDA_VISIBLE_DEVICES="" pytest tests/
```

**"Permission denied"**
```bash
chmod -R 755 tests/data/
```

**Missing conda environment**
```bash
conda env create -f environment.yml
conda activate active_learning
pip install -e .
```