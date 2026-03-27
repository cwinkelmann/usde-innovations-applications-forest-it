---
name: testing-data
description: Manage and validate HILDA test data requirements. Use when users need to check test data completeness, set up missing datasets, understand data requirements for different test categories, or troubleshoot test failures due to missing data. Also triggered when users mention test data, dataset setup, or Phase 0 requirements.
---

# HILDA Test Data Management Skill

Comprehensive test data management for HILDA, covering setup, validation, and troubleshooting of the complete test dataset required for end-to-end pipeline testing.

## Current Test Data Status

### ✅ **Available Test Data** (Ready for Testing)

**Raw Images for Mission Analysis:**
- **Isabela Subset**: `tests/data/Isabela/ISWF01_22012023_subset/` (9 JPGs, ~100 MB)
  - Essential for: Basic mission import, metadata extraction, image database tests
  - Status: ✅ Complete - Mavic 2 Pro images with GPS, timestamps, camera metadata

- **Mavic 2 Pro Dataset**: `tests/data/raw_images/mavic2pro/Floreana/FLMO04_03022021/` (50 JPGs, ~700 MB)
  - Essential for: Mission import pipeline tests, flight metrics calculation
  - Status: ✅ Complete - Full mission with rich EXIF/XMP metadata

- **Matrice 4E Dataset**: `tests/data/raw_images/matrice4e/Eberswalde/rpark_20250718/` (51 JPGs + sidecars)
  - Essential for: Multi-drone support, advanced features (LRF, sensors)
  - Status: ✅ Complete - European coordinates, different naming convention

**Orthomosaics for Inference:**
- **Floreana Full**: `tests/data/Floreana/FMO02_full_orthophoto.tif` (952 MB, EPSG:32715)
- **Isabela Projection**: `tests/data/Isabela/ISWF01_22012023_subset/Orthomosaic_proj.tif` (225 MB)
- **Isabela DEM**: `tests/data/Isabela/ISWF01_22012023_subset/DEM.tif` (49 MB)
- **Additional Tiles**: 93 TIF tiles for testing tile-based processing

**Annotations:**
- **Hasty Labels**: `tests/data/FMO02_03_05_labels.json` (852 KB, covers FMO02/03/05)
- **Crop Segments**: `tests/data/hasty_format_crops_segments.json` (384 KB)

**Geospatial Data:**
- **Shapefiles**: `tests/data/geospatial/FMO02.*` (Complete shapefile set)
- **GeoJSON**: Multiple detection and count files
- **Reference Data**: `tests/data/geospatial/Isa_ISVP01_27012023.*` (orthomosaic + detections)


### **Trained Models** (Required for inference tests):
```
tests/models/
├── dla34/                           # ✅ Available (211 MB)
│   ├── best_model.pth              # Primary HerdNet model
│   └── config.yaml                 # Hydra configuration
├── convnext/                       # ✅ Available (1.0 GB)
│   ├── best_model.pth              # ConvNeXt camouflaged model
│   └── config.yaml                 # Hydra configuration
└── herdnet_general/                # ✅ Available (210 MB)
    ├── 20220413_HerdNet_General_dataset_2022.pth    # Legacy HerdNet model (original name)
    └── 20220413_HerdNet_General_dataset_2022.yaml   # YAML configuration (original name)
```

**Correspondence Tracking Data** (Required for deduplication tests):
```
tests/data/correspondence/
├── cameras_Isa_ISWF01_22012023.xml  # ✅ Available (484 KB) - Metashape camera parameters
├── dem_Isa_ISWF01_22012023.tif      # ✅ Available (46 MB) - DEM for 3D projection
├── detections.csv                   # ✅ Available (16 KB) - HerdNet predictions input
└── ISWF01_22012023_subset/          # ✅ Available (symlink to Isabela images)
```

## Test Data Setup Guide

### Step 1: Validate Current Setup

```bash
# Check what's currently available
python -c "
import subprocess
from pathlib import Path

def check_test_data():
    test_data_dir = Path('tests/data')

    # Check raw images
    isabela_images = list((test_data_dir / 'Isabela/ISWF01_22012023_subset').glob('*.JPG'))
    mavic_images = list((test_data_dir / 'raw_images/mavic2pro').glob('**/*.JPG'))
    matrice_images = list((test_data_dir / 'raw_images/matrice4e').glob('**/*.JPG'))

    print(f'✅ Isabela images: {len(isabela_images)} (expect 9)')
    print(f'✅ Mavic 2 Pro images: {len(mavic_images)} (expect 50)')
    print(f'✅ Matrice 4E images: {len(matrice_images)} (expect 51)')

    # Check orthomosaics
    orthomosaics = list(test_data_dir.glob('**/*.tif'))
    orthomosaics = [o for o in orthomosaics if 'tiles' not in str(o)]
    print(f'✅ Orthomosaics: {len(orthomosaics)} major files')

    # Check models (in separate tests/models/ directory)
    models_dir = Path("tests/models")
    models = list(models_dir.glob('**/*.pth')) if models_dir.exists() else []
    print(f'✅ Trained models: {len(models)} (expect 3)')

    # Check annotations
    annotations = list(test_data_dir.glob('*.json'))
    print(f'✅ Annotation files: {len(annotations)}')

    return len(models) > 0

check_test_data()
"
```

### Step 2: Download Missing Models

**Source Locations:**
```bash
# DLA-34 Primary Model (211 MB)
# From: HerdNet/best_models/10-51-48_20251216_x25_dla34_ys2uq6yf/
mkdir -p tests/models/dla34
cp HerdNet/best_models/10-51-48_*/best_model.pth tests/models/dla34/
cp HerdNet/best_models/10-51-48_*/.hydra/config.yaml tests/models/dla34/

# ConvNeXt Model (1.0 GB) - Optional
# From: HerdNet/best_models/17-02-44_convnext_camouflaged_zr7ljum7/
mkdir -p tests/models/convnext
cp HerdNet/best_models/17-02-44_*/best_model.pth tests/models/convnext/
cp HerdNet/best_models/17-02-44_*/.hydra/config.yaml tests/models/convnext/

# HerdNet General (210 MB)
# From: HerdNet/best_models/20220413_HerdNet_General_dataset_2022.*
mkdir -p tests/models/herdnet_general
cp HerdNet/best_models/20220413_HerdNet_General_dataset_2022.pth tests/models/herdnet_general/
cp HerdNet/best_models/20220413_HerdNet_General_dataset_2022.yaml tests/models/herdnet_general/
```

### Step 3: Set Up Correspondence Data

**Source Location:**
```bash
# From: /Users/christian/data/Manual Counting/single_image_dedublication/flat/
mkdir -p tests/data/correspondence

# Camera parameters (484 KB)
cp "/Users/christian/data/Manual Counting/single_image_dedublication/flat/cameras_Isa_ISWF01_22012023.xml" \
   tests/data/correspondence/

# DEM for 3D projection (46 MB - smaller version)
cp "/Users/christian/data/Manual Counting/single_image_dedublication/flat/ISWF01_22012023_subset/CorrespondanceTracking/Isa_ISWF01_22012023/DEM.tif" \
   tests/data/correspondence/dem_Isa_ISWF01_22012023.tif

# HerdNet detections input (16 KB)
cp "/Users/christian/PycharmProjects/hnee/HerdNet/tools/outputs/2026-01-23/11-35-29/detections.csv" \
   tests/data/correspondence/

# Images (reuse existing)
ln -s ../Isabela/ISWF01_22012023_subset tests/data/correspondence/ISWF01_22012023_subset
```

### Step 4: Alternative: Automated Setup Script

Create `scripts/setup_test_data.py`:

```python
#!/usr/bin/env python3
"""
Automated HILDA test data setup
"""
import shutil
import sys
from pathlib import Path

def setup_test_data():
    """Set up HILDA test data from known source locations"""
    project_root = Path.cwd()
    test_data_dir = project_root / "tests" / "data"
    models_dir = project_root / "tests" / "models"

    # Source paths (adjust for your system)
    sources = {
        "models": {
            "dla34": "HerdNet/best_models/10-51-48_20251216_x25_dla34_ys2uq6yf",
            "convnext": "HerdNet/best_models/17-02-44_convnext_camouflaged_zr7ljum7",
            "herdnet": "HerdNet/best_models/20220413_HerdNet_General_dataset_2022"
        },
        "correspondence": "/Users/christian/data/Manual Counting/single_image_dedublication/flat"
    }

    print("🔍 Checking source availability...")
    available_sources = {}

    # Check model sources
    for model_name, source_path in sources["models"].items():
        source = Path(source_path)
        if source.exists():
            print(f"✅ {model_name}: {source}")
            available_sources[model_name] = source
        else:
            print(f"❌ {model_name}: {source} (not found)")

    # Check correspondence source
    corr_source = Path(sources["correspondence"])
    if corr_source.exists():
        print(f"✅ Correspondence: {corr_source}")
        available_sources["correspondence"] = corr_source
    else:
        print(f"❌ Correspondence: {corr_source} (not found)")

    if not available_sources:
        print("❌ No source data found for automated setup")
        return False

    # Setup models
    for model_name, source_path in available_sources.items():
        if model_name in ["dla34", "convnext"]:
            target_dir = models_dir / model_name
            target_dir.mkdir(parents=True, exist_ok=True)

            # Copy model file
            model_file = source_path / "best_model.pth"
            config_file = source_path / ".hydra" / "config.yaml"

            if model_file.exists():
                shutil.copy2(model_file, target_dir / "best_model.pth")
                print(f"✅ Copied {model_name} model")

            if config_file.exists():
                shutil.copy2(config_file, target_dir / "config.yaml")
                print(f"✅ Copied {model_name} config")

        elif model_name == "herdnet":
            target_dir = models_dir / "herdnet_general"
            target_dir.mkdir(parents=True, exist_ok=True)

            # HerdNet has different file structure
            pth_file = source_path.with_suffix(".pth")
            yaml_file = source_path.with_suffix(".yaml")

            if pth_file.exists():
                shutil.copy2(pth_file, target_dir / "best_model.pth")
            if yaml_file.exists():
                shutil.copy2(yaml_file, target_dir / "config.yaml")

    # Setup correspondence data
    if "correspondence" in available_sources:
        source_path = available_sources["correspondence"]
        target_dir = test_data_dir / "correspondence"
        target_dir.mkdir(parents=True, exist_ok=True)

        # Copy camera parameters
        camera_file = source_path / "cameras_Isa_ISWF01_22012023.xml"
        if camera_file.exists():
            shutil.copy2(camera_file, target_dir / camera_file.name)
            print(f"✅ Copied {camera_file.name}")

        # Copy smaller DEM file from nested location
        dem_file = source_path / "ISWF01_22012023_subset/CorrespondanceTracking/Isa_ISWF01_22012023/DEM.tif"
        if dem_file.exists():
            shutil.copy2(dem_file, target_dir / "dem_Isa_ISWF01_22012023.tif")
            print(f"✅ Copied DEM file (46 MB)")

        # Copy detections input
        detections_file = Path("/Users/christian/PycharmProjects/hnee/HerdNet/tools/outputs/2026-01-23/11-35-29/detections.csv")
        if detections_file.exists():
            shutil.copy2(detections_file, target_dir / "detections.csv")
            print(f"✅ Copied {detections_file.name}")

    print("\n🎉 Test data setup complete!")
    return True

if __name__ == "__main__":
    setup_test_data()
```

## Test Data Validation

### Comprehensive Validation Script

```python
def validate_test_data():
    """Validate HILDA test data completeness and integrity"""
    from pathlib import Path
    import json

    test_data_dir = Path("tests/data")
    models_dir = Path("tests/models")
    issues = []

    # Required datasets with expected counts
    requirements = {
        "Isabela images": {
            "path": test_data_dir / "Isabela/ISWF01_22012023_subset",
            "pattern": "*.JPG",
            "expected": 9,
            "critical": True
        },
        "Mavic 2 Pro images": {
            "path": test_data_dir / "raw_images/mavic2pro",
            "pattern": "**/*.JPG",
            "expected": 50,
            "critical": False
        },
        "Matrice 4E images": {
            "path": test_data_dir / "raw_images/matrice4e",
            "pattern": "**/*.JPG",
            "expected": 51,
            "critical": False
        },
        "DLA-34 model": {
            "path": test_data_dir / "models/dla34",
            "pattern": "*.pth",
            "expected": 1,
            "critical": True
        },
        "Model configs": {
            "path": test_data_dir / "models",
            "pattern": "**/config.yaml",
            "expected": 3,
            "critical": True
        },
        "Orthomosaics": {
            "path": test_data_dir,
            "pattern": "**/*.tif",
            "expected": 5,  # Approximate, varies
            "critical": False
        }
    }

    print("🔍 Validating HILDA test data...")
    print("=" * 50)

    for name, req in requirements.items():
        path = req["path"]
        pattern = req["pattern"]
        expected = req["expected"]
        critical = req["critical"]

        if not path.exists():
            issue = f"❌ {name}: Path {path} does not exist"
            if critical:
                issues.append(issue)
            print(issue)
            continue

        files = list(path.glob(pattern))
        count = len(files)

        if count == expected:
            print(f"✅ {name}: {count} files (perfect)")
        elif count > expected * 0.8:  # 80% threshold
            print(f"⚠️ {name}: {count} files (expected {expected}, close enough)")
        else:
            issue = f"❌ {name}: {count} files (expected {expected})"
            if critical:
                issues.append(issue)
            print(issue)

    # Additional integrity checks
    print("\n🔍 Integrity Checks:")

    # Check annotation files are valid JSON
    for json_file in test_data_dir.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            print(f"✅ {json_file.name}: Valid JSON ({len(data)} items)")
        except Exception as e:
            issue = f"❌ {json_file.name}: Invalid JSON - {e}"
            issues.append(issue)
            print(issue)

    # Summary
    print("\n" + "=" * 50)
    if issues:
        print(f"❌ {len(issues)} critical issues found:")
        for issue in issues:
            print(f"  {issue}")
        print("\n💡 Run test data setup to resolve missing components")
        return False
    else:
        print("✅ All critical test data validated successfully!")
        print("💡 Ready to run: python -m pytest tests/ -v")
        return True

# Run validation
validate_test_data()
```

## Test Categories and Data Requirements

### **Fast Unit Tests** (<30 seconds)
**Required Data:**
- ✅ Isabela subset (9 images) - for metadata extraction tests
- ✅ Sample annotations - for format validation tests

**Commands:**
```bash
# Run fast tests (no models required)
python -m pytest tests/test_image_database.py::TestListImages -v
python -m pytest tests/test_image_database.py::TestMetadataExtraction -v
```

### **Integration Tests** (1-5 minutes)
**Required Data:**
- ✅ Isabela + Mavic 2 Pro images (59 total)
- ✅ Orthomosaics for geospatial tests
- ✅ At least one trained model (DLA-34 minimum)

**Commands:**
```bash
# Run integration tests
python -m pytest tests/test_image_database.py::TestCreateImageDb -v
python -m pytest tests/pipeline/mission_import/ -v
```

### **Model Tests** (5-30 minutes, requires GPU)
**Required Data:**
- ✅ All trained models (DLA-34, ConvNeXt, HerdNet General)
- ✅ Test orthomosaics
- ✅ Reference detection results

**Commands:**
```bash
# Run model inference tests (GPU required)
python -m pytest tests/test_herdnet_predict.py -v
python -m pytest tests/test_geospatial_prediction.py -v
```

### **Full Pipeline Tests** (30+ minutes)
**Required Data:**
- ✅ Complete dataset including correspondence data
- ✅ All trained models
- ✅ All orthomosaics and annotations

**Commands:**
```bash
# Run complete test suite
python -m pytest tests/ -v --durations=10
```

## Database Construction (Analysis-Ready)

### **Test Databases Overview** (Built from test images using current pipeline)

**Test Databases Available:**
```
tests/data/database/
├── floreana_test.parquet          # 93 KB  - Floreana missions (54 images)
├── isabela_test.parquet           # 79 KB  - Isabela missions (81 images)
├── raw_images_test.parquet        # 113 KB - Multi-drone test set (110 images)
├── combined_test_database.parquet # 132 KB - Complete test dataset (175 images)
├── 2020_2021_2022_2023_2024_database.parquet          # 50 MB  - Production database
└── 2020_2021_2022_2023_2024_database_analysis_ready_* # 77 MB  - Production analysis
```

### **Database Schema (Current vs Production)**

**Current Test Schema:** 114 columns (enhanced analysis-ready format)
- **100% Production Compatibility:** All 84 production columns included ✅
- **30 New Analysis Features:** expedition_phase, flight_direction, risk_score, etc.
- **Enhanced Metrics:** GSD calculations, quality scores, correspondence tracking

**Key Columns Available:**
- **Geospatial:** latitude, longitude, geometry (UTM 15S projected)
- **Image Metadata:** datetime, image_name, camera details, EXIF data
- **Flight Metrics:** GSD, altitude, speed, flight direction, quality scores
- **Mission Data:** island, folder_name, expedition_phase, drone_name

### **Database Construction Pipeline**

**Quick Rebuild (if databases corrupted/outdated):**
```python
from active_learning.database import create_image_db
from active_learning.types.image_metadata import convert_to_serialisable_dataframe
from pathlib import Path

# Rebuild specific dataset
def rebuild_database(source_path, output_path):
    gdf_database = create_image_db(
        images_path=Path(source_path),
        local_epsg=32715,  # UTM 15S for Galápagos
        image_extension="JPG"
    )

    # Convert to parquet-compatible format
    gdf_serializable = convert_to_serialisable_dataframe(gdf_database)
    gdf_serializable.to_parquet(output_path)

    print(f"✅ Database: {len(gdf_database)} records, {gdf_database.memory_usage().sum()/1024/1024:.1f} MB")

# Examples
rebuild_database("tests/data/Floreana", "tests/data/database/floreana_test.parquet")
rebuild_database("tests/data/Isabela", "tests/data/database/isabela_test.parquet")
```

**Performance Characteristics:**
- **Processing Speed:** ~55-60 images/second
- **Memory Usage:** Minimal (<100MB peak for test datasets)
- **File Sizes:** 1-2 KB per image record in parquet format
- **Schema Evolution:** Automatic compatibility with enhanced features

**Database Contents:**
- **Floreana:** 54 images, GSD ~0.68 cm/px, 2021-02-02 to 2021-02-03
- **Isabela:** 81 images, mixed GSD, 2023-01-22 missions
- **Raw Images:** 110 images, multi-drone (Mavic2Pro + Matrice4E), 2021-2025
- **Combined:** 175 images, complete test coverage

**Essential for Testing:**
- `tests/test_database_construction.py` - Database pipeline validation
- `scripts/post_flight/010_image_db.py` - Production database creation
- Image discovery, metadata extraction, GSD calculation, quality scoring


## Troubleshooting Test Data Issues

### **Common Problems and Solutions**

**"FileNotFoundError: Test data not found"**
```bash
# Solution 1: Validate test data setup
python -c "from tests.conftest import validate_test_data; validate_test_data()"

# Solution 2: Check specific paths
ls tests/data/Isabela/ISWF01_22012023_subset/*.JPG
ls tests/models/dla34/best_model.pth
```

**"Model loading failed"**
```bash
# Check model files exist and have correct structure
find tests/models -name "*.pth" -exec ls -lh {} \;
find tests/models -name "config.yaml" -exec head -5 {} \;

# Verify model compatibility
python -c "
import torch
model = torch.load('tests/models/dla34/best_model.pth', map_location='cpu')
print(f'Model keys: {list(model.keys())[:5]}')
"
```

**"Permission denied" errors**
```bash
# Fix file permissions
chmod -R 755 tests/data/
find tests/data -type f -name "*.JPG" -exec chmod 644 {} \;
```

**"Disk space issues"**
```bash
# Check test data size
du -sh tests/data/*

# Total expected size: ~3.3 GB
# - Raw images: ~850 MB
# - Orthomosaics: ~1.2 GB
# - Models: ~1.4 GB
# - Correspondence: ~2.2 GB (mostly DEM)
```

## Test Data Maintenance

### **Regular Validation**
```bash
# Add to CI pipeline or run weekly
python scripts/validate_test_data.py --fix-permissions --report-missing
```

### **Cleanup Temporary Files**
```bash
# Clean up test outputs
find tests/data/output -type f -mtime +7 -delete
find tests/temp -type f -delete 2>/dev/null
```

### **Update Test Data**
```bash
# When updating models or datasets
python scripts/update_test_data.py --component models --source HerdNet/best_models/
```

This comprehensive test data management ensures HILDA testing is reliable, reproducible, and covers all critical functionality while providing clear guidance for setup and troubleshooting.