# HILDA Testing Troubleshooting Guide

## Common Test Failures

### Import and Setup Errors

**"ImportError: No module named 'active_learning'"**
- **Cause**: Package not installed in development mode
- **Solution**:
  ```bash
  pip install -e .
  ```

**"ModuleNotFoundError: No module named 'torch'"**
- **Cause**: PyTorch not installed or conda env not active
- **Solution**:
  ```bash
  conda activate active_learning
  # If still fails, reinstall environment
  conda env create -f environment.yml
  ```

### Test Data Errors

**"FileNotFoundError: Test data not found"**
- **Cause**: Missing test data in `tests/data/`
- **Solution**:
  ```bash
  python .claude/skills/run_testing/scripts/check_test_env.py
  # Follow guidance to set up missing data
  ```

**"AssertionError" in metadata tests**
- **Cause**: Corrupted or invalid EXIF data
- **Solution**:
  - Re-download test images from source
  - Check that images are not truncated
  - Verify GPS metadata present

### GPU and Model Errors

**"CUDA out of memory"**
- **Solution**:
  ```bash
  # Run CPU-only tests
  CUDA_VISIBLE_DEVICES="" pytest tests/

  # Or skip model tests
  pytest tests/ -k "not herdnet_predict"
  ```

**"RuntimeError: Expected all tensors to be on the same device"**
- **Cause**: Model/data device mismatch
- **Solution**: Run with CPU-only flag above

**"FileNotFoundError: No such file 'best_model.pth'"**
- **Cause**: Missing model files
- **Solution**: Check that models are present:
  ```bash
  ls tests/models/*/best_model.pth
  # Should show 3 files
  ```

### Permission and File System Errors

**"Permission denied"**
- **Solution**:
  ```bash
  chmod -R 755 tests/data/
  ```

**"OSError: [Errno 24] Too many open files"**
- **Solution**:
  ```bash
  ulimit -n 4096
  pytest tests/
  ```

## Test-Specific Troubleshooting

### Mission Import Tests

**Failing with coordinate errors**
- Check EPSG library: `pip install pyproj`
- Verify test images have GPS data
- Ensure folder structure matches expected layout

**Drone-specific failures**
- Verify both Mavic 2 Pro and Matrice 4E data present
- Check naming conventions match expected patterns
- Validate XMP metadata in test images

### HerdNet Model Tests

**Model loading failures**
- Confirm all 3 model directories present: `dla34/`, `convnext/`, `herdnet_general/`
- Check config.yaml files present alongside .pth files
- Verify model files not corrupted (reasonable file sizes)

**Inference failures**
- Ensure GPU has sufficient memory (>4GB)
- Check PyTorch CUDA compatibility
- Try CPU-only mode first to isolate GPU issues

### Database Tests

**SQLite errors**
- Check write permissions in test directory
- Verify no database files locked by other processes
- Clear any existing test databases: `rm tests/test_*.db`

**Image format issues**
- Ensure images are .JPG (not .jpeg or .JPEG)
- Check file extensions are consistent
- Verify images not corrupted

### Geospatial Tests

**CRS transformation errors**
- Check GDAL installation: `gdalinfo --version`
- Verify EPSG database: `pip install pyproj --upgrade`
- Ensure orthomosaic has valid CRS metadata

**Orthomosaic loading failures**
- Check file permissions on .tif files
- Verify orthomosaic not corrupted
- Test with `gdalinfo tests/data/geospatial/*.tif`

## Test Execution Debugging

### Verbose Debugging Mode
```bash
# Maximum detail for debugging
pytest tests/ -v -s --tb=long --capture=no
```

### Debug Specific Test
```bash
# Single test with debugging
pytest tests/test_image_database.py::TestCreateImageDb::test_create_image_db_isabela -v -s
```

### Check Test Discovery
```bash
# See what tests pytest finds
pytest tests/ --collect-only
```

### Coverage Analysis
```bash
# Generate coverage report
pytest tests/ --cov=active_learning --cov-report=html
open htmlcov/index.html
```

## Recovery Actions

### Full Environment Reset
```bash
# Nuclear option - reset everything
conda deactivate
conda remove -n active_learning --all
conda env create -f environment.yml
conda activate active_learning
pip install -e .
```

### Test Data Reset
```bash
# Remove and redownload test data
rm -rf tests/data/
# Then re-run setup from testing_data skill
```

### Clean Python Cache
```bash
# Remove Python cache files
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +
```

## Performance Optimization

### Parallel Testing
```bash
# Careful with GPU tests
pytest tests/ -n auto --dist=loadscope
```

### Fast Subset for CI
```bash
# Minimal tests for quick feedback
pytest tests/test_image_database.py::TestListImages tests/test_image_database.py::TestMetadataExtraction -v --quiet
```

## Getting Help

1. Check environment with `scripts/check_test_env.py`
2. Run single test in verbose mode to isolate issue
3. Check this troubleshooting guide for common patterns
4. Use pytest's builtin debugging: `pytest --pdb` to drop into debugger on failure