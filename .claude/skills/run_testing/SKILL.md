---
name: run_testing
description: Execute and manage HILDA test suite with comprehensive coverage mapping. Use this skill when users want to run tests, check test coverage, debug test failures, or understand the HILDA test structure. Also triggered when users mention pytest, testing, test suite, unit tests, integration tests, or want to verify code functionality.
---

# Testing Execution

Streamlined testing workflow for the HILDA project covering test execution, environment validation, and failure diagnosis.

## Core Testing Workflow

### Step 1: Assess Test Scope

Ask user to specify:
- **Scope**: "all tests", "specific module", "integration only", "fast tests only"
- **Environment**: "GPU available?", "test models downloaded?"

Common requests:
- "Run all tests" → Full test suite with environment check
- "Test mission import" → Pipeline tests only
- "Quick test" → Fast unit tests only
- "Test HerdNet" → Model inference tests (requires GPU)

### Step 2: Check Environment

Run environment validation:
```bash
python .claude/skills/run_testing/scripts/check_test_env.py
```

**Critical checks:**
- ✅ Conda environment active
- ✅ Test data present (9+50+51 images)
- ✅ Models available (3 model directories)
- ✅ GPU status (available/CPU-only)

If issues found → Point to `references/test-environments.md`

### Step 3: Execute Tests

Based on scope and environment:

**All Tests:**
```bash
pytest tests/ -v --tb=short --durations=10
```

**Fast Tests Only:**
```bash
pytest tests/test_image_database.py::TestListImages tests/test_image_database.py::TestMetadataExtraction -v
```

**Mission Import:**
```bash
pytest tests/pipeline/mission_import/ -v
```

**HerdNet Models (GPU required):**
```bash
pytest tests/test_herdnet_predict.py -v
```

**Specific Test:**
```bash
pytest tests/test_image_database.py::TestCreateImageDb -v
```

### Step 4: Interpret Results

Use result interpreter for actionable feedback:
```bash
# For failures, run:
python .claude/skills/run_testing/scripts/interpret_results.py <return_code> "<output>"
```

**Success (return_code=0):**
- Report: "🎉 ALL TESTS PASSED!"

**Failures:**
- Extract failed test names
- Identify common failure patterns (FileNotFoundError, ImportError, CUDA errors)
- Provide specific solutions from troubleshooting guide

## Test Categories

### Fast Unit Tests (< 30 seconds)
- `TestListImages` - File discovery and basic metadata
- `TestMetadataExtraction` - EXIF/XMP parsing

### Integration Tests (1-5 minutes)
- `TestCreateImageDb` - Full image database creation
- Mission import pipeline tests
- Geospatial prediction tests

### Model Tests (2-10 minutes, GPU-dependent)
- `test_herdnet_predict.py` - Model inference with multiple backends
- Requires: GPU + models in `tests/models/`

## Quick Reference Commands

### Debug Mode
```bash
pytest tests/ -v -s --tb=long --capture=no
```

### Coverage Analysis
```bash
pytest tests/ --cov=active_learning --cov-report=html
```

### CPU-Only Mode
```bash
CUDA_VISIBLE_DEVICES="" pytest tests/
```

### Skip Model Tests
```bash
pytest tests/ -k "not herdnet_predict"
```

## Test Data Structure

Required test data (validate with environment checker):
```
tests/data/
├── Isabela/ISWF01_22012023_subset/     # 9 images
├── raw_images/mavic2pro/               # 50 images
├── raw_images/matrice4e/               # 51 images
├── models/{dla34,convnext,herdnet_general}/  # 3 models
├── correspondence/                     # Deduplication test data
└── geospatial/                        # Orthomosaic test data
```

## Reference Resources

For detailed guidance, see:
- `references/test-environments.md` - Environment setup and validation
- `references/troubleshooting.md` - Common failures and solutions
- `scripts/check_test_env.py` - Automated environment validation
- `scripts/interpret_results.py` - Result analysis and guidance

## Common Test Failures

**Environment Issues:**
- Missing conda environment → `conda activate active_learning`
- Missing package → `pip install -e .`
- Missing test data → Run testing_data skill

**Runtime Issues:**
- GPU memory errors → Use `CUDA_VISIBLE_DEVICES="" pytest`
- Permission errors → `chmod -R 755 tests/data/`
- Import errors → Check PYTHONPATH and package installation

**Test-Specific Issues:**
- Mission import → Verify GPS metadata in test images
- HerdNet models → Check model files and GPU availability
- Database tests → Clear any locked database files

Always start with environment validation and use the reference guides for detailed troubleshooting.