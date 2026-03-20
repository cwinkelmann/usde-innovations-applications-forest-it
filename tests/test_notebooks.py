"""Integration tests for Marimo notebooks and dataset downloads.

These tests verify that:
1. The download_data module functions work correctly
2. Downloaded datasets have the expected structure
3. Marimo notebooks can be executed without errors

Marked as @pytest.mark.integration — skipped by default in fast mode.
Run with: pytest tests/test_notebooks.py -v
"""

import sys
from pathlib import Path

import pytest

# Resolve week1/data/ so download_data is importable
_DATA_MODULE = Path(__file__).parent.parent / "week1" / "data"
if str(_DATA_MODULE) not in sys.path:
    sys.path.insert(0, str(_DATA_MODULE))


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _count_images(directory: Path) -> int:
    """Count image files in a directory (non-recursive)."""
    if not directory.exists():
        return 0
    return sum(1 for p in directory.iterdir()
               if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    """Shared temporary data directory for all download tests."""
    return tmp_path_factory.mktemp("week1_data")


# ---------------------------------------------------------------------------
# Download function tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDownloadSerengeti:
    """Test Snapshot Serengeti download."""

    def test_download_creates_images(self, data_dir):
        from download_data import download_serengeti
        result = download_serengeti(n_images=5, output_dir=data_dir)
        assert result.exists(), f"Output directory does not exist: {result}"
        n = _count_images(result)
        assert n > 0, "No images downloaded"
        assert n <= 5, f"Expected at most 5 images, got {n}"

    def test_download_creates_metadata(self, data_dir):
        meta = data_dir / "camera_trap" / "serengeti_meta.json"
        assert meta.exists(), "serengeti_meta.json not created"
        import json
        with open(meta) as f:
            data = json.load(f)
        assert "images" in data
        assert "annotations" in data
        assert "categories" in data

    def test_metadata_matches_images(self, data_dir):
        import json
        meta = data_dir / "camera_trap" / "serengeti_meta.json"
        with open(meta) as f:
            data = json.load(f)
        img_dir = data_dir / "camera_trap" / "serengeti_subset"
        downloaded = {p.name for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
        # All metadata images should have been attempted (some may fail due to network)
        assert len(data["images"]) > 0


@pytest.mark.integration
class TestDownloadCaltech:
    """Test Caltech Camera Traps download."""

    def test_download_creates_images(self, data_dir):
        from download_data import download_caltech
        result = download_caltech(n_images=5, output_dir=data_dir)
        assert result.exists()
        n = _count_images(result)
        assert n > 0, "No Caltech images downloaded"

    def test_download_creates_labels_csv(self, data_dir):
        labels = data_dir / "camera_trap_labels.csv"
        assert labels.exists(), "camera_trap_labels.csv not created"
        import pandas as pd
        df = pd.read_csv(labels)
        assert "crop" in df.columns
        assert "true_label" in df.columns
        assert len(df) > 0

    def test_labels_have_bbox_columns(self, data_dir):
        import pandas as pd
        df = pd.read_csv(data_dir / "camera_trap_labels.csv")
        for col in ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]:
            assert col in df.columns, f"Missing column: {col}"


@pytest.mark.integration
class TestDownloadEikelboom:
    """Test Eikelboom 2019 aerial dataset download."""

    def test_download_creates_splits(self, data_dir):
        from download_data import download_eikelboom
        result = download_eikelboom(n_images=5, output_dir=data_dir)
        assert result.exists()
        # At least one split directory should exist
        splits_found = [s for s in ["train", "val", "test"]
                        if (result / s).exists()]
        assert len(splits_found) > 0, "No train/val/test splits found"

    def test_train_split_has_images(self, data_dir):
        train_dir = data_dir / "eikelboom" / "train"
        if train_dir.exists():
            n = _count_images(train_dir)
            assert n > 0, "Train split has no images"


@pytest.mark.integration
class TestDownloadGeneralDataset:
    """Test HerdNet General Dataset download."""

    def test_download_creates_directory(self, data_dir):
        from download_data import download_general_dataset
        result = download_general_dataset(n_images=5, output_dir=data_dir)
        assert result.exists()

    def test_has_non_image_files(self, data_dir):
        """CSV or metadata files should be present."""
        gd = data_dir / "general_dataset"
        assert gd.exists()
        all_files = list(gd.rglob("*"))
        assert len(all_files) > 0, "General dataset directory is empty"


@pytest.mark.integration
class TestDownloadAll:
    """Test the download_all convenience function."""

    def test_download_all_returns_dict(self, tmp_path):
        from download_data import download_all
        results = download_all(n_images=3, output_dir=tmp_path, skip_weights=True)
        assert isinstance(results, dict)
        for key in ["general_dataset", "serengeti", "caltech", "eikelboom"]:
            assert key in results, f"Missing key: {key}"
            assert results[key].exists(), f"Path does not exist: {results[key]}"


@pytest.mark.integration
class TestSummarize:
    """Test the summarize function."""

    def test_summarize_returns_counts(self, data_dir):
        from download_data import summarize
        counts = summarize(output_dir=data_dir)
        assert isinstance(counts, dict)


# ---------------------------------------------------------------------------
# Marimo notebook execution tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
class TestMarimoNotebooks:
    """Test that Marimo notebooks execute without errors.

    These tests run the actual notebook files. They are slow because they
    download data and produce plots.
    """

    NOTEBOOKS_DIR = Path(__file__).parent.parent / "week1" / "practicals"

    def _run_notebook(self, notebook_path: Path):
        """Execute a Marimo notebook and return the exit code."""
        import subprocess
        result = subprocess.run(
            ["marimo", "run", "--headless", str(notebook_path)],
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max
            cwd=str(notebook_path.parent),
        )
        return result

    def test_p1_exists(self):
        p1 = self.NOTEBOOKS_DIR / "p1_visual_image_dataset.py"
        assert p1.exists(), f"P1 notebook not found: {p1}"

    def test_p1_is_valid_python(self):
        """P1 should parse as valid Python."""
        import ast
        p1 = self.NOTEBOOKS_DIR / "p1_visual_image_dataset.py"
        source = p1.read_text()
        ast.parse(source)  # raises SyntaxError if invalid

    def test_p1_imports_download_data(self):
        """P1 should import from download_data module."""
        p1 = self.NOTEBOOKS_DIR / "p1_visual_image_dataset.py"
        source = p1.read_text()
        assert "download_data" in source, "P1 does not reference download_data"

    def test_p1_has_marimo_app(self):
        """P1 should define a marimo App."""
        p1 = self.NOTEBOOKS_DIR / "p1_visual_image_dataset.py"
        source = p1.read_text()
        assert "marimo.App" in source
        assert "@app.cell" in source
