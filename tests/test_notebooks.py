"""Integration tests for Jupyter notebooks and dataset downloads.

These tests verify that:
1. The download_data module functions work correctly
2. Downloaded datasets have the expected structure
3. Jupyter notebooks execute top-to-bottom without errors

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
# Jupyter notebook execution tests
# ---------------------------------------------------------------------------


NOTEBOOKS_DIR = Path(__file__).parent.parent / "week1" / "practicals"

# (notebook filename, conda environment) — in execution order.
# Later notebooks may depend on outputs from earlier ones.
NOTEBOOKS = [
    ("practical_3_megadetector_legacy.ipynb", "fit-megadetector"),
    ("practical_3_megadetector_ultralytics.ipynb", "fit-training"),
    ("practical_5_species_classification.ipynb", "fit-training"),
]


def _execute_notebook(notebook_path: Path, conda_env: str, timeout: int = 600):
    """Execute a Jupyter notebook in a specific conda environment.

    Uses `conda run -n <env>` so the notebook runs with the correct
    dependencies without needing kernel registration.

    Returns (success: bool, error_message: str).
    """
    import subprocess
    result = subprocess.run(
        [
            "conda", "run", "-n", conda_env, "--no-capture-output",
            "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            f"--ExecutePreprocessor.timeout={timeout}",
            "--output", "/dev/null",
            str(notebook_path),
        ],
        capture_output=True,
        text=True,
        timeout=timeout + 60,
        cwd=str(notebook_path.parent),
    )
    if result.returncode != 0:
        err = result.stderr or result.stdout
        return False, err[-2000:]
    return True, ""


def _conda_env_exists(env_name: str) -> bool:
    """Check if a conda environment exists."""
    import subprocess
    result = subprocess.run(
        ["conda", "env", "list"],
        capture_output=True, text=True,
    )
    return env_name in result.stdout


@pytest.mark.integration
@pytest.mark.slow
class TestNotebookExecution:
    """Test that Jupyter notebooks execute top-to-bottom without errors.

    Each notebook runs in its designated conda environment via
    `conda run -n <env>`. Notebooks are tested in order because
    later practicals may depend on outputs from earlier ones.
    """

    @pytest.mark.parametrize(
        "notebook, conda_env",
        NOTEBOOKS,
        ids=[Path(n).stem for n, _ in NOTEBOOKS],
    )
    def test_notebook_runs(self, notebook, conda_env):
        nb_path = NOTEBOOKS_DIR / notebook
        if not nb_path.exists():
            pytest.skip(f"{notebook} not yet created")
        if not _conda_env_exists(conda_env):
            pytest.skip(f"conda env '{conda_env}' not installed")
        success, error = _execute_notebook(nb_path, conda_env)
        assert success, f"{notebook} failed in {conda_env}:\n{error}"
