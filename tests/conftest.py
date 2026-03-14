"""Shared test fixtures for wildlife_detection tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def sample_image(tmp_path):
    """Create a small synthetic 3-band JPEG (100x100 pixels)."""
    path = tmp_path / "test_image.jpg"
    data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    Image.fromarray(data).save(path, quality=95)
    return path


@pytest.fixture
def sample_image_array():
    """Return a small synthetic (100, 100, 3) uint8 numpy array."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_1band(tmp_path):
    """Create a small synthetic grayscale JPEG (64x64 pixels)."""
    path = tmp_path / "test_1band.jpg"
    data = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    Image.fromarray(data, mode="L").save(path, quality=95)
    return path


@pytest.fixture
def sample_tiles_dir(tmp_path):
    """Create a directory with synthetic JPEG tiles."""
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    for i in range(5):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(img).save(tiles_dir / f"tile_{i}.jpg")
    return tiles_dir


@pytest.fixture
def sample_masks_dir(tmp_path):
    """Create a directory with synthetic PNG masks matching the tiles."""
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    for i in range(5):
        mask = np.zeros((64, 64), dtype=np.uint8)
        if i < 3:
            mask[20:30, 20:30] = 1
        Image.fromarray(mask).save(masks_dir / f"tile_{i}.png")
    return masks_dir


@pytest.fixture
def sample_annotations_df():
    """Create a sample point annotations DataFrame."""
    return pd.DataFrame({
        "tile_filename": ["tile_0.jpg", "tile_0.jpg", "tile_1.jpg",
                          "tile_2.jpg", "tile_3.jpg"],
        "local_x": [10.0, 50.0, 30.0, 25.0, 40.0],
        "local_y": [15.0, 45.0, 32.0, 28.0, 35.0],
        "class_label": ["iguana"] * 5,
        "source_geom_id": list(range(5)),
    })
