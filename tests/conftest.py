"""Shared test fixtures for wildlife_detection tests."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_bounds


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory."""
    return tmp_path


@pytest.fixture
def sample_geotiff(tmp_path):
    """Create a small synthetic 3-band GeoTIFF (100x100 pixels)."""
    path = tmp_path / "test_ortho.tif"
    width, height = 100, 100
    data = np.random.randint(0, 255, (3, height, width), dtype=np.uint8)
    transform = from_bounds(0, 0, 100, 100, width, height)

    with rasterio.open(
        path, "w", driver="GTiff",
        width=width, height=height, count=3, dtype="uint8",
        crs="EPSG:32717", transform=transform,
    ) as dst:
        dst.write(data)

    return path


@pytest.fixture
def sample_geotiff_1band(tmp_path):
    """Create a single-band GeoTIFF."""
    path = tmp_path / "test_1band.tif"
    width, height = 64, 64
    data = np.random.randint(0, 255, (1, height, width), dtype=np.uint8)
    transform = from_bounds(0, 0, 64, 64, width, height)

    with rasterio.open(
        path, "w", driver="GTiff",
        width=width, height=height, count=1, dtype="uint8",
        crs="EPSG:32717", transform=transform,
    ) as dst:
        dst.write(data)

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
        # Put some foreground pixels in some masks
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
