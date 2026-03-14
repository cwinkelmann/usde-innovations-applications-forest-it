"""Tests for wildlife_detection.tiling.utils."""

import numpy as np
import pytest
from PIL import Image

from wildlife_detection.tiling.utils import (
    TileWindow,
    find_class_column,
    generate_tile_windows,
    load_image_array,
    read_tile,
    save_tile_jpeg,
)


class TestGenerateTileWindows:
    def test_single_tile_no_overlap(self):
        windows = list(generate_tile_windows(512, 512, 512, 0))
        assert len(windows) == 1
        w = windows[0]
        assert w.col_off == 0
        assert w.row_off == 0
        assert w.width == 512
        assert w.height == 512

    def test_exact_grid(self):
        """1024x1024 image with 512 tiles and no overlap → 4 tiles."""
        windows = list(generate_tile_windows(1024, 1024, 512, 0))
        assert len(windows) == 4

    def test_overlap_creates_more_tiles(self):
        without = list(generate_tile_windows(1024, 1024, 512, 0))
        with_overlap = list(generate_tile_windows(1024, 1024, 512, 120))
        assert len(with_overlap) > len(without)

    def test_edge_tiles_are_clipped(self):
        """Non-divisible image produces smaller edge tiles."""
        windows = list(generate_tile_windows(700, 700, 512, 0))
        edge_widths = {w.width for w in windows}
        assert 188 in edge_widths  # 700 - 512 = 188

    def test_small_image(self):
        """Image smaller than tile_size → one clipped tile."""
        windows = list(generate_tile_windows(100, 100, 512, 0))
        assert len(windows) == 1
        assert windows[0].width == 100
        assert windows[0].height == 100

    def test_all_windows_are_within_bounds(self):
        windows = list(generate_tile_windows(1000, 800, 256, 64))
        for w in windows:
            assert w.col_off >= 0
            assert w.row_off >= 0
            assert w.col_off + w.width <= 1000
            assert w.row_off + w.height <= 800

    def test_overlap_coverage(self):
        """With overlap, neighbouring tiles share pixels."""
        windows = list(generate_tile_windows(1024, 512, 512, 120))
        assert windows[0].col_off == 0
        assert windows[1].col_off == 392  # step = 512 - 120
        tile0_end = windows[0].col_off + windows[0].width
        tile1_start = windows[1].col_off
        assert tile0_end > tile1_start


class TestReadTile:
    def test_rgb_output_shape(self, sample_image_array):
        window = TileWindow(0, 0, 50, 50)
        tile = read_tile(sample_image_array, window, 50)
        assert tile.shape == (50, 50, 3)
        assert tile.dtype == np.uint8

    def test_padding_edge_tile(self, sample_image_array):
        """Edge tiles are padded to tile_size with zeros."""
        window = TileWindow(80, 80, 20, 20)  # only 20x20 data
        tile = read_tile(sample_image_array, window, 64)
        assert tile.shape == (64, 64, 3)
        assert np.all(tile[20:, :, :] == 0)
        assert np.all(tile[:, 20:, :] == 0)

    def test_full_tile_no_padding(self, sample_image_array):
        """Full tile with no padding returns a copy of the crop."""
        window = TileWindow(0, 0, 100, 100)
        tile = read_tile(sample_image_array, window, 100)
        assert tile.shape == (100, 100, 3)
        assert np.array_equal(tile, sample_image_array)


class TestLoadImageArray:
    def test_rgb_jpeg(self, sample_image):
        arr = load_image_array(sample_image)
        assert arr.shape == (100, 100, 3)
        assert arr.dtype == np.uint8

    def test_grayscale_converted_to_rgb(self, sample_image_1band):
        arr = load_image_array(sample_image_1band)
        assert arr.shape[2] == 3  # always returns 3 channels


class TestSaveTileJpeg:
    def test_saves_valid_jpeg(self, tmp_dir):
        array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        path = tmp_dir / "test.jpg"
        save_tile_jpeg(array, path)
        assert path.exists()
        img = Image.open(path)
        assert img.size == (64, 64)
        assert img.mode == "RGB"


class TestFindClassColumn:
    def test_finds_class_label(self):
        import pandas as pd
        df = pd.DataFrame({"class_label": ["a"], "x": [0], "y": [0]})
        assert find_class_column(df) == "class_label"

    def test_finds_species(self):
        import pandas as pd
        df = pd.DataFrame({"species": ["iguana"], "value": [1], "x": [0]})
        assert find_class_column(df) == "species"

    def test_returns_none_when_missing(self):
        import pandas as pd
        df = pd.DataFrame({"id": [1], "x": [0], "y": [0]})
        assert find_class_column(df) is None

    def test_priority_order(self):
        """class_label should be preferred over label."""
        import pandas as pd
        df = pd.DataFrame({"label": ["a"], "class_label": ["b"], "x": [0]})
        assert find_class_column(df) == "class_label"
