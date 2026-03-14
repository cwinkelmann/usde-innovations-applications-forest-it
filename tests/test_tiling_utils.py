"""Tests for wildlife_detection.tiling.utils."""

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.windows import Window

from wildlife_detection.tiling.utils import (
    find_class_column,
    generate_tile_windows,
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
        """1024x1024 raster with 512 tiles and no overlap → 4 tiles."""
        windows = list(generate_tile_windows(1024, 1024, 512, 0))
        assert len(windows) == 4

    def test_overlap_creates_more_tiles(self):
        """Overlap increases tile count."""
        without = list(generate_tile_windows(1024, 1024, 512, 0))
        with_overlap = list(generate_tile_windows(1024, 1024, 512, 120))
        assert len(with_overlap) > len(without)

    def test_edge_tiles_are_clipped(self):
        """Non-divisible raster produces smaller edge tiles."""
        windows = list(generate_tile_windows(700, 700, 512, 0))
        edge_widths = {w.width for w in windows}
        assert 188 in edge_widths  # 700 - 512 = 188

    def test_small_raster(self):
        """Raster smaller than tile_size → one clipped tile."""
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
        """With overlap, neighbouring tiles should share pixels."""
        windows = list(generate_tile_windows(1024, 512, 512, 120))
        # Step = 512 - 120 = 392. Tiles at col_off 0 and 392 overlap by 120 px.
        assert windows[0].col_off == 0
        assert windows[1].col_off == 392
        # Verify overlap region exists
        tile0_end = windows[0].col_off + windows[0].width
        tile1_start = windows[1].col_off
        assert tile0_end > tile1_start  # overlapping


class TestReadTile:
    def test_rgb_output_shape(self, sample_geotiff):
        with rasterio.open(sample_geotiff) as src:
            window = Window(0, 0, 50, 50)
            tile = read_tile(src, window, 50)
            assert tile.shape == (50, 50, 3)
            assert tile.dtype == np.uint8

    def test_padding_edge_tile(self, sample_geotiff):
        """Edge tiles should be padded to tile_size."""
        with rasterio.open(sample_geotiff) as src:
            window = Window(80, 80, 20, 20)  # only 20x20 data
            tile = read_tile(src, window, 64)
            assert tile.shape == (64, 64, 3)
            # Padded region should be zeros
            assert np.all(tile[20:, :, :] == 0)
            assert np.all(tile[:, 20:, :] == 0)

    def test_single_band_replicated(self, sample_geotiff_1band):
        """Single-band rasters should produce 3-channel output."""
        with rasterio.open(sample_geotiff_1band) as src:
            window = Window(0, 0, 32, 32)
            tile = read_tile(src, window, 32)
            assert tile.shape == (32, 32, 3)
            # All channels should be identical
            assert np.array_equal(tile[:, :, 0], tile[:, :, 1])
            assert np.array_equal(tile[:, :, 1], tile[:, :, 2])


class TestSaveTileJpeg:
    def test_saves_valid_jpeg(self, tmp_dir):
        array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        path = tmp_dir / "test.jpg"
        save_tile_jpeg(array, path)
        assert path.exists()
        # Re-read and check dimensions
        from PIL import Image
        img = Image.open(path)
        assert img.size == (64, 64)
        assert img.mode == "RGB"


class TestFindClassColumn:
    def test_finds_class_label(self):
        import geopandas as gpd
        from shapely.geometry import Point
        gdf = gpd.GeoDataFrame(
            {"class_label": ["a"], "geometry": [Point(0, 0)]},
        )
        assert find_class_column(gdf) == "class_label"

    def test_finds_species(self):
        import geopandas as gpd
        from shapely.geometry import Point
        gdf = gpd.GeoDataFrame(
            {"species": ["iguana"], "value": [1], "geometry": [Point(0, 0)]},
        )
        assert find_class_column(gdf) == "species"

    def test_returns_none_when_missing(self):
        import geopandas as gpd
        from shapely.geometry import Point
        gdf = gpd.GeoDataFrame(
            {"id": [1], "geometry": [Point(0, 0)]},
        )
        assert find_class_column(gdf) is None

    def test_priority_order(self):
        """class_label should be preferred over label."""
        import geopandas as gpd
        from shapely.geometry import Point
        gdf = gpd.GeoDataFrame(
            {"label": ["a"], "class_label": ["b"], "geometry": [Point(0, 0)]},
        )
        assert find_class_column(gdf) == "class_label"
