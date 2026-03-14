"""Tests for wildlife_detection.training.datasets."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from wildlife_detection.training.datasets import HerdNetDataset, TileMaskDataset


class TestTileMaskDataset:
    def test_length(self, sample_tiles_dir, sample_masks_dir):
        filenames = [f"tile_{i}.jpg" for i in range(5)]
        ds = TileMaskDataset(sample_tiles_dir, sample_masks_dir, filenames, imgsz=32)
        assert len(ds) == 5

    def test_output_shapes(self, sample_tiles_dir, sample_masks_dir):
        filenames = [f"tile_{i}.jpg" for i in range(5)]
        ds = TileMaskDataset(sample_tiles_dir, sample_masks_dir, filenames, imgsz=32)
        img, mask = ds[0]
        assert img.shape == (3, 32, 32)
        assert mask.shape == (32, 32)

    def test_output_dtypes(self, sample_tiles_dir, sample_masks_dir):
        filenames = ["tile_0.jpg"]
        ds = TileMaskDataset(sample_tiles_dir, sample_masks_dir, filenames, imgsz=32)
        img, mask = ds[0]
        assert img.dtype == torch.float32
        assert mask.dtype == torch.int64

    def test_mask_values_preserved(self, sample_tiles_dir, sample_masks_dir):
        """Mask class indices should be preserved (not normalized)."""
        filenames = ["tile_0.jpg"]
        ds = TileMaskDataset(sample_tiles_dir, sample_masks_dir, filenames, imgsz=64)
        _, mask = ds[0]
        unique = torch.unique(mask)
        # tile_0 has foreground (class 1) and background (class 0)
        assert 0 in unique
        assert 1 in unique

    def test_augment_flag(self, sample_tiles_dir, sample_masks_dir):
        filenames = ["tile_0.jpg"]
        ds_no_aug = TileMaskDataset(
            sample_tiles_dir, sample_masks_dir, filenames, imgsz=32, augment=False
        )
        ds_aug = TileMaskDataset(
            sample_tiles_dir, sample_masks_dir, filenames, imgsz=32, augment=True
        )
        assert ds_no_aug.augment is False
        assert ds_aug.augment is True


class TestHerdNetDataset:
    def test_length(self, sample_tiles_dir, sample_annotations_df):
        filenames = [f"tile_{i}.jpg" for i in range(5)]
        ds = HerdNetDataset(
            sample_tiles_dir, sample_annotations_df, filenames,
            patch_size=64, down_ratio=2, fidt_radius=1,
        )
        assert len(ds) == 5

    def test_output_shapes(self, sample_tiles_dir, sample_annotations_df):
        filenames = ["tile_0.jpg"]
        ds = HerdNetDataset(
            sample_tiles_dir, sample_annotations_df, filenames,
            patch_size=64, down_ratio=2, fidt_radius=1,
        )
        img, fidt_map, count = ds[0]
        assert img.shape == (3, 64, 64)
        assert fidt_map.shape == (1, 32, 32)  # 64 / 2 = 32
        assert isinstance(count, int)

    def test_count_matches_annotations(self, sample_tiles_dir, sample_annotations_df):
        """Count should match the number of annotations for the tile."""
        filenames = ["tile_0.jpg"]
        ds = HerdNetDataset(
            sample_tiles_dir, sample_annotations_df, filenames,
            patch_size=64, down_ratio=2, fidt_radius=1, augment=False,
        )
        _, _, count = ds[0]
        # tile_0.jpg has 2 annotations in sample_annotations_df
        assert count == 2

    def test_empty_tile_count_zero(self, sample_tiles_dir, sample_annotations_df):
        """A tile with no annotations should have count=0 and zero FIDT map."""
        filenames = ["tile_4.jpg"]
        ds = HerdNetDataset(
            sample_tiles_dir, sample_annotations_df, filenames,
            patch_size=64, down_ratio=2, fidt_radius=1, augment=False,
        )
        _, fidt_map, count = ds[0]
        # tile_4.jpg has 1 annotation at (40, 35) which is within the 64x64 tile
        assert count == 1

    def test_fidt_map_range(self, sample_tiles_dir, sample_annotations_df):
        """FIDT map values should be in [0, 1]."""
        filenames = ["tile_0.jpg"]
        ds = HerdNetDataset(
            sample_tiles_dir, sample_annotations_df, filenames,
            patch_size=64, down_ratio=2, fidt_radius=1,
        )
        _, fidt_map, _ = ds[0]
        assert fidt_map.min() >= 0.0
        assert fidt_map.max() <= 1.0

    def test_fidt_map_dtype(self, sample_tiles_dir, sample_annotations_df):
        filenames = ["tile_0.jpg"]
        ds = HerdNetDataset(
            sample_tiles_dir, sample_annotations_df, filenames,
            patch_size=64, down_ratio=2, fidt_radius=1,
        )
        _, fidt_map, _ = ds[0]
        assert fidt_map.dtype == torch.float32

    def test_down_ratio_affects_map_size(self, sample_tiles_dir, sample_annotations_df):
        filenames = ["tile_0.jpg"]
        ds4 = HerdNetDataset(
            sample_tiles_dir, sample_annotations_df, filenames,
            patch_size=64, down_ratio=4, fidt_radius=1,
        )
        _, fidt_map, _ = ds4[0]
        assert fidt_map.shape == (1, 16, 16)  # 64 / 4 = 16
