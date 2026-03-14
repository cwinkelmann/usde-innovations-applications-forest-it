"""Tests for wildlife_detection.utils.density."""

import numpy as np
import pytest

from wildlife_detection.utils.density import generate_fidt_map


class TestGenerateFidtMap:
    def test_empty_points_returns_zeros(self):
        result = generate_fidt_map([], 64, 64)
        assert result.shape == (64, 64)
        assert result.dtype == np.float32
        assert np.all(result == 0)

    def test_single_point_peak(self):
        """A single point should produce a peak at its location."""
        result = generate_fidt_map([(32, 32)], 64, 64, radius=1)
        assert result[32, 32] == 1.0  # peak at the point
        # Value should decay with distance
        assert result[32, 32] > result[32, 40]
        assert result[32, 40] > result[32, 50]

    def test_output_range(self):
        """All values should be in [0, 1]."""
        points = [(10, 10), (50, 50), (30, 20)]
        result = generate_fidt_map(points, 64, 64, radius=1)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_multiple_points_distinct_peaks(self):
        """Two well-separated points should each have a distinct peak."""
        result = generate_fidt_map([(10, 10), (50, 50)], 64, 64, radius=1)
        assert result[10, 10] == 1.0
        assert result[50, 50] == 1.0
        # Midpoint should be lower
        assert result[30, 30] < 1.0

    def test_out_of_bounds_points_ignored(self):
        """Points outside the map should not crash."""
        result = generate_fidt_map([(-5, -5), (100, 100)], 64, 64, radius=1)
        assert result.shape == (64, 64)
        assert np.all(result == 0)  # both out of bounds

    def test_larger_radius_spreads_peak(self):
        """A larger radius should create a wider foreground region."""
        r1 = generate_fidt_map([(32, 32)], 64, 64, radius=1)
        r3 = generate_fidt_map([(32, 32)], 64, 64, radius=3)
        # With a larger radius, more pixels are at max value
        assert (r3 == 1.0).sum() > (r1 == 1.0).sum()

    def test_edge_point(self):
        """A point at the edge should still produce a valid map."""
        result = generate_fidt_map([(0, 0)], 64, 64, radius=1)
        assert result[0, 0] == 1.0
        assert result.shape == (64, 64)

    def test_dtype_is_float32(self):
        result = generate_fidt_map([(10, 10)], 32, 32)
        assert result.dtype == np.float32
