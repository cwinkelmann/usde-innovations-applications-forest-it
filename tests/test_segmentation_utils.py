"""Tests for wildlife_detection.training.segmentation utility functions."""

import torch
import pytest

from wildlife_detection.training.segmentation import compute_iou


class TestComputeIou:
    def test_perfect_prediction(self):
        """Perfect prediction should give IoU = 1.0."""
        pred = torch.tensor([[[1, 1], [0, 0]]])
        target = torch.tensor([[[1, 1], [0, 0]]])
        assert compute_iou(pred, target, num_classes=2) == 1.0

    def test_no_overlap(self):
        """No overlap should give IoU = 0.0."""
        pred = torch.tensor([[[1, 1], [0, 0]]])
        target = torch.tensor([[[0, 0], [1, 1]]])
        assert compute_iou(pred, target, num_classes=2) == 0.0

    def test_partial_overlap(self):
        """Partial overlap should give IoU between 0 and 1."""
        pred = torch.tensor([[[1, 1], [1, 0]]])
        target = torch.tensor([[[1, 0], [1, 0]]])
        iou = compute_iou(pred, target, num_classes=2)
        assert 0.0 < iou < 1.0
        # Expected: intersection=2, union=3, IoU=2/3
        assert abs(iou - 2.0 / 3.0) < 1e-6

    def test_all_background(self):
        """All background (class 0) should return 0.0 (no foreground)."""
        pred = torch.zeros(1, 4, 4, dtype=torch.long)
        target = torch.zeros(1, 4, 4, dtype=torch.long)
        assert compute_iou(pred, target, num_classes=2) == 0.0

    def test_multiclass(self):
        """Multi-class IoU should average across foreground classes."""
        pred = torch.tensor([[[1, 2], [0, 0]]])
        target = torch.tensor([[[1, 2], [0, 0]]])
        iou = compute_iou(pred, target, num_classes=3)
        assert iou == 1.0  # both classes perfectly predicted

    def test_multiclass_one_missing(self):
        """If one class is missing from both pred and target, skip it."""
        pred = torch.tensor([[[1, 1], [0, 0]]])
        target = torch.tensor([[[1, 1], [0, 0]]])
        # num_classes=3 but class 2 is absent → only class 1 contributes
        iou = compute_iou(pred, target, num_classes=3)
        assert iou == 1.0
