"""Tests for wildlife_detection.utils.config."""

import pytest
from omegaconf import OmegaConf

from wildlife_detection.utils.config import load_config


class TestLoadConfig:
    def test_load_from_yaml(self, tmp_path):
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text("tile_size: 256\noverlap: 60\n")
        cfg = load_config(yaml_path)
        assert cfg.tile_size == 256
        assert cfg.overlap == 60

    def test_no_config_returns_empty(self):
        cfg = load_config()
        assert len(cfg) == 0

    def test_overrides_applied(self, tmp_path):
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text("tile_size: 256\noverlap: 60\n")
        cfg = load_config(yaml_path, overrides=["tile_size=512", "overlap=120"])
        assert cfg.tile_size == "512"  # overrides are strings
        assert cfg.overlap == "120"

    def test_overrides_without_config(self):
        cfg = load_config(overrides=["key=value", "nested.key=42"])
        assert cfg.key == "value"
        assert cfg.nested.key == "42"

    def test_override_with_equals_in_value(self, tmp_path):
        """Values containing '=' should be handled correctly."""
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text("param: original\n")
        cfg = load_config(yaml_path, overrides=["param=a=b=c"])
        assert cfg.param == "a=b=c"

    def test_nested_yaml(self, tmp_path):
        yaml_path = tmp_path / "nested.yaml"
        yaml_path.write_text("wandb:\n  project: test\n  entity: user\n")
        cfg = load_config(yaml_path)
        assert cfg.wandb.project == "test"
        assert cfg.wandb.entity == "user"
