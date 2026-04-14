"""Smoke tests — import the wrappers; full inference tests require weights + data."""
from __future__ import annotations


def test_import_megadetector_module() -> None:
    from webapp.app.detectors import megadetector as m

    assert m.LABELS[0] == "animal"
    assert m.LABELS[1] == "person"
    assert m.LABELS[2] == "vehicle"


def test_import_speciesnet_module() -> None:
    from webapp.app.detectors import speciesnet as s

    md_results = [
        {
            "file": "/tmp/a.jpg",
            "width": 100,
            "height": 200,
            "detections": [{"category_id": 0, "label": "animal", "conf": 0.9, "bbox_xyxy": [10, 20, 50, 120]}],
        }
    ]
    formatted = s.SpeciesNetClassifier._md_to_speciesnet_format(md_results)
    det = formatted["/tmp/a.jpg"]["detections"][0]
    assert det["category"] == "1"
    assert det["bbox"] == [0.1, 0.1, 0.4, 0.5]


def test_parse_taxonomy() -> None:
    from webapp.app.detectors.speciesnet import SpeciesNetClassifier

    parsed = SpeciesNetClassifier._parse_taxonomy("mammalia;carnivora;felidae;panthera;panthera_leo;lion")
    assert parsed["common_name"] == "lion"
    assert parsed["species"] == "panthera_leo"


def test_config_defaults(monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", "/tmp/usde-test")
    # Reimport to pick up env
    import importlib

    from webapp.app import config

    importlib.reload(config)
    assert str(config.DATA_DIR) == "/tmp/usde-test"
    assert config.UPLOADS_DIR.name == "uploads"
