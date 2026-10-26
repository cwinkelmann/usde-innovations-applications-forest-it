"""Tests for the crown-decline tile index and prediction cache.

Deliberately torch-free and animaloc-free: these cover the parts that decide
whether a 240-tile run is resumable and whether a parameter change is visible
to the cache. The model wrapper itself needs weights + the HerdNet fork, so it
is only smoke-imported where that is available.
"""
from __future__ import annotations

import json

import pytest

from webapp.app import crown_tiles as ct


def test_params_key_is_stable_and_order_independent(tmp_path) -> None:
    weights = tmp_path / "w.pth"
    weights.write_bytes(b"x" * 16)
    a = ct.params_key(weights, {"adapt_ts": 0.2, "kernel_size": 9})
    b = ct.params_key(weights, {"kernel_size": 9, "adapt_ts": 0.2})
    assert a == b, "key must not depend on dict ordering"
    assert len(a) == 10


def test_params_key_changes_with_every_knob(tmp_path) -> None:
    weights = tmp_path / "w.pth"
    weights.write_bytes(b"x" * 16)
    base = {"adapt_ts": 0.2, "kernel_size": 9, "overlap": 120}
    key = ct.params_key(weights, base)
    for field, other in [
        ("adapt_ts", 0.25),
        ("kernel_size", 11),
        ("overlap", 64),
    ]:
        changed = dict(base, **{field: other})
        assert ct.params_key(weights, changed) != key, (
            f"changing {field} must open a new cache namespace, otherwise a "
            "re-run silently reports the previous parameters' results"
        )


def test_params_key_changes_when_weights_change(tmp_path) -> None:
    weights = tmp_path / "w.pth"
    weights.write_bytes(b"x" * 16)
    key = ct.params_key(weights, {"adapt_ts": 0.2})
    weights.write_bytes(b"y" * 32)  # different size => different identity
    assert ct.params_key(weights, {"adapt_ts": 0.2}) != key


def test_cache_roundtrip_and_resume(tmp_path) -> None:
    cache = ct.PredictionCache(tmp_path, "abc1234567", {"adapt_ts": 0.2})
    assert (cache.dir / "params.json").is_file()

    assert cache.has("c000_r000") is False
    assert cache.load("c000_r000") is None

    rec = {"tile": "c000_r000", "count": 2, "points": [{"x": 1.0, "y": 2.0}]}
    cache.save("c000_r000", rec)

    assert cache.has("c000_r000") is True
    assert cache.load("c000_r000") == rec
    # params.json must never be mistaken for a tile record.
    assert cache.cached_names() == {"c000_r000"}
    assert set(cache.load_all()) == {"c000_r000"}


def test_cache_save_is_atomic(tmp_path) -> None:
    """A crashed write must not leave a half-file that later reads as cached."""
    cache = ct.PredictionCache(tmp_path, "key0000000")
    cache.save("t", {"count": 0, "points": []})
    assert not list(cache.dir.glob("*.tmp")), "temp file must be renamed away"


def test_cache_tolerates_corrupt_json(tmp_path) -> None:
    cache = ct.PredictionCache(tmp_path, "key0000000")
    cache.path_for("broken").write_text("{not json")
    assert cache.load("broken") is None
    assert cache.load_all() == {}


def test_color_scale_distinguishes_uncached_from_empty() -> None:
    pytest.importorskip("nicegui")
    from webapp.app.tabs import crown_decline as tab  # noqa: PLC0415

    assert tab._color_for(None) != tab._color_for(0), (
        "'not predicted yet' and 'predicted, found nothing' must look "
        "different on the map"
    )
    assert tab._color_for(100) != tab._color_for(0)


def test_current_params_matches_detector_signature() -> None:
    """The tab's param dict is splatted straight into predict_tile."""
    pytest.importorskip("nicegui")
    import inspect

    from webapp.app.detectors import crown_decline as det  # noqa: PLC0415
    from webapp.app.tabs import crown_decline as tab  # noqa: PLC0415

    params = tab._current_params(0.2, 9, 120, 512, 4, 0.0)
    accepted = set(
        inspect.signature(det.CrownDeclineDetector.predict_tile).parameters
    )
    assert set(params) <= accepted, (
        f"tab passes {set(params) - accepted} which predict_tile rejects"
    )


def test_detector_module_imports() -> None:
    pytest.importorskip("torch")
    from webapp.app.detectors.crown_decline import CrownDeclineDetector

    assert hasattr(CrownDeclineDetector, "predict_tile")


def test_tile_info_bounds_use_all_four_corners(tmp_path) -> None:
    """A projected rectangle is tilted in WGS84; two corners understate it."""
    rasterio = pytest.importorskip("rasterio")
    import numpy as np
    from rasterio.transform import from_origin

    path = tmp_path / "c000_r000.tif"
    # 100 px at 0.2 m in EPSG:25833, far from the central meridian so the
    # grid convergence is visible.
    with rasterio.open(
        path, "w", driver="GTiff", height=100, width=100, count=3,
        dtype="uint8", crs="EPSG:25833",
        transform=from_origin(422849.0, 5762557.0, 0.2, 0.2),
    ) as ds:
        ds.write(np.zeros((3, 100, 100), dtype="uint8"))

    info = ct.read_tile_info(path)
    assert info.lon_min < info.lon_max and info.lat_min < info.lat_max

    from pyproj import Transformer

    tf = Transformer.from_crs(info.crs, "EPSG:4326", always_xy=True)
    for x, y in (
        (info.left, info.bottom), (info.right, info.bottom),
        (info.left, info.top), (info.right, info.top),
    ):
        lon, lat = tf.transform(x, y)
        assert info.lon_min <= lon <= info.lon_max
        assert info.lat_min <= lat <= info.lat_max


def test_list_tiles_memoises_and_survives_a_corrupt_index(tmp_path) -> None:
    rasterio = pytest.importorskip("rasterio")
    import numpy as np
    from rasterio.transform import from_origin

    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    for i in range(2):
        with rasterio.open(
            tiles_dir / f"c00{i}_r000.tif", "w", driver="GTiff", height=8,
            width=8, count=3, dtype="uint8", crs="EPSG:25833",
            transform=from_origin(422849.0 + i * 10, 5762557.0, 0.2, 0.2),
        ) as ds:
            ds.write(np.zeros((3, 8, 8), dtype="uint8"))

    first = ct.list_tiles(tmp_path)
    assert len(first) == 2
    assert (tmp_path / ".tile_index.json").is_file()

    (tmp_path / ".tile_index.json").write_text("{broken")
    again = ct.list_tiles(tmp_path)
    assert [t.name for t in again] == [t.name for t in first]
    # the rebuild should have replaced the corrupt index
    json.loads((tmp_path / ".tile_index.json").read_text())
