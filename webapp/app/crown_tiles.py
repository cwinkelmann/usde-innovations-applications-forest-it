"""Tile-block index, geo helpers and the prediction cache for crown decline.

The input is a *tile block* produced by MapTilesDownloader's ``pull_wms_grid.py``
(``<block>/tiles/c###_r###.tif`` + ``mosaic.vrt``), not an upload session: the
imagery is fixed on disk and shared by every user, so predictions are cached
per tile rather than per session.

Cache key
---------
A run is identified by the model weights (path + mtime + size) plus every
inference parameter that can change the output. Change the threshold and you
get a new cache namespace; re-open the tab with the same settings and the
previous results load instantly. The key is deliberately *not* the job id —
re-running the same configuration must hit the same cache.

Coordinates
-----------
Three systems are in play and mixing them up is the classic failure here:

* **tile pixels** ``(x, y)`` — column/row from the tile's top-left corner.
  This is what the detector returns and what the overlay PNG is drawn in.
* **projected** ``(easting, northing)`` in the tile's own CRS (EPSG:25833 for
  the Brandenburg DOP20cir block).
* **WGS84** ``(lon, lat)`` — only for Leaflet, which speaks nothing else.

Every cached point carries all three so the panel never has to re-open a
GeoTIFF just to place a marker.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


# Rendered overlay previews are capped at this edge length (px). A 4096 tile
# downsamples to 1024, which is plenty for a browser modal and keeps each PNG
# well under a megabyte.
PREVIEW_MAX_EDGE = 1024


@dataclass(frozen=True)
class TileInfo:
    """One GeoTIFF of the block, with just enough geometry to map it."""

    name: str            # file stem, e.g. "c003_r012"
    path: str
    width: int
    height: int
    crs: str             # e.g. "EPSG:25833"
    # Projected bounds in the tile's own CRS.
    left: float
    bottom: float
    right: float
    top: float
    # WGS84 bounds for Leaflet: (south, west), (north, east).
    lat_min: float
    lon_min: float
    lat_max: float
    lon_max: float

    @property
    def bounds_latlon(self) -> list[list[float]]:
        return [[self.lat_min, self.lon_min], [self.lat_max, self.lon_max]]

    @property
    def center_latlon(self) -> list[float]:
        return [(self.lat_min + self.lat_max) / 2, (self.lon_min + self.lon_max) / 2]


def _transformer(src_crs: str):
    from pyproj import Transformer

    return Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)


def read_tile_info(path: Path) -> TileInfo:
    """Open a GeoTIFF header-only and derive its footprint in both CRSs."""
    import rasterio

    with rasterio.open(path) as ds:
        left, bottom, right, top = ds.bounds
        crs = str(ds.crs)
        width, height = ds.width, ds.height

    # All four corners, not two: a rectangle in a projected CRS is not
    # axis-aligned in WGS84 (grid convergence tilts it), so transforming only
    # the SW and NE corners understates the envelope — by ~13 m on these
    # 819 m tiles, enough to put real detections outside their own tile box.
    tf = _transformer(crs)
    lons, lats = zip(*(
        tf.transform(x, y)
        for x, y in ((left, bottom), (right, bottom), (left, top), (right, top))
    ))
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)
    return TileInfo(
        name=path.stem,
        path=str(path),
        width=width,
        height=height,
        crs=crs,
        left=left,
        bottom=bottom,
        right=right,
        top=top,
        lat_min=min(lat_min, lat_max),
        lon_min=min(lon_min, lon_max),
        lat_max=max(lat_min, lat_max),
        lon_max=max(lon_min, lon_max),
    )


def list_tiles(block_dir: Path) -> list[TileInfo]:
    """Index every GeoTIFF in ``<block_dir>/tiles`` (or in ``block_dir`` itself).

    Results are memoised to ``<block_dir>/.tile_index.json`` because opening
    240 GeoTIFF headers over a network mount takes a few seconds and the block
    never changes once pulled. Delete that file to force a re-scan.
    """
    block_dir = Path(block_dir)
    tiles_dir = block_dir / "tiles"
    if not tiles_dir.is_dir():
        tiles_dir = block_dir
    if not tiles_dir.is_dir():
        return []

    index_file = block_dir / ".tile_index.json"
    paths = sorted(
        p for p in tiles_dir.iterdir()
        if p.suffix.lower() in {".tif", ".tiff"} and not p.name.startswith(".")
    )
    if index_file.is_file():
        try:
            cached = json.loads(index_file.read_text())
            if cached.get("n") == len(paths):
                return [TileInfo(**rec) for rec in cached["tiles"]]
        except Exception:  # noqa: BLE001 — a stale/corrupt index just gets rebuilt
            pass

    tiles = [read_tile_info(p) for p in paths]
    try:
        index_file.write_text(
            json.dumps({"n": len(tiles), "tiles": [asdict(t) for t in tiles]})
        )
    except OSError:
        pass  # read-only mount: just don't memoise
    return tiles


def block_bounds(tiles: Iterable[TileInfo]) -> tuple[list[float], int]:
    """Centre + a Leaflet zoom level that fits the whole block."""
    tiles = list(tiles)
    if not tiles:
        return [52.0, 14.0], 10
    lat = (min(t.lat_min for t in tiles) + max(t.lat_max for t in tiles)) / 2
    lon = (min(t.lon_min for t in tiles) + max(t.lon_max for t in tiles)) / 2
    span = max(
        max(t.lat_max for t in tiles) - min(t.lat_min for t in tiles),
        max(t.lon_max for t in tiles) - min(t.lon_min for t in tiles),
    )
    # 360 degrees at zoom 0, halving each level; -1 so the block isn't flush
    # against the viewport edge.
    zoom = 11 if span <= 0 else max(3, min(16, int(math.log2(360.0 / span)) - 1))
    return [lat, lon], zoom


# ── cache ────────────────────────────────────────────────────────────────────

def params_key(weights: Path, params: dict[str, Any]) -> str:
    """Stable short hash over the weights identity and the inference params."""
    weights = Path(weights)
    try:
        st = weights.stat()
        ident = f"{weights.name}:{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        ident = weights.name
    payload = json.dumps({"weights": ident, **params}, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


class PredictionCache:
    """One directory per (weights, params) combination, one JSON per tile."""

    def __init__(self, root: Path, key: str, params: dict[str, Any] | None = None):
        self.dir = Path(root) / key
        self.key = key
        self.dir.mkdir(parents=True, exist_ok=True)
        meta = self.dir / "params.json"
        if params is not None and not meta.is_file():
            try:
                meta.write_text(json.dumps(params, indent=2, default=str))
            except OSError:
                pass

    def path_for(self, tile_name: str) -> Path:
        return self.dir / f"{tile_name}.json"

    def preview_for(self, tile_name: str) -> Path:
        # JPEG, not PNG: a 1024 px aerial crop is ~2 MB as PNG and ~150 KB
        # as JPEG, and at 240 tiles that is the difference between 0.5 GB
        # and 35 MB of cache for images nobody pixel-peeps.
        return self.dir / f"{tile_name}.jpg"

    def has(self, tile_name: str) -> bool:
        return self.path_for(tile_name).is_file()

    def load(self, tile_name: str) -> dict | None:
        p = self.path_for(tile_name)
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def save(self, tile_name: str, record: dict) -> None:
        # Write-then-rename so a crashed job never leaves a half-written JSON
        # that would be cached as if it were a complete result.
        p = self.path_for(tile_name)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record))
        tmp.replace(p)

    def cached_names(self) -> set[str]:
        return {p.stem for p in self.dir.glob("*.json") if p.stem != "params"}

    def load_all(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for p in self.dir.glob("*.json"):
            if p.stem == "params":
                continue
            try:
                out[p.stem] = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
        return out


# ── preview rendering ────────────────────────────────────────────────────────

def render_preview(
    tile: TileInfo,
    points: list[dict],
    out_path: Path,
    max_edge: int = PREVIEW_MAX_EDGE,
    min_dscore: float = 0.0,
) -> Path:
    """Write a downsampled false-colour JPEG of the tile with points marked.

    The block is CIR: band 1 is NIR, 2 is Red, 3 is Green (GDAL mislabels them
    as RGB). Mapping them straight to R,G,B gives the familiar false-colour
    look where healthy canopy is bright red — which is exactly the contrast a
    reviewer needs to judge a crown-decline detection, so we keep it.
    """
    import numpy as np
    import rasterio
    from PIL import Image, ImageDraw

    with rasterio.open(tile.path) as ds:
        scale = max(1, int(math.ceil(max(ds.width, ds.height) / max_edge)))
        out_w, out_h = ds.width // scale, ds.height // scale
        arr = ds.read(
            indexes=[1, 2, 3][: ds.count],
            out_shape=(min(3, ds.count), out_h, out_w),
            resampling=rasterio.enums.Resampling.average,
        )

    if arr.dtype != np.uint8:
        finite = arr[np.isfinite(arr)]
        hi = float(finite.max()) if finite.size else 1.0
        arr = (arr / hi * 255.0).clip(0, 255).astype(np.uint8) if hi > 0 else \
            arr.astype(np.uint8)
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)

    img = Image.fromarray(np.transpose(arr[:3], (1, 2, 0)), mode="RGB")
    draw = ImageDraw.Draw(img)
    r = 7
    for pt in points:
        if pt.get("dscore", 1.0) < min_dscore:
            continue
        cx, cy = pt["x"] / scale, pt["y"] / scale
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 0), width=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=85, optimize=True)
    return out_path
