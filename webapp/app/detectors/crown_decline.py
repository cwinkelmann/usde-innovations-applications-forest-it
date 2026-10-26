"""Crown-decline point detector (HerdNet / animaloc) for CIR orthophoto tiles.

Wraps the ``CamouflageHerdNetConvNeXtV2`` model trained on the Schleppzig
(Spreewald) DOP20cir block to find declining oak crowns. Unlike the other
detectors here this one is *geospatial*: it runs over whole 4096 px GeoTIFF
tiles rather than camera-trap photographs, so it stitches overlapping 512 px
patches and returns points in tile-pixel, projected and WGS84 coordinates.

Dependency note
---------------
``animaloc`` is the HerdNet fork (https://github.com/cwinkelmann/HerdNetPlus),
not a PyPI package and not a dependency of the course environment. It is
imported lazily so the webapp starts fine without it; only this tab fails,
with an actionable message. Install with::

    pip install -e /path/to/HerdNet

or point ``HERDNET_REPO`` at a checkout and it will be added to ``sys.path``.

Inference contract (verified against the trained checkpoint, not inferred
from the source — the code is ambiguous here):

* the bare model returns ``(heatmap[B,1,H/4,W/4], clsmap[B,C,H/32,W/32])``
* ``HerdNetStitcher`` over a large image returns ONE tensor
  ``[1, 1+C, H/4, W/4]`` — heatmap in channel 0, class logits after it
* ``HerdNetLMDS`` must then run with ``up=False``; with ``up=True`` it tries
  to rescale an already-aligned class map and raises a size mismatch
* LMDS locations are ``(row, col)`` at heatmap resolution — multiply by
  ``down_ratio`` for full-resolution tile pixels
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable

# albumentations' Normalize defaults, which is what the training pipeline
# applied. Hard-coded rather than imported so inference does not depend on
# albumentations (and cannot drift if its defaults ever change).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class AnimalocMissing(RuntimeError):
    """Raised when the HerdNet fork is not importable."""


def _import_animaloc():
    repo = os.environ.get("HERDNET_REPO")
    if repo and repo not in sys.path and Path(repo).is_dir():
        sys.path.insert(0, repo)
    try:
        from animaloc.eval.lmds import HerdNetLMDS
        from animaloc.eval.stitchers import HerdNetStitcher
        from animaloc.models import CamouflageHerdNetConvNeXtV2
        from animaloc.models.utils import LossWrapper
    except ImportError as e:  # noqa: PERF203
        raise AnimalocMissing(
            "The crown-decline detector needs the HerdNet fork (`animaloc`), "
            "which is not installed in this environment. Install it with "
            "`pip install -e /path/to/HerdNet`, or set HERDNET_REPO to a "
            f"checkout directory. Original error: {e}"
        ) from e
    return CamouflageHerdNetConvNeXtV2, HerdNetStitcher, HerdNetLMDS, LossWrapper


class CrownDeclineDetector:
    """Loads the checkpoint once; ``predict_tile`` is then cheap to call."""

    def __init__(
        self,
        weights: Path | str,
        device: str = "cpu",
        down_ratio: int = 4,
        backbone_size: str = "tiny",
        num_classes: int = 2,
    ) -> None:
        import torch

        model_cls, self._stitcher_cls, self._lmds_cls, loss_wrapper_cls = (
            _import_animaloc()
        )

        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(
                f"Crown-decline weights not found at {weights}. Copy the "
                "trained checkpoint there or set CROWN_DECLINE_WEIGHTS."
            )

        self.device = device
        self.down_ratio = down_ratio
        self.num_classes = num_classes

        # pretrained=False: every weight comes from the checkpoint, so there
        # is no reason to pull timm's ImageNet weights over the network first.
        model = model_cls(
            num_classes=num_classes,
            down_ratio=down_ratio,
            backbone_size=backbone_size,
            pretrained=False,
        )
        model.reshape_classes(num_classes)

        raw = torch.load(str(weights), map_location="cpu", weights_only=False)
        state = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
        # Training wrapped the net in animaloc's LossWrapper, so every key is
        # prefixed "model.". Strip it; keep the raw dict if it isn't there.
        stripped = {
            k[len("model."):]: v for k, v in state.items() if k.startswith("model.")
        }
        to_load = stripped or state
        missing, unexpected = model.load_state_dict(to_load, strict=False)
        if missing:
            raise RuntimeError(
                f"Checkpoint {weights.name} is missing {len(missing)} weights "
                f"(first: {missing[:3]}). Wrong architecture or down_ratio?"
            )
        self.loaded_keys = len(to_load)
        self.unexpected_keys = len(unexpected)

        model.eval()
        model.to(torch.device(device))
        self.net = model
        # HerdNetStitcher calls `self.model(patch)[0]` and then indexes
        # `[1]` on the result, i.e. it expects a module whose output is
        # `(outputs, extra)` — animaloc's LossWrapper, which is exactly what
        # training and validation ran through. An empty loss list keeps the
        # wrapper's forward on its prediction-only path.
        self.model = loss_wrapper_cls(model, [])
        self.model.eval()
        self.weights = weights

    def _release_memory(self) -> None:
        """Return cached-but-unused allocator blocks to the driver.

        Mirrors what ``worker._release_accelerator_memory`` does at job end,
        but per tile — a single crown-decline job can be hundreds of tiles
        long, so waiting until the end is far too late.
        """
        import gc

        import torch

        gc.collect()
        try:
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            elif self.device == "mps" and getattr(torch, "mps", None):
                empty = getattr(torch.mps, "empty_cache", None)
                if callable(empty):
                    empty()
        except Exception:  # noqa: BLE001 — never fail a tile over cleanup
            pass

    # ── inference ────────────────────────────────────────────────────────

    def _read_tile(self, path: Path):
        """Return the tile as a normalised ``(3, H, W)`` float tensor.

        Band order is preserved as 1,2,3 — on this block that is NIR, Red,
        Green. Training read the crops through PIL's ``convert("RGB")``,
        which also takes the first three bands in file order, so the channel
        semantics the model learned are reproduced exactly.
        """
        import numpy as np
        import rasterio
        import torch

        with rasterio.open(path) as ds:
            idx = [1, 2, 3][: ds.count]
            arr = ds.read(indexes=idx).astype("float32")
            transform = ds.transform
            crs = str(ds.crs)
            h, w = ds.height, ds.width

        if arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)
        arr = arr[:3] / 255.0
        for c in range(3):
            arr[c] = (arr[c] - _IMAGENET_MEAN[c]) / _IMAGENET_STD[c]
        return torch.from_numpy(arr), transform, crs, h, w

    def predict_tile(
        self,
        tile_path: Path | str,
        patch_size: int = 512,
        overlap: int = 120,
        kernel_size: int = 9,
        adapt_ts: float = 0.2,
        batch_size: int = 4,
        score_threshold: float = 0.0,
    ) -> dict[str, Any]:
        """Detect declining crowns in one GeoTIFF tile.

        Returns a cache-ready record: tile metadata plus a ``points`` list,
        each with tile-pixel ``x``/``y``, projected ``easting``/``northing``,
        WGS84 ``lon``/``lat``, ``label``, class ``score`` and heatmap
        ``dscore``.
        """
        import torch
        from pyproj import Transformer

        tile_path = Path(tile_path)
        image, transform, crs, height, width = self._read_tile(tile_path)

        stitcher = self._stitcher_cls(
            model=self.model,
            size=(patch_size, patch_size),
            overlap=overlap,
            down_ratio=self.down_ratio,
            up=False,            # the stitcher already emits down_ratio space
            reduction="mean",    # Hann-weighted mean over overlaps
            batch_size=batch_size,
            device_name=self.device,
        )
        # up=False is mandatory: the stitched class map is already aligned to
        # the heatmap, and LMDS's upsampling path would rescale it again.
        lmds = self._lmds_cls(
            kernel_size=(kernel_size, kernel_size), adapt_ts=adapt_ts, up=False
        )

        with torch.no_grad():
            stitched = stitcher(image)
            heatmap, clsmap = stitched[:, :1, :, :], stitched[:, 1:, :, :]
            counts, locs, labels, scores, dscores = lmds([heatmap, clsmap])

        # A 4096 px tile is cut into ~121 patches, so one call allocates
        # several hundred MB of activations. Without an explicit release the
        # allocator holds them and each successive tile in a 240-tile job
        # gets slower (measured: tile 2 took several times tile 1 on MPS).
        del stitched, heatmap, clsmap, image, stitcher
        self._release_memory()

        to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        points: list[dict[str, Any]] = []
        for (row, col), label, score, dscore in zip(
            locs[0], labels[0], scores[0], dscores[0]
        ):
            if dscore < score_threshold:
                continue
            # LMDS works in down_ratio space; lift back to full resolution.
            x = (float(col) + 0.5) * self.down_ratio
            y = (float(row) + 0.5) * self.down_ratio
            easting, northing = transform * (x, y)
            lon, lat = to_wgs84.transform(easting, northing)
            points.append(
                {
                    "x": round(x, 1),
                    "y": round(y, 1),
                    "easting": round(easting, 2),
                    "northing": round(northing, 2),
                    "lon": round(lon, 7),
                    "lat": round(lat, 7),
                    "label": int(label),
                    "score": round(float(score), 4),
                    "dscore": round(float(dscore), 4),
                }
            )

        return {
            "tile": tile_path.stem,
            "path": str(tile_path),
            "crs": crs,
            "width": width,
            "height": height,
            "count": len(points),
            "lmds_counts": [int(c) for c in counts[0]],
            "params": {
                "patch_size": patch_size,
                "overlap": overlap,
                "kernel_size": kernel_size,
                "adapt_ts": adapt_ts,
                "score_threshold": score_threshold,
                "down_ratio": self.down_ratio,
            },
            "points": points,
        }

    def predict_tiles(
        self,
        tile_paths: list[Path],
        progress_cb: Callable[[int, int], None] | None = None,
        **kwargs: Any,
    ) -> dict[str, dict]:
        out: dict[str, dict] = {}
        total = len(tile_paths)
        for i, p in enumerate(tile_paths, start=1):
            out[Path(p).stem] = self.predict_tile(p, **kwargs)
            if progress_cb:
                progress_cb(i, total)
        return out
