#!/usr/bin/env python3
"""Batch-run the crown-decline detector over a whole tile block.

The webapp tab can do this through the job worker, but a 240-tile block takes
17 minutes on a 4080 and ~11 hours on a laptop's MPS, so in practice the block
is processed once on the GPU host and the resulting cache is copied to wherever
the webapp runs. This script is that headless path.

It writes into the *same* cache the tab reads, so the panel picks the results
up with no import step::

    python scripts/run_crown_block.py --device cuda

Resumable: tiles already in the cache namespace are skipped, so the job can be
killed and restarted. Each tile is committed to disk the moment it finishes.

Running it on a second machine
------------------------------
``crown_tiles.params_key`` identifies the weights by *filename, size and
mtime* rather than by content hash — a 420 MB sha1 on every page load would be
far too slow. The consequence is that the same checkpoint under a different
name or mtime yields a different cache key, i.e. a second cache directory the
panel never looks in. To keep one namespace across hosts, give the file the
same name and stamp it with the same mtime::

    cp best_model.pth schleppzig_convnextv2_1class_ep30.pth
    touch -d @<mtime-on-the-webapp-host> schleppzig_convnextv2_1class_ep30.pth

Verify with md5sum that they really are the same weights and not just the same
stat block.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from webapp.app import crown_tiles as ct  # noqa: E402
from webapp.app import config as cfg  # noqa: E402


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--block", type=Path, default=cfg.CROWN_TILES_DIR,
                   help="tile block directory (expects <block>/tiles/*.tif)")
    p.add_argument("--cache", type=Path, default=cfg.CROWN_CACHE_DIR,
                   help="cache root; a per-params subdirectory is created in it")
    p.add_argument("--weights", type=Path, default=cfg.CROWN_DECLINE_WEIGHTS)
    p.add_argument("--device", default=None, help="cuda|mps|cpu (default: auto)")
    p.add_argument("--limit", type=int, default=0,
                   help="stop after N tiles (0 = all); for a smoke test")
    p.add_argument("--no-preview", action="store_true",
                   help="skip the JPEG previews the inspector shows")
    # Every knob below feeds the cache key: change one and you get a fresh
    # namespace rather than a silent mix of old and new results.
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--overlap", type=int, default=120)
    p.add_argument("--kernel-size", type=int, default=9)
    p.add_argument("--adapt-ts", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--score-threshold", type=float, default=0.0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    params = {
        "patch_size": a.patch_size,
        "overlap": a.overlap,
        "kernel_size": a.kernel_size,
        "adapt_ts": a.adapt_ts,
        "batch_size": a.batch_size,
        "score_threshold": a.score_threshold,
    }

    tiles = ct.list_tiles(a.block)
    if not tiles:
        print(f"[error] no GeoTIFFs under {a.block}", file=sys.stderr)
        return 2

    key = ct.params_key(a.weights, params)
    cache = ct.PredictionCache(a.cache, key, params)
    done = cache.cached_names()
    todo = [t for t in tiles if t.name not in done]
    if a.limit:
        todo = todo[: a.limit]

    print(f"[init] tiles={len(tiles)} cached={len(done)} todo={len(todo)}", flush=True)
    print(f"[init] key={key} dir={cache.dir}", flush=True)
    if not todo:
        print("[done] nothing to do", flush=True)
        return 0

    device = a.device or pick_device()
    print(f"[init] device={device}", flush=True)

    # Imported here, not at module scope, so --help works without torch.
    from webapp.app.detectors.crown_decline import CrownDeclineDetector  # noqa: PLC0415

    t0 = time.time()
    det = CrownDeclineDetector(
        a.weights,
        device=device,
        down_ratio=cfg.CROWN_DOWN_RATIO,
        backbone_size=cfg.CROWN_BACKBONE_SIZE,
        num_classes=cfg.CROWN_NUM_CLASSES,
    )
    print(f"[init] model loaded in {time.time() - t0:.0f}s", flush=True)

    start = time.time()
    total = 0
    failed: list[str] = []
    for i, tile in enumerate(todo, start=1):
        ts = time.time()
        try:
            rec = det.predict_tile(Path(tile.path), **params)
        except Exception:  # noqa: BLE001 — one bad tile must not lose the run
            failed.append(tile.name)
            print(f"[fail] {tile.name}\n{traceback.format_exc()}", flush=True)
            continue
        cache.save(tile.name, rec)
        total += rec["count"]
        if not a.no_preview:
            try:
                ct.render_preview(tile, rec["points"], cache.preview_for(tile.name))
            except Exception as e:  # noqa: BLE001 — a preview is cosmetic
                print(f"[warn] preview {tile.name}: {e}", flush=True)
        elapsed = time.time() - start
        eta = elapsed / i * (len(todo) - i)
        print(
            f"[{i:3d}/{len(todo)}] {tile.name}: {rec['count']:4d} pts "
            f"in {time.time() - ts:5.1f}s  total={total}  eta={eta / 3600:.2f}h",
            flush=True,
        )

    print(
        f"[done] {len(todo) - len(failed)} tiles, {total} detections, "
        f"{(time.time() - start) / 3600:.2f}h, failed={failed}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
