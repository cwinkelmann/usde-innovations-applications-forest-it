"""Crown-decline tab: run the HerdNet crown detector over a GeoTIFF tile block.

Differs from the camera-trap tabs in two ways that drive the whole design:

* **The imagery is fixed.** It is a tile block pulled by MapTilesDownloader,
  shared by every user, so results are cached per (weights, parameters, tile)
  instead of per upload session. Re-opening the tab with the same settings
  shows previous results immediately and the Run button only has to process
  the tiles that are still missing.
* **The results are geographic.** Every detection carries lon/lat, so the
  overview is a Leaflet map of the block with one rectangle per tile, shaded
  by how many declining crowns were found.

Individual detections are *not* all drawn at once: 240 tiles times ~100
points would be 24k Leaflet layers and an unusable page. Tile rectangles
carry the counts; markers are drawn for a selected tile, or for all tiles
once the total is small enough to be safe.
"""
from __future__ import annotations

import json
from pathlib import Path

from nicegui import ui

from ..config import (
    CROWN_CACHE_DIR,
    CROWN_DECLINE_WEIGHTS,
    CROWN_TILES_DIR,
    resolve_batch_size,
    resolve_device,
)
from ..crown_tiles import PredictionCache, block_bounds, list_tiles, params_key

# Above this many individual detections we stop drawing markers for the whole
# block and fall back to tile rectangles only. Chosen so the page stays
# responsive on a laptop; the per-tile view is unaffected.
MAX_MARKERS = 1500

# Tile fill colours by detection count. Grey means "not predicted yet", which
# is deliberately distinct from the pale colour for "predicted, found none".
_COLOR_UNCACHED = "#9e9e9e"
_COLOR_SCALE = [
    (0, "#2e7d32"),     # none found
    (1, "#9ccc65"),
    (10, "#ffd54f"),
    (30, "#fb8c00"),
    (60, "#e53935"),
]


def _color_for(count: int | None) -> str:
    if count is None:
        return _COLOR_UNCACHED
    colour = _COLOR_SCALE[0][1]
    for threshold, c in _COLOR_SCALE:
        if count >= threshold:
            colour = c
    return colour


def _current_params(adapt_ts: float, kernel: int, overlap: int,
                    patch: int, batch: int, score_ts: float) -> dict:
    """Exactly the kwargs ``CrownDeclineDetector.predict_tile`` accepts.

    This dict is both the job payload and the cache key input, so the two can
    never drift apart: any knob that changes a prediction also changes where
    that prediction is stored.
    """
    return {
        "patch_size": patch,
        "overlap": overlap,
        "kernel_size": kernel,
        "adapt_ts": adapt_ts,
        "batch_size": batch,
        "score_threshold": score_ts,
    }


def render(jm) -> None:  # noqa: C901 — one linear UI builder, kept together
    tiles = list_tiles(CROWN_TILES_DIR)
    by_name = {t.name: t for t in tiles}
    device = resolve_device()

    state: dict = {"job_id": None, "cache": None, "key": None, "records": {}}

    with ui.column().classes("w-full gap-4"):
        ui.label("Crown decline (HerdNet)").classes("text-xl font-semibold")
        ui.label(
            "Detects declining oak crowns in CIR orthophoto tiles "
            "(DOP20cir, 0.2 m, bands NIR/Red/Green). Predictions are cached "
            "per tile and per parameter set — running again only processes "
            "tiles that are missing."
        ).classes("text-sm text-gray-700")

        if not tiles:
            ui.label(
                f"No GeoTIFF tiles found under {CROWN_TILES_DIR}. "
                "Set CROWN_TILES_DIR to a MapTilesDownloader block "
                "(the directory containing tiles/)."
            ).classes("text-red-700 text-sm")
            return
        if not Path(CROWN_DECLINE_WEIGHTS).is_file():
            ui.label(
                f"Model weights missing: {CROWN_DECLINE_WEIGHTS}. "
                "Copy the trained checkpoint there or set "
                "CROWN_DECLINE_WEIGHTS."
            ).classes("text-red-700 text-sm")

        ui.label(
            f"Block: {CROWN_TILES_DIR.name} · {len(tiles)} tiles · "
            f"{tiles[0].width}×{tiles[0].height} px · {tiles[0].crs} · "
            f"device {device}"
        ).classes("text-xs text-gray-500")

        # ── controls ────────────────────────────────────────────────────
        with ui.row().classes("items-end gap-4 flex-wrap"):
            adapt_in = ui.number(
                "adapt_ts", value=0.20, min=0.01, max=0.95, step=0.05,
                format="%.2f",
            ).classes("w-28").tooltip(
                "Local-maxima threshold, relative to the peak of the tile's "
                "own heatmap. Training validated at 0.20 on 512 px crops; on "
                "a whole 4096 px tile the peak is taken over a much larger "
                "area, so this is not strictly the same operating point."
            )
            kernel_in = ui.number(
                "LMDS kernel", value=9, min=3, max=31, step=2,
            ).classes("w-28").tooltip(
                "Suppression window in heatmap cells. 9 at down_ratio 4 "
                "= 36 px = ~7.2 m, the spacing the model was tuned with."
            )
            overlap_in = ui.number(
                "overlap px", value=120, min=0, max=256, step=8,
            ).classes("w-28").tooltip(
                "Overlap between the 512 px patches the tile is cut into. "
                "Hann-weighted averaging in the overlap avoids seam artefacts."
            )
            batch_in = ui.number(
                "batch", value=resolve_batch_size(device), min=1, max=64, step=1,
            ).classes("w-24")
            score_in = ui.number(
                "min score", value=0.00, min=0.0, max=1.0, step=0.05,
                format="%.2f",
            ).classes("w-28").tooltip(
                "Drop detections whose heatmap score is below this before "
                "caching. Leave at 0 to keep everything and filter later."
            )

        with ui.row().classes("items-center gap-3 flex-wrap"):
            run_missing_btn = ui.button("Run missing tiles")
            run_all_btn = ui.button("Re-run all tiles").props("outline")
            refresh_btn = ui.button("Refresh").props("flat")
            markers_toggle = ui.switch("Show individual detections", value=False)
        status_label = ui.label("").classes("text-sm text-gray-800")
        progress = ui.linear_progress(value=0.0, show_value=False).classes("w-full")
        progress.set_visibility(False)

        map_container = ui.column().classes("w-full")
        tile_select = ui.select(
            options=["— none cached —"], label="Inspect tile", with_input=True,
        ).classes("w-72")
        detail_container = ui.column().classes("w-full")

    # ── cache plumbing ──────────────────────────────────────────────────

    def _reload_cache() -> dict:
        params = _current_params(
            float(adapt_in.value or 0.2), int(kernel_in.value or 9),
            int(overlap_in.value or 120), 512, int(batch_in.value or 1),
            float(score_in.value or 0.0),
        )
        key = params_key(CROWN_DECLINE_WEIGHTS, params)
        state["key"] = key
        state["cache"] = PredictionCache(CROWN_CACHE_DIR, key, params)
        state["records"] = state["cache"].load_all()
        return params

    def _counts() -> dict[str, int]:
        return {name: rec.get("count", 0) for name, rec in state["records"].items()}

    def _missing() -> list:
        cached = set(state["records"])
        return [t for t in tiles if t.name not in cached]

    # ── map ─────────────────────────────────────────────────────────────

    def _draw_map() -> None:
        map_container.clear()
        counts = _counts()
        total_det = sum(counts.values())
        with map_container:
            center, zoom = block_bounds(tiles)
            lmap = ui.leaflet(center=center, zoom=zoom).classes("w-full h-[560px]")

            popups: dict[str, str] = {}
            for t in tiles:
                count = counts.get(t.name)
                colour = _color_for(count)
                layer = lmap.generic_layer(
                    name="rectangle",
                    args=[
                        t.bounds_latlon,
                        {
                            "color": colour,
                            "weight": 1,
                            "fillColor": colour,
                            "fillOpacity": 0.25 if count is None else 0.45,
                        },
                    ],
                )
                popups[layer.id] = (
                    f"<b>{t.name}</b><br>"
                    + (
                        "not predicted yet"
                        if count is None
                        else f"{count} declining crown(s)"
                    )
                )

            # Individual detections, only when explicitly asked for and only
            # while the total stays inside what Leaflet renders comfortably.
            if markers_toggle.value and 0 < total_det <= MAX_MARKERS:
                for name, rec in state["records"].items():
                    for pt in rec.get("points", []):
                        lmap.generic_layer(
                            name="circleMarker",
                            args=[
                                [pt["lat"], pt["lon"]],
                                {
                                    "radius": 3,
                                    "color": "#ffeb3b",
                                    "fillColor": "#f57f17",
                                    "fillOpacity": 0.8,
                                    "weight": 1,
                                },
                            ],
                        )

            if markers_toggle.value and total_det > MAX_MARKERS:
                # Silently drawing nothing would read as "no detections".
                ui.label(
                    f"{total_det} detections is above the {MAX_MARKERS} "
                    "marker limit — showing tile totals only. Pick a tile "
                    "below to see its individual detections."
                ).classes("text-sm text-amber-800")

            # Popups are bound client-side for the same reason map_view.py
            # does it: Layer.run_method('bindPopup') silently no-ops when the
            # map mounts late (it only mounts when its tab first becomes
            # visible), so we retry from the Leaflet side instead.
            if popups:
                _bind_popups(lmap.id, popups)

    def _bind_popups(map_id: int, popups: dict[str, str]) -> None:
        js = (
            "(function() {"
            "  let attempts = 0;"
            "  function tryBind() {"
            f"    const el = window.getElement({map_id});"
            "    if (el && el.map) {"
            f"      const m = {json.dumps(popups)};"
            "      let bound = 0;"
            "      el.map.eachLayer(function(layer) {"
            "        if (layer && layer.id && m[layer.id]) {"
            "          layer.bindPopup(m[layer.id]); bound++;"
            "        }"
            "      });"
            f"      if (bound >= {len(popups)}) return;"
            "    }"
            "    if (attempts++ < 40) setTimeout(tryBind, 250);"
            "  }"
            "  tryBind();"
            "})()"
        )
        ui.run_javascript(js)

    # ── tile detail ─────────────────────────────────────────────────────

    def _show_tile(tile_name: str | None) -> None:
        detail_container.clear()
        if not tile_name:
            return
        rec = state["records"].get(tile_name)
        tile = by_name.get(tile_name)
        with detail_container:
            if rec is None:
                ui.label(
                    f"{tile_name} has no cached prediction for these settings. "
                    "Run the missing tiles first."
                ).classes("text-sm text-gray-700")
                return
            ui.label(
                f"{tile_name} — {rec['count']} detection(s) · "
                f"{rec['width']}×{rec['height']} px · {rec['crs']}"
            ).classes("text-base font-medium")
            preview = state["cache"].preview_for(tile_name)
            if preview.is_file():
                # Served by the /crown media mount (see main.py). The cache
                # key is part of the path, so switching parameters cannot
                # show a stale overlay from a previous run.
                ui.image(f"/crown/{state['key']}/{tile_name}.jpg").classes(
                    "w-full max-w-[900px] border"
                )
            else:
                ui.label(
                    "No overlay image for this tile (preview rendering was "
                    "skipped or failed); the detections below are unaffected."
                ).classes("text-sm text-gray-600")
            if tile is not None:
                ui.label(
                    f"Centre {tile.center_latlon[0]:.5f}, "
                    f"{tile.center_latlon[1]:.5f}"
                ).classes("text-xs text-gray-500")
            rows = [
                {
                    "x": p["x"], "y": p["y"],
                    "lon": p["lon"], "lat": p["lat"],
                    "score": p["score"], "dscore": p["dscore"],
                }
                for p in sorted(
                    rec.get("points", []),
                    key=lambda p: p["dscore"],
                    reverse=True,
                )
            ]
            if rows:
                ui.table(
                    columns=[
                        {"name": c, "label": lbl, "field": c, "sortable": True}
                        for c, lbl in [
                            ("x", "x px"), ("y", "y px"),
                            ("lon", "lon"), ("lat", "lat"),
                            ("dscore", "heatmap"), ("score", "class"),
                        ]
                    ],
                    rows=rows,
                    pagination=15,
                ).classes("w-full max-w-[900px]")

    # ── actions ─────────────────────────────────────────────────────────

    def _set_status_text() -> None:
        counts = _counts()
        cached = len(counts)
        status_label.set_text(
            f"{cached} / {len(tiles)} tiles predicted · "
            f"{sum(counts.values())} declining crown(s) · cache {state['key']}"
        )

    def _refresh() -> None:
        _reload_cache()
        _set_status_text()
        _draw_map()
        tile_select.options = sorted(state["records"]) or ["— none cached —"]
        tile_select.update()

    def _submit(tile_list) -> None:
        params = _reload_cache()
        if not tile_list:
            ui.notify("Nothing to do — every tile is already cached.",
                      type="positive")
            return
        job_id = jm.submit(
            "crown_decline",
            session="",  # geospatial mode: no upload session involved
            tiles=[t.path for t in tile_list],
            params=params,
            cache_key=state["key"],
            preview=True,
        )
        state["job_id"] = job_id
        progress.set_visibility(True)
        progress.set_value(0.0)
        status_label.set_text(
            f"Queued {len(tile_list)} tile(s) on {device} …"
        )

    def _poll() -> None:
        job_id = state.get("job_id")
        if not job_id:
            return
        st = jm.get_status(job_id)
        if st.get("state") == "running":
            progress.set_value(float(st.get("progress", 0.0)))
            status_label.set_text(
                f"{st.get('stage', 'running')}: "
                f"{st.get('processed', 0)}/{st.get('total', '?')} tiles · "
                f"{st.get('detections', 0)} detection(s) so far"
            )
        elif st.get("state") == "done":
            state["job_id"] = None
            progress.set_value(1.0)
            progress.set_visibility(False)
            _refresh()
            ui.notify("Crown-decline run finished.", type="positive")
        elif st.get("state") == "error":
            state["job_id"] = None
            progress.set_visibility(False)
            status_label.set_text(f"Failed: {st.get('error')}")
            ui.notify(f"Crown-decline run failed: {st.get('error')}",
                      type="negative", timeout=10000)

    # Generic .on("click") rather than .on_click(): it is the Element API
    # that every NiceGUI version in use here exposes.
    run_missing_btn.on("click", lambda: _submit(_missing()))
    run_all_btn.on("click", lambda: _submit(tiles))
    refresh_btn.on("click", lambda: _refresh())
    markers_toggle.on_value_change(lambda _: _draw_map())
    tile_select.on_value_change(lambda e: _show_tile(
        e.value if e.value in state["records"] else None
    ))

    _refresh()
    ui.timer(1.5, _poll)
