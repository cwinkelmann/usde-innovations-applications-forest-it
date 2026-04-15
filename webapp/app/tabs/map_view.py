"""Tab — Map: plot geolocated camera traps with species-density overlays.

For every session that has GPS coords (set in the Admin tab) and SpeciesNet
predictions, a Leaflet ``circleMarker`` is rendered. The radius grows with
the count of images whose top-1 species matches the user's filter — so a
fox-only filter shows traps sized by their fox catches, ignoring other
species. Sqrt scaling is used so circle *area* tracks count linearly
(visually intuitive).
"""
from __future__ import annotations

import json
from html import escape
from math import sqrt

from nicegui import ui

from ..camera_meta import load as load_camera_meta
from ..effective_labels import species_counts as _species_counts
from ..sessions import list_sessions


def _gather_geo_sessions() -> list[dict]:
    """Walk all sessions, return records for those with both lat & lng set."""
    out: list[dict] = []
    for session in list_sessions():
        meta = load_camera_meta(session)
        lat = meta.get("latitude")
        lng = meta.get("longitude")
        if lat is None or lng is None:
            continue
        try:
            lat_f = float(lat)
            lng_f = float(lng)
        except (TypeError, ValueError):
            continue
        out.append(
            {
                "session": session,
                "lat": lat_f,
                "lng": lng_f,
                "name": meta.get("name") or session,
                "description": meta.get("description") or "",
                "species": _species_counts(session),
            }
        )
    return out


def _popup_html(rec: dict, matched: int, selected: set[str]) -> str:
    species = rec["species"]
    if species:
        # Highlight selected species in green; others muted.
        lines = "".join(
            (
                f"<li style='color:{'#16a34a' if sp in selected else '#888'};'>"
                f"{escape(sp)}: {n}</li>"
            )
            for sp, n in sorted(species.items(), key=lambda kv: -kv[1])
        )
        species_block = f"<ul style='margin:4px 0 0 18px;padding:0;'>{lines}</ul>"
    else:
        species_block = (
            "<i style='color:#666;'>No SpeciesNet predictions yet.</i>"
        )
    name = escape(rec["name"])
    session = escape(rec["session"])
    desc = escape(rec["description"])
    desc_block = (
        f"<div style='color:#444;margin-top:2px;'>{desc}</div>" if desc else ""
    )
    return (
        f"<div style='min-width:220px;font-family:system-ui,sans-serif;'>"
        f"<div style='font-weight:600;'>{name}</div>"
        f"<div style='color:#888;font-size:11px;'>session: {session}</div>"
        f"{desc_block}"
        f"<div style='margin-top:6px;font-size:12px;font-weight:500;'>"
        f"Filtered: {matched} image(s)</div>"
        f"<div style='margin-top:4px;font-size:12px;font-weight:500;'>"
        f"All species seen:</div>"
        f"{species_block}"
        f"</div>"
    )


def _center_zoom(geo: list[dict]) -> tuple[tuple[float, float], int]:
    """Pick map centre + zoom that fits all markers."""
    if not geo:
        return (51.5, 10.5), 5
    if len(geo) == 1:
        return (geo[0]["lat"], geo[0]["lng"]), 10
    lats = [g["lat"] for g in geo]
    lngs = [g["lng"] for g in geo]
    cx = (min(lats) + max(lats)) / 2
    cy = (min(lngs) + max(lngs)) / 2
    span = max(max(lats) - min(lats), max(lngs) - min(lngs))
    if span > 20:
        zoom = 3
    elif span > 5:
        zoom = 5
    elif span > 1:
        zoom = 7
    elif span > 0.1:
        zoom = 10
    else:
        zoom = 13
    return (cx, cy), zoom


# Visual scaling — see module docstring for the rationale.
_RADIUS_BASE_PX = 6      # smallest circle (count = 0)
_RADIUS_SCALE = 3        # multiplied by sqrt(count)
_RADIUS_MAX_PX = 60      # clamp so a runaway count doesn't paint half the map
_COLOR_ACTIVE = "#dc2626"
_COLOR_DIM = "#94a3b8"


def _radius_for(count: int) -> float:
    if count <= 0:
        return _RADIUS_BASE_PX
    return min(_RADIUS_MAX_PX, _RADIUS_BASE_PX + sqrt(count) * _RADIUS_SCALE)


def render() -> None:
    with ui.column().classes("w-full gap-4"):
        ui.label("Map").classes("text-xl font-semibold")
        ui.label(
            "Each circle marker is a camera trap with GPS coordinates. "
            "The radius scales with the number of images whose top-1 "
            "species is in the filter — so unticking a species shrinks "
            "the circles where only that species was caught. Click a "
            "marker for the full breakdown."
        ).classes("text-sm text-gray-600")

        # No caching of geo/all_species — disk read on every rebuild.
        # ~10 ms per session, and the species list always matches what's
        # on disk (no "Refresh data" gotcha when a new SpeciesNet run
        # drops a new species into a session).
        # ``sp_select`` is the Quasar QSelect — kept so None/All can push
        # a new value back to the client (otherwise the chip display
        # stays stale after a Python-driven update).
        # ``selected`` is None until first render, then a set.
        # ``prev_all`` lets us distinguish "newly appeared species"
        # (auto-include) from "user deselected" (preserve choice).
        state: dict = {
            "selected": None,
            "sp_select": None,
            "prev_all": set(),
        }

        controls_row = ui.row().classes("items-center gap-3 flex-wrap")
        status_label = ui.label("").classes("text-sm text-gray-700")
        map_container = ui.column().classes("w-full")

        def _rebuild() -> None:
            """Re-read disk, rebuild filter row, rebuild map."""
            geo = _gather_geo_sessions()
            all_species = sorted(
                {sp for g in geo for sp in g["species"]}
            )
            # First render: default to "all selected".
            # Subsequent rebuilds: preserve user's existing choices,
            # auto-include species that just appeared (delta vs prev_all),
            # drop species that no longer exist.
            existing = set(all_species)
            if state["selected"] is None:
                state["selected"] = set(existing)
            else:
                preserved = state["selected"] & existing
                newly_appeared = existing - state["prev_all"]
                state["selected"] = preserved | newly_appeared
            state["prev_all"] = set(existing)

            controls_row.clear()
            with controls_row:
                ui.label("Filter species:").classes("text-sm")

                def on_change(e) -> None:
                    state["selected"] = set(e.value or [])
                    _build_map_only(geo, all_species)

                if all_species:
                    state["sp_select"] = (
                        ui.select(
                            options=all_species,
                            value=sorted(state["selected"]),
                            multiple=True,
                            label=f"Species ({len(all_species)})",
                        )
                        .props("use-chips outlined dense options-dense")
                        .classes("min-w-[260px]")
                        .on_value_change(on_change)
                    )

                    def _set(sel: set[str]) -> None:
                        state["selected"] = sel
                        if state["sp_select"] is not None:
                            state["sp_select"].set_value(sorted(sel))
                        _build_map_only(geo, all_species)

                    ui.button(
                        "None", on_click=lambda: _set(set())
                    ).props("flat dense").tooltip(
                        "Hide all species — circles shrink to base size"
                    )
                    ui.button(
                        "All", on_click=lambda: _set(set(all_species))
                    ).props("flat dense").tooltip("Include every species")
                else:
                    ui.label(
                        "(no SpeciesNet predictions in any geo-tagged session)"
                    ).classes("text-sm text-gray-500")

                # Refresh re-runs the full rebuild including disk read.
                ui.button(
                    "Refresh data", icon="refresh", on_click=_rebuild
                ).props("flat")

            _build_map_only(geo, all_species)

        def _build_map_only(geo: list[dict], all_species: list[str]) -> None:
            """Render just the map, using already-loaded geo data. Called
            on filter changes so we don't re-read disk for every chip
            tweak."""
            map_container.clear()
            with map_container:
                if not geo:
                    ui.label(
                        "No GPS coordinates set yet — open the Admin tab "
                        "to add them."
                    ).classes("text-sm text-gray-500")
                    return

                selected = state["selected"] or set()
                trap_counts: list[int] = []
                for rec in geo:
                    matched = sum(
                        n for sp, n in rec["species"].items() if sp in selected
                    )
                    trap_counts.append(matched)
                total_filtered = sum(trap_counts)
                status_label.set_text(
                    f"{len(geo)} trap(s) on the map · "
                    f"{total_filtered} image(s) match the filter "
                    f"({len(selected)} of {len(all_species)} species selected)."
                )

                center, zoom = _center_zoom(geo)
                lmap = ui.leaflet(center=center, zoom=zoom).classes(
                    "w-full h-[600px]"
                )

                # Two-pass setup: create all layers first, then bind popups
                # in one client-side sweep via L.Map.eachLayer. The per-layer
                # `layer.run_method('bindPopup', ...)` path occasionally
                # didn't produce a clickable popup; iterating from the map
                # side bypasses that and uses Leaflet's API directly.
                popups: dict[str, str] = {}
                for rec, matched in zip(geo, trap_counts):
                    radius = _radius_for(matched)
                    color = _COLOR_ACTIVE if matched > 0 else _COLOR_DIM
                    options = {
                        "radius": radius,
                        "color": color,
                        "fillColor": color,
                        "fillOpacity": 0.55,
                        "weight": 2,
                    }
                    layer = lmap.generic_layer(
                        name="circleMarker",
                        args=[[rec["lat"], rec["lng"]], options],
                    )
                    popups[layer.id] = _popup_html(rec, matched, selected)

                # Bind popups via client-side JS. NiceGUI's Layer.run_method
                # ('bindPopup', ...) and run_map_method routes silently no-op
                # in our setup (probably timing — leaflet mounts when its
                # tab becomes visible, after the initial render queue has
                # drained). Going direct via ui.run_javascript + a small
                # retry loop in JS itself makes this resilient to tab
                # activation timing without needing Python-side timers.
                if popups:
                    popups_json = json.dumps(popups)
                    map_id = lmap.id
                    expected = len(popups)
                    js = (
                        "(function() {"
                        "  let attempts = 0;"
                        "  function tryBind() {"
                        f"    const el = window.getElement({map_id});"
                        "    if (el && el.map) {"
                        f"      const m = {popups_json};"
                        "      let bound = 0;"
                        "      el.map.eachLayer(function(layer) {"
                        "        if (layer && layer.id && m[layer.id]) {"
                        "          layer.bindPopup(m[layer.id]);"
                        "          bound++;"
                        "        }"
                        "      });"
                        f"      if (bound >= {expected}) return;"  # all bound
                        "    }"
                        "    if (++attempts < 60) setTimeout(tryBind, 200);"
                        "  }"
                        "  tryBind();"
                        "})();"
                    )
                    ui.run_javascript(js)

        _rebuild()
