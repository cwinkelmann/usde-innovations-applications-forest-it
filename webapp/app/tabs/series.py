"""Tab — Series: group session images into time-based bursts and summarize
SpeciesNet predictions per series.

Data dependencies:
  - Uploads:       ``/data/uploads/<session>/*.jpg``  (for EXIF extraction)
  - Predictions:   ``/data/outputs/<session>/md_speciesnet/predictions.json``
                   (optional — series still groups without, just no species)
  - Metadata cache: ``/data/outputs/<session>/metadata.json``

The cache means the expensive EXIF scan (~30 s for 3000 images) happens
once. Re-runs read from JSON in milliseconds.
"""
from __future__ import annotations

import json
from pathlib import Path

from nicegui import run, ui

from ..config import OUTPUTS_DIR, THUMBS_DIR, UPLOADS_DIR
from ..metadata import build_metadata, load_metadata
from ..series import aggregate_species, group_into_series, series_time_range
from ..sessions import list_sessions
from ..thumbs import thumb_path_for
from ..user_labels import (
    clear_images,
    label_images,
    load_labels,
    series_assigned_species,
)


def _predictions_by_file(session: str) -> dict[str, list[dict]]:
    """Load SpeciesNet predictions for a session, keyed by basename.

    Returns {} if the SpeciesNet run hasn't happened yet — the series view
    still groups by time, it just can't label them.
    """
    pred_path = OUTPUTS_DIR / session / "md_speciesnet" / "predictions.json"
    if not pred_path.exists():
        return {}
    try:
        data = json.loads(pred_path.read_text())
    except Exception:
        return {}
    out: dict[str, list[dict]] = {}
    for item in data:
        fname = Path(item.get("file", "")).name
        if fname:
            out[fname] = item.get("species") or []
    return out


def _fmt_range(start: str | None, end: str | None) -> str:
    """Compact 'Start → End' string, collapsing to single timestamp when the
    series spans one instant and to 'date H:M:S → H:M:S' when same day."""
    if not start:
        return "—"
    if start == end or not end:
        return start.replace("T", " ")
    s_date = start[:10]
    e_date = end[:10]
    s_time = start[11:] if "T" in start else start[11:]
    e_time = end[11:] if "T" in end else end[11:]
    if s_date == e_date:
        return f"{s_date} {s_time} → {e_time}"
    return f"{start.replace('T', ' ')} → {end.replace('T', ' ')}"


def render() -> None:
    with ui.column().classes("w-full gap-4"):
        ui.label("Series").classes("text-xl font-semibold")
        ui.label(
            "Groups session images into time-based bursts ('series') using "
            "EXIF timestamps, and rolls up SpeciesNet predictions per series. "
            "Default: images ≤ 60 seconds apart belong to the same series."
        ).classes("text-sm text-gray-600")

        session_select = ui.select(list_sessions(), label="Session").classes("w-96")
        session_select.on(
            "popup-show", lambda _e: session_select.set_options(list_sessions())
        )
        ui.button(
            "Refresh sessions",
            on_click=lambda: session_select.set_options(list_sessions()),
        ).props("flat")

        with ui.row().classes("items-center gap-3 w-full"):
            ui.label("Gap (seconds)").classes("text-sm w-28")
            gap_slider = ui.slider(min=10, max=600, step=5, value=60).classes("w-64")
            ui.label().bind_text_from(
                gap_slider, "value", lambda v: f"{int(v)} s"
            ).classes("w-14 font-mono text-sm")

        status_label = ui.label(
            "Pick a session. If metadata is cached it loads automatically."
        ).classes("text-sm text-gray-700")

        # State survives across render passes within this page.
        state: dict = {
            "metadata": [],  # list[dict] — {file, datetime, make, model}
            "preds": {},     # {file: [top-K species dicts]}
            "series_filter": None,  # set[str] | None of species to include
            "user_labels": {},  # {filename: species} — user overrides
        }

        action_row = ui.row().classes("gap-2")
        result_container = ui.column().classes("w-full mt-4 gap-2")

        def _render_empty(msg: str) -> None:
            result_container.clear()
            with result_container:
                ui.label(msg).classes("text-sm text-gray-500")

        def refresh_ui() -> None:
            session = session_select.value
            if not session:
                _render_empty("No session selected.")
                return
            records = state["metadata"]
            if not records:
                _render_empty(
                    "No metadata yet. Click 'Extract metadata' to scan the session."
                )
                return

            preds = state["preds"]
            gap = int(gap_slider.value or 60)
            series_list, untimed = group_into_series(records, max_gap_seconds=gap)
            summaries = [aggregate_species(s, preds) for s in series_list]

            # Species that appear *anywhere* in any series — not just as the
            # majority. Matches the SpeciesNet tab's species list so users
            # don't "lose" species that are always outvoted inside bursts
            # (e.g. grey wolf caught alongside wild boar).
            all_species_seen = sorted(
                {sp for sm in summaries for sp in sm["all_species"]}
            )

            result_container.clear()
            with result_container:
                # Filter row — only meaningful when we have predictions
                if all_species_seen:
                    with ui.row().classes("items-center gap-3 flex-wrap"):
                        ui.label(
                            "Filter — any series containing:"
                        ).classes("text-sm")

                        def on_filter_change(e) -> None:
                            vals = e.value or []
                            state["series_filter"] = (
                                set(vals) if vals else set()
                            )
                            refresh_ui()

                        current = state["series_filter"]
                        initial = (
                            sorted(current)
                            if current is not None
                            else all_species_seen
                        )
                        (
                            ui.select(
                                options=all_species_seen,
                                value=initial,
                                multiple=True,
                                label=f"Species ({len(all_species_seen)})",
                            )
                            .props("use-chips outlined dense options-dense")
                            .classes("min-w-[260px]")
                            .on_value_change(on_filter_change)
                        )

                        def _set(sel: set[str] | None) -> None:
                            state["series_filter"] = sel
                            refresh_ui()

                        ui.button("None", on_click=lambda: _set(set())).props(
                            "flat dense"
                        ).tooltip("Show no species (empty table)")
                        ui.button("All", on_click=lambda: _set(None)).props(
                            "flat dense"
                        ).tooltip("Show all species")

                # Series table. `sortable: True` enables the header-click
                # sort. Numeric columns carry raw ints/floats so Quasar sorts
                # them numerically; the `range` column's display string
                # always starts with `YYYY-MM-DD`, so the alpha sort is
                # effectively chronological.
                columns = [
                    {"name": "idx", "label": "#", "field": "idx",
                     "align": "right", "sortable": True},
                    {"name": "range", "label": "Start → End",
                     "field": "range", "align": "left", "sortable": True},
                    {"name": "count", "label": "Images", "field": "count",
                     "align": "right", "sortable": True},
                    {"name": "assigned", "label": "Assigned",
                     "field": "assigned", "align": "left", "sortable": True},
                    {"name": "majority", "label": "Majority (model)",
                     "field": "majority", "align": "left", "sortable": True},
                    {"name": "others", "label": "Other species",
                     "field": "others", "align": "left"},
                    {"name": "max_conf", "label": "Max conf",
                     "field": "max_conf", "align": "right", "sortable": True,
                     ":format": "v => v == null ? '' : Number(v).toFixed(2)"},
                ]
                rows = []
                visible: dict[int, tuple[list[dict], dict]] = {}
                selected = state["series_filter"]
                for i, (series, summary) in enumerate(zip(series_list, summaries)):
                    majority = summary["majority"] or "—"
                    # A series matches when any of its observed species is
                    # in the selection — not just the majority. This way
                    # a wolf-in-the-mix series shows up when you pick wolf.
                    if selected is not None:
                        series_species = set(summary["all_species"])
                        if not (series_species & selected):
                            continue
                    start, end = series_time_range(series)
                    others = [
                        sp for sp in summary["all_species"]
                        if sp != summary["majority"]
                    ]
                    idx = i + 1
                    # User-assigned species (only if all images in the
                    # series share a single user label). Mixed or absent
                    # labels render as "—".
                    filenames_in_series = [r["file"] for r in series]
                    assigned = series_assigned_species(
                        filenames_in_series, state["user_labels"]
                    )
                    rows.append(
                        {
                            "idx": idx,
                            "range": _fmt_range(start, end),
                            "count": summary["image_count"],
                            "assigned": assigned or "—",
                            "majority": majority,
                            "others": ", ".join(others) if others else "",
                            # Store the raw float for numeric sort; the
                            # column's :format directive renders it to 2dp.
                            # None (no preds) becomes 0.0 so sort is stable.
                            "max_conf": round(summary["max_conf"], 4)
                            if summary["max_conf"]
                            else 0.0,
                        }
                    )
                    visible[idx] = (series, summary)

                total_series = len(series_list)
                ui.label(
                    f"{total_series} series · {len(rows)} shown · "
                    f"{len(untimed)} image(s) without timestamp"
                ).classes("text-sm text-gray-600 mt-2")

                detail_container = ui.column().classes("w-full mt-3 gap-2")

                def _open_fullsize(fname: str) -> None:
                    """Maximised modal with the full-resolution image.

                    Served from /uploads/ (not /thumbs/) because the thumb is
                    256 px — fine for the grid but useless for close-up
                    identification. Reconyx JPEGs are ~2 MB, browser
                    handles it fine even on 3000-file sessions.
                    """
                    src = UPLOADS_DIR / session / fname
                    if not src.exists():
                        ui.notify(f"{fname} not found on disk", type="warning")
                        return
                    rel = src.relative_to(UPLOADS_DIR).as_posix()
                    with ui.dialog().props("maximized") as dialog:
                        with ui.card().classes(
                            "w-full h-full flex flex-col items-center "
                            "justify-center p-4 bg-black"
                        ):
                            ui.html(
                                f'<img src="/uploads/{rel}" '
                                f'style="display:block;max-width:95vw;'
                                f'max-height:85vh;" />'
                            )
                            preds_for_file = state["preds"].get(fname) or []
                            caption = fname
                            if preds_for_file:
                                top = preds_for_file[0]
                                caption = (
                                    f"{fname} — {top.get('common_name', '?')} "
                                    f"({float(top.get('score') or 0):.2f})"
                                )
                            ui.label(caption).classes(
                                "text-sm text-white mt-2"
                            )
                            ui.button(
                                "Close", on_click=dialog.close
                            ).props("flat color=white")
                    dialog.open()

                def _assign_series(
                    fnames: list[str], species: str
                ) -> None:
                    species = (species or "").strip()
                    if not species:
                        ui.notify("Pick or type a species name.", type="warning")
                        return
                    label_images(session, fnames, species)
                    state["user_labels"] = load_labels(session)
                    ui.notify(
                        f"Assigned '{species}' to {len(fnames)} image(s).",
                        type="positive",
                    )
                    refresh_ui()

                def _clear_series(fnames: list[str]) -> None:
                    clear_images(session, fnames)
                    state["user_labels"] = load_labels(session)
                    ui.notify(
                        f"Cleared labels for {len(fnames)} image(s).",
                        type="info",
                    )
                    refresh_ui()

                def _show_series(idx: int) -> None:
                    detail_container.clear()
                    match = visible.get(idx)
                    if not match:
                        return
                    series, summary = match
                    fnames = [r["file"] for r in series]
                    with detail_container:
                        start, end = series_time_range(series)
                        ui.label(
                            f"Series {idx} — {summary['image_count']} images · "
                            f"{_fmt_range(start, end)}"
                        ).classes("text-sm font-semibold")
                        if summary["majority"]:
                            ui.label(
                                f"Majority (model): {summary['majority']} · "
                                f"All species seen: {', '.join(summary['all_species'])}"
                            ).classes("text-xs text-gray-600")

                        # ── Species assignment ────────────────────────────
                        # Options: the union of all predicted species (so
                        # users get autocomplete) plus any already-used
                        # custom labels. `new-value-mode="add"` lets a
                        # biologist type a species SpeciesNet never saw.
                        existing_assigned = series_assigned_species(
                            fnames, state["user_labels"]
                        )
                        # all_species_seen is already a sorted *list*; cast
                        # to set for the union with custom user labels.
                        options = sorted(
                            set(all_species_seen)
                            | {v for v in state["user_labels"].values() if v}
                        )
                        with ui.row().classes(
                            "items-center gap-2 mt-2 flex-wrap"
                        ):
                            ui.label("Assign species:").classes("text-sm")
                            assign_select = (
                                ui.select(
                                    options=options,
                                    value=existing_assigned or None,
                                    with_input=True,
                                    new_value_mode="add-unique",
                                    label="species (type or pick)",
                                )
                                .props("dense outlined use-input")
                                .classes("min-w-[220px]")
                            )
                            ui.button(
                                "Save assignment",
                                icon="save",
                                on_click=lambda: _assign_series(
                                    fnames, assign_select.value or ""
                                ),
                            ).props("color=primary")
                            ui.button(
                                "Clear",
                                on_click=lambda: _clear_series(fnames),
                            ).props("flat")
                            if existing_assigned:
                                ui.label(
                                    f"(currently assigned: {existing_assigned})"
                                ).classes("text-xs text-green-700")

                        # ── Thumbnail grid (click → modal) ────────────────
                        with ui.row().classes("gap-2 flex-wrap mt-3"):
                            for rec in series:
                                fname = rec["file"]
                                src = UPLOADS_DIR / session / fname
                                if not src.exists():
                                    continue
                                tp = thumb_path_for(src)
                                if tp.exists():
                                    rel = tp.relative_to(THUMBS_DIR).as_posix()
                                    url = f"/thumbs/{rel}"
                                else:
                                    rel = src.relative_to(UPLOADS_DIR).as_posix()
                                    url = f"/uploads/{rel}"
                                preds_for_file = state["preds"].get(fname) or []
                                if preds_for_file:
                                    top = preds_for_file[0]
                                    caption = (
                                        f"{top.get('common_name', '?')} "
                                        f"({float(top.get('score') or 0):.2f})"
                                    )
                                else:
                                    caption = fname
                                user_label = state["user_labels"].get(fname)
                                with ui.element("div").classes(
                                    "inline-block"
                                ).style("width:140px;"):
                                    # Click the wrapper (not the image) —
                                    # q-img's loader overlay can swallow
                                    # click events. A plain div as the
                                    # event target, with a raw <img>
                                    # nested inside via ui.html, reliably
                                    # fires the handler.
                                    wrapper = (
                                        ui.element("div")
                                        .classes(
                                            "rounded overflow-hidden shadow "
                                            "cursor-pointer"
                                        )
                                        .style(
                                            "width:140px;height:140px;"
                                            "line-height:0;"
                                        )
                                        .tooltip(fname)
                                        .on(
                                            "click",
                                            lambda _e, fn=fname: _open_fullsize(fn),
                                        )
                                    )
                                    with wrapper:
                                        ui.html(
                                            f'<img src="{url}" '
                                            f'style="width:140px;height:140px;'
                                            f'object-fit:cover;display:block;" '
                                            f'loading="lazy" />'
                                        )
                                    ui.label(caption).classes(
                                        "text-xs truncate block w-full mt-1"
                                    )
                                    if user_label:
                                        ui.label(
                                            f"→ {user_label}"
                                        ).classes(
                                            "text-xs text-green-700 "
                                            "truncate block w-full"
                                        )

                tbl = ui.table(
                    columns=columns, rows=rows, row_key="idx"
                ).classes("w-full")
                # Row click fires Quasar's (evt, row, index). NiceGUI packs
                # those into e.args as a 3-tuple; e.args[1] is the row dict.
                tbl.on(
                    "row-click", lambda e: _show_series(int(e.args[1]["idx"]))
                )
                # Also auto-render the first visible series so the panel
                # isn't blank on initial load.
                if rows:
                    _show_series(rows[0]["idx"])

                if untimed:
                    ui.separator()
                    ui.label(
                        f"{len(untimed)} image(s) had no parseable EXIF "
                        "DateTimeOriginal — not shown in the table above."
                    ).classes("text-sm text-gray-600")

        async def do_extract() -> None:
            session = session_select.value
            if not session:
                ui.notify("Pick a session first", type="warning")
                return
            status_label.set_text(f"Scanning EXIF for '{session}'… (this may take ~30 s for 3000 images)")
            records = await run.io_bound(build_metadata, session)
            state["metadata"] = records
            state["preds"] = _predictions_by_file(session)
            state["user_labels"] = load_labels(session)
            n_timed = sum(1 for r in records if r.get("datetime"))
            status_label.set_text(
                f"Metadata cached — {len(records)} images, "
                f"{n_timed} with timestamps, "
                f"{len(state['preds'])} prediction(s), "
                f"{len(state['user_labels'])} user label(s)."
            )
            state["series_filter"] = None
            refresh_ui()

        def on_session_change(_e) -> None:
            session = session_select.value
            state["metadata"] = []
            state["preds"] = {}
            state["user_labels"] = {}
            state["series_filter"] = None
            if session:
                cached = load_metadata(session)
                state["user_labels"] = load_labels(session)
                if cached:
                    state["metadata"] = cached
                    state["preds"] = _predictions_by_file(session)
                    n_timed = sum(1 for r in cached if r.get("datetime"))
                    status_label.set_text(
                        f"Loaded cached metadata — {len(cached)} images, "
                        f"{n_timed} timed, "
                        f"{len(state['preds'])} prediction(s), "
                        f"{len(state['user_labels'])} user label(s)."
                    )
                else:
                    status_label.set_text(
                        "No cached metadata. Click 'Extract metadata' to scan."
                    )
            refresh_ui()

        with action_row:
            ui.button(
                "Extract metadata", icon="schedule", on_click=do_extract
            ).props("color=primary")

        session_select.on_value_change(on_session_change)
        gap_slider.on_value_change(lambda _e: refresh_ui())
