"""Paginated gallery of MegaDetector (or MD+SpeciesNet) results.

Reads a `detections.json` written by the worker, renders a stats summary, a
class filter, and a paginated grid of thumbnails with CSS-overlaid bounding
boxes. Clicking a thumbnail opens a modal with the full-res image.

`render_gallery` returns a ``current_filenames()`` callable so downstream
callers (e.g. Label Studio export) can act on the same filter the user sees.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from nicegui import ui

from .config import THUMBS_DIR, UPLOADS_DIR
from .thumbs import thumb_path_for

CLASS_COLORS = {
    "animal": "#16a34a",
    "person": "#dc2626",
    "vehicle": "#2563eb",
}

PER_PAGE = 24


def _file_to_url(file: str) -> str:
    p = Path(file)
    try:
        return f"/uploads/{p.relative_to(UPLOADS_DIR).as_posix()}"
    except ValueError:
        return file


def _thumb_url(file: str) -> str:
    """URL for the thumbnail grid. Falls back to the full-res upload if the
    thumbnail doesn't exist (e.g. legacy sessions). Bounding boxes overlay
    correctly either way — they're positioned as percentages of the image's
    original pixel dimensions, which Pillow's aspect-preserving thumbnail
    inherits.
    """
    p = Path(file)
    try:
        tp = thumb_path_for(p)
    except ValueError:
        return _file_to_url(file)
    if tp.exists():
        return f"/thumbs/{tp.relative_to(THUMBS_DIR).as_posix()}"
    return _file_to_url(file)


def _img_tag(url: str, extra_style: str = "") -> None:
    """Inject a raw <img> so the parent inline-block container sizes to the
    image's intrinsic dimensions. `ui.image` wraps q-img, which needs an
    explicit width — that fights the overlay pattern."""
    ui.html(
        f'<img src="{url}" '
        f'style="display:block;{extra_style}" '
        f'loading="lazy" />'
    )


def _draw_boxes(
    item: dict,
    selected: set[str],
    stroke: int = 2,
    label_font_px: int = 11,
) -> None:
    w, h = item["width"], item["height"]
    for d in item.get("detections", []):
        if d["label"] not in selected:
            continue
        x1, y1, x2, y2 = d["bbox_xyxy"]
        left = x1 / w * 100
        top = y1 / h * 100
        bw = (x2 - x1) / w * 100
        bh = (y2 - y1) / h * 100
        color = CLASS_COLORS.get(d["label"], "#f59e0b")
        ui.html(
            f'<div style="position:absolute;left:{left}%;top:{top}%;'
            f'width:{bw}%;height:{bh}%;border:{stroke}px solid {color};'
            f'box-sizing:border-box;pointer-events:none;"></div>'
        )
        ui.html(
            f'<div style="position:absolute;left:{left}%;top:{top}%;'
            f'background:{color};color:white;font-size:{label_font_px}px;'
            f'padding:1px 4px;line-height:1.2;white-space:nowrap;'
            f'pointer-events:none;font-family:ui-monospace,monospace;">'
            f'{d["label"]} {d["conf"]:.2f}</div>'
        )


def _compute_stats(data: list[dict]) -> dict:
    total = len(data)
    images_with_class = {"animal": 0, "person": 0, "vehicle": 0}
    detections_per_class = {"animal": 0, "person": 0, "vehicle": 0}
    occupied = 0
    for item in data:
        dets = item.get("detections", [])
        classes = {d["label"] for d in dets}
        if dets:
            occupied += 1
        for cls in ("animal", "person", "vehicle"):
            if cls in classes:
                images_with_class[cls] += 1
        for d in dets:
            if d["label"] in detections_per_class:
                detections_per_class[d["label"]] += 1
    return {
        "total": total,
        "occupied": occupied,
        "empty": total - occupied,
        "images_with_class": images_with_class,
        "detections_per_class": detections_per_class,
    }


def _render_stats(stats: dict) -> None:
    with ui.card().classes("w-full"):
        ui.label("Summary").classes("text-sm font-semibold")
        with ui.row().classes("gap-6 text-sm"):
            ui.label(f"**Total:** {stats['total']}").classes("font-mono")
            ui.label(f"**Occupied:** {stats['occupied']}").classes(
                "font-mono text-green-700"
            )
            ui.label(f"**Empty:** {stats['empty']}").classes(
                "font-mono text-gray-500"
            )
        columns = [
            {"name": "cls", "label": "Class", "field": "cls", "align": "left"},
            {"name": "imgs", "label": "# images", "field": "imgs", "align": "right"},
            {"name": "dets", "label": "# detections", "field": "dets", "align": "right"},
        ]
        rows = [
            {
                "cls": cls,
                "imgs": stats["images_with_class"][cls],
                "dets": stats["detections_per_class"][cls],
            }
            for cls in ("animal", "person", "vehicle")
        ]
        ui.table(columns=columns, rows=rows, row_key="cls").classes("w-96")


@dataclass
class _State:
    page: int
    per_page: int
    animal: bool = True
    person: bool = True
    vehicle: bool = True
    show_empty: bool = True
    # Populated at render time when the data has SpeciesNet output; otherwise
    # unused. In species mode the "animal" checkbox is replaced by this set
    # and the top per-image species drives inclusion.
    species_selected: set[str] = field(default_factory=set)

    def selected_classes(self) -> set[str]:
        """Classes passed to _draw_boxes — determines which overlays to draw.

        In species mode, the presence of any selected species implies that
        animal boxes should still be rendered (the species filter narrows
        *which images* appear, not which boxes within them).
        """
        s: set[str] = set()
        if self.animal or self.species_selected:
            s.add("animal")
        if self.person:
            s.add("person")
        if self.vehicle:
            s.add("vehicle")
        return s


def render_gallery(
    container: ui.element, detections_path: Path
) -> Callable[[], set[str]]:
    """Render the gallery into `container`. Safe to call repeatedly.

    Returns a ``current_filenames()`` callable that yields the basenames of
    images matching the gallery's *current* filter state. Captures the local
    `_State` in its closure, so each call to ``render_gallery`` produces a
    fresh callable bound to its own gallery instance.
    """
    data: list[dict] = json.loads(Path(detections_path).read_text())
    stats = _compute_stats(data)

    # SpeciesNet results carry a `species` list on each item (ranked
    # predictions). Switch the filter UI to species mode when any item has it.
    has_species = any(item.get("species") for item in data)
    all_species: list[str] = sorted(
        {
            item["species"][0]["common_name"]
            for item in data
            if item.get("species")
        }
    )

    state = _State(page=0, per_page=PER_PAGE)
    if has_species:
        # Default: all species selected (no filter).
        state.species_selected = set(all_species)

    def _top_species(item: dict) -> str | None:
        sp = item.get("species") or []
        return sp[0].get("common_name") if sp else None

    def apply_filter() -> list[dict]:
        out: list[dict] = []
        for item in data:
            dets = item.get("detections", [])
            classes = {d["label"] for d in dets}
            if not classes:
                if state.show_empty:
                    out.append(item)
                continue

            include = False
            # Animal path: filter by selected species if we're in species mode.
            if "animal" in classes:
                if has_species:
                    top = _top_species(item)
                    if top and top in state.species_selected:
                        include = True
                    elif not item.get("species") and state.animal:
                        # Animal detection without any SpeciesNet prediction —
                        # fall back to the plain "animal" checkbox so these
                        # don't become invisible in species mode.
                        include = True
                else:
                    if state.animal:
                        include = True
            # Person / vehicle / mixed-class images: each class's own checkbox.
            if not include and "person" in classes and state.person:
                include = True
            if not include and "vehicle" in classes and state.vehicle:
                include = True
            if include:
                out.append(item)
        return out

    def show_dialog(item: dict) -> None:
        with ui.dialog().props("maximized") as dialog:
            with ui.card().classes(
                "w-full h-full flex flex-col items-center justify-center p-4 bg-black"
            ):
                url = _file_to_url(item["file"])
                with ui.element("div").style(
                    "position:relative;width:fit-content;max-width:100%;line-height:0;"
                ):
                    ui.html(
                        f'<img src="{url}" '
                        f'style="display:block;max-width:95vw;max-height:82vh;" />'
                    )
                    _draw_boxes(
                        item, state.selected_classes(), stroke=3, label_font_px=13
                    )
                with ui.column().classes("gap-1 mt-3 text-white"):
                    ui.label(Path(item["file"]).name).classes("font-semibold text-sm")
                    if item.get("species"):
                        top = item["species"][0]
                        ui.label(
                            f"Species: {top['common_name']} "
                            f"({top['score']:.2f}) — {top['species']}"
                        ).classes("text-base font-semibold").style("color:#86efac;")
                        rest = ", ".join(
                            f"{s['common_name']} {s['score']:.2f}"
                            for s in item["species"][1:4]
                        )
                        if rest:
                            ui.label(f"also: {rest}").classes("text-xs text-gray-300")
                    with ui.row().classes("gap-3 text-sm items-center flex-wrap"):
                        for d in item.get("detections", []):
                            color = CLASS_COLORS.get(d["label"], "#f59e0b")
                            ui.label(f"{d['label']} {d['conf']:.2f}").style(
                                f"color:{color};"
                            )
                ui.button("Close", on_click=dialog.close).props("flat color=white")
        dialog.open()

    def thumb(item: dict) -> None:
        url = _thumb_url(item["file"])
        with ui.element("div").classes(
            "relative inline-block cursor-pointer"
        ).style("width:200px;"):
            with ui.element("div").classes(
                "relative rounded overflow-hidden shadow"
            ).style("width:200px;line-height:0;").on(
                "click", lambda _e, it=item: show_dialog(it)
            ):
                _img_tag(url, "width:100%;")
                _draw_boxes(item, state.selected_classes(), stroke=2, label_font_px=9)
            if item.get("species"):
                top = item["species"][0]
                ui.label(f"{top['common_name']} ({top['score']:.2f})").classes(
                    "text-xs truncate block w-full mt-1"
                ).tooltip(top.get("full_label", ""))

    def rerender() -> None:
        container.clear()
        with container:
            _render_stats(stats)

            with ui.row().classes("items-center gap-4 w-full flex-wrap"):
                ui.label("Filter:").classes("text-sm font-semibold")

                def _tick(attr: str):
                    def handler(e):
                        setattr(state, attr, bool(e.value))
                        state.page = 0
                        rerender()
                    return handler

                # Species multi-select replaces the "animal" checkbox when
                # SpeciesNet predictions are present. `use-chips` renders
                # each selection as a chip with an inline remove button.
                if has_species:
                    def on_species_change(e) -> None:
                        state.species_selected = set(e.value or [])
                        state.page = 0
                        rerender()

                    def _set_species_all(selected: set[str]) -> None:
                        state.species_selected = selected
                        state.page = 0
                        rerender()

                    with ui.element("div").classes("flex items-center gap-1"):
                        color = CLASS_COLORS["animal"]
                        ui.element("div").style(
                            f"width:12px;height:12px;background:{color};border-radius:2px;"
                        )
                        (
                            ui.select(
                                options=all_species,
                                value=sorted(state.species_selected),
                                multiple=True,
                                label=f"Species ({len(all_species)})",
                            )
                            .props("use-chips outlined dense options-dense")
                            .classes("min-w-[260px]")
                            .on_value_change(on_species_change)
                        )
                        # Bulk-select helpers: quickly swap between "all"
                        # (inspect everything) and "none" (start from a
                        # clean slate, then tick only the species you want
                        # to keep — useful with 20+ species).
                        ui.button(
                            "None",
                            on_click=lambda: _set_species_all(set()),
                        ).props("flat dense").tooltip("Deselect all species")
                        ui.button(
                            "All",
                            on_click=lambda: _set_species_all(set(all_species)),
                        ).props("flat dense").tooltip("Select all species")
                else:
                    color = CLASS_COLORS["animal"]
                    with ui.element("div").classes("flex items-center gap-1"):
                        ui.element("div").style(
                            f"width:12px;height:12px;background:{color};border-radius:2px;"
                        )
                        cb = ui.checkbox("animal", value=state.animal)
                        cb.on_value_change(_tick("animal"))

                for cls in ("person", "vehicle"):
                    color = CLASS_COLORS[cls]
                    with ui.element("div").classes("flex items-center gap-1"):
                        ui.element("div").style(
                            f"width:12px;height:12px;background:{color};border-radius:2px;"
                        )
                        cb = ui.checkbox(cls, value=getattr(state, cls))
                        cb.on_value_change(_tick(cls))

                empty_cb = ui.checkbox("no detections", value=state.show_empty)
                empty_cb.on_value_change(_tick("show_empty"))

            filtered = apply_filter()
            total = len(filtered)
            pages = max(1, (total + state.per_page - 1) // state.per_page)
            state.page = min(state.page, pages - 1)
            start = state.page * state.per_page
            page_items = filtered[start : start + state.per_page]

            ui.label(
                f"Showing {len(page_items)} of {total} filtered image(s) "
                f"— page {state.page + 1} / {pages}"
            ).classes("text-sm text-gray-600")

            with ui.row().classes("gap-3 flex-wrap w-full"):
                for item in page_items:
                    thumb(item)

            def go_prev() -> None:
                if state.page > 0:
                    state.page -= 1
                    rerender()

            def go_next() -> None:
                if state.page < pages - 1:
                    state.page += 1
                    rerender()

            with ui.row().classes("gap-2 mt-2 items-center"):
                ui.button("← Prev", on_click=go_prev).props("flat").set_enabled(state.page > 0)
                ui.button("Next →", on_click=go_next).props("flat").set_enabled(state.page < pages - 1)

    rerender()

    def current_filenames() -> set[str]:
        """Basenames of images that pass the gallery's current filter.
        Reflects live state — call at export time, not at render time."""
        return {Path(item["file"]).name for item in apply_filter()}

    return current_filenames
