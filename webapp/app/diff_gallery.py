"""Gallery variant that shows MD vs. Label Studio disagreements.

Colors:
  TP (green)   — predicted box that humans kept
  FP (red)     — predicted box humans removed/disagreed with (over-detection)
  FN (amber)   — human box MD missed (under-detection)

Filters let you hide any of the three categories. An image is included only
if at least one visible category has a box on it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nicegui import ui

from .config import UPLOADS_DIR

COLORS = {"tp": "#16a34a", "fp": "#dc2626", "fn": "#f59e0b"}
PER_PAGE = 24


def _file_to_url(file: str) -> str:
    p = Path(file)
    try:
        return f"/uploads/{p.relative_to(UPLOADS_DIR).as_posix()}"
    except ValueError:
        return file


def _img_tag(url: str, extra_style: str = "") -> None:
    ui.html(
        f'<img src="{url}" style="display:block;{extra_style}" loading="lazy" />'
    )


def _draw(item: dict, kinds: set[str], stroke: int, label_px: int) -> None:
    w, h = item["width"], item["height"]
    for kind in ("tp", "fp", "fn"):
        if kind not in kinds:
            continue
        color = COLORS[kind]
        for x1, y1, x2, y2 in item.get(kind, []):
            left = x1 / w * 100
            top = y1 / h * 100
            bw = (x2 - x1) / w * 100
            bh = (y2 - y1) / h * 100
            ui.html(
                f'<div style="position:absolute;left:{left}%;top:{top}%;'
                f'width:{bw}%;height:{bh}%;border:{stroke}px solid {color};'
                f'box-sizing:border-box;pointer-events:none;"></div>'
            )
            ui.html(
                f'<div style="position:absolute;left:{left}%;top:{top}%;'
                f'background:{color};color:white;font-size:{label_px}px;'
                f'padding:1px 4px;line-height:1.2;pointer-events:none;'
                f'font-family:ui-monospace,monospace;">{kind.upper()}</div>'
            )


@dataclass
class _State:
    page: int
    tp: bool
    fp: bool
    fn: bool

    def kinds(self) -> set[str]:
        s: set[str] = set()
        if self.tp:
            s.add("tp")
        if self.fp:
            s.add("fp")
        if self.fn:
            s.add("fn")
        return s


def render_diff_gallery(container: ui.element, diff: list[dict]) -> None:
    state = _State(page=0, tp=False, fp=True, fn=True)  # default: show disagreements

    def apply_filter() -> list[dict]:
        kinds = state.kinds()
        out: list[dict] = []
        for item in diff:
            if any(item.get(k) for k in kinds):
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
                    _img_tag(url, "max-width:95vw;max-height:82vh;")
                    _draw(item, state.kinds(), stroke=3, label_px=13)
                with ui.row().classes("gap-3 mt-3 flex-wrap text-white text-sm"):
                    ui.label(Path(item["file"]).name).classes("font-semibold")
                    ui.label(f"TP {len(item.get('tp', []))}").style(f"color:{COLORS['tp']};")
                    ui.label(f"FP {len(item.get('fp', []))}").style(f"color:{COLORS['fp']};")
                    ui.label(f"FN {len(item.get('fn', []))}").style(f"color:{COLORS['fn']};")
                ui.button("Close", on_click=dialog.close).props("flat color=white")
        dialog.open()

    def thumb(item: dict) -> None:
        url = _file_to_url(item["file"])
        with ui.element("div").classes("relative inline-block cursor-pointer").style(
            "width:220px;"
        ):
            with ui.element("div").classes(
                "relative rounded overflow-hidden shadow"
            ).style("width:220px;line-height:0;").on(
                "click", lambda _e, it=item: show_dialog(it)
            ):
                _img_tag(url, "width:100%;")
                _draw(item, state.kinds(), stroke=2, label_px=9)
            with ui.row().classes("gap-2 text-xs mt-1"):
                ui.label(f"TP {len(item.get('tp', []))}").style(f"color:{COLORS['tp']};")
                ui.label(f"FP {len(item.get('fp', []))}").style(f"color:{COLORS['fp']};")
                ui.label(f"FN {len(item.get('fn', []))}").style(f"color:{COLORS['fn']};")

    def rerender() -> None:
        container.clear()
        with container:
            with ui.row().classes("items-center gap-4"):
                ui.label("Diff filter:").classes("text-sm font-semibold")

                def _mk(attr: str):
                    def handler(e):
                        setattr(state, attr, bool(e.value))
                        state.page = 0
                        rerender()
                    return handler

                for kind, color in COLORS.items():
                    with ui.element("div").classes("flex items-center gap-1"):
                        ui.element("div").style(
                            f"width:12px;height:12px;background:{color};border-radius:2px;"
                        )
                        cb = ui.checkbox(kind.upper(), value=getattr(state, kind))
                        cb.on_value_change(_mk(kind))

            filtered = apply_filter()
            total = len(filtered)
            pages = max(1, (total + PER_PAGE - 1) // PER_PAGE)
            state.page = min(state.page, pages - 1)
            start = state.page * PER_PAGE
            items = filtered[start : start + PER_PAGE]

            ui.label(
                f"{total} image(s) with visible differences — page {state.page + 1} / {pages}"
            ).classes("text-sm text-gray-600")

            with ui.row().classes("gap-3 flex-wrap w-full"):
                for it in items:
                    thumb(it)

            def prev():
                if state.page > 0:
                    state.page -= 1
                    rerender()

            def nxt():
                if state.page < pages - 1:
                    state.page += 1
                    rerender()

            with ui.row().classes("gap-2 mt-2"):
                ui.button("← Prev", on_click=prev).props("flat").set_enabled(state.page > 0)
                ui.button("Next →", on_click=nxt).props("flat").set_enabled(state.page < pages - 1)

    rerender()
