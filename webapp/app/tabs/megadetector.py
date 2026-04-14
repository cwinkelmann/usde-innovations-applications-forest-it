"""Tab 2 — MegaDetector detection + Label Studio bridge."""
from __future__ import annotations

import os
from pathlib import Path

from nicegui import run, ui

from ..config import OUTPUTS_DIR, UPLOADS_DIR
from ..diff_gallery import render_diff_gallery
from ..evaluation import diff_paths, evaluate_paths
from ..exports import (
    MD_CLASSES,
    build_class_zip,
    coco_annotations_to_csv,
    md_detections_to_csv,
    md_per_image_csv,
)
from ..gallery import render_gallery
from ..job_manager import JobManager
from ..labelstudio import export_session, import_session
from ..sessions import list_sessions
from .. import user_config


def _md_results_for(session: str) -> Path:
    return OUTPUTS_DIR / session / "md" / "detections.json"


def _default_project_name(session: str) -> str:
    return f"FIT — {session}"


def _ls_export_path(session: str) -> Path:
    return OUTPUTS_DIR / session / "labelstudio_export.json"


def _exported_sessions() -> dict:
    return user_config.load().get("ls_exports", {})


def _mark_exported(session: str, project: str, link: str) -> None:
    from datetime import datetime

    existing = _exported_sessions()
    existing[session] = {
        "project": project,
        "link": link,
        "at": datetime.now().isoformat(timespec="seconds"),
    }
    user_config.save({"ls_exports": existing})


def _forget_export(session: str) -> None:
    existing = _exported_sessions()
    if session in existing:
        del existing[session]
        user_config.save({"ls_exports": existing})


def _render_eval_card(container: ui.element, m: dict) -> None:
    """Show precision/recall/F1 + per-class breakdown after LS import."""
    container.clear()
    with container:
        with ui.card().classes("w-full"):
            ui.label(
                f"MegaDetector vs. Label Studio agreement  (IoU ≥ {m['iou_threshold']:.2f})"
            ).classes("font-semibold")

            with ui.row().classes("gap-6 text-sm items-center flex-wrap"):
                ui.label(f"MD boxes: {m['n_predicted_boxes']}").classes("font-mono")
                ui.label(f"Human boxes: {m['n_human_boxes']}").classes("font-mono")
                ui.label(f"TP: {m['tp']}").classes("font-mono text-green-700")
                ui.label(f"FP: {m['fp']}").classes("font-mono text-red-700")
                ui.label(f"FN: {m['fn']}").classes("font-mono text-amber-700")

            with ui.row().classes("gap-6 text-base font-semibold mt-1 flex-wrap"):
                ui.label(f"Precision: {m['precision']:.3f}")
                ui.label(f"Recall: {m['recall']:.3f}")
                ui.label(f"F1: {m['f1']:.3f}")

            if m["per_class"]:
                columns = [
                    {"name": "class", "label": "Class", "field": "class", "align": "left"},
                    {"name": "tp", "label": "TP", "field": "tp", "align": "right"},
                    {"name": "fp", "label": "FP", "field": "fp", "align": "right"},
                    {"name": "fn", "label": "FN", "field": "fn", "align": "right"},
                    {"name": "precision", "label": "Precision", "field": "precision", "align": "right"},
                    {"name": "recall", "label": "Recall", "field": "recall", "align": "right"},
                    {"name": "f1", "label": "F1", "field": "f1", "align": "right"},
                ]
                rows = [
                    {
                        **r,
                        "precision": f"{r['precision']:.3f}",
                        "recall": f"{r['recall']:.3f}",
                        "f1": f"{r['f1']:.3f}",
                    }
                    for r in m["per_class"]
                ]
                ui.table(columns=columns, rows=rows, row_key="class").classes(
                    "w-full mt-2"
                )


def render(jm: JobManager) -> None:
    with ui.column().classes("w-full gap-4"):
        ui.label("MegaDetector").classes("text-xl font-semibold")
        ui.label("Detects animal / person / vehicle bounding boxes.").classes(
            "text-sm text-gray-600"
        )

        session_select = ui.select(list_sessions(), label="Session").classes("w-96")
        session_select.on(
            "popup-show", lambda _e: session_select.set_options(list_sessions())
        )
        ui.button(
            "Refresh sessions",
            on_click=lambda: session_select.set_options(list_sessions()),
        ).props("flat")

        with ui.row().classes("items-center gap-3 w-full"):
            ui.label("Confidence").classes("text-sm w-24")
            conf = ui.slider(min=0.05, max=0.95, step=0.05, value=0.2).classes(
                "w-64"
            )
            # Live value readout — binds the label to whatever conf currently is.
            ui.label().bind_text_from(
                conf, "value", lambda v: f"{float(v):.2f}"
            ).classes("w-10 font-mono text-sm")
        imgsz = ui.number("Image size", value=1280, min=320, max=4096, step=64).classes(
            "w-32"
        )

        status_label = ui.label("Idle.").classes("text-sm")
        progress = ui.linear_progress(value=0, show_value=False).classes("w-full")
        result_area = ui.markdown("")

        # ── Run button sits right under the configs so it's reachable without
        # scrolling past the Label Studio card. ─────────────────────────────
        job_id_ref: dict = {"id": None}
        poll_timer: dict = {"t": None}
        gallery_container = ui.column().classes("w-full mt-4")
        # Holder for the gallery's "currently filtered filenames" callable.
        # Updated on each render; consulted at export time so only images the
        # user actually sees are pushed to Label Studio.
        gallery_filter: dict = {"fn": None}

        def poll() -> None:
            jid = job_id_ref["id"]
            if not jid:
                return
            s = jm.get_status(jid)
            state = s.get("state", "unknown")
            progress.value = float(s.get("progress", 0.0))
            worker = s.get("worker", "-")
            stage = s.get("stage")
            processed = s.get("processed")
            total = s.get("total")
            if state == "running" and stage and processed is not None and total:
                status_label.text = (
                    f"{stage}: {processed} / {total} images ({worker})"
                )
            else:
                status_label.text = f"{state} ({worker})"
            if state in ("done", "error"):
                if poll_timer["t"]:
                    poll_timer["t"].deactivate()
                    poll_timer["t"] = None
                if state == "done":
                    res = s.get("result", {})
                    out_dir = Path(res.get("out_dir", ""))
                    det_path = out_dir / "detections.json"
                    result_area.content = (
                        f"**Done** — {res.get('num_images')} images processed. "
                        f"Output: `{out_dir}`"
                    )
                    if det_path.exists():
                        gallery_filter["fn"] = render_gallery(gallery_container, det_path)
                else:
                    result_area.content = f"**Error:** {s.get('error', 'unknown')}"

        def run_md() -> None:
            if not session_select.value:
                ui.notify("Pick a session first", type="warning")
                return
            gallery_container.clear()
            jid = jm.submit(
                mode="md",
                session=session_select.value,
                conf=float(conf.value),
                imgsz=int(imgsz.value),
            )
            job_id_ref["id"] = jid
            status_label.text = "queued"
            result_area.content = ""
            poll_timer["t"] = ui.timer(0.5, poll)

        ui.button("Run MegaDetector", on_click=run_md).props("color=primary")

        # ── Label Studio card ───────────────────────────────────────────────
        saved = user_config.load()
        default_url = saved.get(
            "ls_url", os.environ.get("LABEL_STUDIO_URL", "http://localhost:8081")
        )
        default_token = saved.get(
            "ls_token", os.environ.get("LABEL_STUDIO_TOKEN", "")
        )
        with ui.card().classes("w-full"):
            ui.label("Label Studio").classes("font-semibold")
            ui.label(
                "Export pushes session images + MD pre-annotations to a Label "
                "Studio project. Import pulls the corrected COCO JSON back to "
                "outputs/<session>/labelstudio_export.json. "
                "Settings save to DATA_DIR/config.json."
            ).classes("text-xs text-gray-600")
            ls_url = ui.input("URL", value=default_url).classes("w-96")
            ls_token = ui.input(
                "API token",
                value=default_token,
                password=True,
                password_toggle_button=True,
            ).classes("w-96")
            ls_project = ui.input("Project name").classes("w-96")

            def save_ls_settings() -> None:
                user_config.save({"ls_url": ls_url.value, "ls_token": ls_token.value})
                ui.notify("Saved Label Studio settings", type="positive")

            ui.button("Save URL + token", on_click=save_ls_settings).props("flat")
            ls_status = ui.label("").classes("text-sm")
            eval_container = ui.column().classes("w-full mt-2")
            diff_container = ui.column().classes("w-full mt-2")

            async def _run_export(session: str, project: str) -> None:
                det_path = _md_results_for(session)
                # Snapshot the gallery filter at click time; if no gallery is
                # rendered (pre-run, or load failed) we fall through with None
                # which means "no filter — export everything".
                allowlist = (
                    gallery_filter["fn"]() if gallery_filter["fn"] else None
                )
                if allowlist is not None:
                    ls_status.text = (
                        f"Exporting {len(allowlist)} filtered image(s) to {project}..."
                    )
                else:
                    ls_status.text = f"Exporting to {project}..."
                try:
                    link = await run.io_bound(
                        export_session,
                        ls_url.value,
                        ls_token.value,
                        project,
                        UPLOADS_DIR / session,
                        det_path,
                        None,            # species_map — unused on MD tab
                        allowlist,       # file_allowlist
                    )
                    _mark_exported(session, project, link)
                    ls_status.text = f"Exported → {link}"
                    ui.notify("Export complete", type="positive")
                except Exception as e:  # noqa: BLE001
                    ls_status.text = f"Error: {e}"
                    ui.notify(f"Export failed: {e}", type="negative")

            async def do_export() -> None:
                session = session_select.value
                if not session:
                    ui.notify("Pick a session first", type="warning")
                    return
                det_path = _md_results_for(session)
                if not det_path.exists():
                    ui.notify("Run MegaDetector first", type="warning")
                    return
                if not ls_token.value:
                    ui.notify("Label Studio API token required", type="warning")
                    return
                project = ls_project.value or _default_project_name(session)

                prev = _exported_sessions().get(session)
                if prev:
                    with ui.dialog() as dialog, ui.card():
                        ui.label(
                            f"Session '{session}' was already exported on "
                            f"{prev.get('at', '?')} to project '{prev.get('project', '?')}'."
                        ).classes("font-semibold")
                        ui.label(
                            "Re-exporting skips already-uploaded images but may add new "
                            "pre-annotations to images not yet in the project."
                        ).classes("text-sm text-gray-600")
                        with ui.row().classes("gap-2 mt-2"):
                            ui.button(
                                "Re-export", on_click=lambda: dialog.submit("go")
                            ).props("color=primary")
                            ui.button(
                                "Forget previous",
                                on_click=lambda: dialog.submit("forget"),
                            ).props("flat")
                            ui.button("Cancel", on_click=lambda: dialog.submit("")).props(
                                "flat"
                            )
                    choice = await dialog
                    if choice == "":
                        return
                    if choice == "forget":
                        _forget_export(session)
                        ui.notify("Forgot previous export marker", type="info")
                        return
                await _run_export(session, project)

            async def do_import() -> None:
                session = session_select.value
                if not session:
                    ui.notify("Pick a session first", type="warning")
                    return
                if not ls_token.value:
                    ui.notify("Label Studio API token required", type="warning")
                    return
                project = ls_project.value or _default_project_name(session)
                out_path = OUTPUTS_DIR / session / "labelstudio_export.json"
                ls_status.text = f"Importing from {project}..."
                eval_container.clear()
                try:
                    n = await run.io_bound(
                        import_session,
                        ls_url.value,
                        ls_token.value,
                        project,
                        out_path,
                    )
                    ls_status.text = f"Imported {n} annotations → {out_path}"
                    ui.notify(f"Imported {n} annotations", type="positive")
                except Exception as e:  # noqa: BLE001
                    ls_status.text = f"Error: {e}"
                    ui.notify(f"Import failed: {e}", type="negative")
                    return

                # Compare corrected annotations against the MD predictions for
                # this session. Skip silently if no MD detections exist.
                det_path = _md_results_for(session)
                if not det_path.exists() or n == 0:
                    return
                try:
                    metrics = await run.io_bound(evaluate_paths, det_path, out_path)
                except Exception as e:  # noqa: BLE001
                    ui.notify(f"Evaluation failed: {e}", type="negative")
                    return
                _render_eval_card(eval_container, metrics)

                # Build a visual diff gallery (FP = over-detection, FN = missed)
                try:
                    diff = await run.io_bound(diff_paths, det_path, out_path)
                except Exception as e:  # noqa: BLE001
                    ui.notify(f"Diff build failed: {e}", type="negative")
                    return
                with diff_container:
                    diff_container.clear()
                    with diff_container:
                        ui.label(
                            "Disagreements — FN (missed by MD, amber) and FP "
                            "(added by MD, red). Toggle TP (green) to see agreement."
                        ).classes("text-sm font-semibold mt-2")
                render_diff_gallery(diff_container, diff)

        # ── Session-change hook — needs ls_project from the LS card. ────────

            with ui.row().classes("gap-2"):
                ui.button("Export to Label Studio", on_click=do_export).props(
                    "color=secondary"
                )
                ui.button("Import from Label Studio", on_click=do_import).props("flat")

        # ── Download card (CSV / JSON, both MD predictions and LS export) ──
        with ui.card().classes("w-full"):
            ui.label("Downloads").classes("font-semibold")
            ui.label(
                "MegaDetector predictions and the reviewed Label Studio "
                "annotations can be downloaded as JSON (native format) or CSV "
                "(one row per box)."
            ).classes("text-xs text-gray-600")

            def _session_or_warn() -> str | None:
                s = session_select.value
                if not s:
                    ui.notify("Pick a session first", type="warning")
                return s

            def _dl(path: Path, filename: str) -> None:
                if not path.exists():
                    ui.notify(f"{path.name} not available yet", type="warning")
                    return
                ui.download(path, filename=filename)

            def _dl_md_json() -> None:
                s = _session_or_warn()
                if not s:
                    return
                _dl(_md_results_for(s), f"{s}_md_detections.json")

            def _dl_md_csv() -> None:
                s = _session_or_warn()
                if not s:
                    return
                src = _md_results_for(s)
                if not src.exists():
                    ui.notify("Run MegaDetector first", type="warning")
                    return
                import json as _json

                csv_text = md_detections_to_csv(_json.loads(src.read_text()))
                ui.download(csv_text.encode("utf-8"), filename=f"{s}_md_detections.csv")

            def _dl_corr_json() -> None:
                s = _session_or_warn()
                if not s:
                    return
                _dl(_ls_export_path(s), f"{s}_labelstudio_export.json")

            def _dl_corr_csv() -> None:
                s = _session_or_warn()
                if not s:
                    return
                src = _ls_export_path(s)
                if not src.exists():
                    ui.notify("Import from Label Studio first", type="warning")
                    return
                import json as _json

                csv_text = coco_annotations_to_csv(_json.loads(src.read_text()))
                ui.download(csv_text.encode("utf-8"), filename=f"{s}_corrected.csv")

            with ui.row().classes("gap-2 flex-wrap"):
                ui.button("MD JSON", on_click=_dl_md_json).props("flat")
                ui.button("MD CSV (boxes)", on_click=_dl_md_csv).props("flat")
                ui.button("Corrected JSON", on_click=_dl_corr_json).props("flat")
                ui.button("Corrected CSV", on_click=_dl_corr_csv).props("flat")

            def _dl_per_image_csv() -> None:
                s = _session_or_warn()
                if not s:
                    return
                src = _md_results_for(s)
                if not src.exists():
                    ui.notify("Run MegaDetector first", type="warning")
                    return
                import json as _json

                csv_text = md_per_image_csv(_json.loads(src.read_text()))
                ui.download(
                    csv_text.encode("utf-8"), filename=f"{s}_per_image.csv"
                )

            ui.button("Per-image CSV (class counts)", on_click=_dl_per_image_csv).props(
                "flat"
            )

            ui.label("Download images as zip per class:").classes(
                "text-xs text-gray-600 mt-2"
            )

            async def _dl_class_zip(cls: str) -> None:
                s = _session_or_warn()
                if not s:
                    return
                src = _md_results_for(s)
                if not src.exists():
                    ui.notify("Run MegaDetector first", type="warning")
                    return
                import json as _json

                md = _json.loads(src.read_text())
                ui.notify(f"Building {cls} zip...", type="info")
                try:
                    data = await run.io_bound(build_class_zip, md, cls)
                except Exception as e:  # noqa: BLE001
                    ui.notify(f"Zip failed: {e}", type="negative")
                    return
                if not data or len(data) <= 22:  # empty zip sig is 22 bytes
                    ui.notify(f"No {cls} images found", type="warning")
                    return
                ui.download(data, filename=f"{s}_{cls}.zip")

            with ui.row().classes("gap-2 flex-wrap"):
                for _cls in (*MD_CLASSES, "empty"):
                    ui.button(
                        f"Zip {_cls}",
                        on_click=lambda _e, c=_cls: _dl_class_zip(c),
                    ).props("flat")

        # ── Session-change hook — needs ls_project from the LS card. ────────
        def load_existing(session: str | None) -> None:
            gallery_container.clear()
            if not session:
                ls_project.set_value("")
                return
            ls_project.set_value(_default_project_name(session))
            det_path = _md_results_for(session)
            if det_path.exists():
                result_area.content = (
                    f"**Previous run** loaded from `{det_path}`. "
                    f"Run again to overwrite."
                )
                gallery_filter["fn"] = render_gallery(gallery_container, det_path)
            else:
                gallery_filter["fn"] = None
                result_area.content = ""

        session_select.on_value_change(lambda _e: load_existing(session_select.value))
