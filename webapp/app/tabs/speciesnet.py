"""Tab 3 — MegaDetector + SpeciesNet, with Label Studio + downloads + evaluation.

Feature-parity with the MegaDetector tab, but:
- operates on `outputs/<session>/md_speciesnet/predictions.json`
- LS export uses a species_map built from SpeciesNet's top predictions, so
  animal boxes arrive in Label Studio pre-labelled with the species name
  instead of the generic "animal" class
- per-image CSV includes the top species + its score
"""
from __future__ import annotations

import json as _json
import os
from pathlib import Path

from nicegui import run, ui

from ..config import OUTPUTS_DIR, UPLOADS_DIR
from ..evaluation import evaluate_paths
from ..exports import (
    MD_CLASSES,
    build_class_zip,
    coco_annotations_to_csv,
    md_detections_to_csv,
    md_per_image_csv,
)
from ..gallery import render_gallery
from ..job_manager import JobManager
from ..labelstudio import build_species_map, export_session, import_session
from ..sessions import list_sessions
from .. import user_config


def _results_for(session: str) -> Path:
    return OUTPUTS_DIR / session / "md_speciesnet" / "predictions.json"


def _ls_export_path(session: str) -> Path:
    return OUTPUTS_DIR / session / "labelstudio_speciesnet_export.json"


def _default_project_name(session: str) -> str:
    return f"FIT-SN — {session}"


def _exported_sessions() -> dict:
    return user_config.load().get("sn_ls_exports", {})


def _mark_exported(session: str, project: str, link: str) -> None:
    from datetime import datetime

    existing = _exported_sessions()
    existing[session] = {
        "project": project,
        "link": link,
        "at": datetime.now().isoformat(timespec="seconds"),
    }
    user_config.save({"sn_ls_exports": existing})


def _forget_export(session: str) -> None:
    existing = _exported_sessions()
    if session in existing:
        del existing[session]
        user_config.save({"sn_ls_exports": existing})


def _render_eval_card(container: ui.element, m: dict) -> None:
    container.clear()
    with container:
        with ui.card().classes("w-full"):
            ui.label(
                f"Predictions vs. Label Studio agreement  (IoU ≥ {m['iou_threshold']:.2f})"
            ).classes("font-semibold")
            with ui.row().classes("gap-6 text-sm items-center flex-wrap"):
                ui.label(f"Predicted boxes: {m['n_predicted_boxes']}").classes("font-mono")
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
                ui.table(columns=columns, rows=rows, row_key="class").classes("w-full mt-2")


def render(jm: JobManager) -> None:
    with ui.column().classes("w-full gap-4"):
        ui.label("MegaDetector + SpeciesNet").classes("text-xl font-semibold")
        ui.label(
            "Runs MegaDetector to find animals, then SpeciesNet (classifier-only) "
            "to label each animal's species."
        ).classes("text-sm text-gray-600")

        session_select = ui.select(list_sessions(), label="Session").classes("w-96")
        session_select.on(
            "popup-show", lambda _e: session_select.set_options(list_sessions())
        )
        ui.button(
            "Refresh sessions",
            on_click=lambda: session_select.set_options(list_sessions()),
        ).props("flat")

        conf = ui.number(
            "MD confidence", value=0.2, min=0.01, max=1.0, step=0.05, format="%.2f"
        ).classes("w-32")
        country = ui.select(
            options={"": "(none / global)", "DEU": "DEU — Germany", "TZA": "TZA — Tanzania"},
            value="",
            label="Country (SpeciesNet geofence)",
        ).classes("w-64")

        status_label = ui.label("Idle.").classes("text-sm")
        progress = ui.linear_progress(value=0, show_value=False).classes("w-full")
        result_area = ui.markdown("")

        job_id_ref: dict = {"id": None}
        poll_timer: dict = {"t": None}
        gallery_container = ui.column().classes("w-full mt-4")

        def poll() -> None:
            jid = job_id_ref["id"]
            if not jid:
                return
            s = jm.get_status(jid)
            state = s.get("state", "unknown")
            progress.value = float(s.get("progress", 0.0))
            status_label.text = f"{state} ({s.get('worker', '-')})"
            if state in ("done", "error"):
                if poll_timer["t"]:
                    poll_timer["t"].deactivate()
                    poll_timer["t"] = None
                if state == "done":
                    res = s.get("result", {})
                    out_dir = Path(res.get("out_dir", ""))
                    pred_path = out_dir / "predictions.json"
                    result_area.content = (
                        f"**Done** — {res.get('num_images')} images processed. "
                        f"Output: `{out_dir}`"
                    )
                    if pred_path.exists():
                        render_gallery(gallery_container, pred_path)
                else:
                    result_area.content = f"**Error:** {s.get('error', 'unknown')}"

        def run_job() -> None:
            if not session_select.value:
                ui.notify("Pick a session first", type="warning")
                return
            gallery_container.clear()
            jid = jm.submit(
                mode="md+speciesnet",
                session=session_select.value,
                conf=float(conf.value),
                country=(country.value or None),
            )
            job_id_ref["id"] = jid
            status_label.text = "queued"
            result_area.content = ""
            poll_timer["t"] = ui.timer(0.5, poll)

        ui.button("Run MD + SpeciesNet", on_click=run_job).props("color=primary")

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
                "Export pushes session images with species-labelled pre-annotations. "
                "Import pulls corrected COCO JSON to "
                "outputs/<session>/labelstudio_speciesnet_export.json."
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

            async def _run_export(session: str, project: str) -> None:
                pred_path = _results_for(session)
                species_map = build_species_map(pred_path)
                ls_status.text = (
                    f"Exporting to {project} with {len(set(species_map.values()))} species labels..."
                )
                try:
                    link = await run.io_bound(
                        export_session,
                        ls_url.value,
                        ls_token.value,
                        project,
                        UPLOADS_DIR / session,
                        pred_path,
                        species_map,
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
                pred_path = _results_for(session)
                if not pred_path.exists():
                    ui.notify("Run MD + SpeciesNet first", type="warning")
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
                out_path = _ls_export_path(session)
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

                pred_path = _results_for(session)
                if not pred_path.exists() or n == 0:
                    return
                try:
                    metrics = await run.io_bound(evaluate_paths, pred_path, out_path)
                except Exception as e:  # noqa: BLE001
                    ui.notify(f"Evaluation failed: {e}", type="negative")
                    return
                _render_eval_card(eval_container, metrics)

            with ui.row().classes("gap-2"):
                ui.button("Export to Label Studio", on_click=do_export).props(
                    "color=secondary"
                )
                ui.button("Import from Label Studio", on_click=do_import).props("flat")

        # ── Downloads card ──────────────────────────────────────────────────
        with ui.card().classes("w-full"):
            ui.label("Downloads").classes("font-semibold")
            ui.label(
                "SpeciesNet predictions (JSON / CSV with species), reviewed "
                "Label Studio annotations, per-image class counts, and per-class "
                "image zips."
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

            def _dl_sn_json() -> None:
                s = _session_or_warn()
                if not s:
                    return
                _dl(_results_for(s), f"{s}_sn_predictions.json")

            def _dl_sn_csv() -> None:
                s = _session_or_warn()
                if not s:
                    return
                src = _results_for(s)
                if not src.exists():
                    ui.notify("Run MD + SpeciesNet first", type="warning")
                    return
                csv_text = md_detections_to_csv(_json.loads(src.read_text()))
                ui.download(csv_text.encode("utf-8"), filename=f"{s}_sn_boxes.csv")

            def _dl_corr_json() -> None:
                s = _session_or_warn()
                if not s:
                    return
                _dl(_ls_export_path(s), f"{s}_ls_sn_export.json")

            def _dl_corr_csv() -> None:
                s = _session_or_warn()
                if not s:
                    return
                src = _ls_export_path(s)
                if not src.exists():
                    ui.notify("Import from Label Studio first", type="warning")
                    return
                csv_text = coco_annotations_to_csv(_json.loads(src.read_text()))
                ui.download(csv_text.encode("utf-8"), filename=f"{s}_corrected_sn.csv")

            with ui.row().classes("gap-2 flex-wrap"):
                ui.button("SN JSON", on_click=_dl_sn_json).props("flat")
                ui.button("SN CSV (boxes)", on_click=_dl_sn_csv).props("flat")
                ui.button("Corrected JSON", on_click=_dl_corr_json).props("flat")
                ui.button("Corrected CSV", on_click=_dl_corr_csv).props("flat")

            def _dl_per_image_csv() -> None:
                s = _session_or_warn()
                if not s:
                    return
                src = _results_for(s)
                if not src.exists():
                    ui.notify("Run MD + SpeciesNet first", type="warning")
                    return
                csv_text = md_per_image_csv(
                    _json.loads(src.read_text()), with_species=True
                )
                ui.download(
                    csv_text.encode("utf-8"), filename=f"{s}_per_image_sn.csv"
                )

            ui.button(
                "Per-image CSV (class counts + top species)",
                on_click=_dl_per_image_csv,
            ).props("flat")

            ui.label("Download images as zip per class:").classes(
                "text-xs text-gray-600 mt-2"
            )

            async def _dl_class_zip(cls: str) -> None:
                s = _session_or_warn()
                if not s:
                    return
                src = _results_for(s)
                if not src.exists():
                    ui.notify("Run MD + SpeciesNet first", type="warning")
                    return
                md = _json.loads(src.read_text())
                ui.notify(f"Building {cls} zip...", type="info")
                try:
                    data = await run.io_bound(build_class_zip, md, cls)
                except Exception as e:  # noqa: BLE001
                    ui.notify(f"Zip failed: {e}", type="negative")
                    return
                if not data or len(data) <= 22:
                    ui.notify(f"No {cls} images found", type="warning")
                    return
                ui.download(data, filename=f"{s}_{cls}.zip")

            with ui.row().classes("gap-2 flex-wrap"):
                for _cls in (*MD_CLASSES, "empty"):
                    ui.button(
                        f"Zip {_cls}",
                        on_click=lambda _e, c=_cls: _dl_class_zip(c),
                    ).props("flat")

        # ── Session-change hook ─────────────────────────────────────────────
        def load_existing(session: str | None) -> None:
            gallery_container.clear()
            eval_container.clear()
            if not session:
                ls_project.set_value("")
                return
            ls_project.set_value(_default_project_name(session))
            pred_path = _results_for(session)
            if pred_path.exists():
                result_area.content = (
                    f"**Previous run** loaded from `{pred_path}`. Run again to overwrite."
                )
                render_gallery(gallery_container, pred_path)
            else:
                result_area.content = ""

        session_select.on_value_change(lambda _e: load_existing(session_select.value))
