"""Tab — SpeciesNet (canonical / full pipeline).

Runs SpeciesNet end-to-end via ``model.predict(filepaths, country)``:
its own bundled detector finds animals, the classifier runs on each box
(or the whole image if the detector finds nothing), and the ensemble
applies geofence + confidence thresholds + taxonomic rollup. This is the
API ``speciesnet/scripts/run_model.py`` recommends.

Unlike the ``MD + SpeciesNet`` tab (which runs our MegaDetector first and
feeds its boxes into the classifier), this tab ignores our MD entirely.
Pro: catches animals MD missed, and labels reflect the library's own
confidence + rollup logic (so low-confidence predictions get demoted to
family / order / "animal" instead of being surfaced as implausible
species). Con: the boxes in the gallery come from SpeciesNet's MDv5,
not our MDv1000 — expect slightly different detection quality.
"""
from __future__ import annotations

import json as _json
from pathlib import Path

from nicegui import ui

from ..config import OUTPUTS_DIR
from ..exports import md_per_image_csv
from ..gallery import render_gallery
from ..job_manager import JobManager
from ..sessions import list_sessions


def _results_for(session: str) -> Path:
    return OUTPUTS_DIR / session / "speciesnet" / "predictions.json"


def render(jm: JobManager) -> None:
    with ui.column().classes("w-full gap-4"):
        ui.label("SpeciesNet (full pipeline)").classes("text-xl font-semibold")
        ui.label(
            "Runs SpeciesNet's canonical model.predict() — its own detector "
            "+ classifier + geofence + rollup in one pass. Catches animals "
            "our MegaDetector might miss. The ensemble's confidence + "
            "rollup logic demotes low-confidence predictions to higher "
            "taxonomic ranks (e.g. 'cervidae family') or falls back to the "
            "detector's generic 'animal' label rather than committing to "
            "an implausible species."
        ).classes("text-sm text-gray-600")

        session_select = ui.select(list_sessions(), label="Session").classes("w-96")
        session_select.on(
            "popup-show", lambda _e: session_select.set_options(list_sessions())
        )
        ui.button(
            "Refresh sessions",
            on_click=lambda: session_select.set_options(list_sessions()),
        ).props("flat")

        country = ui.select(
            options={
                "": "(none / global)",
                "DEU": "DEU — Germany",
                "SWE": "SWE — Sweden",
                "TZA": "TZA — Tanzania",
            },
            value="",
            label="Country (SpeciesNet geofence)",
        ).classes("w-64")

        status_label = ui.label("Idle.").classes("text-sm")
        progress = ui.linear_progress(value=0, show_value=False).classes("w-full")
        result_area = ui.markdown("")

        job_id_ref: dict = {"id": None}
        poll_timer: dict = {"t": None}
        gallery_container = ui.column().classes("w-full mt-4")
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
                    pred_path = out_dir / "predictions.json"
                    result_area.content = (
                        f"**Done** — {res.get('num_images')} images processed. "
                        f"Output: `{out_dir}`"
                    )
                    if pred_path.exists():
                        gallery_filter["fn"] = render_gallery(
                            gallery_container, pred_path
                        )
                else:
                    result_area.content = f"**Error:** {s.get('error', 'unknown')}"

        def run_job() -> None:
            if not session_select.value:
                ui.notify("Pick a session first", type="warning")
                return
            gallery_container.clear()
            jid = jm.submit(
                mode="speciesnet",
                session=session_select.value,
                country=(country.value or None),
            )
            job_id_ref["id"] = jid
            status_label.text = "queued"
            result_area.content = ""
            poll_timer["t"] = ui.timer(0.5, poll)

        ui.button("Run SpeciesNet (full)", on_click=run_job).props("color=primary")

        # ── Downloads ───────────────────────────────────────────────────────
        with ui.card().classes("w-full"):
            ui.label("Downloads").classes("font-semibold")
            ui.label(
                "Raw predictions JSON (includes prediction_source so you "
                "can see whether a label came from the classifier, a "
                "rollup, or the detector fallback), and a per-image CSV "
                "summarizing the committed label for each image."
            ).classes("text-xs text-gray-600")

            def _session_or_warn() -> str | None:
                s = session_select.value
                if not s:
                    ui.notify("Pick a session first", type="warning")
                return s

            def _dl_json() -> None:
                s = _session_or_warn()
                if not s:
                    return
                path = _results_for(s)
                if not path.exists():
                    ui.notify("Run SpeciesNet (full) first", type="warning")
                    return
                ui.download(path, filename=f"{s}_speciesnet_full.json")

            def _dl_per_image_csv() -> None:
                s = _session_or_warn()
                if not s:
                    return
                src = _results_for(s)
                if not src.exists():
                    ui.notify("Run SpeciesNet (full) first", type="warning")
                    return
                csv_text = md_per_image_csv(
                    _json.loads(src.read_text()), with_species=True
                )
                ui.download(
                    csv_text.encode("utf-8"),
                    filename=f"{s}_per_image_speciesnet_full.csv",
                )

            with ui.row().classes("gap-2 flex-wrap"):
                ui.button("Predictions JSON", on_click=_dl_json).props("flat")
                ui.button("Per-image CSV", on_click=_dl_per_image_csv).props("flat")

        # ── Session-change hook ─────────────────────────────────────────────
        def load_existing(session: str | None) -> None:
            gallery_container.clear()
            if not session:
                return
            pred_path = _results_for(session)
            if pred_path.exists():
                result_area.content = (
                    f"**Previous run** loaded from `{pred_path}`. "
                    "Run again to overwrite."
                )
                gallery_filter["fn"] = render_gallery(gallery_container, pred_path)
            else:
                gallery_filter["fn"] = None
                result_area.content = ""

        session_select.on_value_change(lambda _e: load_existing(session_select.value))
