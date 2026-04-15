"""Tab — MegaDetector + DeepFaune Europe.

Runs MegaDetector for animal detection, then DeepFaune's DINOv2 ViT-L
classifier on each animal crop. Output (per-image top-5 species) lives at
``outputs/<session>/md_deepfaune/predictions.json`` and is consumed by the
gallery / Series / Map tabs the same way SpeciesNet predictions are.

Weights:  deepfaune-vit_large_patch14_dinov2.lvd142m.v3.pt (~1.1 GB)
License:  CC BY-NC-SA 4.0 — non-commercial / share-alike
Source:   https://www.deepfaune.cnrs.fr/
"""
from __future__ import annotations

from pathlib import Path

from nicegui import ui

from ..config import OUTPUTS_DIR
from ..gallery import render_gallery
from ..job_manager import JobManager
from ..sessions import list_sessions


def _results_for(session: str) -> Path:
    return OUTPUTS_DIR / session / "md_deepfaune" / "predictions.json"


def render(jm: JobManager) -> None:
    with ui.column().classes("w-full gap-4"):
        ui.label("MegaDetector + DeepFaune Europe").classes(
            "text-xl font-semibold"
        )
        ui.label(
            "Runs MegaDetector to find animals, then DeepFaune (DINOv2 "
            "ViT-L, 34 European species) on each animal crop. License: "
            "CC BY-NC-SA 4.0 — non-commercial use only."
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
            ui.label("MD confidence").classes("text-sm w-32")
            conf = ui.slider(min=0.05, max=0.95, step=0.05, value=0.2).classes(
                "w-64"
            )
            ui.label().bind_text_from(
                conf, "value", lambda v: f"{float(v):.2f}"
            ).classes("w-10 font-mono text-sm")

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
                    err = s.get("error", "unknown")
                    result_area.content = f"**Error:** {err}"

        def run_job() -> None:
            if not session_select.value:
                ui.notify("Pick a session first", type="warning")
                return
            gallery_container.clear()
            jid = jm.submit(
                mode="md+deepfaune",
                session=session_select.value,
                conf=float(conf.value),
            )
            job_id_ref["id"] = jid
            status_label.text = "queued"
            result_area.content = ""
            poll_timer["t"] = ui.timer(0.5, poll)

        ui.button("Run MD + DeepFaune", on_click=run_job).props("color=primary")

        # Auto-load existing run if the session already has predictions.
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
                result_area.content = ""

        session_select.on_value_change(
            lambda _e: load_existing(session_select.value)
        )
