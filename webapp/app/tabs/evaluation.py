"""Tab — Evaluation: model predictions vs. Lissl ground-truth (wolf / no-wolf).

For a chosen session, joins per-image model outputs against the Lissl
``lissl_labels.json`` (built at upload time by SHA-256 lookup) and reports
per-model precision / recall / F1 plus a confusion matrix on the binary
*wolf-vs-no-wolf* task. Three model sources are evaluated when present:

  - SpeciesNet (``md_speciesnet/predictions.json``)
  - DeepFaune  (``md_deepfaune/predictions.json``)
  - MegaDetector only (``md/detections.json``) — uninformative for wolf
    discrimination; reported as "no comparison" with detection counts only.

If a session has no Lissl labels yet, the user can run a one-shot
``Tag from Lissl`` action that walks the session's images, hashes them,
and writes the labels file.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from nicegui import run, ui

from ..config import OUTPUTS_DIR, UPLOADS_DIR
from ..lissl_groundtruth import (
    index_size,
    labels_path,
    load_session_labels,
    tag_session,
)
from ..sessions import list_sessions

# Model output files we know how to evaluate.
_MODEL_PATHS = [
    ("SpeciesNet", "md_speciesnet/predictions.json"),
    ("DeepFaune", "md_deepfaune/predictions.json"),
]

# Heuristic: a model says "wolf" when its top-1 species name contains
# "wolf" as a whole word (handles "grey wolf", "red wolf", "timber wolf").
def _is_wolf(common_name: str | None) -> bool:
    if not common_name:
        return False
    cn = common_name.strip().lower()
    return cn == "wolf" or " wolf" in cn or cn.startswith("wolf ")


def _model_top1(item: dict) -> str | None:
    species = item.get("species") or []
    if species:
        return (species[0] or {}).get("common_name")
    return None


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else 0.0


def _evaluate(model_path: Path, gt: dict[str, str]) -> tuple[Confusion, int, int]:
    """Returns (confusion, predicted_only, no_overlap).

    Only images that appear in *both* the model output and the ground truth
    contribute to the confusion. Other images are reported as "predicted
    but no GT" so the user knows the coverage gap.
    """
    cm = Confusion()
    if not model_path.exists():
        return cm, 0, len(gt)
    try:
        data = json.loads(model_path.read_text())
    except Exception:
        return cm, 0, len(gt)
    pred_files = set()
    matched = 0
    for item in data:
        fname = Path(item.get("file", "")).name
        if not fname:
            continue
        pred_files.add(fname)
        gt_label = gt.get(fname)
        if gt_label is None:
            continue
        matched += 1
        is_pred_wolf = _is_wolf(_model_top1(item))
        is_gt_wolf = gt_label == "wolf"
        if is_gt_wolf and is_pred_wolf:
            cm.tp += 1
        elif is_gt_wolf and not is_pred_wolf:
            cm.fn += 1
        elif not is_gt_wolf and is_pred_wolf:
            cm.fp += 1
        else:
            cm.tn += 1
    pred_only = len(pred_files - set(gt))
    no_overlap = len(set(gt) - pred_files)
    return cm, pred_only, no_overlap


def render() -> None:
    with ui.column().classes("w-full gap-4"):
        ui.label("Evaluation").classes("text-xl font-semibold")
        ui.label(
            "Compares model predictions (SpeciesNet, DeepFaune) against "
            "Lissl's ground-truth wolf / no-wolf labels by SHA-256 image "
            "hash. Auto-tagged at upload; click 'Tag from Lissl' to "
            "(re-)scan an existing session."
        ).classes("text-sm text-gray-600")

        gt_status = ui.label(
            f"Lissl ground-truth index: {index_size():,} entries from "
            f"{Path('webapp/_data/lissl_occurances').name}/"
        ).classes("text-xs text-gray-500")

        session_select = ui.select(list_sessions(), label="Session").classes("w-96")
        session_select.on(
            "popup-show", lambda _e: session_select.set_options(list_sessions())
        )

        results_container = ui.column().classes("w-full gap-3 mt-2")

        def _render_results() -> None:
            results_container.clear()
            sess = session_select.value
            if not sess:
                with results_container:
                    ui.label("Pick a session above.").classes("text-sm text-gray-500")
                return
            gt = load_session_labels(sess)
            with results_container:
                # Coverage summary
                if not gt:
                    ui.label(
                        "No Lissl labels for this session yet. "
                        "Click 'Tag from Lissl' below to scan."
                    ).classes("text-sm text-orange-700")
                else:
                    n_wolf = sum(1 for v in gt.values() if v == "wolf")
                    n_no = sum(1 for v in gt.values() if v == "no_wolf")
                    ui.label(
                        f"Ground truth: {len(gt):,} labelled image(s) — "
                        f"{n_wolf:,} wolf · {n_no:,} no-wolf."
                    ).classes("text-sm text-gray-700 font-semibold")

                # Per-model evaluation
                for model_name, rel in _MODEL_PATHS:
                    model_path = OUTPUTS_DIR / sess / rel
                    if not model_path.exists():
                        ui.label(
                            f"{model_name}: no predictions.json yet."
                        ).classes("text-sm text-gray-500")
                        continue
                    cm, pred_only, no_overlap = _evaluate(model_path, gt)
                    if cm.total == 0:
                        ui.label(
                            f"{model_name}: predictions exist but none "
                            f"overlap with the ground truth set."
                        ).classes("text-sm text-gray-500")
                        continue
                    with ui.card().classes("w-full"):
                        ui.label(model_name).classes("font-semibold")
                        with ui.row().classes("gap-6 text-sm font-mono"):
                            ui.label(f"Precision: {cm.precision:.3f}")
                            ui.label(f"Recall: {cm.recall:.3f}")
                            ui.label(f"F1: {cm.f1:.3f}")
                            ui.label(f"Accuracy: {cm.accuracy:.3f}")
                        # Confusion matrix
                        rows = [
                            {
                                "row": "GT wolf",
                                "pred_wolf": cm.tp,
                                "pred_no": cm.fn,
                            },
                            {
                                "row": "GT no-wolf",
                                "pred_wolf": cm.fp,
                                "pred_no": cm.tn,
                            },
                        ]
                        cols = [
                            {"name": "row", "label": "", "field": "row", "align": "left"},
                            {"name": "pred_wolf", "label": "Pred wolf",
                             "field": "pred_wolf", "align": "right"},
                            {"name": "pred_no", "label": "Pred no-wolf",
                             "field": "pred_no", "align": "right"},
                        ]
                        ui.table(columns=cols, rows=rows, row_key="row").classes("w-96")
                        ui.label(
                            f"Evaluated {cm.total} image(s) "
                            f"({cm.tp} TP · {cm.fp} FP · {cm.tn} TN · "
                            f"{cm.fn} FN). "
                            f"{pred_only} prediction(s) had no GT; "
                            f"{no_overlap} GT row(s) had no prediction."
                        ).classes("text-xs text-gray-600")

        async def do_tag() -> None:
            sess = session_select.value
            if not sess:
                ui.notify("Pick a session first", type="warning")
                return
            ui.notify(
                f"Hashing images in '{sess}'… (~600 images/min on SSD)",
                type="info",
            )
            summary = await run.io_bound(tag_session, sess)
            ui.notify(
                f"Tagged {summary['tagged']} of {summary['images']} images "
                f"({summary['wolf']} wolf, {summary['no_wolf']} no-wolf).",
                type="positive",
            )
            _render_results()

        with ui.row().classes("gap-2 mt-2"):
            ui.button(
                "Tag from Lissl",
                icon="fingerprint",
                on_click=do_tag,
            ).props("color=primary")
            ui.button("Refresh", icon="refresh", on_click=_render_results).props(
                "flat"
            )

        session_select.on_value_change(lambda _e: _render_results())
        # Initial render covers "no session" state.
        _render_results()
