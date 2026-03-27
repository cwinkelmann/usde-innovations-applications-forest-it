"""Three-phase YOLO fine-tuning for aerial wildlife detection.

Preserves MegaDetector backbone knowledge through progressive unfreezing:
  Phase 1 — Freeze backbone (layers 0-9), train detection head only
  Phase 2 — Unfreeze later backbone stages (freeze layers 0-5)
  Phase 3 — Full fine-tune at very low LR

Based on the strategy in doc/fine_tuning_yolo11.md and the evidence that
domain-specific fine-tuning closes a 20-50 point mAP gap on aerial imagery.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class PhaseConfig:
    name: str
    freeze: int
    epochs: int
    lr0: float
    lrf: float
    warmup_epochs: float


@dataclass
class TrainConfig:
    weights: str
    data: str
    project: str = "wildlife-detection"
    device: str = "0"
    imgsz: int = 640
    batch: int = 16
    workers: int = 8
    patience: int = 10
    save_period: int = 5
    seed: int = 0
    amp: bool = True
    cos_lr: bool = False
    lrf: float = 0.01
    weight_decay: float = 0.0001
    # Augmentation — match single-phase baseline for fair comparison
    mosaic: float = 1.0
    mixup: float = 0.1
    copy_paste: float = 0.1
    degrees: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    scale: float = 0.5

    phases: list[PhaseConfig] = field(default_factory=lambda: [
        PhaseConfig(
            name="phase1_head_only",
            freeze=10,
            epochs=15,
            lr0=1e-3,
            lrf=0.01,
            warmup_epochs=5.0,
        ),
        PhaseConfig(
            name="phase2_partial_unfreeze",
            freeze=6,
            epochs=20,
            lr0=1e-5,
            lrf=0.01,
            warmup_epochs=5.0,
        ),
        PhaseConfig(
            name="phase3_full_finetune",
            freeze=0,
            epochs=15,
            lr0=5e-6,
            lrf=0.01,
            warmup_epochs=3.0,
        ),
    ])


def _best_weights(run_dir: Path) -> Path:
    """Return best.pt from a completed run, falling back to last.pt."""
    for name in ["best.pt", "last.pt"]:
        candidate = run_dir / "weights" / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No weights found in {run_dir}")


def _build_train_kwargs(cfg: TrainConfig, phase: PhaseConfig, name: str) -> dict:
    return dict(
        data=cfg.data,
        epochs=phase.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        workers=cfg.workers,
        device=cfg.device,
        project=cfg.project,
        name=name,
        freeze=phase.freeze,
        lr0=phase.lr0,
        lrf=phase.lrf,
        warmup_epochs=phase.warmup_epochs,
        weight_decay=cfg.weight_decay,
        patience=cfg.patience,
        save_period=cfg.save_period,
        seed=cfg.seed,
        amp=cfg.amp,
        cos_lr=cfg.cos_lr,
        mosaic=cfg.mosaic,
        mixup=cfg.mixup,
        copy_paste=cfg.copy_paste,
        degrees=cfg.degrees,
        perspective=cfg.perspective,
        flipud=cfg.flipud,
        fliplr=cfg.fliplr,
        hsv_h=cfg.hsv_h,
        hsv_s=cfg.hsv_s,
        hsv_v=cfg.hsv_v,
        scale=cfg.scale,
        half=True,
        resume=False,
        exist_ok=False,
        verbose=True,
        plots=True,
    )


def run_phased_training(cfg: TrainConfig) -> Path:
    """Execute the 3-phase training pipeline.

    Returns the path to the final best weights.
    """
    import os
    os.environ["WANDB_PROJECT"] = cfg.project

    from ultralytics import YOLO

    current_weights = cfg.weights
    run_dirs: list[Path] = []

    for i, phase in enumerate(cfg.phases):
        log.info("=" * 60)
        log.info(f"Phase {i+1}/{len(cfg.phases)}: {phase.name}")
        log.info(f"  freeze={phase.freeze}, lr0={phase.lr0}, epochs={phase.epochs}")
        log.info(f"  Weights: {current_weights}")
        log.info("=" * 60)

        model = YOLO(current_weights)

        # Inject F1 metric so WandB logs it alongside P/R/mAP
        def _inject_f1(trainer):
            p = trainer.metrics.get("metrics/precision(B)", 0)
            r = trainer.metrics.get("metrics/recall(B)", 0)
            trainer.metrics["metrics/F1(B)"] = 2 * p * r / max(p + r, 1e-8)

        model.add_callback("on_fit_epoch_end", _inject_f1)

        run_name = f"{i+1:02d}_{phase.name}"
        kwargs = _build_train_kwargs(cfg, phase, run_name)
        results = model.train(**kwargs)

        # Get the actual save directory from the trainer (ultralytics prepends runs_dir)
        run_dir = Path(model.trainer.save_dir)
        log.info(f"Run directory: {run_dir}")

        run_dirs.append(run_dir)
        current_weights = str(_best_weights(run_dir))
        log.info(f"Phase {i+1} complete. Best weights: {current_weights}")

    # Copy final weights to a clear location next to the last run
    final_out = run_dirs[-1].parent / "phased_final_best.pt"
    shutil.copy2(current_weights, final_out)

    log.info("=" * 60)
    log.info("All phases complete!")
    log.info(f"Final best weights: {final_out}")
    for d in run_dirs:
        log.info(f"  {d}")
    log.info("=" * 60)

    # Final validation
    log.info("Running final validation...")
    final_model = YOLO(str(final_out))
    val_results = final_model.val(
        data=cfg.data,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        project=cfg.project,
        name="final_validation",
        verbose=True,
    )
    log.info(f"mAP@50:    {val_results.box.map50:.4f}")
    log.info(f"mAP@50:95: {val_results.box.map:.4f}")
    log.info(f"Precision: {val_results.box.mp:.4f}")
    log.info(f"Recall:    {val_results.box.mr:.4f}")

    return final_out
