"""
Fine-tune YOLO11l on drone imagery while preserving MegaDetector knowledge.

Classes: 0=animal, 1=person, 2=vehicle  (MegaDetector convention)

Three-phase training strategy:
  Phase 1  – Freeze entire backbone, train detection head only
  Phase 2  – Progressive unfreeze (later backbone stages open)
  Phase 3  – Full fine-tune at very low LR

Usage:
    python finetune_yolo11_megadetector.py \
        --weights path/to/megadetector_yolo11l.pt \
        --data    path/to/dataset.yaml \
        --project runs/md_drone \
        --device  0

dataset.yaml must follow Ultralytics format:
    path: /data/drone_dataset
    train: images/train
    val:   images/val
    nc: 3
    names: [animal, person, vehicle]
"""

from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class PhaseConfig:
    name: str
    freeze: int          # number of layers to freeze from the start
    epochs: int
    lr0: float           # initial LR
    lrf: float           # final LR as fraction of lr0 (cosine decay)
    warmup_epochs: float


@dataclass
class TrainConfig:
    weights: str
    data: str
    project: str         = "runs/md_drone"
    device: str          = "0"
    imgsz: int           = 1280          # drone images benefit from high res
    batch: int           = 8
    workers: int         = 8
    patience: int        = 20            # early-stopping patience per phase
    save_period: int     = 5
    seed: int            = 42
    amp: bool            = True
    cos_lr: bool         = True
    # Augmentation — keep strong to regularise against camera-trap→drone shift
    mosaic: float        = 1.0
    mixup: float         = 0.15
    copy_paste: float    = 0.1
    degrees: float       = 15.0          # rotation: drones have variable gimbal yaw
    perspective: float   = 0.0005
    flipud: float        = 0.5           # nadir imagery: up/down symmetry is valid
    fliplr: float        = 0.5
    hsv_h: float         = 0.015
    hsv_s: float         = 0.7
    hsv_v: float         = 0.4

    phases: list[PhaseConfig] = field(default_factory=lambda: [
        PhaseConfig(
            name="phase1_head_only",
            freeze=10,          # freeze all backbone layers (0–9)
            epochs=10,
            lr0=1e-3,
            lrf=0.1,
            warmup_epochs=3.0,
        ),
        PhaseConfig(
            name="phase2_partial_unfreeze",
            freeze=6,           # unfreeze last 4 backbone stages (layers 6–9)
            epochs=20,
            lr0=1e-4,
            lrf=0.1,
            warmup_epochs=1.0,
        ),
        PhaseConfig(
            name="phase3_full_finetune",
            freeze=0,           # train everything
            epochs=20,
            lr0=1e-5,
            lrf=0.1,
            warmup_epochs=1.0,
        ),
    ])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def best_weights(run_dir: Path) -> Path:
    """Return the best.pt from a completed training run directory."""
    candidate = run_dir / "weights" / "best.pt"
    if not candidate.exists():
        # Fall back to last.pt if best.pt wasn't written (e.g. early stop epoch 0)
        candidate = run_dir / "weights" / "last.pt"
    if not candidate.exists():
        raise FileNotFoundError(f"No weights found in {run_dir}")
    return candidate


def build_train_kwargs(cfg: TrainConfig, phase: PhaseConfig, weights: str, name: str) -> dict:
    """Assemble keyword arguments for model.train()."""
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
        patience=cfg.patience,
        save_period=cfg.save_period,
        seed=cfg.seed,
        amp=cfg.amp,
        cos_lr=cfg.cos_lr,
        # Augmentation
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
        # Always resume=False between phases — we load new weights explicitly
        resume=False,
        exist_ok=False,
        verbose=True,
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def run_training(cfg: TrainConfig) -> None:
    project_dir = Path(cfg.project)
    project_dir.mkdir(parents=True, exist_ok=True)

    current_weights = cfg.weights
    run_dirs: list[Path] = []

    for i, phase in enumerate(cfg.phases):
        log.info("=" * 60)
        log.info(f"Starting {phase.name}  (freeze={phase.freeze}, "
                 f"lr0={phase.lr0}, epochs={phase.epochs})")
        log.info(f"  Weights: {current_weights}")
        log.info("=" * 60)

        model = YOLO(current_weights)

        run_name = f"{i+1:02d}_{phase.name}"
        kwargs = build_train_kwargs(cfg, phase, current_weights, run_name)

        results = model.train(**kwargs)

        # Locate the run directory Ultralytics created
        run_dir = project_dir / run_name
        if not run_dir.exists():
            # Ultralytics may append a number suffix if name existed
            candidates = sorted(project_dir.glob(f"{run_name}*"), reverse=True)
            run_dir = candidates[0] if candidates else project_dir / run_name

        run_dirs.append(run_dir)
        log.info(f"Phase complete. Run dir: {run_dir}")

        # Hand best weights to the next phase
        current_weights = str(best_weights(run_dir))
        log.info(f"Best weights for next phase: {current_weights}")

    # ------------------------------------------------------------------
    # Copy final best weights to a clearly named location
    # ------------------------------------------------------------------
    final_best = Path(current_weights)
    final_out = project_dir / "md_drone_final_best.pt"
    shutil.copy2(final_best, final_out)

    log.info("=" * 60)
    log.info("Training complete!")
    log.info(f"Final best weights → {final_out}")
    log.info("Phase run directories:")
    for d in run_dirs:
        log.info(f"  {d}")
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # Quick validation on final model
    # ------------------------------------------------------------------
    log.info("Running final validation …")
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
    log.info(f"mAP50     : {val_results.box.map50:.4f}")
    log.info(f"mAP50-95  : {val_results.box.map:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3-phase YOLO11l fine-tuning for drone imagery (MegaDetector classes)"
    )
    p.add_argument("--weights", required=True,
                   help="Path to pretrained MegaDetector YOLO11l .pt weights")
    p.add_argument("--data", required=True,
                   help="Path to dataset YAML (nc=3, names=[animal,person,vehicle])")
    p.add_argument("--project", default="runs/md_drone",
                   help="Root directory for all run outputs")
    p.add_argument("--device", default="0",
                   help="CUDA device index or 'cpu'")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="Input image size (default 1280 for drone imagery)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    # Per-phase epoch overrides (optional)
    p.add_argument("--epochs-p1", type=int, default=None,
                   help="Override Phase 1 epoch count")
    p.add_argument("--epochs-p2", type=int, default=None,
                   help="Override Phase 2 epoch count")
    p.add_argument("--epochs-p3", type=int, default=None,
                   help="Override Phase 3 epoch count")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = TrainConfig(
        weights=args.weights,
        data=args.data,
        project=args.project,
        device=args.device,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
    )

    # Apply any CLI epoch overrides
    overrides = [args.epochs_p1, args.epochs_p2, args.epochs_p3]
    for phase, override in zip(cfg.phases, overrides):
        if override is not None:
            log.info(f"Overriding {phase.name} epochs: {phase.epochs} → {override}")
            phase.epochs = override

    log.info("Training configuration:")
    log.info(f"  Weights : {cfg.weights}")
    log.info(f"  Data    : {cfg.data}")
    log.info(f"  Project : {cfg.project}")
    log.info(f"  Device  : {cfg.device}")
    log.info(f"  ImgSz   : {cfg.imgsz}")
    log.info(f"  Batch   : {cfg.batch}")
    for phase in cfg.phases:
        log.info(f"  {phase.name}: freeze={phase.freeze}, "
                 f"lr0={phase.lr0}, epochs={phase.epochs}")

    run_training(cfg)


if __name__ == "__main__":
    main()