"""Fine-tune YOLO or RT-DETR models on wildlife detection datasets.

Supports MegaDetector variants (larch, sorrel, rtdetr-c) and any ultralytics-
compatible .pt weights. Automatically detects RT-DETR models and applies the
required settings (amp=False, lower LR).
"""

import sys
from pathlib import Path


def _is_rtdetr(weights: str) -> bool:
    """Check if weights are an RT-DETR model."""
    name = Path(weights).stem.lower()
    return "rtdetr" in name


def train_combined(data: str, weights: str, epochs: int = 50, batch: int = 16,
                   imgsz: int = 640, freeze: int = 10, lr0: float = None,
                   patience: int = 10, device: str = "0",
                   project: str = "wildlife-detection",
                   name: str = "larch-freeze10-combined",
                   warmup_epochs: float = 3.0,
                   weight_decay: float = 0.0005):
    """Train YOLO or RT-DETR on a detection dataset.

    Automatically handles RT-DETR-specific settings (amp=False, lower LR).

    Parameters
    ----------
    data : str
        Path to dataset.yaml.
    weights : str
        Path to pretrained .pt weights.
    epochs, batch, imgsz, freeze, lr0, patience : training hyperparameters.
    device : str
        CUDA device id.
    project, name : str
        Output directory and WandB project name.
    warmup_epochs : float
        Number of warmup epochs.
    weight_decay : float
        Weight decay for optimizer.

    Returns
    -------
    results
        Ultralytics training results object.
    """
    import os
    os.environ["WANDB_PROJECT"] = project

    from ultralytics import YOLO

    if not Path(weights).exists():
        print(f"Error: Weights not found at {weights}")
        sys.exit(1)

    if not Path(data).exists():
        print(f"Error: Dataset YAML not found at {data}")
        sys.exit(1)

    is_rtdetr = _is_rtdetr(weights)

    # Default LR: 0.0001 for RT-DETR, 0.001 for YOLO
    if lr0 is None:
        lr0 = 0.0001 if is_rtdetr else 0.001

    if is_rtdetr:
        from ultralytics import RTDETR
        model = RTDETR(weights)
    else:
        model = YOLO(weights)

    # Inject F1 metric so WandB logs it alongside P/R/mAP
    def _inject_f1(trainer):
        p = trainer.metrics.get("metrics/precision(B)", 0)
        r = trainer.metrics.get("metrics/recall(B)", 0)
        trainer.metrics["metrics/F1(B)"] = 2 * p * r / max(p + r, 1e-8)

    model.add_callback("on_fit_epoch_end", _inject_f1)

    print(f"Model weights: {weights}")
    print(f"Model type: {'RT-DETR' if is_rtdetr else 'YOLO'}")
    print(f"Dataset: {data}")
    print(f"Epochs: {epochs}, Batch: {batch}, ImgSz: {imgsz}")
    print(f"Freeze: {freeze} layers, LR0: {lr0}, Warmup: {warmup_epochs}")
    print(f"Project: {project}/{name}")
    print()

    train_kwargs = dict(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        lr0=lr0,
        freeze=freeze,
        patience=patience,
        project=project,
        name=name,
        warmup_epochs=warmup_epochs,
        weight_decay=weight_decay,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        scale=0.5,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    if is_rtdetr:
        # RT-DETR: AMP causes NaN in bipartite matching, deterministic breaks grid_sample
        train_kwargs["amp"] = False
        train_kwargs["deterministic"] = False
        # Don't set half for RT-DETR
    else:
        train_kwargs["half"] = True

    results = model.train(**train_kwargs)

    print("\nTraining complete.")
    print(f"Best weights: {project}/{name}/weights/best.pt")
    return results


def resume_training(last_pt: str):
    """Resume an interrupted training run."""
    from ultralytics import YOLO

    print(f"Resuming training from {last_pt}")
    model = YOLO(last_pt)
    return model.train(resume=True)
