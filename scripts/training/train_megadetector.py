#!/usr/bin/env python
"""
Fine-tune any MegaDetector model (v5, v6) on custom YOLO-format datasets.

All MegaDetector ultralytics models (.pt format) are supported:
  MDV6-yolov9-c, MDV6-yolov9-e, MDV6-yolov10-c, MDV6-yolov10-e, MDV6-rtdetr-c
  MDv5a, MDv5b

Dataset must be in YOLO format with dataset.yaml. Convert from COCO if needed:
    python scripts/training/convert_dataset.py --from coco --to yolo --src <coco_dir> --dst <yolo_dir>

Usage:
    # Fine-tune MDV6 RT-DETR on Eikelboom data
    python scripts/training/train_megadetector.py \
        --model MDV6-rtdetr-c \
        --data ./week1/data/eikelboom_yolo_tiled/dataset.yaml \
        --epochs 50 --batch 8 --imgsz 640

    # Fine-tune MDV6 YOLOv9 compact
    python scripts/training/train_megadetector.py \
        --model MDV6-yolov9-c \
        --data ./week1/data/eikelboom_yolo_tiled/dataset.yaml \
        --epochs 50 --batch 16 --imgsz 640

    # Fine-tune from local weights (e.g. previous run)
    python scripts/training/train_megadetector.py \
        --model ./output/mdv6_finetune/weights/best.pt \
        --data ./week1/data/eikelboom_yolo_tiled/dataset.yaml \
        --epochs 30

    # Evaluate only (no training)
    python scripts/training/train_megadetector.py \
        --model ./output/mdv6_finetune/weights/best.pt \
        --data ./week1/data/eikelboom_yolo_tiled/dataset.yaml \
        --eval_only
"""

import argparse
import os
import sys
from pathlib import Path

import torch


# ─── Known MegaDetector weights ─────────────────────────────────────────────

MEGADETECTOR_MODELS = {
    # MDV6 ultralytics (AGPL)
    "MDV6-rtdetr-c": {
        "url": "https://zenodo.org/records/15398270/files/MDV6-rtdetr-c.pt?download=1",
        "filename": "MDV6-rtdetr-c.pt",
        "arch": "rtdetr",
        "native_imgsz": 1280,
    },
    "MDV6-yolov9-c": {
        "url": "https://zenodo.org/records/15398270/files/MDV6-yolov9-c.pt?download=1",
        "filename": "MDV6-yolov9-c.pt",
        "arch": "yolo",
        "native_imgsz": 1280,
    },
    "MDV6-yolov9-e": {
        "url": "https://zenodo.org/records/15398270/files/MDV6-yolov9-e-1280.pt?download=1",
        "filename": "MDV6-yolov9-e-1280.pt",
        "arch": "yolo",
        "native_imgsz": 1280,
    },
    "MDV6-yolov10-c": {
        "url": "https://zenodo.org/records/15398270/files/MDV6-yolov10-c.pt?download=1",
        "filename": "MDV6-yolov10-c.pt",
        "arch": "yolo",
        "native_imgsz": 1280,
    },
    "MDV6-yolov10-e": {
        "url": "https://zenodo.org/records/15398270/files/MDV6-yolov10-e-1280.pt?download=1",
        "filename": "MDV6-yolov10-e-1280.pt",
        "arch": "yolo",
        "native_imgsz": 1280,
    },
    # MD v1000 (agentmorris/MegaDetector, YOLO11)
    "MD1000-cedar": {
        "url": "https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/md_v1000.0.0-cedar.pt",
        "filename": "md_v1000.0.0-cedar.pt",
        "arch": "yolo",
        "native_imgsz": 640,
    },
    "MD1000-larch": {
        "url": "https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/md_v1000.0.0-larch.pt",
        "filename": "md_v1000.0.0-larch.pt",
        "arch": "yolo",
        "native_imgsz": 640,
    },
    "MD1000-redwood": {
        "url": "https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/md_v1000.0.0-redwood.pt",
        "filename": "md_v1000.0.0-redwood.pt",
        "arch": "yolo",
        "native_imgsz": 640,
    },
    "MD1000-sorrel": {
        "url": "https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/md_v1000.0.0-sorrel.pt",
        "filename": "md_v1000.0.0-sorrel.pt",
        "arch": "yolo",
        "native_imgsz": 640,
    },
    "MD1000-spruce": {
        "url": "https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/md_v1000.0.0-spruce.pt",
        "filename": "md_v1000.0.0-spruce.pt",
        "arch": "yolo",
        "native_imgsz": 640,
    },
    # MDv5 (YOLOv5)
    "MDv5a": {
        "url": "https://zenodo.org/records/13357337/files/md_v5a.0.0.pt?download=1",
        "filename": "md_v5a.0.0.pt",
        "arch": "yolo",
        "native_imgsz": 1280,
    },
    "MDv5b": {
        "url": "https://zenodo.org/records/10023414/files/MegaDetector_v5b.0.0.pt?download=1",
        "filename": "MegaDetector_v5b.0.0.pt",
        "arch": "yolo",
        "native_imgsz": 1280,
    },
}


def resolve_weights(model_name: str) -> str:
    """Resolve model name to local weights path, downloading if needed."""
    # If it's a local path, use directly
    if os.path.exists(model_name):
        return model_name

    if model_name not in MEGADETECTOR_MODELS:
        available = ", ".join(sorted(MEGADETECTOR_MODELS.keys()))
        print(f"Unknown model: {model_name}")
        print(f"Available: {available}")
        print(f"Or provide a local .pt file path")
        sys.exit(1)

    info = MEGADETECTOR_MODELS[model_name]
    cache_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    weights_path = os.path.join(cache_dir, info["filename"])

    if not os.path.exists(weights_path):
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Downloading {model_name} from Zenodo...")
        import wget
        wget.download(info["url"], out=weights_path)
        print()

    return weights_path


def is_rtdetr(model_name: str, weights_path: str) -> bool:
    """Check if model is RT-DETR (needs special training settings)."""
    if model_name in MEGADETECTOR_MODELS:
        return MEGADETECTOR_MODELS[model_name]["arch"] == "rtdetr"
    # Check by loading
    from ultralytics import YOLO
    model = YOLO(weights_path)
    return "rtdetr" in type(model.model).__name__.lower()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    available = ", ".join(sorted(MEGADETECTOR_MODELS.keys()))
    p = argparse.ArgumentParser(
        description="Fine-tune MegaDetector models on custom datasets",
        epilog=f"Available models: {available}",
    )
    p.add_argument("--model", required=True,
                   help="Model name (e.g. MDV6-rtdetr-c) or path to .pt file")
    p.add_argument("--data", required=True,
                   help="Path to dataset.yaml (YOLO format)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640,
                   help="Image size (default: 640, MDV6 native: 1280)")
    p.add_argument("--device", default="0",
                   help="CUDA device (0, 1, cpu)")
    p.add_argument("--lr0", type=float, default=None,
                   help="Initial learning rate (default: auto)")
    p.add_argument("--weight_decay", type=float, default=0.0001)
    p.add_argument("--freeze", type=int, default=None,
                   help="Freeze first N layers (prevents early F1 dip)")
    p.add_argument("--warmup_epochs", type=float, default=None,
                   help="LR warmup epochs (default: 3, increase to 5-10 if early dip)")
    p.add_argument("--patience", type=int, default=None,
                   help="Early stopping patience (epochs without improvement)")
    p.add_argument("--project", default="output")
    p.add_argument("--name", default=None,
                   help="Run name (default: auto from model name)")
    p.add_argument("--resume", action="store_true",
                   help="Resume training from last checkpoint")
    p.add_argument("--eval_only", action="store_true",
                   help="Only run validation, no training")
    p.add_argument("--log", nargs="+", choices=["wandb", "tensorboard"],
                   default=[], help="Logging backends (optional)")
    p.add_argument("--wandb_project", default="megadetector-finetune")
    p.add_argument("--wandb_run", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    weights_path = resolve_weights(args.model)
    rtdetr = is_rtdetr(args.model, weights_path)

    # Auto-set run name
    if args.name is None:
        model_stem = Path(weights_path).stem
        args.name = f"{model_stem}_finetune"

    print(f"Model: {args.model}")
    print(f"Weights: {weights_path}")
    print(f"Architecture: {'RT-DETR' if rtdetr else 'YOLO'}")
    print(f"Dataset: {args.data}")
    print(f"Image size: {args.imgsz}")

    # Load model
    from ultralytics import YOLO, RTDETR
    if rtdetr:
        model = RTDETR(weights_path)
    else:
        model = YOLO(weights_path)

    print(f"Classes: {model.names}")
    model.info()

    if args.eval_only:
        print("\nRunning validation only...")
        metrics = model.val(
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            split="val",
        )
        print(f"\nmAP@0.5:     {metrics.box.map50:.4f}")
        print(f"mAP@0.5:0.95:{metrics.box.map:.4f}")
        return

    # Build training kwargs
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        weight_decay=args.weight_decay,
        project=args.project,
        name=args.name,
        exist_ok=True,
        verbose=True,
        deterministic=False,  # Required for RT-DETR F.grid_sample
    )

    # RT-DETR specific settings
    if rtdetr:
        train_kwargs["amp"] = False  # AMP causes NaN with RT-DETR
        if args.lr0 is not None:
            train_kwargs["lr0"] = args.lr0
        # Let ultralytics auto-select optimizer for RT-DETR
        print("\nRT-DETR: amp=False (stability), optimizer=auto")
    else:
        if args.lr0 is not None:
            train_kwargs["lr0"] = args.lr0

    if args.freeze is not None:
        train_kwargs["freeze"] = args.freeze

    if args.warmup_epochs is not None:
        train_kwargs["warmup_epochs"] = args.warmup_epochs

    if args.patience is not None:
        train_kwargs["patience"] = args.patience

    if args.resume:
        train_kwargs["resume"] = True

    # Logging — WandB
    if "wandb" in args.log:
        import wandb as wb

        # Init wandb BEFORE ultralytics does — ultralytics checks `if not wb.run`
        # and reuses our run. This lets us control the project name independently
        # of the local --project directory.
        run_name = args.wandb_run or args.name
        wb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model": args.model,
                "epochs": args.epochs,
                "batch": args.batch,
                "imgsz": args.imgsz,
                "lr0": args.lr0,
                "weight_decay": args.weight_decay,
                "freeze": args.freeze,
                "architecture": "RT-DETR" if rtdetr else "YOLO",
            },
        )

        # Inject F1 into trainer.metrics so ultralytics' own wandb callback logs it
        # alongside P, R, mAP in the same wb.run.log() call
        def inject_f1_callback(trainer):
            try:
                p = trainer.metrics.get("metrics/precision(B)", 0)
                r = trainer.metrics.get("metrics/recall(B)", 0)
                f1 = 2 * p * r / max(p + r, 1e-8)
                trainer.metrics["metrics/F1(B)"] = f1
            except Exception:
                pass

        model.add_callback("on_fit_epoch_end", inject_f1_callback)
        print(f"WandB: project={args.wandb_project}, run={run_name}")
        print("  Logging: P, R, F1, mAP@0.5, mAP@0.5:0.95, train/val losses")

    if "tensorboard" in args.log:
        train_kwargs["tensorboard"] = True

    print(f"\nStarting training: {args.epochs} epochs, batch={args.batch}, imgsz={args.imgsz}")
    results = model.train(**train_kwargs)

    # Print final metrics including F1
    if hasattr(results, "results_dict"):
        m = results.results_dict
        p = m.get("metrics/precision(B)", 0)
        r = m.get("metrics/recall(B)", 0)
        f1 = 2 * p * r / max(p + r, 1e-8)
        print(f"\nFinal metrics:")
        print(f"  Precision: {p:.4f}")
        print(f"  Recall:    {r:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  mAP@0.5:   {m.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  mAP@.5:.95:{m.get('metrics/mAP50-95(B)', 0):.4f}")

    print(f"\nTraining complete!")
    print(f"Best weights: {args.project}/{args.name}/weights/best.pt")

    if "wandb" in args.log:
        try:
            import wandb as wb
            if wb.run is not None:
                wb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
