"""Evaluate a trained YOLO model on the Eikelboom test set.

Remaps the original Eikelboom species labels (Elephant=0, Giraffe=1, Zebra=2)
to MegaDetector class 0 (animal) before evaluation.
"""

import os
import sys
from pathlib import Path


def setup_eval_dataset(eikelboom_src: Path, eval_dir: Path):
    """Create a copy of Eikelboom test set with labels remapped to animal=0.

    Parameters
    ----------
    eikelboom_src : Path
        Path to the original eikelboom_yolo_tiled directory.
    eval_dir : Path
        Destination for the remapped evaluation dataset.
    """
    print("Setting up Eikelboom evaluation dataset with MegaDetector classes...")

    (eval_dir / "images" / "test").mkdir(parents=True, exist_ok=True)
    (eval_dir / "labels" / "test").mkdir(parents=True, exist_ok=True)

    # Symlink images
    src_img_dir = eikelboom_src / "images" / "test"
    dst_img_dir = eval_dir / "images" / "test"
    count = 0
    for img in sorted(src_img_dir.glob("*.jpg")):
        dst = dst_img_dir / img.name
        if not dst.exists():
            try:
                os.symlink(img.resolve(), dst)
            except OSError:
                import shutil
                shutil.copy2(str(img.resolve()), str(dst))
        count += 1

    # Remap labels
    src_lbl_dir = eikelboom_src / "labels" / "test"
    dst_lbl_dir = eval_dir / "labels" / "test"
    total_boxes = 0
    for lbl in sorted(src_lbl_dir.glob("*.txt")):
        lines = []
        for line in lbl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            lines.append(f"0 {' '.join(parts[1:])}")
            total_boxes += 1
        (dst_lbl_dir / lbl.name).write_text("\n".join(lines))

    # Also create train/val references (required by ultralytics)
    for split in ["train", "val"]:
        src_split_img = eikelboom_src / "images" / split
        src_split_lbl = eikelboom_src / "labels" / split
        dst_split_img = eval_dir / "images" / split
        dst_split_lbl = eval_dir / "labels" / split
        dst_split_img.mkdir(parents=True, exist_ok=True)
        dst_split_lbl.mkdir(parents=True, exist_ok=True)

        for img in sorted(src_split_img.glob("*.jpg")):
            dst = dst_split_img / img.name
            if not dst.exists():
                try:
                    os.symlink(img.resolve(), dst)
                except OSError:
                    import shutil
                    shutil.copy2(str(img.resolve()), str(dst))

        for lbl in sorted(src_split_lbl.glob("*.txt")):
            lines = []
            for line in lbl.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                lines.append(f"0 {' '.join(parts[1:])}")
            (dst_split_lbl / lbl.name).write_text("\n".join(lines))

    yaml_content = f"""path: {eval_dir}
train: images/train
val: images/val
test: images/test

nc: 2
names: ['animal', 'person']
"""
    (eval_dir / "dataset.yaml").write_text(yaml_content)

    print(f"  Test images: {count}")
    print(f"  Test boxes (remapped to animal=0): {total_boxes}")
    print(f"  Dataset YAML: {eval_dir / 'dataset.yaml'}")


def evaluate(weights_path: str, eval_dir: Path, eikelboom_src: Path = None,
             conf: float = 0.25, iou: float = 0.5):
    """Run evaluation on Eikelboom test set.

    Parameters
    ----------
    weights_path : str
        Path to trained model weights.
    eval_dir : Path
        Path to the remapped evaluation dataset.
    eikelboom_src : Path, optional
        Original eikelboom_yolo_tiled path (used to create eval dataset if missing).
    conf : float
        Confidence threshold.
    iou : float
        IoU threshold for NMS.

    Returns
    -------
    metrics
        Ultralytics validation metrics.
    """
    from ultralytics import YOLO

    dataset_yaml = eval_dir / "dataset.yaml"
    if not dataset_yaml.exists():
        if eikelboom_src is None:
            print("Error: Eval dataset not found and eikelboom_src not provided.")
            sys.exit(1)
        setup_eval_dataset(eikelboom_src, eval_dir)

    if not Path(weights_path).exists():
        print(f"Error: Weights not found at {weights_path}")
        sys.exit(1)

    print(f"\nEvaluating: {weights_path}")
    print(f"Dataset: {dataset_yaml}")
    print(f"Conf: {conf}, IoU: {iou}")
    print()

    model = YOLO(weights_path)
    metrics = model.val(
        data=str(dataset_yaml),
        split="test",
        imgsz=640,
        conf=conf,
        iou=iou,
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Eikelboom Test Set Results")
    print("=" * 60)
    print(f"  mAP@50:      {metrics.box.map50:.4f}")
    print(f"  mAP@50:95:   {metrics.box.map:.4f}")
    print(f"  Precision:    {metrics.box.mp:.4f}")
    print(f"  Recall:       {metrics.box.mr:.4f}")
    print()
    print("Baseline comparison:")
    print("  Eikelboom (2019) RetinaNet: mAP ~0.77")
    print("  May et al. (2025) YOLOv8: competitive on oblique imagery")

    return metrics
