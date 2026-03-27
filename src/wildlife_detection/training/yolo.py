"""YOLO training utilities for wildlife detection."""

import os
import tempfile
from pathlib import Path

import pandas as pd
import wandb
import yaml
from omegaconf import OmegaConf
from ultralytics import YOLO


def setup_yolo_splits(data_dir, split_manifest, tmpdir):
    """Create symlinked split directories in Ultralytics format.

    Parameters
    ----------
    data_dir : str or Path
        Root data directory containing tiles/ and annotations_yolo/.
    split_manifest : str or Path
        CSV with columns: tile_filename, split (train/val/test).
    tmpdir : str or Path
        Temporary directory for symlinked splits.

    Returns
    -------
    pathlib.Path
        Path to the temporary dataset root.
    """
    data_dir = Path(data_dir)
    tiles_dir = data_dir / "tiles"
    yolo_dir = data_dir / "annotations_yolo"
    split_df = pd.read_csv(split_manifest)

    for split in ["train", "val", "test"]:
        img_dir = Path(tmpdir) / "images" / split
        lbl_dir = Path(tmpdir) / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        subset = split_df[split_df["split"] == split]
        for _, row in subset.iterrows():
            fname = row["tile_filename"]
            src_img = tiles_dir / fname
            src_lbl = yolo_dir / fname.replace(".jpg", ".txt")

            if src_img.exists():
                os.symlink(src_img.resolve(), img_dir / fname)
            if src_lbl.exists():
                os.symlink(src_lbl.resolve(), lbl_dir / fname.replace(".jpg", ".txt"))

    return Path(tmpdir)


def count_class_distribution(data_root):
    """Count annotations per class across training label files.

    Parameters
    ----------
    data_root : str or Path
        Dataset root containing labels/train/.

    Returns
    -------
    tuple of (dict, int, int)
        (class_counts, total_files, empty_files)
    """
    counts = {}
    labels_dir = Path(data_root) / "labels" / "train"
    total_files = 0
    empty_files = 0
    for txt_file in labels_dir.glob("*.txt"):
        total_files += 1
        content = txt_file.read_text().strip()
        if not content:
            empty_files += 1
            continue
        for line in content.split("\n"):
            cls = int(line.split()[0])
            counts[cls] = counts.get(cls, 0) + 1
    return counts, total_files, empty_files


def train_yolo(cfg):
    """Run YOLO training with Weights & Biases logging.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Configuration with keys: data_dir, split_manifest, model, imgsz,
        epochs, batch, lr0, wandb.{project, entity, tags}.
    """
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=list(cfg.wandb.tags),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        data_root = setup_yolo_splits(cfg.data_dir, cfg.split_manifest, tmpdir)

        class_counts, total_tiles, empty_tiles = count_class_distribution(data_root)
        nc = max(class_counts.keys()) + 1 if class_counts else 1
        names = {i: f"class_{i}" for i in range(nc)}

        dataset_yaml = {
            "path": str(data_root),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": nc,
            "names": names,
        }

        dataset_yaml_path = data_root / "dataset.yaml"
        with open(dataset_yaml_path, "w") as f:
            yaml.dump(dataset_yaml, f)

        wandb.log({
            "data/total_train_tiles": total_tiles,
            "data/empty_train_tiles": empty_tiles,
            "data/empty_tile_ratio": empty_tiles / max(total_tiles, 1),
        })

        output_project = str(Path(cfg.data_dir) / "runs")
        model = YOLO(cfg.model)
        model.train(
            data=str(dataset_yaml_path),
            imgsz=cfg.imgsz,
            epochs=cfg.epochs,
            batch=cfg.batch,
            lr0=cfg.lr0,
            project=output_project,
            name="train",
        )

        # Log test predictions
        test_img_dir = data_root / "images" / "test"
        test_images = list(test_img_dir.glob("*.jpg"))[:20]

        if test_images:
            results = model.predict(source=test_images, save=False)
            table = wandb.Table(columns=["image", "predictions", "num_detections"])
            for img_path, result in zip(test_images, results):
                boxes = result.boxes
                n_det = len(boxes) if boxes is not None else 0
                table.add_data(
                    wandb.Image(str(img_path)),
                    wandb.Image(result.plot()),
                    n_det,
                )
            wandb.log({"test/predictions": table})

    wandb.finish()
    print("Training complete.")
