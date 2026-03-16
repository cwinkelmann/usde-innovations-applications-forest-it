"""
upload_eikelboom.py — Upload Eikelboom2019 aerial wildlife dataset to HuggingFace
==================================================================================

Uploads the Eikelboom et al. (2019) drone survey dataset to karisu/Eikelboom2019.

Dataset structure (local):
    <data_dir>/
        train/          393 .JPG tiles
        val/             56 .JPG tiles
        test/           112 .JPG tiles
        annotations_train.csv   8 981 rows
        annotations_val.csv       451 rows
        annotations_test.csv      849 rows
        annotations_images.csv  4 306 rows (all raw images, not splits)
        resnet50_csv_36.h5      pre-trained RetinaNet weights
        readme.txt

Annotation format (no header):
    filename, x1, y1, x2, y2, class
    (Pascal VOC pixel bounding boxes; classes: Elephant, Zebra, Giraffe)

HuggingFace repo:  karisu/Eikelboom2019  (dataset)

Usage
-----
    python upload_eikelboom.py --data-dir /Users/christian/Downloads/data
    python upload_eikelboom.py --data-dir /Users/christian/Downloads/data --dry-run

Authentication
--------------
Run `huggingface-cli login` once before uploading. No token argument needed.
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "karisu/Eikelboom2019"
REPO_TYPE = "dataset"

SPLITS = ["train", "val", "test"]
ANNOTATION_FILES = {
    "train": "annotations_train.csv",
    "val": "annotations_val.csv",
    "test": "annotations_test.csv",
    "images": "annotations_images.csv",
}
EXTRA_FILES = [
    "readme.txt",
    "resnet50_csv_36.h5",
]

DATASET_CARD = """\
---
license: cc-by-4.0
task_categories:
  - object-detection
tags:
  - wildlife
  - aerial
  - drone
  - retinanet
  - bounding-boxes
pretty_name: Eikelboom 2019 Aerial Wildlife Detection
size_categories:
  - 100K<n<1M
---

# Eikelboom 2019 — Aerial Wildlife Detection

Aerial drone survey dataset for wildlife detection from:

> Eikelboom, J. A. J., Wind, J., van de Ven, E., Kenana, L. M., Schroder, B.,
> de Knegt, H. J., van Langevelde, F., & Prins, H. H. T. (2019).
> Improving the precision and accuracy of animal population estimates with
> aerial image object detection. *Methods in Ecology and Evolution*, 10(11), 1875–1887.
> https://doi.org/10.1111/2041-210X.13277

Dataset archived at: https://data.4tu.nl/articles/_/12713903/1

## Contents

| Split | Images | Annotations |
|-------|-------:|------------:|
| train |    393 |       8 981 |
| val   |     56 |         451 |
| test  |    112 |         849 |

**Species:** Elephant, Zebra, Giraffe

## Annotation Format

CSV files with no header row:

```
filename, x1, y1, x2, y2, class
```

Coordinates are pixel-level bounding boxes in Pascal VOC format (x1, y1 = top-left corner;
x2, y2 = bottom-right corner).

## Model Weights

`resnet50_csv_36.h5` — best-performing RetinaNet model from the paper (Keras/TensorFlow).
Trained with [keras-retinanet 0.5.0](https://github.com/fizyr/keras-retinanet).

## License

Creative Commons Attribution 4.0 International (CC BY 4.0)
"""


def upload_dataset(data_dir: Path, dry_run: bool) -> None:
    api = HfApi()

    # Create repo (no-op if it already exists)
    print(f"Creating/verifying repo: {REPO_ID}")
    if not dry_run:
        create_repo(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            exist_ok=True,
        )

    # Upload dataset card
    card_path = data_dir / "_README.md"
    card_path.write_text(DATASET_CARD)
    print("Uploading dataset card (README.md)...")
    if not dry_run:
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )
    card_path.unlink()

    # Upload annotation CSVs
    for key, fname in ANNOTATION_FILES.items():
        local = data_dir / fname
        if not local.exists():
            print(f"  WARNING: {fname} not found, skipping")
            continue
        repo_path = f"annotations/{fname}"
        print(f"Uploading {fname} → {repo_path}")
        if not dry_run:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
            )

    # Upload extra files (readme, model weights)
    for fname in EXTRA_FILES:
        local = data_dir / fname
        if not local.exists():
            print(f"  WARNING: {fname} not found, skipping")
            continue
        print(f"Uploading {fname}")
        if not dry_run:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=fname,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
            )

    # Upload images per split
    for split in SPLITS:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"  WARNING: {split}/ directory not found, skipping")
            continue

        images = sorted(
            p for p in split_dir.iterdir() if p.suffix.upper() == ".JPG"
        )
        print(f"\nUploading {split}/ — {len(images)} images...")

        if dry_run:
            print(f"  [dry-run] would upload {len(images)} files to {split}/")
            continue

        api.upload_folder(
            folder_path=str(split_dir),
            path_in_repo=split,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            allow_patterns=["*.JPG", "*.jpg", "*.jpeg", "*.JPEG"],
        )
        print(f"  Done — {split}/")

    print("\n" + "=" * 50)
    print(f"Upload complete: https://huggingface.co/datasets/{REPO_ID}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload Eikelboom2019 dataset to HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/Users/christian/Downloads/data"),
        help="Local path to the dataset root (default: /Users/christian/Downloads/data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without actually uploading",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"ERROR: data directory not found: {args.data_dir}")
        raise SystemExit(1)

    upload_dataset(
        data_dir=args.data_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
