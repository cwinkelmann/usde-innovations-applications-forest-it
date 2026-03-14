"""
download_data.py — Week 1 dataset setup
========================================
Downloads all datasets needed for Week 1 practicals into week1/data/.

Usage
-----
    python download_data.py --sample   # minimal set for quick testing (~500 MB)
    python download_data.py --full     # full teaching subsets (~5 GB)

After running, copy the iguana tile subset manually — see DATASETS.md section 2.
"""
import argparse
import io
import json
import os
import zipfile
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm

BASE_DIR = Path(__file__).parent  # week1/data/


def download_general_dataset(full: bool) -> None:
    """karisu/General_Dataset — HerdNet aerial dataset from HuggingFace."""
    out = BASE_DIR / "general_dataset"
    print("\n=== karisu/General_Dataset ===")
    if full:
        snapshot_download(
            repo_id="karisu/General_Dataset",
            repo_type="dataset",
            local_dir=str(out),
        )
    else:
        snapshot_download(
            repo_id="karisu/General_Dataset",
            repo_type="dataset",
            local_dir=str(out),
            allow_patterns=["test_sample/*", "test_sample.csv"],
        )
    print(f"  Saved to {out}")


def download_herdnet_weights() -> None:
    """Download pretrained HerdNet weights from karisu/HerdNet."""
    models_dir = BASE_DIR.parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    print("\n=== HerdNet pretrained weights ===")
    hf_hub_download(
        repo_id="karisu/HerdNet",
        filename="general_2022/20220413_HerdNet_General_dataset_2022.pth",
        local_dir=str(models_dir),
    )
    hf_hub_download(
        repo_id="karisu/HerdNet",
        filename="general_2022/config.yaml",
        local_dir=str(models_dir),
    )
    print(f"  Saved to {models_dir}")


def download_serengeti(n_images: int = 500) -> None:
    """Snapshot Serengeti subset from LILA / Azure Blob Storage."""
    out_dir = BASE_DIR / "camera_trap" / "serengeti_subset"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Snapshot Serengeti (n={n_images}) ===")

    META_URL = (
        "https://lilablobssc.blob.core.windows.net"
        "/snapshotserengeti-v-2-0/SnapshotSerengeti_S1-11_v2.1.json.zip"
    )
    print("  Downloading metadata...")
    r = requests.get(META_URL, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            meta = json.load(f)

    print(f"  {len(meta['images'])} total images in metadata")

    # Sample from one site to avoid spatial leakage between P3 and P4 exercises
    TARGET_SITE = "B04"
    sampled = [
        img for img in meta["images"]
        if img.get("location", "").startswith(TARGET_SITE)
    ][:n_images]

    if len(sampled) < n_images:
        # Fallback: take from any site if B04 doesn't have enough
        sampled = meta["images"][:n_images]

    print(f"  Sampled {len(sampled)} images")

    meta_path = BASE_DIR / "camera_trap" / "serengeti_meta.json"
    sampled_ids = {img["id"] for img in sampled}
    with open(meta_path, "w") as f:
        json.dump({
            "images": sampled,
            "annotations": [a for a in meta["annotations"] if a["image_id"] in sampled_ids],
            "categories": meta["categories"],
        }, f)

    BASE = "https://lilablobssc.blob.core.windows.net/snapshotserengeti-v-2-0/"
    for img in tqdm(sampled, desc="Serengeti images"):
        dest = out_dir / os.path.basename(img["file_name"])
        if dest.exists():
            continue
        resp = requests.get(BASE + img["file_name"], timeout=30)
        if resp.status_code == 200:
            dest.write_bytes(resp.content)
        else:
            print(f"  WARNING: {img['file_name']} → HTTP {resp.status_code}")

    print(f"  Done. Images in {out_dir}")


def download_caltech(n_images: int = 500) -> None:
    """Caltech Camera Traps subset (annotated images only) from LILA / S3."""
    import pandas as pd

    out_dir = BASE_DIR / "camera_trap" / "caltech_subset"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Caltech Camera Traps (n={n_images}) ===")

    META_URL = (
        "https://caltech-camera-traps.s3.amazonaws.com"
        "/cct_images/caltech_images_20200316.json.zip"
    )
    print("  Downloading metadata...")
    r = requests.get(META_URL, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            meta = json.load(f)

    print(f"  {len(meta['images'])} total images in metadata")

    annotated_ids = {ann["image_id"] for ann in meta["annotations"]}
    sampled = [img for img in meta["images"] if img["id"] in annotated_ids][:n_images]
    sampled_id_set = {img["id"] for img in sampled}

    print(f"  Sampled {len(sampled)} annotated images")

    id_to_cat = {c["id"]: c["name"] for c in meta["categories"]}
    id_to_file = {img["id"]: img["file_name"] for img in sampled}

    rows = [
        {
            "crop": Path(id_to_file[a["image_id"]]).name,
            "true_label": id_to_cat.get(a["category_id"], "unknown"),
            "bbox_x": a["bbox"][0],
            "bbox_y": a["bbox"][1],
            "bbox_w": a["bbox"][2],
            "bbox_h": a["bbox"][3],
        }
        for a in meta["annotations"]
        if a["image_id"] in sampled_id_set
    ]
    labels_path = BASE_DIR / "camera_trap_labels.csv"
    pd.DataFrame(rows).to_csv(labels_path, index=False)
    print(f"  Reference labels → {labels_path}")

    BASE = "https://caltech-camera-traps.s3.amazonaws.com/cct_images/"
    for img in tqdm(sampled, desc="Caltech images"):
        dest = out_dir / Path(img["file_name"]).name
        if dest.exists():
            continue
        resp = requests.get(BASE + img["file_name"], timeout=30)
        if resp.status_code == 200:
            dest.write_bytes(resp.content)
        else:
            print(f"  WARNING: {img['file_name']} → HTTP {resp.status_code}")

    print(f"  Done. Images in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Week 1 datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--sample", action="store_true", help="Minimal set (~500 MB)")
    parser.add_argument("--full", action="store_true", help="Full teaching subsets (~5 GB)")
    args = parser.parse_args()

    if not args.sample and not args.full:
        parser.print_help()
        return

    n = 5000 if args.full else 500

    download_herdnet_weights()
    download_general_dataset(full=args.full)
    download_serengeti(n_images=n)
    download_caltech(n_images=n)

    print("\n" + "=" * 50)
    print("Week 1 datasets ready.")
    print(f"  general_dataset/    ← HerdNet aerial data")
    print(f"  camera_trap/serengeti_subset/  ← {n} Serengeti images")
    print(f"  camera_trap/caltech_subset/    ← {n} Caltech images")
    print(f"  camera_trap_labels.csv         ← P6 reference labels")
    print()
    print("Still needed (manual copy from C. Winkelmann):")
    print("  iguanas/tiles/       ← ~500 iguana tile JPEGs")
    print("  iguanas/iguana_counts.csv")
    print("See DATASETS.md section 2 for details.")


if __name__ == "__main__":
    main()
