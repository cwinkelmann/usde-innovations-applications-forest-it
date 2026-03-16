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


def download_serengeti(n_images: int = 50) -> None:
    """Snapshot Serengeti subset from LILA / Azure Blob Storage.

    Uses Season 1 metadata only (~18 MB) rather than the full 11-season
    combined file (~180 MB), then fetches individual images from the
    unzipped mirror.
    """
    out_dir = BASE_DIR / "camera_trap" / "serengeti_subset"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Snapshot Serengeti (n={n_images}) ===")

    # Season 1 metadata only — ~18 MB vs ~180 MB for all seasons
    META_URL = (
        "https://lilawildlife.blob.core.windows.net"
        "/lila-wildlife/snapshotserengeti-v-2-0/SnapshotSerengetiS01.json.zip"
    )
    print("  Downloading Season 1 metadata (~18 MB)...")
    r = requests.get(META_URL, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            meta = json.load(f)

    print(f"  {len(meta['images'])} images in Season 1 metadata")

    # Prefer images that have at least one animal annotation (not empty)
    animal_cat_ids = {
        c["id"] for c in meta["categories"] if c["name"].lower() != "empty"
    }
    images_with_animals = {
        a["image_id"] for a in meta["annotations"]
        if a.get("category_id") in animal_cat_ids
    }
    animal_images = [img for img in meta["images"] if img["id"] in images_with_animals]
    sampled = animal_images[:n_images]
    if len(sampled) < n_images:
        sampled = meta["images"][:n_images]  # fallback
    print(f"  Sampled {len(sampled)} images ({len(images_with_animals)} with animals available)")

    meta_path = BASE_DIR / "camera_trap" / "serengeti_meta.json"
    sampled_ids = {img["id"] for img in sampled}
    with open(meta_path, "w") as f:
        json.dump({
            "images": sampled,
            "annotations": [a for a in meta["annotations"] if a["image_id"] in sampled_ids],
            "categories": meta["categories"],
        }, f)

    # Individual images accessible from the unzipped mirror
    BASE = "https://lilawildlife.blob.core.windows.net/lila-wildlife/snapshotserengeti-unzipped/"
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


def download_caltech(n_images: int = 50) -> None:
    """Caltech Camera Traps subset (annotated images only) from LILA / GCS."""
    import pandas as pd

    out_dir = BASE_DIR / "camera_trap" / "caltech_subset"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Caltech Camera Traps (n={n_images}) ===")

    META_URL = (
        "https://storage.googleapis.com/public-datasets-lila"
        "/caltechcameratraps/labels/caltech_camera_traps.json.zip"
    )
    print("  Downloading metadata...")
    r = requests.get(META_URL, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            meta = json.load(f)

    print(f"  {len(meta['images'])} total images in metadata")

    # Sample evenly across species so no single class dominates
    import random as _random
    _random.seed(42)
    id_to_cat = {c["id"]: c["name"] for c in meta["categories"]}
    wildlife_names = {"empty", "dog", "cat", "person"}  # exclude from priority
    id_to_anns = {}
    for ann in meta["annotations"]:
        id_to_anns.setdefault(ann["image_id"], []).append(ann)

    # Build per-species pools, exclude domestic/empty
    species_pools: dict = {}
    for img in meta["images"]:
        for ann in id_to_anns.get(img["id"], []):
            sp = id_to_cat.get(ann.get("category_id"), "unknown")
            if sp not in wildlife_names:
                species_pools.setdefault(sp, []).append(img)

    # Round-robin across species
    per_species = max(1, n_images // max(len(species_pools), 1))
    sampled = []
    seen_ids: set = set()
    for pool in species_pools.values():
        for img in _random.sample(pool, min(per_species, len(pool))):
            if img["id"] not in seen_ids:
                sampled.append(img)
                seen_ids.add(img["id"])
    # Top up if needed
    if len(sampled) < n_images:
        annotated_ids = {ann["image_id"] for ann in meta["annotations"]}
        extras = [img for img in meta["images"]
                  if img["id"] in annotated_ids and img["id"] not in seen_ids]
        sampled += extras[:n_images - len(sampled)]
    sampled = sampled[:n_images]
    sampled_id_set = {img["id"] for img in sampled}

    species_present = {id_to_cat.get(ann.get("category_id"), "?")
                       for ann in meta["annotations"] if ann["image_id"] in sampled_id_set}
    print(f"  Sampled {len(sampled)} images across species: {sorted(species_present)}")

    id_to_cat = {c["id"]: c["name"] for c in meta["categories"]}
    id_to_file = {img["id"]: img["file_name"] for img in sampled}

    # Bounding boxes are in a separate file — download and match
    BBOX_URL = (
        "https://storage.googleapis.com/public-datasets-lila"
        "/caltechcameratraps/labels/caltech_bboxes_20200316.json"
    )
    print("  Downloading bbox annotations...")
    r_bbox = requests.get(BBOX_URL, timeout=60)
    img_id_to_bbox: dict = {}
    if r_bbox.ok:
        bbox_data = r_bbox.json()
        for ann in bbox_data.get("annotations", []):
            # keep first bbox per image (most images have exactly one animal)
            if ann["image_id"] not in img_id_to_bbox and ann.get("bbox"):
                img_id_to_bbox[ann["image_id"]] = ann["bbox"]  # [x, y, w, h]
        print(f"  {len(img_id_to_bbox)} bbox annotations loaded")
    else:
        print(f"  WARNING: bbox download failed ({r_bbox.status_code}), skipping")

    rows = [
        {
            "crop": Path(id_to_file[a["image_id"]]).name,
            "true_label": id_to_cat.get(a["category_id"], "unknown"),
            "bbox_x": img_id_to_bbox.get(a["image_id"], [None, None, None, None])[0],
            "bbox_y": img_id_to_bbox.get(a["image_id"], [None, None, None, None])[1],
            "bbox_w": img_id_to_bbox.get(a["image_id"], [None, None, None, None])[2],
            "bbox_h": img_id_to_bbox.get(a["image_id"], [None, None, None, None])[3],
        }
        for a in meta["annotations"]
        if a["image_id"] in sampled_id_set
    ]
    labels_path = BASE_DIR / "camera_trap_labels.csv"
    pd.DataFrame(rows).to_csv(labels_path, index=False)
    print(f"  Reference labels → {labels_path}")

    BASE = "https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images/"
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


def download_eikelboom() -> None:
    """Eikelboom 2019 — aerial wildlife detection dataset from HuggingFace."""
    out = BASE_DIR / "eikelboom"
    print("\n=== Eikelboom 2019 (aerial wildlife, 3 species) ===")
    snapshot_download(
        repo_id="karisu/Eikelboom2019",
        repo_type="dataset",
        local_dir=str(out),
    )
    print(f"  Saved to {out}")

    # Count what we got
    for split in ["train", "val", "test"]:
        split_dir = out / split
        if split_dir.exists():
            imgs = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
            print(f"  {split}: {len(imgs)} images")


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

    n = 500 if args.full else 50

    download_herdnet_weights()
    download_general_dataset(full=args.full)
    download_serengeti(n_images=n)
    download_caltech(n_images=n)
    download_eikelboom()

    print("\n" + "=" * 50)
    print("Week 1 datasets ready.")
    print(f"  general_dataset/    ← HerdNet aerial data")
    print(f"  camera_trap/serengeti_subset/  ← {n} Serengeti images")
    print(f"  camera_trap/caltech_subset/    ← {n} Caltech images")
    print(f"  camera_trap_labels.csv         ← P6 reference labels")
    print(f"  eikelboom/          ← Eikelboom 2019 aerial drone data")
    print()
    print("Still needed (manual copy from C. Winkelmann):")
    print("  iguanas/tiles/       ← ~500 iguana tile JPEGs")
    print("  iguanas/iguana_counts.csv")
    print("See DATASETS.md section 2 for details.")


if __name__ == "__main__":
    main()
