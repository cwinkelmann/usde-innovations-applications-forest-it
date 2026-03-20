"""
download_data.py — Week 1 dataset setup
========================================
Downloads datasets needed for Week 1 practicals into week1/data/.

CLI usage::

    python download_data.py --sample          # ~50 images per source
    python download_data.py --full            # ~500 images per source
    python download_data.py --n-images 20     # exactly 20 images per source

Library usage::

    from week1.data.download_data import download_serengeti, download_all

    download_serengeti(n_images=10)                          # 10 Serengeti images
    download_all(n_images=25)                                # 25 images from each source
    download_all(n_images=25, output_dir=Path("/tmp/data"))  # custom output directory
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import zipfile
from pathlib import Path
from typing import Optional

import requests
from huggingface_hub import hf_hub_download, list_repo_tree
from tqdm import tqdm

_DEFAULT_DIR = Path(__file__).parent  # week1/data/

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _is_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS


# ---------------------------------------------------------------------------
# Individual dataset downloaders
# ---------------------------------------------------------------------------


def download_general_dataset(
    n_images: Optional[int] = None,
    output_dir: Path = _DEFAULT_DIR,
) -> Path:
    """karisu/General_Dataset — HerdNet aerial dataset from HuggingFace.

    Parameters
    ----------
    n_images : int or None
        Max images to download. None downloads the full dataset.
    output_dir : Path
        Parent directory. Dataset goes into ``output_dir / general_dataset``.

    Returns
    -------
    Path to the dataset directory.
    """
    out = output_dir / "general_dataset"
    out.mkdir(parents=True, exist_ok=True)
    repo_id = "karisu/General_Dataset"
    print(f"\n=== karisu/General_Dataset (n={n_images or 'all'}) ===")

    if n_images is None:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id, repo_type="dataset", local_dir=str(out),
        )
    else:
        all_files = [
            entry.rfilename
            for entry in list_repo_tree(repo_id, repo_type="dataset", recursive=True)
            if hasattr(entry, "rfilename")
        ]
        images = sorted(f for f in all_files if _is_image(f))
        non_images = [f for f in all_files if not _is_image(f)]

        for f in non_images:
            hf_hub_download(repo_id=repo_id, repo_type="dataset",
                            filename=f, local_dir=str(out))

        for f in images[:n_images]:
            hf_hub_download(repo_id=repo_id, repo_type="dataset",
                            filename=f, local_dir=str(out))
        print(f"  Downloaded {min(n_images, len(images))}/{len(images)} images")

    print(f"  Saved to {out}")
    return out


def download_herdnet_weights(output_dir: Path = _DEFAULT_DIR) -> Path:
    """Download pretrained HerdNet weights from karisu/HerdNet.

    Returns
    -------
    Path to the models directory.
    """
    models_dir = output_dir.parent.parent / "models" if output_dir == _DEFAULT_DIR \
        else output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
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
    return models_dir


def download_serengeti(
    n_images: int = 50,
    output_dir: Path = _DEFAULT_DIR,
) -> Path:
    """Snapshot Serengeti subset from LILA / Azure Blob Storage.

    Uses Season 1 metadata only (~18 MB). Prefers images that contain
    at least one animal annotation.

    Parameters
    ----------
    n_images : int
        Number of images to download.
    output_dir : Path
        Parent directory. Images go into ``output_dir / camera_trap / serengeti_subset``.

    Returns
    -------
    Path to the image directory.
    """
    out_dir = output_dir / "camera_trap" / "serengeti_subset"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Snapshot Serengeti (n={n_images}) ===")

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

    sampled_ids = {img["id"] for img in sampled}
    sampled_annotations = [a for a in meta["annotations"] if a["image_id"] in sampled_ids]

    # ── Download bounding box annotations (separate file on LILA) ─────────
    BBOX_URL = (
        "https://lila.science/public"
        "/snapshot-serengeti-bounding-boxes-20190903.json.zip"
    )
    print("  Downloading bounding box annotations...")
    try:
        r_bbox = requests.get(BBOX_URL, timeout=120)
        r_bbox.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r_bbox.content)) as z:
            with z.open(z.namelist()[0]) as f:
                bbox_meta = json.load(f)
        # Merge bbox annotations for our sampled images
        bbox_anns = [a for a in bbox_meta.get("annotations", [])
                     if a["image_id"] in sampled_ids and "bbox" in a]
        # Add bboxes to the sampled annotations list
        sampled_annotations.extend(bbox_anns)
        n_with_bbox = len({a["image_id"] for a in bbox_anns})
        print(f"  {len(bbox_anns)} bounding boxes for {n_with_bbox}/{len(sampled)} images")
    except Exception as e:
        print(f"  WARNING: bbox download failed ({e}), continuing with species labels only")

    meta_path = output_dir / "camera_trap" / "serengeti_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "images": sampled,
            "annotations": sampled_annotations,
            "categories": meta["categories"],
        }, f)

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
    return out_dir


def download_caltech(
    n_images: int = 50,
    output_dir: Path = _DEFAULT_DIR,
) -> Path:
    """Caltech Camera Traps subset (annotated images only) from LILA / GCS.

    Samples evenly across species so no single class dominates.

    Parameters
    ----------
    n_images : int
        Number of images to download.
    output_dir : Path
        Parent directory. Images go into ``output_dir / camera_trap / caltech_subset``.

    Returns
    -------
    Path to the image directory.
    """
    import pandas as pd

    out_dir = output_dir / "camera_trap" / "caltech_subset"
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
    rng = random.Random(42)
    id_to_cat = {c["id"]: c["name"] for c in meta["categories"]}
    exclude_names = {"empty", "dog", "cat", "person"}
    id_to_anns: dict = {}
    for ann in meta["annotations"]:
        id_to_anns.setdefault(ann["image_id"], []).append(ann)

    # Build per-species pools, exclude domestic/empty
    species_pools: dict = {}
    for img in meta["images"]:
        for ann in id_to_anns.get(img["id"], []):
            sp = id_to_cat.get(ann.get("category_id"), "unknown")
            if sp not in exclude_names:
                species_pools.setdefault(sp, []).append(img)

    # Round-robin across species
    per_species = max(1, n_images // max(len(species_pools), 1))
    sampled = []
    seen_ids: set = set()
    for pool in species_pools.values():
        for img in rng.sample(pool, min(per_species, len(pool))):
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
            if ann["image_id"] not in img_id_to_bbox and ann.get("bbox"):
                img_id_to_bbox[ann["image_id"]] = ann["bbox"]
        print(f"  {len(img_id_to_bbox)} bbox annotations loaded")
    else:
        print(f"  WARNING: bbox download failed ({r_bbox.status_code}), skipping")

    _none4 = [None, None, None, None]
    rows = [
        {
            "crop": Path(id_to_file[a["image_id"]]).name,
            "true_label": id_to_cat.get(a["category_id"], "unknown"),
            "bbox_x": img_id_to_bbox.get(a["image_id"], _none4)[0],
            "bbox_y": img_id_to_bbox.get(a["image_id"], _none4)[1],
            "bbox_w": img_id_to_bbox.get(a["image_id"], _none4)[2],
            "bbox_h": img_id_to_bbox.get(a["image_id"], _none4)[3],
        }
        for a in meta["annotations"]
        if a["image_id"] in sampled_id_set
    ]
    labels_path = output_dir / "camera_trap_labels.csv"
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
    return out_dir


def download_eikelboom(
    n_images: Optional[int] = None,
    output_dir: Path = _DEFAULT_DIR,
) -> Path:
    """Eikelboom 2019 — aerial wildlife detection dataset from HuggingFace.

    Parameters
    ----------
    n_images : int or None
        Max images to download per split (train/val/test). None downloads all.
    output_dir : Path
        Parent directory. Dataset goes into ``output_dir / eikelboom``.

    Returns
    -------
    Path to the dataset directory.
    """
    out = output_dir / "eikelboom"
    out.mkdir(parents=True, exist_ok=True)
    repo_id = "karisu/Eikelboom2019"
    print(f"\n=== Eikelboom 2019 (n={n_images or 'all'} per split) ===")

    if n_images is None:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id, repo_type="dataset", local_dir=str(out),
        )
    else:
        all_files = [
            entry.rfilename
            for entry in list_repo_tree(repo_id, repo_type="dataset", recursive=True)
            if hasattr(entry, "rfilename")
        ]
        # Download non-image files (metadata, labels, configs)
        non_images = [f for f in all_files if not _is_image(f)]
        for f in non_images:
            hf_hub_download(repo_id=repo_id, repo_type="dataset",
                            filename=f, local_dir=str(out))

        # Group images by split directory and limit per split
        from collections import defaultdict
        split_files: dict[str, list[str]] = defaultdict(list)
        for f in sorted(all_files):
            if _is_image(f):
                split = f.split("/")[0] if "/" in f else ""
                split_files[split].append(f)

        total = 0
        for split, files in split_files.items():
            to_download = files[:n_images]
            for f in to_download:
                hf_hub_download(repo_id=repo_id, repo_type="dataset",
                                filename=f, local_dir=str(out))
            total += len(to_download)
            print(f"  {split or 'root'}: {len(to_download)}/{len(files)} images")
        print(f"  Downloaded {total} images total")

    # Count what we got
    for split in ["train", "val", "test"]:
        split_dir = out / split
        if split_dir.exists():
            imgs = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
            print(f"  {split}: {len(imgs)} images")

    print(f"  Saved to {out}")
    return out


def download_mmla_wilds(
    n_images: Optional[int] = None,
    output_dir: Path = _DEFAULT_DIR,
) -> Path:
    """imageomics/mmla_wilds — drone wildlife dataset from HuggingFace.

    Multi-species aerial imagery (giraffe, Grevy's zebra, Persian onager,
    African painted dog) with YOLO-format bounding box annotations.
    Collected at The Wilds, Ohio using DJI Mavic Mini and Parrot Anafi drones.

    Source: https://huggingface.co/datasets/imageomics/mmla_wilds

    Parameters
    ----------
    n_images : int or None
        Max images to download per split. None downloads all.
    output_dir : Path
        Parent directory. Dataset goes into ``output_dir / mmla_wilds``.

    Returns
    -------
    Path to the dataset directory.
    """
    out = output_dir / "mmla_wilds"
    out.mkdir(parents=True, exist_ok=True)
    repo_id = "imageomics/mmla_wilds"
    print(f"\n=== MMLA Wilds (n={n_images or 'all'} per split) ===")

    if n_images is None:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id, repo_type="dataset", local_dir=str(out),
        )
    else:
        all_files = [
            entry.rfilename
            for entry in list_repo_tree(repo_id, repo_type="dataset", recursive=True)
            if hasattr(entry, "rfilename")
        ]
        # Download non-image files (metadata, labels, configs, yaml)
        non_images = [f for f in all_files if not _is_image(f)]
        for f in non_images:
            hf_hub_download(repo_id=repo_id, repo_type="dataset",
                            filename=f, local_dir=str(out))

        # Group images by directory (splits or species folders) and limit per group
        from collections import defaultdict
        group_files: dict[str, list[str]] = defaultdict(list)
        for f in sorted(all_files):
            if _is_image(f):
                group = f.split("/")[0] if "/" in f else ""
                group_files[group].append(f)

        total = 0
        for group, files in group_files.items():
            to_download = files[:n_images]
            for f in to_download:
                hf_hub_download(repo_id=repo_id, repo_type="dataset",
                                filename=f, local_dir=str(out))
            total += len(to_download)
            print(f"  {group or 'root'}: {len(to_download)}/{len(files)} images")
        print(f"  Downloaded {total} images total")

    print(f"  Saved to {out}")
    return out


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def download_all(
    n_images: int = 50,
    output_dir: Path = _DEFAULT_DIR,
    skip_weights: bool = False,
) -> dict[str, Path]:
    """Download all Week 1 datasets.

    Parameters
    ----------
    n_images : int
        Number of images to download per source.
    output_dir : Path
        Root directory for all downloads.
    skip_weights : bool
        If True, skip HerdNet weight download.

    Returns
    -------
    Dict mapping dataset name to its output path.
    """
    results: dict[str, Path] = {}

    if not skip_weights:
        results["herdnet_weights"] = download_herdnet_weights(output_dir)

    results["general_dataset"] = download_general_dataset(n_images, output_dir)
    results["serengeti"] = download_serengeti(n_images, output_dir)
    results["caltech"] = download_caltech(n_images, output_dir)
    results["eikelboom"] = download_eikelboom(n_images, output_dir)
    results["mmla_wilds"] = download_mmla_wilds(n_images, output_dir)

    print("\n" + "=" * 50)
    print("Week 1 datasets ready.")
    for name, path in results.items():
        print(f"  {name}: {path}")
    return results


# ---------------------------------------------------------------------------
# Exploration helpers
# ---------------------------------------------------------------------------
#
# All plotting functions create figures but do NOT call plt.show().
# This makes them work in both Marimo notebooks and standalone scripts.
# In Marimo, the figure is captured automatically by the cell.
# In scripts/REPL, call plt.show() yourself after calling these functions.

DATASETS = ("general_dataset", "serengeti", "caltech", "eikelboom", "mmla_wilds")


def _list_images(directory: Path) -> list[Path]:
    """Return sorted image paths in *directory*."""
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _load_serengeti_meta(output_dir: Path) -> Optional[dict]:
    """Load Serengeti COCO JSON metadata, return None if missing."""
    meta_path = output_dir / "camera_trap" / "serengeti_meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


def _load_caltech_labels(output_dir: Path):
    """Load Caltech labels CSV, return None if missing."""
    import pandas as pd
    csv_path = output_dir / "camera_trap_labels.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def _load_herdnet_csv(output_dir: Path):
    """Load HerdNet General Dataset CSV, return (df, tile_dir) or (None, None)."""
    import pandas as pd
    csv = output_dir / "general_dataset" / "test_sample.csv"
    img_dir = output_dir / "general_dataset" / "test_sample"
    if not csv.exists():
        return None, img_dir
    return pd.read_csv(csv), img_dir


def _parse_yolo_labels(label_dir: Path) -> dict[str, int]:
    """Parse YOLO .txt label files, return {class_id: count}."""
    counts: dict[str, int] = {}
    if not label_dir.exists():
        return counts
    for lf in sorted(label_dir.glob("*.txt")):
        for line in lf.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                counts[parts[0]] = counts.get(parts[0], 0) + 1
    return counts


def summarize(output_dir: Path = _DEFAULT_DIR) -> dict[str, int]:
    """Print overview of downloaded datasets and return image counts."""
    import pandas as pd

    counts: dict[str, int] = {}
    print("Week 1 data summary")
    print("=" * 50)

    gd = output_dir / "general_dataset" / "test_sample"
    if gd.exists():
        n = len(_list_images(gd))
        csv = output_dir / "general_dataset" / "test_sample.csv"
        n_ann = len(pd.read_csv(csv)) if csv.exists() else 0
        counts["general_dataset"] = n
        print(f"  general_dataset  : {n} tiles, {n_ann} point annotations")
    else:
        print("  general_dataset  : not downloaded")

    sd = output_dir / "camera_trap" / "serengeti_subset"
    if sd.exists():
        counts["serengeti"] = len(_list_images(sd))
        print(f"  serengeti        : {counts['serengeti']} images")
    else:
        print("  serengeti        : not downloaded")

    cd = output_dir / "camera_trap" / "caltech_subset"
    if cd.exists():
        counts["caltech"] = len(_list_images(cd))
        cdf = _load_caltech_labels(output_dir)
        extra = f", {cdf['true_label'].nunique()} species" if cdf is not None else ""
        print(f"  caltech          : {counts['caltech']} images{extra}")
    else:
        print("  caltech          : not downloaded")

    ed = output_dir / "eikelboom"
    if ed.exists():
        total = 0
        parts = []
        for split in ["train", "val", "test"]:
            sp = ed / split
            if sp.exists():
                n = len(_list_images(sp))
                total += n
                parts.append(f"{split}={n}")
        counts["eikelboom"] = total
        print(f"  eikelboom        : {total} images ({', '.join(parts)})")
    else:
        print("  eikelboom        : not downloaded")

    md = output_dir / "mmla_wilds"
    if md.exists():
        all_imgs = []
        for d in md.rglob("*"):
            if d.is_file() and d.suffix.lower() in IMAGE_EXTENSIONS:
                all_imgs.append(d)
        counts["mmla_wilds"] = len(all_imgs)
        print(f"  mmla_wilds       : {len(all_imgs)} images")
    else:
        print("  mmla_wilds       : not downloaded")

    return counts


def show_samples(
    dataset: str,
    n: int = 6,
    output_dir: Path = _DEFAULT_DIR,
    title: Optional[str] = None,
) -> None:
    """Display a grid of sample images from a dataset.

    Images are labelled with their species where labels are available.

    Example::

        show_samples("caltech", n=9)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image as PILImage

    # Build label lookup
    label_fn = None
    if dataset == "serengeti":
        img_dir = output_dir / "camera_trap" / "serengeti_subset"
        display_title = "Snapshot Serengeti"
        meta = _load_serengeti_meta(output_dir)
        if meta:
            cat_map = {c["id"]: c["name"] for c in meta["categories"]}
            # Build image_id → filename → label (O(n), not O(n^2))
            id_to_fname = {img["id"]: os.path.basename(img["file_name"])
                           for img in meta["images"]}
            fname_to_label = {}
            for a in meta["annotations"]:
                fname = id_to_fname.get(a["image_id"])
                if fname:
                    fname_to_label[fname] = cat_map.get(a["category_id"], "?")
            label_fn = lambda p: fname_to_label.get(p.name, p.name)
    elif dataset == "caltech":
        img_dir = output_dir / "camera_trap" / "caltech_subset"
        display_title = "Caltech Camera Traps"
        cdf = _load_caltech_labels(output_dir)
        if cdf is not None:
            crop_to_label = dict(zip(cdf["crop"], cdf["true_label"]))
            label_fn = lambda p: crop_to_label.get(p.name, p.name)
    elif dataset == "general_dataset":
        img_dir = output_dir / "general_dataset" / "test_sample"
        display_title = "HerdNet General Dataset"
    elif dataset == "eikelboom":
        base = output_dir / "eikelboom"
        img_dir = base / "train" if (base / "train").exists() else base
        display_title = "Eikelboom 2019 (aerial)"
    elif dataset == "mmla_wilds":
        base = output_dir / "mmla_wilds"
        img_dir = base / "train" / "images" if (base / "train" / "images").exists() else base
        display_title = "MMLA Wilds (drone wildlife)"
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Choose from {DATASETS}")

    images = _list_images(img_dir)[:n]
    if not images:
        print(f"No images found in {img_dir}")
        return

    ncols = min(len(images), 4)
    nrows = (len(images) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    axes_flat = np.array(axes).flatten() if len(images) > 1 else [axes]

    for i, path in enumerate(images):
        axes_flat[i].imshow(np.array(PILImage.open(path)))
        label = label_fn(path) if label_fn else path.name
        axes_flat[i].set_title(label, fontsize=8)
        axes_flat[i].axis("off")

    for ax in axes_flat[len(images):]:
        ax.axis("off")

    plt.suptitle(title or f"{display_title} — {len(images)} samples", fontsize=11)
    plt.tight_layout()


def show_class_distribution(
    dataset: str,
    output_dir: Path = _DEFAULT_DIR,
    ax=None,
) -> None:
    """Bar chart of species/class distribution for a dataset.

    Pass *ax* to draw into an existing matplotlib axes.

    Example::

        show_class_distribution("serengeti")
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    if dataset == "general_dataset":
        df, _ = _load_herdnet_csv(output_dir)
        if df is None:
            print("General Dataset CSV not found"); return
        counts = df["labels"].value_counts()
        title = "General Dataset — annotations per species"
    elif dataset == "serengeti":
        meta = _load_serengeti_meta(output_dir)
        if meta is None:
            print("Serengeti metadata not found"); return
        cat_map = {c["id"]: c["name"] for c in meta["categories"]}
        labels = [cat_map.get(a["category_id"], "unknown") for a in meta["annotations"]]
        counts = pd.Series(labels).value_counts()
        title = "Snapshot Serengeti — species distribution"
    elif dataset == "caltech":
        cdf = _load_caltech_labels(output_dir)
        if cdf is None:
            print("Caltech labels not found"); return
        counts = cdf["true_label"].value_counts()
        title = "Caltech Camera Traps — species distribution"
    elif dataset == "eikelboom":
        yolo = _parse_yolo_labels(output_dir / "eikelboom" / "train")
        if not yolo:
            print("No Eikelboom YOLO labels found"); return
        counts = pd.Series(yolo).sort_values(ascending=False)
        counts.index = [f"class {c}" for c in counts.index]
        title = "Eikelboom — bounding boxes per class"
    elif dataset == "mmla_wilds":
        # Try train/labels first, then labels/ at root
        labels_dir = output_dir / "mmla_wilds" / "train" / "labels"
        if not labels_dir.exists():
            labels_dir = output_dir / "mmla_wilds" / "labels"
        yolo = _parse_yolo_labels(labels_dir)
        if not yolo:
            print("No MMLA Wilds YOLO labels found"); return
        counts = pd.Series(yolo).sort_values(ascending=False)
        counts.index = [f"class {c}" for c in counts.index]
        title = "MMLA Wilds — bounding boxes per class"
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Choose from {DATASETS}")

    if ax is None:
        _, ax = plt.subplots(figsize=(7, max(3, len(counts) * 0.35)))
    ax.barh(counts.index, counts.values, color="steelblue")
    ax.set_xlabel("Count")
    ax.set_title(title)
    plt.tight_layout()


def show_bboxes(
    dataset: str = "caltech",
    n: int = 6,
    output_dir: Path = _DEFAULT_DIR,
) -> None:
    """Show images with bounding boxes drawn.

    Supports ``'serengeti'`` (COCO format from metadata JSON),
    ``'caltech'`` (COCO format from CSV), and ``'eikelboom'``
    (YOLO format from .txt label files).

    Example::

        show_bboxes("serengeti", n=6)
        show_bboxes("eikelboom", n=6)
    """
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image as PILImage

    if dataset == "serengeti":
        meta = _load_serengeti_meta(output_dir)
        if meta is None:
            print("Serengeti metadata not found"); return
        cat_map = {c["id"]: c["name"] for c in meta["categories"]}
        # Build image_id → bbox list
        id_to_bboxes: dict[str, list] = {}
        for a in meta["annotations"]:
            if "bbox" in a:
                id_to_bboxes.setdefault(a["image_id"], []).append(a)
        # Pick images that have at least one bbox
        imgs_with_box = [img for img in meta["images"] if img["id"] in id_to_bboxes]
        if not imgs_with_box:
            print("No bounding box annotations in Serengeti subset"); return
        imgs_with_box = imgs_with_box[:n]

        img_dir = output_dir / "camera_trap" / "serengeti_subset"
        ncols = min(len(imgs_with_box), 3)
        nrows = (len(imgs_with_box) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
        axes_flat = np.array(axes).flatten() if len(imgs_with_box) > 1 else [axes]

        shown = 0
        for img_data in imgs_with_box:
            path = img_dir / os.path.basename(img_data["file_name"])
            if not path.exists():
                continue
            ax = axes_flat[shown]
            ax.imshow(np.array(PILImage.open(path)))
            for a in id_to_bboxes[img_data["id"]]:
                x, y, w, h = a["bbox"]
                label = cat_map.get(a.get("category_id"), "")
                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=2, edgecolor="lime", facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(x, y - 2, label, fontsize=7, color="lime",
                        backgroundcolor="black")
            ax.set_title(os.path.basename(img_data["file_name"]), fontsize=8)
            ax.axis("off")
            shown += 1

        for ax in axes_flat[shown:]:
            ax.axis("off")
        plt.suptitle(f"Snapshot Serengeti — COCO bounding boxes "
                     f"({len(id_to_bboxes)} images have boxes)", fontsize=11)
        plt.tight_layout()

    elif dataset == "caltech":
        cdf = _load_caltech_labels(output_dir)
        if cdf is None:
            print("Caltech labels not found"); return
        with_box = cdf.dropna(subset=["bbox_x"])
        if with_box.empty:
            print("No bounding box annotations in Caltech subset"); return

        img_dir = output_dir / "camera_trap" / "caltech_subset"
        rows = list(with_box.itertuples())[:n]
        ncols = min(len(rows), 3)
        nrows = (len(rows) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
        axes_flat = np.array(axes).flatten() if len(rows) > 1 else [axes]

        for i, row in enumerate(rows):
            path = img_dir / row.crop
            if not path.exists():
                continue
            ax = axes_flat[i]
            ax.imshow(np.array(PILImage.open(path)))
            rect = patches.Rectangle(
                (row.bbox_x, row.bbox_y), row.bbox_w, row.bbox_h,
                linewidth=2, edgecolor="lime", facecolor="none",
            )
            ax.add_patch(rect)
            ax.set_title(f"{row.true_label}\n"
                         f"[{int(row.bbox_x)}, {int(row.bbox_y)}, "
                         f"{int(row.bbox_w)}, {int(row.bbox_h)}]", fontsize=8)
            ax.axis("off")

        for ax in axes_flat[len(rows):]:
            ax.axis("off")
        plt.suptitle(f"Camera trap — COCO bounding boxes "
                     f"({len(with_box)}/{len(cdf)} have boxes)", fontsize=11)
        plt.tight_layout()

    elif dataset == "eikelboom":
        train_dir = output_dir / "eikelboom" / "train"
        images = _list_images(train_dir)[:n]
        if not images:
            print("No Eikelboom train images found"); return

        ncols = min(len(images), 3)
        nrows = (len(images) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
        axes_flat = np.array(axes).flatten() if len(images) > 1 else [axes]

        for i, img_path in enumerate(images):
            ax = axes_flat[i]
            img = PILImage.open(img_path)
            img_arr = np.array(img)
            ax.imshow(img_arr)
            h, w = img_arr.shape[:2]

            # Draw YOLO boxes if label file exists
            label_path = img_path.with_suffix(".txt")
            if label_path.exists():
                for line in label_path.read_text().strip().splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cx, cy, bw, bh = (float(x) for x in parts[1:5])
                        x0 = (cx - bw / 2) * w
                        y0 = (cy - bh / 2) * h
                        rect = patches.Rectangle(
                            (x0, y0), bw * w, bh * h,
                            linewidth=2, edgecolor="lime", facecolor="none",
                        )
                        ax.add_patch(rect)
            ax.set_title(img_path.name, fontsize=8)
            ax.axis("off")

        for ax in axes_flat[len(images):]:
            ax.axis("off")
        plt.suptitle("Eikelboom — aerial tiles with YOLO bounding boxes", fontsize=11)
        plt.tight_layout()
    else:
        raise ValueError(f"show_bboxes supports 'caltech' or 'eikelboom', got {dataset!r}")


def show_annotated_tiles(
    n: int = 3,
    output_dir: Path = _DEFAULT_DIR,
) -> None:
    """Show General Dataset tiles with point annotations overlaid.

    Picks the *n* most densely annotated tiles.

    Example::

        show_annotated_tiles(n=4)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image as PILImage

    df, img_dir = _load_herdnet_csv(output_dir)
    if df is None:
        print("General Dataset CSV not found"); return

    per_tile = df.groupby("images").size()
    print(f"Tiles: {len(_list_images(img_dir))}  |  "
          f"Annotations: {len(df)}  |  "
          f"Annotated tiles: {df['images'].nunique()}  |  "
          f"Median per tile: {per_tile.median():.0f}  |  "
          f"Max: {per_tile.max()}")

    top = per_tile.sort_values(ascending=False).head(n)
    ncols = min(n, 4)
    nrows = (len(top) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4.5))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, (name, count) in enumerate(top.items()):
        path = img_dir / name
        if not path.exists():
            axes_flat[i].set_title(f"{name} — not found")
            continue
        tile_df = df[df["images"] == name]
        axes_flat[i].imshow(np.array(PILImage.open(path)))
        axes_flat[i].scatter(
            tile_df["x"], tile_df["y"],
            c="red", s=25, marker="+", linewidths=1.5,
        )
        axes_flat[i].set_title(f"{name}\n{count} animals", fontsize=9)
        axes_flat[i].axis("off")

    for ax in axes_flat[len(top):]:
        ax.axis("off")

    plt.suptitle("Most densely annotated tiles — red + = one animal", fontsize=11)
    plt.tight_layout()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Week 1 datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--sample", action="store_true",
                        help="Minimal set (~50 images per source)")
    parser.add_argument("--full", action="store_true",
                        help="Full teaching subsets (~500 images per source)")
    parser.add_argument("--n-images", type=int, default=None,
                        help="Exact number of images per source (overrides --sample/--full)")
    args = parser.parse_args()

    if args.n_images is not None:
        n = args.n_images
    elif args.full:
        n = 500
    elif args.sample:
        n = 50
    else:
        parser.print_help()
        return

    download_all(n_images=n)


if __name__ == "__main__":
    main()
