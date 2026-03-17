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

    meta_path = output_dir / "camera_trap" / "serengeti_meta.json"
    sampled_ids = {img["id"] for img in sampled}
    with open(meta_path, "w") as f:
        json.dump({
            "images": sampled,
            "annotations": [a for a in meta["annotations"] if a["image_id"] in sampled_ids],
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

    print("\n" + "=" * 50)
    print("Week 1 datasets ready.")
    for name, path in results.items():
        print(f"  {name}: {path}")
    return results


# ---------------------------------------------------------------------------
# Exploration helpers
# ---------------------------------------------------------------------------

DATASETS = ("general_dataset", "serengeti", "caltech", "eikelboom")


def _list_images(directory: Path) -> list[Path]:
    """Return sorted image paths in *directory*."""
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def summarize(output_dir: Path = _DEFAULT_DIR) -> dict[str, int]:
    """Print an overview of downloaded datasets and return image counts.

    Example::

        from week1.data.download_data import summarize
        counts = summarize()
    """
    counts: dict[str, int] = {}
    print("Week 1 data summary")
    print("=" * 50)

    # General dataset
    gd = output_dir / "general_dataset" / "test_sample"
    if gd.exists():
        n = len(_list_images(gd))
        csv = output_dir / "general_dataset" / "test_sample.csv"
        n_ann = 0
        if csv.exists():
            import pandas as pd
            n_ann = len(pd.read_csv(csv))
        counts["general_dataset"] = n
        print(f"  general_dataset  : {n} tiles, {n_ann} point annotations")
    else:
        print(f"  general_dataset  : not downloaded")

    # Serengeti
    sd = output_dir / "camera_trap" / "serengeti_subset"
    if sd.exists():
        n = len(_list_images(sd))
        counts["serengeti"] = n
        print(f"  serengeti        : {n} images")
    else:
        print(f"  serengeti        : not downloaded")

    # Caltech
    cd = output_dir / "camera_trap" / "caltech_subset"
    if cd.exists():
        n = len(_list_images(cd))
        counts["caltech"] = n
        label_csv = output_dir / "camera_trap_labels.csv"
        extra = ""
        if label_csv.exists():
            import pandas as pd
            extra = f", {pd.read_csv(label_csv)['true_label'].nunique()} species in labels"
        print(f"  caltech          : {n} images{extra}")
    else:
        print(f"  caltech          : not downloaded")

    # Eikelboom
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
        print(f"  eikelboom        : not downloaded")

    # HerdNet weights
    wt = (output_dir.parent.parent / "models" / "general_2022"
          / "20220413_HerdNet_General_dataset_2022.pth")
    print(f"  herdnet_weights  : {'present' if wt.exists() else 'not downloaded'}")

    return counts


def show_samples(
    dataset: str = "general_dataset",
    n: int = 6,
    output_dir: Path = _DEFAULT_DIR,
) -> None:
    """Display a grid of sample images from a dataset.

    Parameters
    ----------
    dataset : str
        One of ``'general_dataset'``, ``'serengeti'``, ``'caltech'``, ``'eikelboom'``.
    n : int
        Number of images to show.
    output_dir : Path
        Root data directory.

    Example::

        from week1.data.download_data import show_samples
        show_samples("caltech", n=9)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    img_dir, title, label_fn = _resolve_dataset(dataset, output_dir)
    images = _list_images(img_dir)
    if not images:
        print(f"No images found in {img_dir}")
        return

    images = images[:n]
    ncols = min(n, 4)
    nrows = (len(images) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, path in enumerate(images):
        img = Image.open(path)
        axes_flat[i].imshow(np.array(img))
        label = label_fn(path) if label_fn else path.name
        axes_flat[i].set_title(label, fontsize=8)
        axes_flat[i].axis("off")

    for ax in axes_flat[len(images):]:
        ax.axis("off")

    plt.suptitle(f"{title} — {len(images)} samples", fontsize=11)
    plt.tight_layout()
    plt.show()


def show_class_distribution(
    dataset: str = "general_dataset",
    output_dir: Path = _DEFAULT_DIR,
) -> None:
    """Bar chart of species/class distribution for a dataset.

    Parameters
    ----------
    dataset : str
        One of ``'general_dataset'``, ``'serengeti'``, ``'caltech'``, ``'eikelboom'``.
    output_dir : Path
        Root data directory.

    Example::

        from week1.data.download_data import show_class_distribution
        show_class_distribution("serengeti")
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    if dataset == "general_dataset":
        csv = output_dir / "general_dataset" / "test_sample.csv"
        if not csv.exists():
            print(f"Annotation CSV not found: {csv}")
            return
        df = pd.read_csv(csv)
        counts = df["labels"].value_counts()
        title = "General Dataset — annotations per species"

    elif dataset == "serengeti":
        meta_path = output_dir / "camera_trap" / "serengeti_meta.json"
        if not meta_path.exists():
            print(f"Metadata not found: {meta_path}")
            return
        with open(meta_path) as f:
            meta = json.load(f)
        cat_map = {c["id"]: c["name"] for c in meta["categories"]}
        labels = [cat_map.get(a["category_id"], "unknown") for a in meta["annotations"]]
        counts = pd.Series(labels).value_counts()
        title = "Snapshot Serengeti — annotations per species"

    elif dataset == "caltech":
        csv = output_dir / "camera_trap_labels.csv"
        if not csv.exists():
            print(f"Labels not found: {csv}")
            return
        df = pd.read_csv(csv)
        counts = df["true_label"].value_counts()
        title = "Caltech Camera Traps — images per species"

    elif dataset == "eikelboom":
        ed = output_dir / "eikelboom"
        split_counts = {}
        for split in ["train", "val", "test"]:
            sp = ed / split
            if sp.exists():
                split_counts[split] = len(_list_images(sp))
        if not split_counts:
            print(f"No splits found in {ed}")
            return
        counts = pd.Series(split_counts)
        title = "Eikelboom 2019 — images per split"
    else:
        print(f"Unknown dataset: {dataset!r}. Choose from {DATASETS}")
        return

    fig, ax = plt.subplots(figsize=(7, max(3, len(counts) * 0.35)))
    ax.barh(counts.index, counts.values, color="steelblue")
    ax.set_xlabel("Count")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def show_annotated_tiles(
    n: int = 3,
    output_dir: Path = _DEFAULT_DIR,
) -> None:
    """Show General Dataset tiles with point annotations overlaid.

    Picks the *n* most densely annotated tiles.

    Example::

        from week1.data.download_data import show_annotated_tiles
        show_annotated_tiles(n=4)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from PIL import Image

    csv = output_dir / "general_dataset" / "test_sample.csv"
    img_dir = output_dir / "general_dataset" / "test_sample"
    if not csv.exists():
        print(f"Annotation CSV not found: {csv}")
        return

    df = pd.read_csv(csv)
    top = df.groupby("images").size().sort_values(ascending=False).head(n)
    ncols = min(n, 4)
    nrows = (len(top) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4.5))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, (name, count) in enumerate(top.items()):
        path = img_dir / name
        if not path.exists():
            axes_flat[i].set_title(f"{name} — not found")
            continue
        img = np.array(Image.open(path))
        tile_df = df[df["images"] == name]
        axes_flat[i].imshow(img)
        axes_flat[i].scatter(
            tile_df["x"], tile_df["y"],
            c="red", s=25, marker="+", linewidths=1.5,
        )
        axes_flat[i].set_title(f"{name}\n{count} animals", fontsize=9)
        axes_flat[i].axis("off")

    for ax in axes_flat[len(top):]:
        ax.axis("off")

    plt.suptitle("Most densely annotated tiles (General Dataset)", fontsize=11)
    plt.tight_layout()
    plt.show()


def _resolve_dataset(
    dataset: str, output_dir: Path,
) -> tuple[Path, str, Optional[callable]]:
    """Return (image_dir, display_title, label_function) for a dataset name."""
    if dataset == "general_dataset":
        return (
            output_dir / "general_dataset" / "test_sample",
            "General Dataset (aerial tiles)",
            None,
        )
    elif dataset == "serengeti":
        meta_path = output_dir / "camera_trap" / "serengeti_meta.json"
        label_fn = None
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            cat_map = {c["id"]: c["name"] for c in meta["categories"]}
            fname_to_label = {}
            for a in meta["annotations"]:
                for img in meta["images"]:
                    if img["id"] == a["image_id"]:
                        fname_to_label[os.path.basename(img["file_name"])] = (
                            cat_map.get(a["category_id"], "?")
                        )
            label_fn = lambda p: fname_to_label.get(p.name, p.name)
        return (
            output_dir / "camera_trap" / "serengeti_subset",
            "Snapshot Serengeti",
            label_fn,
        )
    elif dataset == "caltech":
        labels_csv = output_dir / "camera_trap_labels.csv"
        label_fn = None
        if labels_csv.exists():
            import pandas as pd
            ldf = pd.read_csv(labels_csv)
            crop_to_label = dict(zip(ldf["crop"], ldf["true_label"]))
            label_fn = lambda p: crop_to_label.get(p.name, p.name)
        return (
            output_dir / "camera_trap" / "caltech_subset",
            "Caltech Camera Traps",
            label_fn,
        )
    elif dataset == "eikelboom":
        # Show from the train split by default
        base = output_dir / "eikelboom"
        img_dir = base / "train" if (base / "train").exists() else base
        return (img_dir, "Eikelboom 2019 (aerial)", None)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Choose from {DATASETS}")


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
