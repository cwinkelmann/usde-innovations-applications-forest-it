"""Prepare a combined YOLO dataset from multiple aerial wildlife sources.

Converts Eikelboom, Koger Ungulates, Koger Geladas, Liege Multispecies,
and MMLA Wilds into a unified YOLO directory with MegaDetector classes:
  0 = animal
  1 = person

All wildlife species are mapped to class 0 (animal).
Koger Gelada 'human' annotations map to class 1 (person).
"""

import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from wildlife_detection.tiling.utils import (
    generate_tile_windows,
    load_image_array,
    read_tile,
    save_tile_jpeg,
)

# Number of parallel workers for tiling (0 = auto based on CPU count)
NUM_WORKERS = 0

# ---------------------------------------------------------------------------
# Default dataset paths (can be overridden via function arguments)
# ---------------------------------------------------------------------------
DATASETS_ROOT = Path("/data/mnt/storage/Datasets")

DEFAULT_PATHS = {
    "eikelboom": None,  # Set dynamically relative to repo root
    "koger_ungulates": (
        DATASETS_ROOT
        / "Quantifying the movement, behaviour and environmental context of group-living animals using drones and computer vision"
        / "data-repo" / "kenyan-ungulates" / "ungulate-annotations"
    ),
    "koger_geladas": (
        DATASETS_ROOT
        / "Quantifying the movement, behaviour and environmental context of group-living animals using drones and computer vision"
        / "data-repo" / "geladas" / "gelada-annotations"
    ),
    "liege": (
        DATASETS_ROOT
        / "Multispecies detection and identification of African mammals in aerial imagery using convolutional neural networks"
        / "general_dataset"
    ),
    "mmla": DATASETS_ROOT / "mmla_wilds",
}


def get_default_paths(repo_root: Path = None) -> dict:
    """Return default dataset paths, resolving repo-relative paths."""
    paths = dict(DEFAULT_PATHS)
    if repo_root is None:
        # Try to find repo root from this file's location
        repo_root = Path(__file__).resolve().parents[3]
    paths["eikelboom"] = repo_root / "week1" / "data" / "eikelboom_yolo_tiled"
    return paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _link_or_copy(src: Path, dst: Path):
    """Symlink if possible, otherwise copy. Resolves symlinks in src first."""
    if dst.exists():
        return
    real_src = src.resolve()
    try:
        os.symlink(real_src, dst)
    except OSError:
        shutil.copy2(str(real_src), str(dst))


def remap_yolo_labels(src_label_dir: Path, dst_label_dir: Path, class_map: dict,
                      src_image_dir: Path = None, dst_image_dir: Path = None):
    """Copy YOLO label files with remapped class IDs. Optionally link/copy images."""
    dst_label_dir.mkdir(parents=True, exist_ok=True)
    if dst_image_dir:
        dst_image_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for label_file in sorted(src_label_dir.glob("*.txt")):
        lines = []
        for line in label_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            old_cls = int(parts[0])
            new_cls = class_map.get(old_cls)
            if new_cls is None:
                continue
            lines.append(f"{new_cls} {' '.join(parts[1:])}")

        (dst_label_dir / label_file.name).write_text("\n".join(lines))

        if src_image_dir and dst_image_dir:
            img_name = label_file.stem + ".jpg"
            src_img = src_image_dir / img_name
            dst_img = dst_image_dir / img_name
            _link_or_copy(src_img, dst_img)
        count += 1
    return count


def coco_to_per_image_csv(coco_json_path: Path, class_map: dict):
    """Parse COCO JSON and return per-image annotations with remapped classes.

    Returns
    -------
    annotations : dict
        ``{image_filename: [(class_id, x1, y1, x2, y2), ...]}``
    img_id_to_name : dict
    img_id_to_size : dict
    """
    with open(coco_json_path) as f:
        coco = json.load(f)

    img_id_to_name = {}
    img_id_to_size = {}
    for img in coco["images"]:
        img_id_to_name[img["id"]] = img["file_name"]
        img_id_to_size[img["id"]] = (img["width"], img["height"])

    annotations = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        fname = img_id_to_name.get(img_id)
        if fname is None:
            continue
        x, y, w, h = ann["bbox"]
        old_cls = ann["category_id"]
        new_cls = class_map.get(old_cls)
        if new_cls is None:
            continue
        annotations.setdefault(fname, []).append((new_cls, x, y, x + w, y + h))

    return annotations, img_id_to_name, img_id_to_size


def _tile_single_image(args):
    """Worker function: tile one image and return (n_tiles, n_boxes)."""
    fname, image_dir, boxes, prefix, tile_size, overlap, images_out, labels_out = args

    img_path = image_dir / fname
    if not img_path.exists():
        return 0, 0

    try:
        image_array = load_image_array(str(img_path))
    except Exception as e:
        return 0, 0

    img_h, img_w = image_array.shape[:2]
    stem = f"{prefix}_{Path(fname).stem}"
    n_tiles = 0
    n_boxes = 0

    for window in generate_tile_windows(img_w, img_h, tile_size, overlap):
        col_off, row_off, win_w, win_h = window
        tile_name = f"{stem}_{col_off}_{row_off}"

        tile_array = read_tile(image_array, window, tile_size)
        save_tile_jpeg(tile_array, images_out / f"{tile_name}.jpg")

        lines = []
        for cls_id, x1, y1, x2, y2 in boxes:
            ix1 = max(x1, col_off)
            iy1 = max(y1, row_off)
            ix2 = min(x2, col_off + win_w)
            iy2 = min(y2, row_off + win_h)

            if ix2 <= ix1 or iy2 <= iy1:
                continue

            orig_area = (x2 - x1) * (y2 - y1)
            inter_area = (ix2 - ix1) * (iy2 - iy1)
            if orig_area > 0 and inter_area / orig_area < 0.5:
                continue

            lx1 = ix1 - col_off
            ly1 = iy1 - row_off
            lx2 = ix2 - col_off
            ly2 = iy2 - row_off

            cx = (lx1 + lx2) / 2 / tile_size
            cy = (ly1 + ly2) / 2 / tile_size
            w = (lx2 - lx1) / tile_size
            h = (ly2 - ly1) / tile_size

            lines.append(f"{int(cls_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            n_boxes += 1

        (labels_out / f"{tile_name}.txt").write_text("\n".join(lines))
        n_tiles += 1

    return n_tiles, n_boxes


def _get_num_workers():
    """Return number of workers: NUM_WORKERS if set, else cpu_count - 2."""
    if NUM_WORKERS > 0:
        return NUM_WORKERS
    return max(1, (os.cpu_count() or 4) - 2)


def tile_from_coco(coco_json_path: Path, image_dir: Path, output_dir: Path,
                   class_map: dict, tile_size: int, overlap: int, prefix: str):
    """Convert COCO JSON annotations to tiled YOLO format (parallelized)."""
    annotations, img_id_to_name, _ = coco_to_per_image_csv(coco_json_path, class_map)

    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    all_image_names = sorted(set(img_id_to_name.values()))

    # Build work items
    work = [
        (fname, image_dir, annotations.get(fname, []), prefix,
         tile_size, overlap, images_out, labels_out)
        for fname in all_image_names
    ]

    workers = _get_num_workers()
    total_tiles = 0
    total_boxes = 0

    if workers <= 1:
        # Sequential fallback
        for args in tqdm(work, desc=f"Tiling {prefix}"):
            nt, nb = _tile_single_image(args)
            total_tiles += nt
            total_boxes += nb
    else:
        print(f"  Tiling {prefix} with {workers} workers...")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_tile_single_image, w): w[0] for w in work}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Tiling {prefix}"):
                nt, nb = future.result()
                total_tiles += nt
                total_boxes += nb

    print(f"  {prefix}: {total_tiles} tiles, {total_boxes} boxes")
    return total_tiles, total_boxes


def _tile_single_yolo_image(args):
    """Worker: tile one image with YOLO labels, return (n_tiles, n_boxes)."""
    img_path, label_dir, class_map, prefix, tile_size, overlap, images_out, labels_out = args

    try:
        image_array = load_image_array(str(img_path))
    except Exception:
        return 0, 0

    img_h, img_w = image_array.shape[:2]
    label_path = label_dir / (img_path.stem + ".txt")

    boxes = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            old_cls = int(parts[0])
            new_cls = class_map.get(old_cls)
            if new_cls is None:
                continue
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append((new_cls, x1, y1, x2, y2))

    stem = f"{prefix}_{img_path.stem}"
    n_tiles = 0
    n_boxes = 0

    for window in generate_tile_windows(img_w, img_h, tile_size, overlap):
        col_off, row_off, win_w, win_h = window
        tile_name = f"{stem}_{col_off}_{row_off}"

        tile_array = read_tile(image_array, window, tile_size)
        save_tile_jpeg(tile_array, images_out / f"{tile_name}.jpg")

        lines = []
        for cls_id, x1, y1, x2, y2 in boxes:
            ix1 = max(x1, col_off)
            iy1 = max(y1, row_off)
            ix2 = min(x2, col_off + win_w)
            iy2 = min(y2, row_off + win_h)

            if ix2 <= ix1 or iy2 <= iy1:
                continue

            orig_area = (x2 - x1) * (y2 - y1)
            inter_area = (ix2 - ix1) * (iy2 - iy1)
            if orig_area > 0 and inter_area / orig_area < 0.5:
                continue

            lx1 = ix1 - col_off
            ly1 = iy1 - row_off
            lx2 = ix2 - col_off
            ly2 = iy2 - row_off

            cx_n = (lx1 + lx2) / 2 / tile_size
            cy_n = (ly1 + ly2) / 2 / tile_size
            w_n = (lx2 - lx1) / tile_size
            h_n = (ly2 - ly1) / tile_size

            lines.append(f"{int(cls_id)} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")
            n_boxes += 1

        (labels_out / f"{tile_name}.txt").write_text("\n".join(lines))
        n_tiles += 1

    return n_tiles, n_boxes


def tile_from_yolo_labels(image_dir: Path, label_dir: Path, output_dir: Path,
                          class_map: dict, tile_size: int, overlap: int, prefix: str):
    """Tile large images with existing YOLO .txt labels (parallelized)."""
    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    image_files = sorted(image_dir.glob("*.jpg"))
    work = [
        (img_path, label_dir, class_map, prefix, tile_size, overlap, images_out, labels_out)
        for img_path in image_files
    ]

    workers = _get_num_workers()
    total_tiles = 0
    total_boxes = 0

    if workers <= 1:
        for args in tqdm(work, desc=f"Tiling {prefix}"):
            nt, nb = _tile_single_yolo_image(args)
            total_tiles += nt
            total_boxes += nb
    else:
        print(f"  Tiling {prefix} with {workers} workers...")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_tile_single_yolo_image, w): w[0] for w in work}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Tiling {prefix}"):
                nt, nb = future.result()
                total_tiles += nt
                total_boxes += nb

    print(f"  {prefix}: {total_tiles} tiles, {total_boxes} boxes")
    return total_tiles, total_boxes


# ---------------------------------------------------------------------------
# Per-dataset converters
# ---------------------------------------------------------------------------

def convert_eikelboom(output_dir: Path, split: str, paths: dict):
    """Remap Eikelboom tiled YOLO classes to MegaDetector (all -> animal=0)."""
    src = paths["eikelboom"]
    class_map = {0: 0, 1: 0, 2: 0}

    src_label_dir = src / "labels" / split
    src_image_dir = src / "images" / split
    dst_label_dir = output_dir / "labels" / split
    dst_image_dir = output_dir / "images" / split

    dst_label_dir.mkdir(parents=True, exist_ok=True)
    dst_image_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for label_file in sorted(src_label_dir.glob("*.txt")):
        new_name = f"eik_{label_file.name}"
        lines = []
        for line in label_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            old_cls = int(parts[0])
            new_cls = class_map.get(old_cls, 0)
            lines.append(f"{new_cls} {' '.join(parts[1:])}")
        (dst_label_dir / new_name).write_text("\n".join(lines))

        img_name = label_file.stem + ".jpg"
        new_img = f"eik_{img_name}"
        src_img = src_image_dir / img_name
        dst_img = dst_image_dir / new_img
        if src_img.exists():
            _link_or_copy(src_img, dst_img)
        count += 1

    print(f"  Eikelboom {split}: {count} tiles remapped")
    return count


def convert_koger_ungulates(output_dir: Path, split: str, tile_size: int,
                            overlap: int, paths: dict):
    """Convert Koger Kenyan Ungulates from COCO JSON to tiled YOLO."""
    src = paths["koger_ungulates"]
    ann_dir = src / "annotations-clean-name-pruned"

    split_map = {"train": "train.json", "val": "val.json"}
    json_file = ann_dir / split_map[split]
    if not json_file.exists():
        print(f"  Warning: {json_file} not found, skipping")
        return 0, 0

    class_map = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    tmp_out = output_dir / f"_tmp_koger_ung_{split}"
    tiles, boxes = tile_from_coco(
        json_file, src, tmp_out, class_map, tile_size, overlap, prefix="kung"
    )

    _merge_tiles(tmp_out, output_dir, split)
    shutil.rmtree(tmp_out, ignore_errors=True)
    return tiles, boxes


def convert_koger_geladas(output_dir: Path, split: str, tile_size: int,
                          overlap: int, paths: dict):
    """Convert Koger Geladas from COCO JSON to tiled YOLO."""
    src = paths["koger_geladas"]

    split_map = {
        "train": "train_males.json",
        "val": "coco_males_export-2022-01-05T15_54_11.401Z-val.json",
    }
    json_file = src / split_map[split]
    if not json_file.exists():
        print(f"  Warning: {json_file} not found, skipping")
        return 0, 0

    image_dir = src / "annotated_images"
    class_map = {1: 0, 2: 0, 3: 1}

    tmp_out = output_dir / f"_tmp_koger_gel_{split}"
    tiles, boxes = tile_from_coco(
        json_file, image_dir, tmp_out, class_map, tile_size, overlap, prefix="kgel"
    )

    _merge_tiles(tmp_out, output_dir, split)
    shutil.rmtree(tmp_out, ignore_errors=True)
    return tiles, boxes


def convert_liege(output_dir: Path, split: str, tile_size: int,
                  overlap: int, paths: dict):
    """Convert Liege Multispecies from COCO JSON to tiled YOLO."""
    src = paths["liege"]
    json_dir = src / "groundtruth" / "json" / "big_size"

    split_map = {
        "train": "train_big_size_A_B_E_K_WH_WB.json",
        "val": "val_big_size_A_B_E_K_WH_WB.json",
    }
    json_file = json_dir / split_map[split]
    if not json_file.exists():
        print(f"  Warning: {json_file} not found, skipping")
        return 0, 0

    class_map = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

    image_dir = src / split
    if not image_dir.exists():
        image_dir = src / "train"

    tmp_out = output_dir / f"_tmp_liege_{split}"
    tiles, boxes = tile_from_coco(
        json_file, image_dir, tmp_out, class_map, tile_size, overlap, prefix="liege"
    )

    _merge_tiles(tmp_out, output_dir, split)
    shutil.rmtree(tmp_out, ignore_errors=True)
    return tiles, boxes


def download_mmla(paths: dict):
    """Download MMLA Wilds dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  Warning: 'datasets' package not installed. Install with: pip install datasets")
        print("  Skipping MMLA download.")
        return False

    mmla_dir = paths["mmla"]
    hf_cache = mmla_dir / "hf_cache"
    hf_cache.mkdir(parents=True, exist_ok=True)

    print("  Downloading MMLA Wilds from HuggingFace (this may take a while)...")
    try:
        ds = load_dataset("imageomics/mmla_wilds", cache_dir=str(hf_cache))
        print(f"  Download complete. Splits: {list(ds.keys())}")

        extracted_dir = mmla_dir / "extracted"
        extracted_dir.mkdir(parents=True, exist_ok=True)

        for split_name in ds:
            split_dir = extracted_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Extracting {split_name}: {len(ds[split_name])} items...")

            for i, item in enumerate(tqdm(ds[split_name], desc=f"  Extracting {split_name}")):
                if "image" in item and item["image"] is not None:
                    img = item["image"]
                    if hasattr(img, "save"):
                        fname = item.get("file_name", item.get("image_id", f"frame_{i:06d}"))
                        if not fname.endswith(".jpg"):
                            fname = f"{fname}.jpg"
                        img.save(split_dir / fname)

        return True
    except Exception as e:
        print(f"  Warning: MMLA download failed: {e}")
        print("  Continuing without MMLA data.")
        return False


def convert_mmla(output_dir: Path, split: str, tile_size: int,
                 overlap: int, paths: dict):
    """Convert MMLA Wilds (YOLO format, large images) to tiled YOLO."""
    src = paths["mmla"]
    class_map = {0: 0, 1: 0, 2: 0, 3: 0}

    total_tiles = 0
    total_boxes = 0

    for session in ["session_1", "session_2", "session_3", "session_4"]:
        session_dir = src / session
        if not session_dir.exists():
            continue

        for video_dir in sorted(session_dir.iterdir()):
            if not video_dir.is_dir():
                continue

            jpg_files = list(video_dir.glob("*.jpg"))
            if not jpg_files:
                continue

            if jpg_files[0].stat().st_size < 1000:
                extracted = src / "extracted" / session / video_dir.name
                if extracted.exists():
                    video_dir = extracted
                    jpg_files = list(video_dir.glob("*.jpg"))
                    if not jpg_files:
                        continue
                else:
                    continue

            label_source = src / session / video_dir.name

            tiles, boxes = tile_from_yolo_labels(
                video_dir, label_source, output_dir / f"_tmp_mmla_{split}",
                class_map, tile_size, overlap,
                prefix=f"mmla_{session}_{video_dir.name}"
            )
            total_tiles += tiles
            total_boxes += boxes

    if total_tiles > 0:
        tmp_out = output_dir / f"_tmp_mmla_{split}"
        if tmp_out.exists():
            _merge_tiles(tmp_out, output_dir, split)
            shutil.rmtree(tmp_out, ignore_errors=True)

    return total_tiles, total_boxes


def _merge_tiles(tmp_dir: Path, output_dir: Path, split: str):
    """Move tiles from temp directory into the combined output."""
    dst_images = output_dir / "images" / split
    dst_labels = output_dir / "labels" / split
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    src_images = tmp_dir / "images"
    src_labels = tmp_dir / "labels"

    if src_images.exists():
        for f in src_images.glob("*.jpg"):
            dst = dst_images / f.name
            if not dst.exists():
                shutil.move(str(f), str(dst))

    if src_labels.exists():
        for f in src_labels.glob("*.txt"):
            dst = dst_labels / f.name
            if not dst.exists():
                shutil.move(str(f), str(dst))


def write_dataset_yaml(output_dir: Path):
    """Write the ultralytics dataset.yaml."""
    yaml_content = f"""path: {output_dir}
train: images/train
val: images/val

nc: 2
names: ['animal', 'person']
"""
    (output_dir / "dataset.yaml").write_text(yaml_content)
    print(f"\nDataset YAML written to {output_dir / 'dataset.yaml'}")


def print_dataset_stats(output_dir: Path):
    """Print class distribution and tile counts."""
    for split in ["train", "val"]:
        label_dir = output_dir / "labels" / split
        image_dir = output_dir / "images" / split

        if not label_dir.exists():
            continue

        n_images = len(list(image_dir.glob("*.jpg")))
        n_labels = len(list(label_dir.glob("*.txt")))

        class_counts = {0: 0, 1: 0}
        empty_tiles = 0
        for lf in label_dir.glob("*.txt"):
            content = lf.read_text().strip()
            if not content:
                empty_tiles += 1
                continue
            for line in content.splitlines():
                cls = int(line.split()[0])
                class_counts[cls] = class_counts.get(cls, 0) + 1

        total_boxes = sum(class_counts.values())
        print(f"\n{split}:")
        print(f"  Images: {n_images}, Labels: {n_labels}")
        print(f"  Total boxes: {total_boxes}")
        print(f"  animal (0): {class_counts.get(0, 0)}")
        print(f"  person (1): {class_counts.get(1, 0)}")
        print(f"  Empty tiles: {empty_tiles} ({100*empty_tiles/max(n_labels,1):.1f}%)")


def prepare_combined_dataset(output_dir: Path, sources: list, tile_size: int = 640,
                             overlap: int = 120, do_download_mmla: bool = False,
                             paths: dict = None):
    """Main entry point: prepare the combined dataset.

    Parameters
    ----------
    output_dir : Path
        Destination for the unified YOLO dataset.
    sources : list of str
        Dataset names to include (eikelboom, koger_ungulates, koger_geladas, liege, mmla).
    tile_size : int
    overlap : int
    do_download_mmla : bool
        Whether to download MMLA from HuggingFace first.
    paths : dict, optional
        Override default dataset paths.
    """
    if paths is None:
        paths = get_default_paths()

    print(f"Output: {output_dir}")
    print(f"Tile size: {tile_size}, Overlap: {overlap}")
    print(f"Sources: {sources}")
    print()

    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    if do_download_mmla and "mmla" in sources:
        print("=" * 60)
        print("Step 0: Downloading MMLA Wilds")
        print("=" * 60)
        download_mmla(paths)

    for source in sources:
        print("=" * 60)
        print(f"Converting: {source}")
        print("=" * 60)

        if source == "eikelboom":
            for split in ["train", "val"]:
                convert_eikelboom(output_dir, split, paths)

        elif source == "koger_ungulates":
            for split in ["train", "val"]:
                convert_koger_ungulates(output_dir, split, tile_size, overlap, paths)

        elif source == "koger_geladas":
            for split in ["train", "val"]:
                convert_koger_geladas(output_dir, split, tile_size, overlap, paths)

        elif source == "liege":
            for split in ["train", "val"]:
                convert_liege(output_dir, split, tile_size, overlap, paths)

        elif source == "mmla":
            convert_mmla(output_dir, "train", tile_size, overlap, paths)

        else:
            print(f"  Unknown source: {source}, skipping")

    write_dataset_yaml(output_dir)

    print("\n" + "=" * 60)
    print("Combined Dataset Statistics")
    print("=" * 60)
    print_dataset_stats(output_dir)
