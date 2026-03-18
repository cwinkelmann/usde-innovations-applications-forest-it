#!/usr/bin/env python
"""
Dataset format converter for detection annotations.

Supported conversions:
    eikelboom-csv  → coco     Pascal VOC CSV (no header) to COCO JSON
    pascal-voc-csv → coco     Generic Pascal VOC CSV to COCO JSON
    coco           → yolo     COCO JSON to YOLO .txt labels
    yolo           → coco     YOLO .txt labels to COCO JSON

Usage:
    # Eikelboom CSV → COCO JSON (ready for train_detection.py)
    python convert_dataset.py \\
        --from eikelboom-csv \\
        --src  ./data/eikelboom \\
        --dst  ./data/eikelboom_coco

    # Generic Pascal VOC CSV → COCO JSON
    python convert_dataset.py \\
        --from pascal-voc-csv \\
        --src  ./data/my_dataset \\
        --dst  ./data/my_dataset_coco \\
        --csv-train annotations_train.csv \\
        --csv-val   annotations_val.csv \\
        --header    # if CSV has a header row

    # COCO JSON → YOLO .txt
    python convert_dataset.py \\
        --from coco --to yolo \\
        --src ./data/my_coco_dataset \\
        --dst ./data/my_yolo_dataset

    # YOLO .txt → COCO JSON
    python convert_dataset.py \\
        --from yolo --to coco \\
        --src ./data/my_yolo_dataset \\
        --dst ./data/my_coco_dataset
"""

import argparse
import csv
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert between detection annotation formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--from", dest="src_fmt", required=True,
                   choices=["eikelboom-csv", "pascal-voc-csv", "coco", "yolo"],
                   help="Source annotation format")
    p.add_argument("--to", dest="dst_fmt", default="coco",
                   choices=["coco", "yolo"],
                   help="Target annotation format (default: coco)")
    p.add_argument("--src", required=True,
                   help="Source dataset directory")
    p.add_argument("--dst", required=True,
                   help="Output directory (will be created)")
    p.add_argument("--csv-train", default="annotations_train.csv",
                   help="Training CSV filename (for CSV sources)")
    p.add_argument("--csv-val", default="annotations_val.csv",
                   help="Validation CSV filename (for CSV sources)")
    p.add_argument("--csv-test", default="annotations_test.csv",
                   help="Test CSV filename (for CSV sources)")
    p.add_argument("--csv-dir", default="annotations",
                   help="Subdirectory containing CSV files (default: annotations)")
    p.add_argument("--header", action="store_true",
                   help="CSV has a header row (skipped during parsing)")
    p.add_argument("--symlink", action="store_true",
                   help="Symlink images instead of copying (saves disk space)")
    p.add_argument("--tile", type=int, default=0,
                   help="Tile images into NxN px crops (e.g. --tile 640). "
                        "Annotations are remapped to tile coordinates. "
                        "Tiles with no annotations are included as negatives.")
    p.add_argument("--tile-overlap", type=float, default=0.2,
                   help="Overlap ratio between tiles (default: 0.2)")
    p.add_argument("--tile-min-bbox-area", type=int, default=16,
                   help="Min bbox area in px to keep after clipping (default: 16)")
    return p.parse_args()


# ─── PASCAL VOC CSV → COCO JSON ─────────────────────────────────────────────

def _parse_voc_csv(csv_path: Path, has_header: bool) -> list[dict]:
    """Parse a Pascal VOC CSV file (filepath,x1,y1,x2,y2,class).

    Returns list of dicts with keys: filepath, x1, y1, x2, y2, class_name.
    """
    rows = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        if has_header:
            next(reader)
        for line in reader:
            if len(line) < 6:
                continue
            filepath, x1, y1, x2, y2, class_name = (
                line[0].strip(), int(line[1]), int(line[2]),
                int(line[3]), int(line[4]), line[5].strip(),
            )
            rows.append({
                "filepath": filepath,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "class_name": class_name,
            })
    return rows


def _build_coco_json(
    rows: list[dict],
    img_dir: Path,
    class_names: list[str],
) -> dict:
    """Build a COCO JSON dict from parsed VOC rows."""
    class_to_id = {name: i + 1 for i, name in enumerate(class_names)}

    images = []
    annotations = []
    seen_images: dict[str, int] = {}  # filename → image_id
    image_sizes: dict[int, tuple[int, int]] = {}  # image_id → (w, h)
    ann_id = 1

    for row in rows:
        # Extract just the filename (CSV may have split-relative paths like train/img.JPG)
        basename = Path(row["filepath"]).name
        img_path = img_dir / basename

        if basename not in seen_images:
            img_id = len(seen_images) + 1
            seen_images[basename] = img_id

            # Get image dimensions
            if img_path.exists():
                with Image.open(img_path) as im:
                    w, h = im.size
            else:
                w, h = 0, 0
                print(f"  WARNING: image not found: {img_path}")

            images.append({
                "id": img_id,
                "file_name": basename,
                "width": w,
                "height": h,
            })
            image_sizes[img_id] = (w, h)

        img_id = seen_images[basename]
        iw, ih = image_sizes[img_id]

        # Clip to image bounds
        x1 = max(0, min(row["x1"], iw))
        y1 = max(0, min(row["y1"], ih))
        x2 = max(0, min(row["x2"], iw))
        y2 = max(0, min(row["y2"], ih))
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        if bbox_w <= 0 or bbox_h <= 0:
            continue

        annotations.append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": class_to_id[row["class_name"]],
            "bbox": [x1, y1, bbox_w, bbox_h],  # COCO format: [x, y, w, h]
            "area": bbox_w * bbox_h,
            "iscrowd": 0,
        })
        ann_id += 1

    categories = [{"id": i + 1, "name": name} for i, name in enumerate(class_names)]

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def _tile_coco_dataset(
    coco: dict,
    img_src: Path,
    img_dst: Path,
    tile_size: int,
    overlap: float,
    min_bbox_area: int,
) -> dict:
    """Tile images and remap COCO annotations to tile coordinates.

    Each image is sliced into tile_size x tile_size crops with overlap.
    Bounding boxes are clipped to tile boundaries. Tiles with no annotations
    are kept as negative examples.

    Returns a new COCO dict with tiled images and remapped annotations.
    """
    img_dst.mkdir(parents=True, exist_ok=True)
    stride = int(tile_size * (1 - overlap))

    new_images = []
    new_annotations = []
    tile_img_id = 0
    ann_id = 0

    # Index annotations by image_id
    anns_by_image: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    images_meta = {img["id"]: img for img in coco["images"]}

    for img_id, img_meta in images_meta.items():
        fname = img_meta["file_name"]
        src_path = img_src / fname
        if not src_path.exists():
            continue

        pil_img = Image.open(src_path).convert("RGB")
        W, H = pil_img.size
        anns = anns_by_image.get(img_id, [])

        for y0 in range(0, H, stride):
            for x0 in range(0, W, stride):
                x1 = min(x0 + tile_size, W)
                y1 = min(y0 + tile_size, H)

                # Skip very small edge tiles
                if (x1 - x0) < tile_size // 2 or (y1 - y0) < tile_size // 2:
                    continue

                tile_img_id += 1
                tile_name = f"{Path(fname).stem}_{x0}_{y0}.jpg"

                # Crop and save tile
                tile = pil_img.crop((x0, y0, x1, y1))
                tile.save(img_dst / tile_name, quality=95)

                tile_w = x1 - x0
                tile_h = y1 - y0

                new_images.append({
                    "id": tile_img_id,
                    "file_name": tile_name,
                    "width": tile_w,
                    "height": tile_h,
                })

                # Remap annotations to tile coordinates
                for ann in anns:
                    bx, by, bw, bh = ann["bbox"]  # COCO [x, y, w, h]
                    bx2, by2 = bx + bw, by + bh

                    # Skip if box doesn't overlap this tile at all
                    if bx2 <= x0 or bx >= x1 or by2 <= y0 or by >= y1:
                        continue

                    # Clip to tile bounds
                    cx1 = max(bx, x0) - x0
                    cy1 = max(by, y0) - y0
                    cx2 = min(bx2, x1) - x0
                    cy2 = min(by2, y1) - y0

                    cw = cx2 - cx1
                    ch = cy2 - cy1

                    # Must have positive area
                    if cw < 1 or ch < 1:
                        continue

                    if cw * ch < min_bbox_area:
                        continue

                    # Skip if less than 30% of original box is visible
                    if cw * ch < 0.3 * bw * bh:
                        continue

                    ann_id += 1
                    new_annotations.append({
                        "id": ann_id,
                        "image_id": tile_img_id,
                        "category_id": ann["category_id"],
                        "bbox": [cx1, cy1, cw, ch],
                        "area": cw * ch,
                        "iscrowd": 0,
                    })

    print(f"    Tiled: {len(images_meta)} images → {len(new_images)} tiles "
          f"({tile_size}px, {overlap:.0%} overlap)")
    print(f"    Annotations: {len(coco['annotations'])} → {len(new_annotations)} "
          f"(after clipping)")
    n_with_anns = len({a["image_id"] for a in new_annotations})
    n_empty = len(new_images) - n_with_anns
    print(f"    Tiles with objects: {n_with_anns}, empty (negatives): {n_empty}")

    return {
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco["categories"],
    }


def _copy_or_link_images(
    src_dir: Path, dst_dir: Path,
    filenames: set[str], use_symlink: bool,
):
    """Copy or symlink image files from src to dst."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for fname in sorted(filenames):
        src = src_dir / fname
        dst = dst_dir / fname
        if dst.exists():
            copied += 1
            continue
        if not src.exists():
            continue
        if use_symlink:
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)
        copied += 1
    return copied


def convert_eikelboom_to_coco(args):
    """Convert Eikelboom dataset to COCO JSON format.

    The Eikelboom dataset stores raw images in train/val/test directories
    and has a single annotations_images.csv with full-image coordinates.
    The per-split CSVs (annotations_train.csv etc.) reference tiles that
    need to be generated with ImageMagick — we use the raw annotations instead,
    splitting by which files exist in which directory.
    """
    src = Path(args.src)
    dst = Path(args.dst)
    csv_dir = src / args.csv_dir if (src / args.csv_dir).exists() else src

    # Use annotations_images.csv (raw, full-image coords) if available
    raw_csv = csv_dir / "annotations_images.csv"
    if raw_csv.exists():
        print(f"  Using raw annotations: {raw_csv.name}")
        rows = _parse_voc_csv(raw_csv, has_header=True)
    else:
        # Fallback to per-split CSVs
        all_rows = []
        for csv_name in [args.csv_train, args.csv_val, args.csv_test]:
            csv_path = csv_dir / csv_name
            if csv_path.exists():
                all_rows.extend(_parse_voc_csv(csv_path, has_header=args.header))
        rows = all_rows

    if not rows:
        print(f"No annotations found in {csv_dir}")
        sys.exit(1)

    # Build file-to-split mapping from directory contents
    file_to_split: dict[str, str] = {}
    for split in ["train", "val", "test"]:
        split_dir = src / split
        if split_dir.exists():
            for p in split_dir.iterdir():
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                    file_to_split[p.name] = split

    # Split annotations by directory membership
    split_rows: dict[str, list[dict]] = defaultdict(list)
    all_classes: set[str] = set()
    for row in rows:
        basename = Path(row["filepath"]).name
        split = file_to_split.get(basename)
        if split:
            split_rows[split].append(row)
            all_classes.add(row["class_name"])

    class_names = sorted(all_classes)
    print(f"  Classes ({len(class_names)}): {class_names}")
    for split, split_data in sorted(split_rows.items()):
        print(f"  {split}: {len(split_data)} annotations")

    # Convert each split
    for split, split_data in split_rows.items():
        img_src = src / split
        coco = _build_coco_json(split_data, img_src, class_names)

        split_dst = dst / split
        img_dst = split_dst / "images"
        img_dst.mkdir(parents=True, exist_ok=True)

        if args.tile > 0:
            # Tile images and remap annotations
            coco = _tile_coco_dataset(
                coco, img_src, img_dst,
                tile_size=args.tile,
                overlap=args.tile_overlap,
                min_bbox_area=args.tile_min_bbox_area,
            )
        else:
            filenames = {img["file_name"] for img in coco["images"]}
            _copy_or_link_images(img_src, img_dst, filenames, args.symlink)

        ann_path = split_dst / "annotations.json"
        with open(ann_path, "w") as f:
            json.dump(coco, f, indent=2)

        print(f"  {split}: {len(coco['images'])} images, "
              f"{len(coco['annotations'])} annotations → {ann_path}")

    print(f"\nDone. COCO dataset ready at: {dst}")


def convert_voc_csv_to_coco(args):
    """Convert generic Pascal VOC CSV annotations to COCO JSON format."""
    src = Path(args.src)
    dst = Path(args.dst)
    csv_dir = src / args.csv_dir if (src / args.csv_dir).exists() else src

    splits = {}
    for split, csv_name in [("train", args.csv_train), ("val", args.csv_val),
                             ("test", args.csv_test)]:
        csv_path = csv_dir / csv_name
        if csv_path.exists():
            splits[split] = csv_path

    if not splits:
        print(f"No CSV files found in {csv_dir}")
        print(f"  Looked for: {args.csv_train}, {args.csv_val}, {args.csv_test}")
        sys.exit(1)

    # Parse all CSVs to discover classes
    all_rows: dict[str, list[dict]] = {}
    all_classes: set[str] = set()
    for split, csv_path in splits.items():
        rows = _parse_voc_csv(csv_path, has_header=args.header)
        all_rows[split] = rows
        all_classes.update(r["class_name"] for r in rows)
        print(f"  {split}: {len(rows)} annotations from {csv_path.name}")

    class_names = sorted(all_classes)
    print(f"  Classes ({len(class_names)}): {class_names}")

    # Convert each split
    for split, rows in all_rows.items():
        # Determine image source directory
        img_src = src / split
        if not img_src.exists():
            img_src = src

        # Build COCO JSON
        coco = _build_coco_json(rows, img_src, class_names)

        split_dst = dst / split
        img_dst = split_dst / "images"
        img_dst.mkdir(parents=True, exist_ok=True)

        filenames = {img["file_name"] for img in coco["images"]}
        _copy_or_link_images(img_src, img_dst, filenames, args.symlink)

        ann_path = split_dst / "annotations.json"
        with open(ann_path, "w") as f:
            json.dump(coco, f, indent=2)

        print(f"  {split}: {len(coco['images'])} images, "
              f"{len(coco['annotations'])} annotations → {ann_path}")

    print(f"\nDone. COCO dataset ready at: {dst}")


# ─── COCO JSON → YOLO .txt ──────────────────────────────────────────────────

def convert_coco_to_yolo(args):
    """Convert COCO JSON to YOLO .txt labels."""
    src = Path(args.src)
    dst = Path(args.dst)

    # Discover splits
    splits = []
    for split in ["train", "val", "test"]:
        ann_path = src / split / "annotations.json"
        if ann_path.exists():
            splits.append(split)

    if not splits:
        print(f"No COCO splits found in {src} (expected train/annotations.json, etc.)")
        sys.exit(1)

    all_class_names: list[str] = []

    for split in splits:
        ann_path = src / split / "annotations.json"
        img_src = src / split / "images"

        with open(ann_path) as f:
            coco = json.load(f)

        categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
        sorted_cat_ids = sorted(categories.keys())
        cat_id_to_idx = {cid: i for i, cid in enumerate(sorted_cat_ids)}

        if not all_class_names:
            all_class_names = [categories[cid] for cid in sorted_cat_ids]

        images = {img["id"]: img for img in coco["images"]}

        # Group annotations by image
        anns_by_image: dict[int, list] = defaultdict(list)
        for ann in coco["annotations"]:
            anns_by_image[ann["image_id"]].append(ann)

        # Output dirs
        img_dst = dst / "images" / split
        lbl_dst = dst / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        n_images = 0
        n_labels = 0
        for img_id, img_meta in images.items():
            fname = img_meta["file_name"]
            stem = Path(fname).stem
            img_w = img_meta["width"]
            img_h = img_meta["height"]

            # Copy/link image
            src_img = img_src / fname
            dst_img = img_dst / fname
            if src_img.exists() and not dst_img.exists():
                if args.symlink:
                    dst_img.symlink_to(src_img.resolve())
                else:
                    shutil.copy2(src_img, dst_img)
            n_images += 1

            # Write YOLO label
            lines = []
            for ann in anns_by_image.get(img_id, []):
                x, y, w, h = ann["bbox"]  # COCO: [x, y, w, h] pixels
                if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
                    continue
                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h
                cls_idx = cat_id_to_idx[ann["category_id"]]
                lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                n_labels += 1

            (lbl_dst / f"{stem}.txt").write_text("\n".join(lines))

        print(f"  {split}: {n_images} images, {n_labels} labels")

    # Write dataset.yaml
    yaml_content = f"""path: {dst.resolve()}
train: images/train
val: images/val

nc: {len(all_class_names)}
names: {all_class_names}
"""
    (dst / "dataset.yaml").write_text(yaml_content)
    print(f"\n  dataset.yaml written with {len(all_class_names)} classes: {all_class_names}")
    print(f"\nDone. YOLO dataset ready at: {dst}")


# ─── YOLO .txt → COCO JSON ──────────────────────────────────────────────────

def convert_yolo_to_coco(args):
    """Convert YOLO .txt labels to COCO JSON format."""
    import yaml

    src = Path(args.src)
    dst = Path(args.dst)

    # Find dataset.yaml
    yaml_path = None
    for candidate in [src / "dataset.yaml", src / "data.yaml"]:
        if candidate.exists():
            yaml_path = candidate
            break

    if yaml_path is None:
        print(f"No dataset.yaml found in {src}")
        sys.exit(1)

    with open(yaml_path) as f:
        ds = yaml.safe_load(f)

    nc = ds["nc"]
    names = ds["names"]
    ds_root = Path(ds.get("path", src))

    print(f"  Classes ({nc}): {names}")

    for split in ["train", "val", "test"]:
        split_rel = ds.get(split)
        if split_rel is None:
            continue

        img_dir = ds_root / split_rel
        lbl_dir = Path(str(img_dir).replace("/images/", "/labels/")
                       .replace("\\images\\", "\\labels\\"))

        if not img_dir.exists() or not lbl_dir.exists():
            continue

        images = []
        annotations = []
        categories = [{"id": i + 1, "name": name} for i, name in enumerate(names)]
        ann_id = 1

        img_files = sorted(
            p for p in img_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        )

        for img_id_0, img_path in enumerate(img_files):
            img_id = img_id_0 + 1
            with Image.open(img_path) as im:
                w, h = im.size

            images.append({
                "id": img_id,
                "file_name": img_path.name,
                "width": w,
                "height": h,
            })

            txt_path = lbl_dir / (img_path.stem + ".txt")
            if not txt_path.exists():
                continue

            for line in txt_path.read_text().strip().split("\n"):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_idx = int(parts[0])
                cx, cy, bw, bh = (float(v) for v in parts[1:])

                # YOLO normalised → COCO pixel [x, y, w, h]
                px = (cx - bw / 2) * w
                py = (cy - bh / 2) * h
                pw = bw * w
                ph = bh * h

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_idx + 1,  # COCO is 1-indexed
                    "bbox": [round(px, 2), round(py, 2), round(pw, 2), round(ph, 2)],
                    "area": round(pw * ph, 2),
                    "iscrowd": 0,
                })
                ann_id += 1

        # Output
        split_dst = dst / split
        img_dst = split_dst / "images"
        img_dst.mkdir(parents=True, exist_ok=True)

        # Copy/link images
        filenames = {img["file_name"] for img in images}
        _copy_or_link_images(img_dir, img_dst, filenames, args.symlink)

        coco = {"images": images, "annotations": annotations, "categories": categories}
        ann_path = split_dst / "annotations.json"
        with open(ann_path, "w") as f:
            json.dump(coco, f, indent=2)

        print(f"  {split}: {len(images)} images, {len(annotations)} annotations → {ann_path}")

    print(f"\nDone. COCO dataset ready at: {dst}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    src_fmt = args.src_fmt
    dst_fmt = args.dst_fmt

    print(f"\nConverting: {src_fmt} → {dst_fmt}")
    print(f"  Source: {args.src}")
    print(f"  Output: {args.dst}\n")

    if src_fmt == "eikelboom-csv" and dst_fmt == "coco":
        convert_eikelboom_to_coco(args)
    elif src_fmt == "pascal-voc-csv" and dst_fmt == "coco":
        convert_voc_csv_to_coco(args)
    elif src_fmt == "coco" and dst_fmt == "yolo":
        convert_coco_to_yolo(args)
    elif src_fmt == "yolo" and dst_fmt == "coco":
        convert_yolo_to_coco(args)
    elif src_fmt == "coco" and dst_fmt == "coco":
        print("Source and target are both COCO — nothing to convert.")
    elif src_fmt == "yolo" and dst_fmt == "yolo":
        print("Source and target are both YOLO — nothing to convert.")
    elif src_fmt in ("eikelboom-csv", "pascal-voc-csv") and dst_fmt == "yolo":
        # Two-step: CSV → COCO → YOLO
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            args_step1 = argparse.Namespace(**vars(args))
            args_step1.dst = tmp
            args_step1.dst_fmt = "coco"
            convert_voc_csv_to_coco(args_step1)

            print(f"\n  Step 2: COCO → YOLO\n")
            args_step2 = argparse.Namespace(**vars(args))
            args_step2.src = tmp
            args_step2.src_fmt = "coco"
            convert_coco_to_yolo(args_step2)
    else:
        print(f"Unsupported conversion: {src_fmt} → {dst_fmt}")
        sys.exit(1)


if __name__ == "__main__":
    main()