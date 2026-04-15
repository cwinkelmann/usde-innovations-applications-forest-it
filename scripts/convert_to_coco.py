"""
Convert Tier 1 aerial datasets to a unified COCO JSON format.

Superclasses: animal (1), person (2), vehicle (3)

Usage:
    python scripts/convert_to_coco.py --dataset izembek --src /Volumes/storage/Datasets/Izembek-lagoon-birds --dst /Volumes/storage/Datasets/aerial_megadetector/processed/animal/izembek
    python scripts/convert_to_coco.py --dataset all --base /Volumes/storage/Datasets --dst /Volumes/storage/Datasets/aerial_megadetector/processed
"""

import argparse
import csv
import glob
import json
import os
import shutil
from pathlib import Path
from typing import Any

# Unified categories
CATEGORIES = [
    {"id": 1, "name": "animal", "supercategory": "object"},
    {"id": 2, "name": "person", "supercategory": "object"},
    {"id": 3, "name": "vehicle", "supercategory": "object"},
]

SUPERCLASS_MAP = {
    # Animal classes
    "brant": 1, "canada": 1, "gull": 1, "emperor": 1, "other_bird": 1,
    "elephant": 1, "zebra": 1, "giraffe": 1,
    "buffalo": 1, "kob": 1, "warthog": 1, "waterbuck": 1, "alcelaphinae": 1,
    "bird": 1, "animal": 1,
    "sheep": 1, "cattle": 1, "seal": 1, "camel": 1, "kiang": 1,
    # Person classes
    "pedestrian": 2, "person": 2,
    # Vehicle classes
    "small-vehicle": 3, "large-vehicle": 3, "ship": 3, "plane": 3,
    "helicopter": 3, "harbor": 3,
    "car": 3, "van": 3, "bus": 3, "truck": 3, "motor": 3, "bicycle": 3,
    "awning-tricycle": 3, "tricycle": 3,
}

# DOTA classes to exclude (infrastructure, not our target)
DOTA_EXCLUDE = {
    "storage-tank", "baseball-diamond", "tennis-court", "basketball-court",
    "ground-track-field", "soccer-ball-field", "roundabout", "swimming-pool",
    "bridge", "container-crane", "airport", "helipad",
}


def make_coco_skeleton():
    return {
        "images": [],
        "annotations": [],
        "categories": CATEGORIES,
        "info": {"description": "Aerial MegaDetector unified dataset", "version": "1.0"},
    }


def convert_izembek(src: str, dst: str):
    """A1: Izembek Lagoon Waterfowl -- already COCO, remap categories."""
    meta_path = os.path.join(src, "izembek-lagoon-birds-metadata.json")
    with open(meta_path) as f:
        data = json.load(f)

    cat_map = {}
    for cat in data["categories"]:
        name = cat["name"].lower()
        if name == "empty":
            continue
        if name == "other":
            name = "other_bird"
        cat_map[cat["id"]] = SUPERCLASS_MAP.get(name, 1)

    coco = make_coco_skeleton()
    img_dir = os.path.join(src, "images")

    for img in data["images"]:
        coco["images"].append({
            "id": img["id"],
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
        })

    ann_id = 0
    skipped = 0
    for ann in data["annotations"]:
        cat_id_orig = ann["category_id"]
        if cat_id_orig not in cat_map:
            skipped += 1
            continue
        bbox = ann["bbox"]  # [x, y, w, h]
        coco["annotations"].append({
            "id": ann_id,
            "image_id": ann["image_id"],
            "category_id": cat_map[cat_id_orig],
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "iscrowd": 0,
            "attributes": {"original_class": _get_cat_name(data["categories"], cat_id_orig)},
        })
        ann_id += 1

    os.makedirs(dst, exist_ok=True)
    out_path = os.path.join(dst, "annotations.json")
    with open(out_path, "w") as f:
        json.dump(coco, f)

    # Symlink images directory
    img_link = os.path.join(dst, "images")
    if not os.path.exists(img_link):
        os.symlink(img_dir, img_link)

    print(f"[Izembek] {len(coco['images'])} images, {len(coco['annotations'])} annotations (skipped {skipped} empty), saved to {out_path}")
    return coco


def convert_eikelboom(src: str, dst: str):
    """A2: Eikelboom Savanna -- CSV format (path,x1,y1,x2,y2,Class)."""
    base = os.path.join(src, "Improving the precision and accuracy of animal population estimates with aerial image object detection_1_all")
    coco = make_coco_skeleton()
    images_by_name = {}
    ann_id = 0
    img_id = 0

    class_map = {"Elephant": "elephant", "Zebra": "zebra", "Giraffe": "giraffe"}

    for split in ["annotations_train.csv", "annotations_val.csv", "annotations_test.csv"]:
        csv_path = os.path.join(base, split)
        if not os.path.exists(csv_path):
            print(f"  [Eikelboom] Warning: {csv_path} not found, skipping")
            continue
        with open(csv_path) as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6:
                    continue
                img_rel, x1, y1, x2, y2, cls_name = row[0], int(row[1]), int(row[2]), int(row[3]), int(row[4]), row[5]

                if img_rel not in images_by_name:
                    img_path = os.path.join(base, img_rel)
                    images_by_name[img_rel] = img_id
                    # We don't have image dimensions in CSV; use placeholder
                    coco["images"].append({
                        "id": img_id,
                        "file_name": img_rel,
                        "width": 0,
                        "height": 0,
                    })
                    img_id += 1

                w = x2 - x1
                h = y2 - y1
                original_class = class_map.get(cls_name, cls_name.lower())
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": images_by_name[img_rel],
                    "category_id": SUPERCLASS_MAP.get(original_class, 1),
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "attributes": {"original_class": original_class},
                })
                ann_id += 1

    os.makedirs(dst, exist_ok=True)
    out_path = os.path.join(dst, "annotations.json")
    with open(out_path, "w") as f:
        json.dump(coco, f)

    # Symlink image dirs
    for subdir in ["train", "val", "test"]:
        src_dir = os.path.join(base, subdir)
        link = os.path.join(dst, subdir)
        if os.path.isdir(src_dir) and not os.path.exists(link):
            os.symlink(src_dir, link)

    print(f"[Eikelboom] {len(coco['images'])} images, {len(coco['annotations'])} annotations, saved to {out_path}")
    return coco


def convert_delplanque(src: str, dst: str):
    """A4: Delplanque African Mammals -- CSV (Image,x1,y1,x2,y2,Label)."""
    gt_dir = os.path.join(src, "general_dataset", "groundtruth", "csv")
    coco = make_coco_skeleton()
    images_by_name = {}
    ann_id = 0
    img_id = 0

    # Label mapping from the dataset's numeric labels
    label_map = {
        "1": "alcelaphinae",
        "2": "buffalo",
        "3": "elephant",
        "4": "kob",
        "5": "warthog",
        "6": "waterbuck",
    }

    for csv_file in sorted(glob.glob(os.path.join(gt_dir, "*.csv"))):
        split_name = os.path.basename(csv_file).replace("_big_size_A_B_E_K_WH_WB.csv", "")
        with open(csv_file) as f:
            reader = csv.reader(f)
            header = next(reader, None)  # Skip header
            for row in reader:
                if len(row) < 5:
                    continue
                img_name = row[0]
                x1, y1, x2, y2 = int(row[1]), int(row[2]), int(row[3]), int(row[4])
                label_num = row[5] if len(row) > 5 else "0"

                img_key = f"{split_name}/{img_name}"
                if img_key not in images_by_name:
                    images_by_name[img_key] = img_id
                    coco["images"].append({
                        "id": img_id,
                        "file_name": img_key,
                        "width": 6000,
                        "height": 4000,
                    })
                    img_id += 1

                w = x2 - x1
                h = y2 - y1
                original_class = label_map.get(label_num, f"unknown_{label_num}")
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": images_by_name[img_key],
                    "category_id": 1,  # All are animals
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "attributes": {"original_class": original_class},
                })
                ann_id += 1

    os.makedirs(dst, exist_ok=True)
    out_path = os.path.join(dst, "annotations.json")
    with open(out_path, "w") as f:
        json.dump(coco, f)

    # Symlink image directories
    for subdir in ["train", "val", "test"]:
        src_dir = os.path.join(src, "general_dataset", subdir)
        link = os.path.join(dst, subdir)
        if os.path.isdir(src_dir) and not os.path.exists(link):
            os.symlink(src_dir, link)

    print(f"[Delplanque] {len(coco['images'])} images, {len(coco['annotations'])} annotations, saved to {out_path}")
    return coco


def convert_weinstein(src: str, dst: str):
    """A5: Weinstein Global Birds -- aggregate COCO JSONs from multiple sites."""
    coco = make_coco_skeleton()
    ann_id = 0
    img_id = 0
    images_seen = set()

    # Find all coco_format JSON files
    json_files = sorted(glob.glob(os.path.join(src, "**", "coco_format*.json"), recursive=True))
    if not json_files:
        # Try deepforest CSV format
        csv_files = sorted(glob.glob(os.path.join(src, "**", "deep_forest_format.csv"), recursive=True))
        for csv_file in csv_files:
            site = Path(csv_file).parts[-3] if len(Path(csv_file).parts) > 3 else "unknown"
            with open(csv_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_name = row.get("image_path", "")
                    if not img_name:
                        continue
                    img_key = f"{site}/{img_name}"
                    if img_key not in images_seen:
                        images_seen.add(img_key)
                        coco["images"].append({
                            "id": img_id,
                            "file_name": img_key,
                            "width": 0,
                            "height": 0,
                        })
                        img_id += 1

                    x1 = float(row.get("xmin", 0))
                    y1 = float(row.get("ymin", 0))
                    x2 = float(row.get("xmax", 0))
                    y2 = float(row.get("ymax", 0))
                    w, h = x2 - x1, y2 - y1
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": coco["images"][-1]["id"] if img_key in images_seen else img_id - 1,
                        "category_id": 1,
                        "bbox": [x1, y1, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                        "attributes": {"original_class": "bird", "site": site},
                    })
                    ann_id += 1
        print(f"[Weinstein] Parsed {len(csv_files)} CSV files")
    else:
        for jf in json_files:
            site = Path(jf).parts[-3] if len(Path(jf).parts) > 3 else "unknown"
            with open(jf) as f:
                data = json.load(f)

            id_remap = {}
            for img in data.get("images", []):
                old_id = img["id"]
                img_key = f"{site}/{img.get('file_name', '')}"
                if img_key not in images_seen:
                    images_seen.add(img_key)
                    id_remap[old_id] = img_id
                    coco["images"].append({
                        "id": img_id,
                        "file_name": img_key,
                        "width": img.get("width", 0),
                        "height": img.get("height", 0),
                    })
                    img_id += 1
                else:
                    # Find existing id
                    for existing in coco["images"]:
                        if existing["file_name"] == img_key:
                            id_remap[old_id] = existing["id"]
                            break

            for ann in data.get("annotations", []):
                old_img_id = ann["image_id"]
                if old_img_id not in id_remap:
                    continue
                bbox = ann.get("bbox", [0, 0, 0, 0])
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": id_remap[old_img_id],
                    "category_id": 1,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3] if len(bbox) == 4 else 0,
                    "iscrowd": 0,
                    "attributes": {"original_class": "bird", "site": site},
                })
                ann_id += 1

    os.makedirs(dst, exist_ok=True)
    out_path = os.path.join(dst, "annotations.json")
    with open(out_path, "w") as f:
        json.dump(coco, f)
    print(f"[Weinstein] {len(coco['images'])} images, {len(coco['annotations'])} annotations, saved to {out_path}")
    return coco


def convert_dota(src: str, dst: str):
    """V1: DOTA -- OBB format to axis-aligned HBB, vehicle classes only."""
    coco = make_coco_skeleton()
    ann_id = 0
    img_id = 0

    # Process train and val
    splits = []
    # Train: images in src/images/, labels in src/labelTxt-v1.0/labelTxt/
    train_img_dir = os.path.join(src, "images")
    train_lbl_dir = os.path.join(src, "labelTxt-v1.0", "labelTxt")
    if os.path.isdir(train_img_dir) and os.path.isdir(train_lbl_dir):
        splits.append(("train", train_img_dir, train_lbl_dir))

    # Val: images in src/val/images/images/, labels need to be unzipped
    val_img_dir = os.path.join(src, "val", "images", "images")
    # Check if val labels exist
    val_lbl_zip = os.path.join(src, "val", "labelTxt-v1.0", "labelTxt.zip")
    val_lbl_dir = os.path.join(src, "val", "labelTxt-v1.0", "labelTxt")
    if os.path.isfile(val_lbl_zip) and not os.path.isdir(val_lbl_dir):
        import zipfile
        print(f"  Extracting val labels from {val_lbl_zip}...")
        with zipfile.ZipFile(val_lbl_zip, 'r') as zf:
            zf.extractall(os.path.join(src, "val", "labelTxt-v1.0"))
    if os.path.isdir(val_img_dir) and os.path.isdir(val_lbl_dir):
        splits.append(("val", val_img_dir, val_lbl_dir))

    for split_name, img_dir, lbl_dir in splits:
        label_files = sorted(glob.glob(os.path.join(lbl_dir, "*.txt")))
        for lbl_file in label_files:
            stem = Path(lbl_file).stem
            # Find corresponding image
            img_path = None
            for ext in [".png", ".jpg", ".jpeg", ".tif"]:
                candidate = os.path.join(img_dir, stem + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            if not img_path:
                # Images might be one level up
                for ext in [".png", ".jpg", ".jpeg"]:
                    candidate = os.path.join(img_dir, "..", stem + ext)
                    if os.path.exists(candidate):
                        img_path = candidate
                        break

            with open(lbl_file) as f:
                lines = f.readlines()

            # Parse DOTA format: first 2 lines are metadata (imagesource, gsd)
            annotations_for_image = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith("imagesource") or line.startswith("gsd"):
                    continue
                parts = line.split()
                if len(parts) < 9:
                    continue
                coords = [float(parts[i]) for i in range(8)]
                cls_name = parts[8]
                difficult = int(parts[9]) if len(parts) > 9 else 0

                if cls_name in DOTA_EXCLUDE:
                    continue

                superclass_id = SUPERCLASS_MAP.get(cls_name)
                if superclass_id is None:
                    continue

                # Convert OBB (8 points) to axis-aligned HBB
                xs = coords[0::2]
                ys = coords[1::2]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                w = x_max - x_min
                h = y_max - y_min

                annotations_for_image.append({
                    "category_id": superclass_id,
                    "bbox": [x_min, y_min, w, h],
                    "original_class": cls_name,
                    "difficult": difficult,
                })

            if img_path or annotations_for_image:
                rel_path = f"{split_name}/{stem}{Path(img_path).suffix}" if img_path else f"{split_name}/{stem}.png"
                coco["images"].append({
                    "id": img_id,
                    "file_name": rel_path,
                    "width": 0,
                    "height": 0,
                })
                for ann_data in annotations_for_image:
                    bbox = ann_data["bbox"]
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": ann_data["category_id"],
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                        "attributes": {"original_class": ann_data["original_class"], "difficult": ann_data["difficult"]},
                    })
                    ann_id += 1
                img_id += 1

    os.makedirs(dst, exist_ok=True)
    out_path = os.path.join(dst, "annotations.json")
    with open(out_path, "w") as f:
        json.dump(coco, f)

    print(f"[DOTA] {len(coco['images'])} images, {len(coco['annotations'])} vehicle/person annotations, saved to {out_path}")
    return coco


def convert_waid(src: str, dst: str):
    """A3: WAID -- YOLO TXT format."""
    coco = make_coco_skeleton()
    ann_id = 0
    img_id = 0

    waid_classes = {0: "sheep", 1: "cattle", 2: "seal", 3: "camel", 4: "kiang", 5: "zebra"}

    # Find image directories
    img_dirs = sorted(glob.glob(os.path.join(src, "WAID", "**", "images"), recursive=True))
    if not img_dirs:
        img_dirs = sorted(glob.glob(os.path.join(src, "**", "images"), recursive=True))

    for img_dir in img_dirs:
        lbl_dir = img_dir.replace("images", "labels")
        if not os.path.isdir(lbl_dir):
            continue

        for img_file in sorted(glob.glob(os.path.join(img_dir, "**", "*.jpg"), recursive=True) + glob.glob(os.path.join(img_dir, "**", "*.png"), recursive=True)):
            stem = Path(img_file).stem
            # Mirror the subdirectory structure for labels
            img_rel = os.path.relpath(img_file, img_dir)
            lbl_rel = os.path.join(os.path.dirname(img_rel), stem + ".txt")
            lbl_file = os.path.join(lbl_dir, lbl_rel)

            rel_path = os.path.relpath(img_file, src)
            coco["images"].append({
                "id": img_id,
                "file_name": rel_path,
                "width": 640,
                "height": 640,
            })

            if os.path.exists(lbl_file):
                with open(lbl_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls_idx = int(parts[0])
                        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        # Convert YOLO normalized to pixel coords (640x640)
                        px = (cx - w / 2) * 640
                        py = (cy - h / 2) * 640
                        pw = w * 640
                        ph = h * 640

                        original_class = waid_classes.get(cls_idx, f"class_{cls_idx}")
                        coco["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": 1,  # All animals
                            "bbox": [px, py, pw, ph],
                            "area": pw * ph,
                            "iscrowd": 0,
                            "attributes": {"original_class": original_class},
                        })
                        ann_id += 1

            img_id += 1

    os.makedirs(dst, exist_ok=True)
    out_path = os.path.join(dst, "annotations.json")
    with open(out_path, "w") as f:
        json.dump(coco, f)
    print(f"[WAID] {len(coco['images'])} images, {len(coco['annotations'])} annotations, saved to {out_path}")
    return coco


def convert_aed(src: str, dst: str):
    """A6: AED -- Point annotations in CSV.
    Format: training_elephants.csv / test_elephants.csv with columns: image_hash, x, y
    Image metadata: training_images.csv / test_images.csv with columns: image_hash, sortie_id, width, height, GSD, ...
    """
    coco = make_coco_skeleton()
    ann_id = 0
    img_id = 0
    images_by_name = {}

    # Load image metadata
    img_meta = {}
    for meta_csv in ["training_images.csv", "test_images.csv"]:
        meta_path = os.path.join(src, meta_csv)
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            for row in csv.reader(f):
                if len(row) >= 4:
                    img_meta[row[0]] = {"width": int(row[2]), "height": int(row[3])}

    # Load elephant point annotations
    for split, ann_csv, img_dir_name in [
        ("train", "training_elephants.csv", "training_images"),
        ("test", "test_elephants.csv", "test_images"),
    ]:
        ann_path = os.path.join(src, ann_csv)
        if not os.path.exists(ann_path):
            print(f"  [AED] {ann_path} not found, skipping {split}")
            continue

        with open(ann_path) as f:
            for row in csv.reader(f):
                if len(row) < 3:
                    continue
                img_hash = row[0]
                try:
                    x = float(row[1])
                    y = float(row[2])
                except ValueError:
                    continue

                img_key = f"{img_dir_name}/{img_hash}.jpg"
                if img_key not in images_by_name:
                    images_by_name[img_key] = img_id
                    meta = img_meta.get(img_hash, {"width": 0, "height": 0})
                    coco["images"].append({
                        "id": img_id,
                        "file_name": img_key,
                        "width": meta["width"],
                        "height": meta["height"],
                    })
                    img_id += 1

                # Point -> pseudo-box (radius 40px for elephants)
                radius = 40
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": images_by_name[img_key],
                    "category_id": 1,
                    "bbox": [x - radius, y - radius, 2 * radius, 2 * radius],
                    "area": (2 * radius) ** 2,
                    "iscrowd": 0,
                    "attributes": {"original_class": "elephant", "annotation_type": "point", "point": [x, y]},
                })
                ann_id += 1

    os.makedirs(dst, exist_ok=True)
    out_path = os.path.join(dst, "annotations.json")
    with open(out_path, "w") as f:
        json.dump(coco, f)

    # Symlink image directories
    for subdir in ["training_images", "test_images"]:
        src_dir = os.path.join(src, subdir)
        link = os.path.join(dst, subdir)
        if os.path.isdir(src_dir) and not os.path.exists(link):
            os.symlink(src_dir, link)

    print(f"[AED] {len(coco['images'])} images, {len(coco['annotations'])} annotations, saved to {out_path}")
    return coco


def convert_heridal(src: str, dst: str):
    """P5: HERIDAL -- person detection in aerial SAR imagery.
    Supports Roboflow CSV format: filename,width,height,class,xmin,ymin,xmax,ymax
    and Pascal VOC XML format.
    """
    coco = make_coco_skeleton()
    ann_id = 0
    img_id = 0
    images_by_name = {}

    # Look for Roboflow-style CSV annotations (preferred)
    csv_files = sorted(glob.glob(os.path.join(src, "**", "_annotations.csv"), recursive=True))
    xml_files = sorted(glob.glob(os.path.join(src, "**", "*.xml"), recursive=True))

    if csv_files:
        print(f"  [HERIDAL] Found {len(csv_files)} Roboflow CSV annotation files")
        for csv_file in csv_files:
            split_dir = os.path.dirname(csv_file)
            split_name = os.path.basename(split_dir)
            with open(csv_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row.get("filename", "")
                    if not filename:
                        continue
                    img_key = f"{split_name}/{filename}"
                    if img_key not in images_by_name:
                        images_by_name[img_key] = img_id
                        coco["images"].append({
                            "id": img_id,
                            "file_name": img_key,
                            "width": int(row.get("width", 0)),
                            "height": int(row.get("height", 0)),
                        })
                        img_id += 1

                    cls_name = row.get("class", "human").lower()
                    x1 = int(row.get("xmin", 0))
                    y1 = int(row.get("ymin", 0))
                    x2 = int(row.get("xmax", 0))
                    y2 = int(row.get("ymax", 0))
                    w = x2 - x1
                    h = y2 - y1
                    if w <= 0 or h <= 0:
                        continue
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": images_by_name[img_key],
                        "category_id": 2,  # person
                        "bbox": [x1, y1, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                        "attributes": {"original_class": cls_name},
                    })
                    ann_id += 1

    elif xml_files:
        import xml.etree.ElementTree as ET
        print(f"  [HERIDAL] Found {len(xml_files)} VOC XML files")
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                filename = root.find("filename")
                if filename is None:
                    continue
                img_name = filename.text
                size = root.find("size")
                w_img = int(size.find("width").text) if size is not None and size.find("width") is not None else 0
                h_img = int(size.find("height").text) if size is not None and size.find("height") is not None else 0

                coco["images"].append({
                    "id": img_id,
                    "file_name": img_name,
                    "width": w_img,
                    "height": h_img,
                })

                for obj in root.findall("object"):
                    cls_name = obj.find("name").text.lower() if obj.find("name") is not None else "person"
                    bndbox = obj.find("bndbox")
                    if bndbox is None:
                        continue
                    x1 = float(bndbox.find("xmin").text)
                    y1 = float(bndbox.find("ymin").text)
                    x2 = float(bndbox.find("xmax").text)
                    y2 = float(bndbox.find("ymax").text)
                    w = x2 - x1
                    h = y2 - y1
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 2,  # person
                        "bbox": [x1, y1, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                        "attributes": {"original_class": cls_name},
                    })
                    ann_id += 1
                img_id += 1
            except ET.ParseError:
                continue
    else:
        print(f"  [HERIDAL] No annotations found!")

    os.makedirs(dst, exist_ok=True)
    out_path = os.path.join(dst, "annotations.json")
    with open(out_path, "w") as f:
        json.dump(coco, f)
    print(f"[HERIDAL] {len(coco['images'])} images, {len(coco['annotations'])} annotations, saved to {out_path}")
    return coco


def convert_uavdt(src: str, dst: str):
    """V4: UAVDT -- vehicle detection from drone video frames."""
    coco = make_coco_skeleton()
    ann_id = 0
    img_id = 0

    uavdt_classes = {1: "car", 2: "truck", 3: "bus"}

    # UAVDT structure: sequences with frame images and GT annotations
    gt_files = sorted(glob.glob(os.path.join(src, "**", "gt_whole.txt"), recursive=True))
    if not gt_files:
        gt_files = sorted(glob.glob(os.path.join(src, "**", "gt.txt"), recursive=True))
    if not gt_files:
        # Try finding annotation files
        gt_files = sorted(glob.glob(os.path.join(src, "**", "*_gt.txt"), recursive=True))

    print(f"  [UAVDT] Found {len(gt_files)} GT files")

    images_seen = {}
    for gt_file in gt_files:
        seq_dir = os.path.dirname(gt_file)
        seq_name = os.path.basename(seq_dir)
        img_dir = os.path.join(seq_dir, "img1") if os.path.isdir(os.path.join(seq_dir, "img1")) else seq_dir

        with open(gt_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue
                frame_id = int(parts[0])
                obj_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                # parts[6] = score/flag, parts[7] = class
                cls_id = int(parts[7]) if len(parts) > 7 else 1

                img_name = f"{seq_name}/img{frame_id:06d}.jpg"
                if img_name not in images_seen:
                    images_seen[img_name] = img_id
                    coco["images"].append({
                        "id": img_id,
                        "file_name": img_name,
                        "width": 1024,
                        "height": 540,
                    })
                    img_id += 1

                original_class = uavdt_classes.get(cls_id, "vehicle")
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": images_seen[img_name],
                    "category_id": 3,  # vehicle
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "attributes": {"original_class": original_class, "track_id": obj_id},
                })
                ann_id += 1

    os.makedirs(dst, exist_ok=True)
    out_path = os.path.join(dst, "annotations.json")
    with open(out_path, "w") as f:
        json.dump(coco, f)
    print(f"[UAVDT] {len(coco['images'])} images, {len(coco['annotations'])} annotations, saved to {out_path}")
    return coco


def convert_tinyperson(src: str, dst: str):
    """P3: TinyPerson -- already COCO format, remap categories to person."""
    coco = make_coco_skeleton()
    ann_id = 0
    img_id = 0

    # Try multiple annotation paths
    for ann_name in [
        "annotations/tiny_set_train.json",
        "annotations/tiny_set_test.json",
        "tiny_set/annotations/tiny_set_train.json",
        "tiny_set/annotations/tiny_set_test.json",
    ]:
        ann_path = os.path.join(src, ann_name)
        if not os.path.exists(ann_path):
            continue

        split = "train" if "train" in ann_name else "test"
        print(f"  [TinyPerson] Loading {ann_path}")
        with open(ann_path) as f:
            data = json.load(f)

        id_remap = {}
        for img in data.get("images", []):
            old_id = img["id"]
            id_remap[old_id] = img_id
            coco["images"].append({
                "id": img_id,
                "file_name": f"{split}/{img.get('file_name', '')}",
                "width": img.get("width", 0),
                "height": img.get("height", 0),
            })
            img_id += 1

        for ann in data.get("annotations", []):
            # Skip ignored, uncertain, and logo annotations
            if ann.get("ignore", False) or ann.get("uncertain", False) or ann.get("logo", False):
                continue
            old_img_id = ann["image_id"]
            if old_img_id not in id_remap:
                continue
            bbox = ann.get("bbox", [0, 0, 0, 0])
            orig_cat = ann.get("category_id", 1)
            # TinyPerson categories: 1=sea_person, 2=earth_person -> both map to person
            original_class = "sea_person" if orig_cat == 1 else "earth_person"
            coco["annotations"].append({
                "id": ann_id,
                "image_id": id_remap[old_img_id],
                "category_id": 2,  # person
                "bbox": bbox,
                "area": bbox[2] * bbox[3] if len(bbox) == 4 else 0,
                "iscrowd": 0,
                "attributes": {"original_class": original_class},
            })
            ann_id += 1

    os.makedirs(dst, exist_ok=True)
    out_path = os.path.join(dst, "annotations.json")
    with open(out_path, "w") as f:
        json.dump(coco, f)
    print(f"[TinyPerson] {len(coco['images'])} images, {len(coco['annotations'])} annotations, saved to {out_path}")
    return coco


def convert_visdrone(src: str, dst: str):
    """P1/V2: VisDrone-DET -- person + vehicle classes from drone imagery.
    VisDrone annotation format: per-image TXT with lines:
    bbox_left,bbox_top,bbox_width,bbox_height,score,object_category,truncation,occlusion
    Categories: 0=ignored, 1=pedestrian, 2=people, 3=bicycle, 4=car, 5=van,
                6=truck, 7=tricycle, 8=awning-tricycle, 9=bus, 10=motor, 11=others
    """
    coco = make_coco_skeleton()
    ann_id = 0
    img_id = 0

    visdrone_to_super = {
        1: (2, "pedestrian"), 2: (2, "person"),
        3: (3, "bicycle"), 4: (3, "car"), 5: (3, "van"),
        6: (3, "truck"), 7: (3, "tricycle"), 8: (3, "awning-tricycle"),
        9: (3, "bus"), 10: (3, "motor"),
    }

    for split in ["VisDrone2019-DET-train", "VisDrone2019-DET-val", "VisDrone2019-DET-test-dev"]:
        split_dir = os.path.join(src, split)
        img_dir = os.path.join(split_dir, "images")
        ann_dir = os.path.join(split_dir, "annotations")
        if not os.path.isdir(img_dir) or not os.path.isdir(ann_dir):
            continue

        print(f"  [VisDrone] Processing {split}")
        for ann_file in sorted(glob.glob(os.path.join(ann_dir, "*.txt"))):
            stem = Path(ann_file).stem
            img_path = None
            for ext in [".jpg", ".png", ".jpeg"]:
                candidate = os.path.join(img_dir, stem + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break

            rel_name = f"{split}/{stem}{Path(img_path).suffix}" if img_path else f"{split}/{stem}.jpg"
            coco["images"].append({
                "id": img_id,
                "file_name": rel_name,
                "width": 0,
                "height": 0,
            })

            with open(ann_file) as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 6:
                        continue
                    x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                    cat_id_vd = int(parts[5])
                    if cat_id_vd not in visdrone_to_super:
                        continue
                    super_id, orig_cls = visdrone_to_super[cat_id_vd]
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": super_id,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                        "attributes": {"original_class": orig_cls},
                    })
                    ann_id += 1
            img_id += 1

    os.makedirs(dst, exist_ok=True)
    out_path = os.path.join(dst, "annotations.json")
    with open(out_path, "w") as f:
        json.dump(coco, f)
    print(f"[VisDrone] {len(coco['images'])} images, {len(coco['annotations'])} annotations, saved to {out_path}")
    return coco


def _get_cat_name(categories, cat_id):
    for c in categories:
        if c["id"] == cat_id:
            return c["name"].lower()
    return f"unknown_{cat_id}"


# --- Dataset registry ---
DATASET_REGISTRY = {
    "izembek": {
        "src_default": "Izembek-lagoon-birds",
        "dst_subdir": "animal/izembek",
        "converter": convert_izembek,
    },
    "eikelboom": {
        "src_default": "ImprovingPrecisionAccuracy_Eikelboom2019data",
        "dst_subdir": "animal/eikelboom",
        "converter": convert_eikelboom,
    },
    "delplanque": {
        "src_default": "africa_elephants_uliege",
        "dst_subdir": "animal/delplanque",
        "converter": convert_delplanque,
    },
    "weinstein": {
        "src_default": "deep_forest_birds",
        "dst_subdir": "animal/weinstein_birds",
        "converter": convert_weinstein,
    },
    "dota": {
        "src_default": "DOTA",
        "dst_subdir": "vehicle/dota",
        "converter": convert_dota,
    },
    "waid": {
        "src_default": "WAID",
        "dst_subdir": "animal/waid",
        "converter": convert_waid,
    },
    "aed": {
        "src_default": "AED_extracted",
        "dst_subdir": "animal/aed_elephants",
        "converter": convert_aed,
    },
    "heridal": {
        "src_default": "HERIDAL",
        "dst_subdir": "person/heridal",
        "converter": convert_heridal,
    },
    "tinyperson": {
        "src_default": "TinyPerson",
        "dst_subdir": "person/tinyperson",
        "converter": convert_tinyperson,
    },
    "uavdt": {
        "src_default": "UAVDT",
        "dst_subdir": "vehicle/uavdt",
        "converter": convert_uavdt,
    },
    "visdrone": {
        "src_default": "VisDrone",
        "dst_subdir": "person_vehicle/visdrone",
        "converter": convert_visdrone,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Convert aerial datasets to unified COCO format")
    parser.add_argument("--dataset", required=True, help="Dataset name or 'all'")
    parser.add_argument("--base", default="/Volumes/storage/Datasets", help="Base directory containing raw datasets")
    parser.add_argument("--src", default=None, help="Override source directory")
    parser.add_argument("--dst", default=None, help="Override destination directory")
    args = parser.parse_args()

    base_dst = os.path.join(args.base, "aerial_megadetector", "processed")

    if args.dataset == "all":
        datasets = list(DATASET_REGISTRY.keys())
    else:
        datasets = [args.dataset]

    for name in datasets:
        if name not in DATASET_REGISTRY:
            print(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
            continue

        reg = DATASET_REGISTRY[name]
        src = args.src or os.path.join(args.base, reg["src_default"])
        dst = args.dst or os.path.join(base_dst, reg["dst_subdir"])

        if not os.path.exists(src):
            print(f"[{name}] Source not found: {src} -- skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Converting: {name}")
        print(f"  Source: {src}")
        print(f"  Destination: {dst}")
        print(f"{'='*60}")
        reg["converter"](src, dst)


if __name__ == "__main__":
    main()
