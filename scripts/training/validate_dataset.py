#!/usr/bin/env python
"""
Pre-flight dataset validator.
Checks your dataset before training starts to catch common issues early.

Usage:
    python validate_dataset.py --task detection       --dataset_dir ./data/my_coco_dataset
    python validate_dataset.py --task classification  --dataset_dir ./data/my_images
    python validate_dataset.py --task seg_instance    --dataset_dir ./data/my_coco_dataset
    python validate_dataset.py --task seg_semantic    --dataset_dir ./data/my_semantic_dataset
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True,
                   choices=["detection", "classification", "seg_instance", "seg_semantic",
                            "yolo"])
    p.add_argument("--dataset_dir", required=True,
                   help="Root dir of the dataset (for yolo: dir containing dataset.yaml, "
                        "or the dataset.yaml path itself)")
    p.add_argument("--max_images", type=int, default=500,
                   help="Max images to open for integrity checks")
    return p.parse_args()


# ─── DETECTION ────────────────────────────────────────────────────────────────

def validate_detection(dataset_dir: str, max_images: int):
    print(f"\n🔍  Validating COCO detection dataset: {dataset_dir}\n")
    errors, warnings = [], []
    root = Path(dataset_dir)
    ok = True

    for split in ["train", "val"]:
        split_dir = root / split
        ann_path  = split_dir / "annotations.json"
        img_dir   = split_dir / "images"

        print(f"── {split}/")

        # Check structure
        if not split_dir.exists():
            errors.append(f"  ❌  Missing split directory: {split_dir}")
            ok = False
            continue
        if not ann_path.exists():
            errors.append(f"  ❌  Missing annotations.json in {split_dir}")
            ok = False
            continue
        if not img_dir.exists():
            errors.append(f"  ❌  Missing images/ directory in {split_dir}")
            ok = False
            continue

        # Load annotations
        with open(ann_path) as f:
            coco = json.load(f)

        required_keys = ["images", "annotations", "categories"]
        for k in required_keys:
            if k not in coco:
                errors.append(f"  ❌  '{k}' key missing from annotations.json")
                ok = False

        if not ok:
            continue

        images_meta = {img["id"]: img for img in coco["images"]}
        categories   = {cat["id"]: cat["name"] for cat in coco["categories"]}
        ann_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            ann_by_image[ann["image_id"]].append(ann)

        print(f"   Images in JSON : {len(images_meta)}")
        print(f"   Annotations    : {len(coco['annotations'])}")
        print(f"   Categories ({len(categories)}): {list(categories.values())}")

        # Check image files exist
        missing_files = []
        for img_meta in coco["images"]:
            img_path = img_dir / img_meta["file_name"]
            if not img_path.exists():
                missing_files.append(img_meta["file_name"])
        if missing_files:
            n = len(missing_files)
            errors.append(f"  ❌  {n} image file(s) listed in JSON but missing on disk")
            if n <= 5:
                for f in missing_files:
                    errors.append(f"       {f}")
            ok = False
        else:
            print(f"   All image files : ✓ found on disk")

        # Check bounding boxes
        zero_area, negative, out_of_bounds = 0, 0, 0
        for ann in coco["annotations"]:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                zero_area += 1
            if x < 0 or y < 0:
                negative += 1
            img = images_meta.get(ann["image_id"])
            if img:
                if x + w > img["width"] + 5 or y + h > img["height"] + 5:
                    out_of_bounds += 1

        if zero_area:
            warnings.append(f"  ⚠️   {zero_area} zero/negative-area bboxes (will be filtered)")
        if negative:
            warnings.append(f"  ⚠️   {negative} annotations with negative x/y coords")
        if out_of_bounds:
            warnings.append(f"  ⚠️   {out_of_bounds} bboxes extend outside image bounds")

        # Class balance
        class_counts = Counter(
            categories[ann["category_id"]] for ann in coco["annotations"]
            if ann["category_id"] in categories
        )
        print(f"   Annotation counts per class:")
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
            bar = "█" * min(40, cnt // max(1, max(class_counts.values()) // 40))
            print(f"     {cls:30s} {cnt:6d}  {bar}")

        # Images without annotations
        unannotated = [img_id for img_id in images_meta
                       if img_id not in ann_by_image]
        if unannotated:
            warnings.append(f"  ⚠️   {len(unannotated)} images have zero annotations (empty/background)")

        # Spot-check image integrity
        files = sorted(img_dir.iterdir())[:max_images]
        corrupt = []
        for fpath in files:
            try:
                with Image.open(fpath) as im:
                    im.verify()
            except Exception as e:
                corrupt.append((fpath.name, str(e)))
        if corrupt:
            for fname, err in corrupt[:5]:
                errors.append(f"  ❌  Corrupt image: {fname} — {err}")
            ok = False
        else:
            print(f"   Image integrity : ✓ ({len(files)} checked)")

        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    for w in warnings:
        print(w)
    for e in errors:
        print(e)

    if ok and not errors:
        print("✅  Detection dataset looks good — ready to train!\n")
    else:
        print("❌  Fix the errors above before training.\n")
        sys.exit(1)


# ─── CLASSIFICATION ───────────────────────────────────────────────────────────

def validate_classification(dataset_dir: str, max_images: int):
    print(f"\n🔍  Validating ImageFolder classification dataset: {dataset_dir}\n")
    errors, warnings = [], []
    root = Path(dataset_dir)
    ok = True

    for split in ["train", "val"]:
        split_dir = root / split
        print(f"── {split}/")

        if not split_dir.exists():
            errors.append(f"  ❌  Missing split directory: {split_dir}")
            ok = False
            continue

        classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        if not classes:
            errors.append(f"  ❌  No class subdirectories found in {split_dir}")
            ok = False
            continue

        print(f"   Classes ({len(classes)}): {classes}")

        class_counts = {}
        all_images = []
        for cls in classes:
            cls_dir = split_dir / cls
            files = [f for f in cls_dir.iterdir()
                     if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
            class_counts[cls] = len(files)
            all_images.extend(files[:max_images // len(classes) + 1])

        print(f"   Image counts per class:")
        total = sum(class_counts.values())
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
            pct = cnt / total * 100 if total else 0
            bar = "█" * min(40, int(pct * 40 / 100))
            print(f"     {cls:30s} {cnt:6d}  ({pct:5.1f}%)  {bar}")

        # Warn on class imbalance
        counts = list(class_counts.values())
        if counts:
            ratio = max(counts) / (min(counts) + 1)
            if ratio > 10:
                warnings.append(
                    f"  ⚠️   High class imbalance in {split}: "
                    f"max/min ratio = {ratio:.1f}x — consider weighted sampling"
                )

        # Check for empty classes
        empty = [c for c, n in class_counts.items() if n == 0]
        if empty:
            errors.append(f"  ❌  Empty class directories: {empty}")
            ok = False

        # Spot-check image integrity
        corrupt = []
        for fpath in all_images[:max_images]:
            try:
                with Image.open(fpath) as im:
                    im.verify()
            except Exception as e:
                corrupt.append((fpath.name, str(e)))
        if corrupt:
            for fname, err in corrupt[:5]:
                errors.append(f"  ❌  Corrupt image: {fname} — {err}")
            ok = False
        else:
            print(f"   Image integrity : ✓ ({min(len(all_images), max_images)} checked)")

        # Check train/val have same classes
        print()

    # Cross-check splits
    train_classes = set(
        d.name for d in (root / "train").iterdir()
        if d.is_dir() and (root / "train").exists()
    )
    val_classes = set(
        d.name for d in (root / "val").iterdir()
        if d.is_dir() and (root / "val").exists()
    )
    if train_classes and val_classes:
        only_train = train_classes - val_classes
        only_val   = val_classes - train_classes
        if only_train:
            warnings.append(f"  ⚠️   Classes in train but not val: {only_train}")
        if only_val:
            warnings.append(f"  ⚠️   Classes in val but not train: {only_val}")

    for w in warnings:
        print(w)
    for e in errors:
        print(e)

    if ok and not errors:
        print("✅  Classification dataset looks good — ready to train!\n")
    else:
        print("❌  Fix the errors above before training.\n")
        sys.exit(1)


# ─── SEGMENTATION — INSTANCE (COCO POLYGONS) ─────────────────────────────────

def validate_seg_instance(dataset_dir: str, max_images: int):
    """Validates a COCO-format instance segmentation dataset (polygon annotations)."""
    print(f"\n🔍  Validating COCO instance segmentation dataset: {dataset_dir}\n")
    errors, warnings = [], []
    root = Path(dataset_dir)
    ok = True

    for split in ["train", "val"]:
        split_dir = root / split
        ann_path  = split_dir / "annotations.json"
        img_dir   = split_dir / "images"

        print(f"── {split}/")

        for path, label in [(split_dir, "split dir"), (ann_path, "annotations.json"),
                            (img_dir, "images/")]:
            if not path.exists():
                errors.append(f"  ❌  Missing {label}: {path}")
                ok = False

        if not ok:
            continue

        with open(ann_path) as f:
            coco = json.load(f)

        images_meta  = {img["id"]: img for img in coco.get("images", [])}
        categories   = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}
        annotations  = coco.get("annotations", [])

        print(f"   Images     : {len(images_meta)}")
        print(f"   Annotations: {len(annotations)}")
        print(f"   Categories ({len(categories)}): {list(categories.values())}")

        # Check segmentation fields
        no_seg = sum(1 for a in annotations
                     if not a.get("segmentation") or a.get("iscrowd", 0))
        if no_seg:
            warnings.append(f"  ⚠️   {no_seg} annotations missing 'segmentation' or marked as crowd "
                            "(will be skipped at training time)")

        # Check for zero/degenerate polygons
        degenerate = 0
        for ann in annotations:
            seg = ann.get("segmentation", [])
            if isinstance(seg, list):
                for poly in seg:
                    if len(poly) < 6:
                        degenerate += 1
        if degenerate:
            warnings.append(f"  ⚠️   {degenerate} degenerate polygon(s) with < 3 points")

        # Class balance (annotation count)
        class_counts = Counter(
            categories.get(a["category_id"], "unknown") for a in annotations
        )
        print(f"   Annotations per class:")
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
            bar = "█" * min(40, cnt // max(1, max(class_counts.values()) // 40))
            print(f"     {cls:30s} {cnt:6d}  {bar}")

        # Image file check
        missing = [img["file_name"] for img in coco["images"]
                   if not (img_dir / img["file_name"]).exists()]
        if missing:
            errors.append(f"  ❌  {len(missing)} image file(s) missing from disk")
            ok = False
        else:
            print(f"   Image files: ✓ all found on disk")

        # Spot-check integrity
        files = list(img_dir.iterdir())[:max_images]
        corrupt = []
        for fpath in files:
            try:
                with Image.open(fpath) as im:
                    im.verify()
            except Exception as e:
                corrupt.append((fpath.name, str(e)))
        if corrupt:
            for fname, err in corrupt[:5]:
                errors.append(f"  ❌  Corrupt: {fname} — {err}")
            ok = False
        else:
            print(f"   Image integrity: ✓ ({len(files)} checked)")
        print()

    for w in warnings:
        print(w)
    for e in errors:
        print(e)

    if ok and not errors:
        print("✅  Instance segmentation dataset looks good — ready to train!\n")
    else:
        print("❌  Fix errors above before training.\n")
        sys.exit(1)


# ─── SEGMENTATION — SEMANTIC (IMAGE / MASK PNGs) ─────────────────────────────

def validate_seg_semantic(dataset_dir: str, max_images: int):
    """Validates an image/mask-PNG semantic segmentation dataset."""
    print(f"\n🔍  Validating semantic segmentation dataset: {dataset_dir}\n")
    errors, warnings = [], []
    root = Path(dataset_dir)
    ok = True

    for split in ["train", "val"]:
        split_dir = root / split
        img_dir   = split_dir / "images"
        mask_dir  = split_dir / "masks"

        print(f"── {split}/")

        for path, label in [(img_dir, "images/"), (mask_dir, "masks/")]:
            if not path.exists():
                errors.append(f"  ❌  Missing {label}: {path}")
                ok = False

        if not ok:
            continue

        img_stems  = {f.stem for f in img_dir.iterdir()
                      if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}}
        mask_stems = {f.stem for f in mask_dir.iterdir()
                      if f.suffix.lower() in {".png", ".tif", ".tiff"}}

        paired = img_stems & mask_stems
        print(f"   Images      : {len(img_stems)}")
        print(f"   Masks       : {len(mask_stems)}")
        print(f"   Paired      : {len(paired)}")

        only_img  = img_stems - mask_stems
        only_mask = mask_stems - img_stems
        if only_img:
            warnings.append(f"  ⚠️   {len(only_img)} images have no matching mask")
        if only_mask:
            warnings.append(f"  ⚠️   {len(only_mask)} masks have no matching image")
        if not paired:
            errors.append(f"  ❌  No matched image/mask pairs found — check filenames match")
            ok = False
            continue

        # Inspect masks: collect unique class ids and size consistency
        unique_ids: set[int] = set()
        size_mismatches = 0
        sample_masks = sorted(mask_dir.iterdir())[:min(max_images, len(paired))]

        for mask_path in sample_masks:
            if mask_path.stem not in paired:
                continue
            try:
                mask_arr = np.array(Image.open(mask_path))
                ids = [int(v) for v in np.unique(mask_arr) if v != 255]
                unique_ids.update(ids)

                # Check mask matches image size
                img_candidates = list(img_dir.glob(f"{mask_path.stem}.*"))
                if img_candidates:
                    with Image.open(img_candidates[0]) as im:
                        iw, ih = im.size
                    mh, mw = mask_arr.shape[:2]
                    if (ih, iw) != (mh, mw):
                        size_mismatches += 1
            except Exception as e:
                errors.append(f"  ❌  Cannot read mask {mask_path.name}: {e}")
                ok = False

        print(f"   Class ids found (first {min(max_images, len(paired))} masks): "
              f"{sorted(unique_ids)}")
        print(f"   Inferred num_classes: {max(unique_ids) + 1 if unique_ids else '?'}")

        if size_mismatches:
            warnings.append(f"  ⚠️   {size_mismatches} image/mask pairs have mismatched sizes "
                            "(Albumentations will resize both — OK if using augmentation)")

        # Check for label_map.json
        label_map = root / "label_map.json"
        if label_map.exists():
            with open(label_map) as f:
                lm = json.load(f)
            print(f"   label_map.json: ✓ ({len(lm)} classes: {lm})")
        else:
            warnings.append(
                f"  ⚠️   No label_map.json found — classes will be named numerically.\n"
                f"      Create {root}/label_map.json: {{\"0\": \"background\", \"1\": \"zebra\", ...}}"
            )

        # Spot-check image integrity
        files = sorted(img_dir.iterdir())[:max_images]
        corrupt = []
        for fpath in files:
            try:
                with Image.open(fpath) as im:
                    im.verify()
            except Exception as e:
                corrupt.append((fpath.name, str(e)))
        if corrupt:
            for fname, err in corrupt[:5]:
                errors.append(f"  ❌  Corrupt image: {fname} — {err}")
            ok = False
        else:
            print(f"   Image integrity: ✓ ({len(files)} checked)")
        print()

    for w in warnings:
        print(w)
    for e in errors:
        print(e)

    if ok and not errors:
        print("✅  Semantic segmentation dataset looks good — ready to train!\n")
    else:
        print("❌  Fix errors above before training.\n")
        sys.exit(1)


# ─── YOLO FORMAT ─────────────────────────────────────────────────────────────

def validate_yolo(dataset_dir: str, max_images: int):
    """
    Validates a YOLO-format detection dataset.

    Accepted layouts:
      1. dataset_dir is a directory containing dataset.yaml
      2. dataset_dir is the path to dataset.yaml itself

    YOLO format:
        path/
          images/train/  *.jpg
          images/val/    *.jpg
          labels/train/  *.txt   (class_id cx cy w h — normalised)
          labels/val/    *.txt
          dataset.yaml
    """
    import yaml

    root = Path(dataset_dir)
    errors, warnings = [], []
    ok = True

    # Find dataset.yaml
    if root.suffix in {".yaml", ".yml"}:
        yaml_path = root
    elif (root / "dataset.yaml").exists():
        yaml_path = root / "dataset.yaml"
    elif (root / "data.yaml").exists():
        yaml_path = root / "data.yaml"
    else:
        print(f"\n❌  No dataset.yaml found in {root}")
        sys.exit(1)

    print(f"\n🔍  Validating YOLO dataset: {yaml_path}\n")

    with open(yaml_path) as f:
        ds = yaml.safe_load(f)

    # Check required keys
    for key in ["train", "val", "nc", "names"]:
        if key not in ds:
            errors.append(f"  ❌  Missing key '{key}' in dataset.yaml")
            ok = False

    if not ok:
        for e in errors:
            print(e)
        sys.exit(1)

    ds_root = Path(ds.get("path", yaml_path.parent))
    nc = ds["nc"]
    names = ds["names"]

    print(f"   Root    : {ds_root}")
    print(f"   Classes ({nc}): {names}")

    if len(names) != nc:
        errors.append(f"  ❌  nc={nc} but names has {len(names)} entries")
        ok = False

    for split in ["train", "val"]:
        split_rel = ds.get(split, f"images/{split}")
        img_dir = ds_root / split_rel
        # Derive label dir: images/ → labels/
        lbl_dir = Path(str(img_dir).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\"))

        print(f"\n── {split}/")
        print(f"   Images dir: {img_dir}")
        print(f"   Labels dir: {lbl_dir}")

        if not img_dir.exists():
            errors.append(f"  ❌  Image directory not found: {img_dir}")
            ok = False
            continue

        if not lbl_dir.exists():
            errors.append(f"  ❌  Label directory not found: {lbl_dir}")
            ok = False
            continue

        # Count images and labels
        img_files = {
            p.stem: p for p in img_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        }
        lbl_files = {p.stem: p for p in lbl_dir.iterdir() if p.suffix == ".txt"}

        paired = set(img_files.keys()) & set(lbl_files.keys())
        only_img = set(img_files.keys()) - set(lbl_files.keys())
        only_lbl = set(lbl_files.keys()) - set(img_files.keys())

        print(f"   Images     : {len(img_files)}")
        print(f"   Labels     : {len(lbl_files)}")
        print(f"   Paired     : {len(paired)}")

        if only_img:
            # Images without labels is common (background/negative images) — just warn
            warnings.append(f"  ⚠️   {len(only_img)} images have no label file (treated as background)")
        if only_lbl:
            warnings.append(f"  ⚠️   {len(only_lbl)} label files have no matching image")

        if not img_files:
            errors.append(f"  ❌  No image files found in {img_dir}")
            ok = False
            continue

        # Validate label file contents
        class_counter: Counter = Counter()
        bad_lines = 0
        out_of_range_cls = 0
        out_of_bounds = 0
        total_annotations = 0
        empty_labels = 0

        for stem in sorted(paired)[:max_images]:
            txt_path = lbl_files[stem]
            content = txt_path.read_text().strip()

            if not content:
                empty_labels += 1
                continue

            for line_num, line in enumerate(content.split("\n"), 1):
                parts = line.strip().split()
                if not parts:
                    continue

                if len(parts) != 5:
                    bad_lines += 1
                    continue

                try:
                    cls_id = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    bad_lines += 1
                    continue

                total_annotations += 1
                class_counter[cls_id] += 1

                if cls_id < 0 or cls_id >= nc:
                    out_of_range_cls += 1

                # Check normalised values are in [0, 1]
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    out_of_bounds += 1

        print(f"   Annotations: {total_annotations} (in {min(len(paired), max_images)} files checked)")

        if empty_labels:
            print(f"   Empty labels: {empty_labels} (background images)")

        if bad_lines:
            errors.append(f"  ❌  {bad_lines} malformed lines (expected: class_id cx cy w h)")
            ok = False
        if out_of_range_cls:
            errors.append(f"  ❌  {out_of_range_cls} annotations with class_id outside [0, {nc - 1}]")
            ok = False
        if out_of_bounds:
            warnings.append(f"  ⚠️   {out_of_bounds} annotations with values outside [0, 1] range")

        # Class distribution
        if class_counter:
            print(f"   Annotations per class:")
            for cls_id in sorted(class_counter.keys()):
                name = names[cls_id] if cls_id < len(names) else f"class_{cls_id}"
                cnt = class_counter[cls_id]
                bar = "█" * min(40, cnt // max(1, max(class_counter.values()) // 40))
                print(f"     {name:25s} {cnt:6d}  {bar}")

        # Missing classes
        missing_cls = set(range(nc)) - set(class_counter.keys())
        if missing_cls:
            missing_names = [names[i] if i < len(names) else f"class_{i}" for i in missing_cls]
            warnings.append(f"  ⚠️   Classes not seen in {split}: {missing_names}")

        # Spot-check image integrity
        files_to_check = sorted(img_files.values())[:max_images]
        corrupt = []
        for fpath in files_to_check:
            try:
                with Image.open(fpath) as im:
                    im.verify()
            except Exception as e:
                corrupt.append((fpath.name, str(e)))
        if corrupt:
            for fname, err in corrupt[:5]:
                errors.append(f"  ❌  Corrupt image: {fname} — {err}")
            ok = False
        else:
            print(f"   Image integrity: ✓ ({len(files_to_check)} checked)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    for w in warnings:
        print(w)
    for e in errors:
        print(e)

    if ok and not errors:
        print("\n✅  YOLO dataset looks good — ready to train!\n")
    else:
        print("\n❌  Fix the errors above before training.\n")
        sys.exit(1)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.task == "detection":
        validate_detection(args.dataset_dir, args.max_images)
    elif args.task == "classification":
        validate_classification(args.dataset_dir, args.max_images)
    elif args.task == "seg_instance":
        validate_seg_instance(args.dataset_dir, args.max_images)
    elif args.task == "yolo":
        validate_yolo(args.dataset_dir, args.max_images)
    else:
        validate_seg_semantic(args.dataset_dir, args.max_images)


if __name__ == "__main__":
    main()
