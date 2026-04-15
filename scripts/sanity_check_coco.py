"""
Sanity check converted COCO JSON files for the Aerial MegaDetector dataset.

Checks:
1. JSON structure validity
2. Category IDs match expected {1: animal, 2: person, 3: vehicle}
3. Annotation counts per category
4. Bbox validity (non-negative, non-zero area)
5. Image file existence (spot check)
6. Basic statistics

Usage:
    python scripts/sanity_check_coco.py --dir /Volumes/storage/Datasets/aerial_megadetector/processed
    python scripts/sanity_check_coco.py --file /Volumes/storage/Datasets/aerial_megadetector/processed/animal/izembek/annotations.json
"""

import argparse
import glob
import json
import os
import random
from collections import Counter
from pathlib import Path


EXPECTED_CATEGORIES = {1: "animal", 2: "person", 3: "vehicle"}


def check_one(json_path: str, spot_check_images: int = 5):
    """Run sanity checks on a single COCO JSON file."""
    dataset_name = Path(json_path).parent.name
    print(f"\n{'='*60}")
    print(f"  {dataset_name}")
    print(f"  {json_path}")
    print(f"{'='*60}")

    errors = []
    warnings = []

    # 1. Load JSON
    try:
        with open(json_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  FAIL: Invalid JSON: {e}")
        return False

    # 2. Check top-level keys
    required_keys = {"images", "annotations", "categories"}
    missing = required_keys - set(data.keys())
    if missing:
        errors.append(f"Missing top-level keys: {missing}")

    # 3. Check categories
    cats = {c["id"]: c["name"] for c in data.get("categories", [])}
    for cat_id, cat_name in EXPECTED_CATEGORIES.items():
        if cat_id not in cats:
            warnings.append(f"Missing category {cat_id} ({cat_name})")
        elif cats[cat_id] != cat_name:
            warnings.append(f"Category {cat_id} name mismatch: expected '{cat_name}', got '{cats[cat_id]}'")

    # 4. Image stats
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    image_ids = {img["id"] for img in images}

    print(f"  Images:      {len(images)}")
    print(f"  Annotations: {len(annotations)}")

    # 5. Annotation stats
    cat_counts = Counter()
    orphan_anns = 0
    bad_bbox = 0
    negative_coords = 0
    zero_area = 0
    original_classes = Counter()

    for ann in annotations:
        cat_id = ann.get("category_id")
        cat_counts[cat_id] += 1

        if ann.get("image_id") not in image_ids:
            orphan_anns += 1

        bbox = ann.get("bbox", [])
        if len(bbox) == 4:
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                zero_area += 1
            if x < 0 or y < 0:
                negative_coords += 1
        else:
            bad_bbox += 1

        orig_cls = ann.get("attributes", {}).get("original_class", "unknown")
        original_classes[orig_cls] += 1

    print(f"\n  Per-category counts:")
    for cat_id in sorted(cat_counts.keys()):
        cat_name = EXPECTED_CATEGORIES.get(cat_id, f"unknown_{cat_id}")
        print(f"    {cat_name} (id={cat_id}): {cat_counts[cat_id]:,}")

    print(f"\n  Original class distribution (top 10):")
    for cls, count in original_classes.most_common(10):
        print(f"    {cls}: {count:,}")

    if orphan_anns:
        errors.append(f"{orphan_anns} annotations reference non-existent image IDs")
    if bad_bbox:
        errors.append(f"{bad_bbox} annotations have malformed bboxes")
    if zero_area:
        warnings.append(f"{zero_area} annotations have zero/negative area")
    if negative_coords:
        warnings.append(f"{negative_coords} annotations have negative coordinates (may be from point->pseudobox near image edge)")

    # 6. Image dimension stats
    widths = [img.get("width", 0) for img in images]
    heights = [img.get("height", 0) for img in images]
    zero_dim = sum(1 for w, h in zip(widths, heights) if w == 0 or h == 0)
    if zero_dim:
        warnings.append(f"{zero_dim}/{len(images)} images have zero dimensions (may need resolution)")
    else:
        print(f"\n  Image dimensions: width [{min(widths)}-{max(widths)}], height [{min(heights)}-{max(heights)}]")

    # 7. Spot check image file existence
    if spot_check_images > 0 and images:
        base_dir = os.path.dirname(json_path)
        sample = random.sample(images, min(spot_check_images, len(images)))
        found = 0
        checked = 0
        for img in sample:
            fname = img["file_name"]
            # Try multiple base directories
            candidates = [
                os.path.join(base_dir, fname),
                os.path.join(base_dir, os.path.basename(fname)),
            ]
            if any(os.path.exists(c) for c in candidates):
                found += 1
            checked += 1

        print(f"\n  Image spot check: {found}/{checked} found on disk")
        if found == 0:
            warnings.append(f"No images found on disk during spot check (images may be symlinked or stored elsewhere)")

    # 8. Summary
    annotations_per_image = len(annotations) / len(images) if images else 0
    print(f"\n  Avg annotations/image: {annotations_per_image:.1f}")

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    [ERROR] {e}")
    if warnings:
        print(f"\n  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"    [WARN] {w}")

    if not errors:
        print(f"\n  PASS")
    else:
        print(f"\n  FAIL")

    return len(errors) == 0


def main():
    parser = argparse.ArgumentParser(description="Sanity check COCO JSON files")
    parser.add_argument("--dir", default=None, help="Directory to recursively search for annotations.json")
    parser.add_argument("--file", default=None, help="Single JSON file to check")
    parser.add_argument("--spot-check", type=int, default=5, help="Number of images to spot-check")
    args = parser.parse_args()

    if args.file:
        files = [args.file]
    elif args.dir:
        files = sorted(glob.glob(os.path.join(args.dir, "**", "annotations.json"), recursive=True))
    else:
        files = sorted(glob.glob("/Volumes/storage/Datasets/aerial_megadetector/processed/**/annotations.json", recursive=True))

    print(f"Found {len(files)} COCO annotation files to check\n")

    results = {}
    total_images = 0
    total_annotations = 0
    cat_totals = Counter()

    for f in files:
        ok = check_one(f, args.spot_check)
        dataset = Path(f).parent.name
        results[dataset] = ok

        with open(f) as fp:
            data = json.load(fp)
        total_images += len(data.get("images", []))
        total_annotations += len(data.get("annotations", []))
        for ann in data.get("annotations", []):
            cat_totals[ann.get("category_id")] += 1

    # Grand summary
    print(f"\n{'='*60}")
    print(f"  GRAND SUMMARY")
    print(f"{'='*60}")
    print(f"  Datasets checked: {len(results)}")
    print(f"  Passed:           {sum(results.values())}")
    print(f"  Failed:           {sum(1 for v in results.values() if not v)}")
    print(f"\n  Total images:      {total_images:,}")
    print(f"  Total annotations: {total_annotations:,}")
    print(f"\n  Per-category totals:")
    for cat_id in sorted(cat_totals.keys()):
        cat_name = EXPECTED_CATEGORIES.get(cat_id, f"unknown_{cat_id}")
        print(f"    {cat_name}: {cat_totals[cat_id]:,}")

    for name, ok in sorted(results.items()):
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")


if __name__ == "__main__":
    main()
