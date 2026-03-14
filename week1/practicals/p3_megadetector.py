import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _context(mo):
    mo.md(r"""
    # Practical 3 — Running MegaDetector

    **Context:** A camera trap deployment in Serengeti produces thousands of images per
    week. More than 80 % of frames are empty — triggered by wind, rain, or passing humans.
    MegaDetector is a pre-trained detector that filters those empty frames automatically,
    reducing manual review workload by ~80 %.

    MegaDetector detects three classes: **animal**, **person**, **vehicle**.
    It does *not* identify species — that is a downstream classification problem.

    Today you will:
    - Run MegaDetector on a folder of camera trap images via PyTorch Wildlife
    - Parse the JSON output: confidence scores, bounding box coordinates
    - Filter by confidence threshold and extract animal crops
    - Visualise detections and spot failure modes

    **Install:** `pip install PytorchWildlife`
    """)


@app.cell
def _imports():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from PIL import Image, ImageDraw

    return Image, ImageDraw, Path, np, pd, plt


@app.cell
def _step1(mo):
    mo.md(r"""
    ## Step 1 — Load MegaDetector

    PyTorch Wildlife wraps MegaDetector v5 (YOLOv5-based) in a simple Python API.
    The model downloads automatically on first use (~600 MB).
    """)


@app.cell
def _load_model():
    from PytorchWildlife.models import detection as pw_detection

    detector = pw_detection.MegaDetectorV5(device="cpu", pretrained=True)
    print("MegaDetector v5 loaded.")
    return (detector,)


@app.cell
def _step2(mo):
    mo.md(r"""
    ## Step 2 — Run detection on a single image

    The detector returns a list of detections per image. Each detection has:
    - `category` — 1 = animal, 2 = person, 3 = vehicle
    - `conf` — confidence score (0–1)
    - `bbox` — `[x_min, y_min, width, height]` in relative coordinates (0–1)
    """)


@app.cell
def _single_image(Image, ImageDraw, Path, detector, plt):
    IMAGE_PATH = Path("../data/camera_trap/sample_01.jpg")
    CONF_THRESHOLD = 0.2  # detections below this are ignored

    result = detector.single_image_detection(str(IMAGE_PATH), conf_thres=CONF_THRESHOLD)

    print(f"Detections found: {len(result['detections'])}")
    for det in result["detections"]:
        label = {1: "animal", 2: "person", 3: "vehicle"}.get(det["category"], "?")
        print(f"  {label:8s}  conf={det['conf']:.3f}  bbox={[round(v, 3) for v in det['bbox']]}")

    # Draw bounding boxes
    img = Image.open(IMAGE_PATH).convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size
    COLOURS = {1: "lime", 2: "red", 3: "yellow"}

    for det in result["detections"]:
        x, y, bw, bh = det["bbox"]
        x1, y1 = int(x * W), int(y * H)
        x2, y2 = int((x + bw) * W), int((y + bh) * H)
        draw.rectangle([x1, y1, x2, y2], outline=COLOURS.get(det["category"], "white"), width=2)
        label = {1: "animal", 2: "person", 3: "vehicle"}.get(det["category"], "?")
        draw.text((x1, max(0, y1 - 14)), f"{label} {det['conf']:.2f}", fill="white")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img)
    ax.set_title(f"MegaDetector — {IMAGE_PATH.name}  (threshold = {CONF_THRESHOLD})")
    ax.axis("off")
    plt.tight_layout()
    plt.show()
    return CONF_THRESHOLD, IMAGE_PATH, W, H, ax, bh, bw, det, draw, fig, img, label, result, x, x1, x2, y, y1, y2


@app.cell
def _step3(mo):
    mo.md(r"""
    ## Step 3 — Batch detection and results table

    Run MegaDetector on a whole folder and collect results into a DataFrame.
    This is the starting point for the two-stage pipeline:
    **detect → crop → classify**.
    """)


@app.cell
def _batch(Path, detector, pd):
    IMAGE_DIR = Path("../data/camera_trap")
    CONF_THRESHOLD = 0.2

    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))
    records = []

    for img_path in image_paths:
        result = detector.single_image_detection(str(img_path), conf_thres=CONF_THRESHOLD)
        for det in result["detections"]:
            category_name = {1: "animal", 2: "person", 3: "vehicle"}.get(det["category"], "unknown")
            records.append({
                "filename": img_path.name,
                "category": category_name,
                "confidence": round(det["conf"], 4),
                "bbox_x": round(det["bbox"][0], 4),
                "bbox_y": round(det["bbox"][1], 4),
                "bbox_w": round(det["bbox"][2], 4),
                "bbox_h": round(det["bbox"][3], 4),
            })

    detections_df = pd.DataFrame(records)

    print(f"Processed {len(image_paths)} images, found {len(detections_df)} detections")
    print(f"\nCategory breakdown:")
    print(detections_df["category"].value_counts().to_string())
    print(f"\nConfidence distribution:")
    print(detections_df.groupby("category")["confidence"].describe().round(3).to_string())
    return CONF_THRESHOLD, IMAGE_DIR, category_name, det, detections_df, image_paths, img_path, records, result


@app.cell
def _step4(mo):
    mo.md(r"""
    ## Step 4 — Extract animal crops

    Crops are the input to the downstream species classifier.
    We save each crop as a JPEG named `{original_stem}_{crop_index}.jpg`.
    """)


@app.cell
def _extract_crops(Path, detections_df):
    from PIL import Image

    CROPS_DIR = Path("../data/camera_trap_crops")
    CROPS_DIR.mkdir(exist_ok=True)

    animal_detections = detections_df[detections_df["category"] == "animal"].copy()
    crop_paths = []

    for _, row in animal_detections.iterrows():
        img_path = Path("../data/camera_trap") / row["filename"]
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        x1 = int(row["bbox_x"] * W)
        y1 = int(row["bbox_y"] * H)
        x2 = int((row["bbox_x"] + row["bbox_w"]) * W)
        y2 = int((row["bbox_y"] + row["bbox_h"]) * H)

        crop = img.crop((x1, y1, x2, y2))
        stem = img_path.stem
        idx = len(crop_paths)
        out_path = CROPS_DIR / f"{stem}_crop{idx:04d}.jpg"
        crop.save(out_path, quality=90)
        crop_paths.append(out_path)

    print(f"Saved {len(crop_paths)} animal crops to {CROPS_DIR}")
    return CROPS_DIR, Image, W, H, animal_detections, crop, crop_paths, idx, img, img_path, out_path, row, stem, x1, x2, y1, y2


@app.cell
def _exercise(mo):
    mo.md(r"""
    ## Exercise

    > **Change `CONF_THRESHOLD` from 0.2 to 0.5, then to 0.8. What happens to the
    > number of detections?**

    Browse your saved crops and find:
    1. One detection that is clearly correct
    2. One false positive (non-animal detected as animal)
    3. One missed detection (animal visible but not detected)

    For each: what was the confidence score? What do you think caused the error?
    """)


@app.cell
def _reflection(mo):
    mo.md(r"""
    ## Reflection

    - MegaDetector was trained primarily on camera-trap perspective images. Would you
      expect it to work on overhead drone imagery? Why or why not?
    - If you set the threshold very low (0.05), you catch more true animals but also
      more false positives. Who decides what threshold to use in a real field deployment?
    - What would happen to the downstream classifier if MegaDetector misses 20 % of animals?
    """)


if __name__ == "__main__":
    app.run()
