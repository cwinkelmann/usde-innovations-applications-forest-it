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
    _IMAGE_PATH = Path("../data/camera_trap/sample_01.jpg")
    _CONF_THRESHOLD = 0.2  # detections below this are ignored

    _result = detector.single_image_detection(str(_IMAGE_PATH), conf_thres=_CONF_THRESHOLD)

    print(f"Detections found: {len(_result['detections'])}")
    for _det in _result["detections"]:
        _label = {1: "animal", 2: "person", 3: "vehicle"}.get(_det["category"], "?")
        print(f"  {_label:8s}  conf={_det['conf']:.3f}  bbox={[round(v, 3) for v in _det['bbox']]}")

    # Draw bounding boxes
    _img = Image.open(_IMAGE_PATH).convert("RGB")
    _draw = ImageDraw.Draw(_img)
    _W, _H = _img.size
    _COLOURS = {1: "lime", 2: "red", 3: "yellow"}

    for _det in _result["detections"]:
        _x, _y, _bw, _bh = _det["bbox"]
        _x1, _y1 = int(_x * _W), int(_y * _H)
        _x2, _y2 = int((_x + _bw) * _W), int((_y + _bh) * _H)
        _draw.rectangle([_x1, _y1, _x2, _y2], outline=_COLOURS.get(_det["category"], "white"), width=2)
        _label = {1: "animal", 2: "person", 3: "vehicle"}.get(_det["category"], "?")
        _draw.text((_x1, max(0, _y1 - 14)), f"{_label} {_det['conf']:.2f}", fill="white")

    _fig, _ax = plt.subplots(figsize=(10, 7))
    _ax.imshow(_img)
    _ax.set_title(f"MegaDetector — {_IMAGE_PATH.name}  (threshold = {_CONF_THRESHOLD})")
    _ax.axis("off")
    plt.tight_layout()
    plt.gca()
    return


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
    _IMAGE_DIR = Path("../data/camera_trap")
    _CONF_THRESHOLD = 0.2

    image_paths = sorted(_IMAGE_DIR.glob("*.jpg"))
    records = []

    for _img_path in image_paths:
        _result = detector.single_image_detection(str(_img_path), conf_thres=_CONF_THRESHOLD)
        for _det in _result["detections"]:
            _category_name = {1: "animal", 2: "person", 3: "vehicle"}.get(_det["category"], "unknown")
            records.append({
                "filename": _img_path.name,
                "category": _category_name,
                "confidence": round(_det["conf"], 4),
                "bbox_x": round(_det["bbox"][0], 4),
                "bbox_y": round(_det["bbox"][1], 4),
                "bbox_w": round(_det["bbox"][2], 4),
                "bbox_h": round(_det["bbox"][3], 4),
            })

    detections_df = pd.DataFrame(records)

    print(f"Processed {len(image_paths)} images, found {len(detections_df)} detections")
    print(f"\nCategory breakdown:")
    print(detections_df["category"].value_counts().to_string())
    print(f"\nConfidence distribution:")
    print(detections_df.groupby("category")["confidence"].describe().round(3).to_string())
    return detections_df, image_paths, records


@app.cell
def _step4(mo):
    mo.md(r"""
    ## Step 4 — Extract animal crops

    Crops are the input to the downstream species classifier.
    We save each crop as a JPEG named `{original_stem}_{crop_index}.jpg`.
    """)


@app.cell
def _extract_crops(Image, Path, detections_df):

    CROPS_DIR = Path("../data/camera_trap_crops")
    CROPS_DIR.mkdir(exist_ok=True)

    animal_detections = detections_df[detections_df["category"] == "animal"].copy()
    crop_paths = []

    for _, row in animal_detections.iterrows():
        _img_path2 = Path("../data/camera_trap") / row["filename"]
        _img2 = Image.open(_img_path2).convert("RGB")
        _W2, _H2 = _img2.size

        _x1 = int(row["bbox_x"] * _W2)
        _y1 = int(row["bbox_y"] * _H2)
        _x2 = int((row["bbox_x"] + row["bbox_w"]) * _W2)
        _y2 = int((row["bbox_y"] + row["bbox_h"]) * _H2)

        _crop = _img2.crop((_x1, _y1, _x2, _y2))
        _stem = _img_path2.stem
        _idx = len(crop_paths)
        _out_path = CROPS_DIR / f"{_stem}_crop{_idx:04d}.jpg"
        _crop.save(_out_path, quality=90)
        crop_paths.append(_out_path)

    print(f"Saved {len(crop_paths)} animal crops to {CROPS_DIR}")
    return CROPS_DIR, crop_paths


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
