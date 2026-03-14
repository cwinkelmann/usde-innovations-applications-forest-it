import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _imports():
    from pathlib import Path

    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from PIL import Image

    DATA_DIR = Path(__file__).parent.parent / "data"

    return DATA_DIR, Image, Path, mpatches, np, pd, plt


@app.cell(hide_code=True)
def _context(mo):
    mo.md(r"""
    # Practical 3 — Animal Detection with MegaDetector & SAHI

    **Context:** A camera trap deployment produces thousands of images per week.
    More than 80 % are empty — triggered by wind or vegetation movement.
    MegaDetector is a pre-trained detector that filters those empty frames automatically,
    reducing manual review by ~80 %.

    MegaDetector detects three classes: **animal · person · vehicle**.
    It does *not* identify species — species classification is a downstream problem (P5).

    Drone orthomosaics present a different challenge: images can be 10,000+ pixels wide
    with animals occupying only 20–50 pixels. **SAHI** (Slicing Aided Hyper Inference)
    addresses this by running detection on overlapping tiles and merging results.

    In Part 3 you fine-tune a lightweight **YOLOv8** detector on your own annotations.

    **Install:**
    ```bash
    pip install megadetector          # MegaDetector (agentmorris fork)
    pip install ultralytics           # YOLOv8 for custom training + SAHI backend
    pip install sahi                  # Slicing Aided Hyper Inference
    ```
    """)
    return


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — MEGADETECTOR INFERENCE
# ═══════════════════════════════════════════════════════════════════════════


@app.cell(hide_code=True)
def _part1_header(mo):
    mo.md(r"""
    ## Part 1 — Inference with MegaDetector
    """)
    return


@app.cell(hide_code=True)
def _step1_md(mo):
    mo.md(r"""
    ### Step 1 — Load MegaDetector v5

    `load_detector('MDV5A')` downloads the ~600 MB model weights on first use
    and caches them locally. MDV5A is a YOLOv5-based model trained on
    ~4.5 million camera trap images from 89 data sources worldwide.

    Two variants exist — A and B — trained on different data splits.
    MDV5A is the default recommendation.
    """)
    return


@app.cell
def _load_detector():
    from megadetector.detection.run_detector import load_detector

    detector = load_detector("MDV5A")
    print("MegaDetector v5a loaded.")
    print(f"  Type : {type(detector)}")
    return (detector,)


@app.cell(hide_code=True)
def _step2_md(mo):
    mo.md(r"""
    ### Step 2 — Single-image detection

    `generate_detections_one_image` returns a dict with:
    - `detections` — list of hits, each with `category`, `conf`, `bbox`
    - `bbox` format — `[x_min, y_min, width, height]` **normalised 0–1**
    - `category` — string: `'1'` = animal, `'2'` = person, `'3'` = vehicle

    Note the category is a **string**, not an integer.
    """)
    return


@app.cell
def _single_image(DATA_DIR, Image, detector, mpatches, np, plt):
    _LABELS = {"1": "animal", "2": "person", "3": "vehicle"}
    _COLORS = {"1": "#E74C3C", "2": "#3498DB", "3": "#F39C12"}

    _img_dir = DATA_DIR / "camera_trap" / "caltech_subset"
    _img_files = sorted(
        p for p in _img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}
    )

    if not _img_files:
        print("No Caltech images found — run the download cell in p1 first.")
    else:
        _img_path = _img_files[0]
        _pil = Image.open(_img_path).convert("RGB")
        _arr = np.array(_pil)

        _result = detector.generate_detections_one_image(
            _arr, image_id=_img_path.name, detection_threshold=0.1
        )

        print(f"Image : {_img_path.name}  ({_pil.width}×{_pil.height} px)")
        print(f"Hits  : {len(_result['detections'])}")
        for _d in _result["detections"]:
            _lbl = _LABELS.get(_d["category"], "?")
            _bb  = [round(v, 3) for v in _d["bbox"]]
            print(f"  {_lbl:8s}  conf={_d['conf']:.3f}  bbox={_bb}")

        _W, _H = _pil.width, _pil.height
        _fig, _ax = plt.subplots(figsize=(10, 7))
        _ax.imshow(_arr)

        for _d in _result["detections"]:
            _x, _y, _bw, _bh = _d["bbox"]
            _lbl = _LABELS.get(_d["category"], "?")
            _col = _COLORS.get(_d["category"], "white")
            _ax.add_patch(mpatches.Rectangle(
                (_x * _W, _y * _H), _bw * _W, _bh * _H,
                linewidth=2, edgecolor=_col, facecolor="none",
            ))
            _ax.text(
                _x * _W, _y * _H - 4,
                f"{_lbl} {_d['conf']:.2f}",
                fontsize=8, color=_col, fontweight="bold",
            )

        _ax.set_title(f"MegaDetector v5a — {_img_path.name}", fontsize=10)
        _ax.axis("off")
        plt.tight_layout()

    plt.gca()
    return


@app.cell(hide_code=True)
def _step3_md(mo):
    mo.md(r"""
    ### Step 3 — Batch detection

    Run MegaDetector on all Caltech and Serengeti images and collect results
    into a single DataFrame. This is the starting point for the two-stage pipeline:

    ```
    images  →  MegaDetector  →  animal crops  →  species classifier (P5)
    ```
    """)
    return


@app.cell
def _batch(DATA_DIR, Image, detector, np, pd):
    _CONF_THRESHOLD = 0.2
    _LABELS = {"1": "animal", "2": "person", "3": "vehicle"}

    _sources = {
        "caltech":   DATA_DIR / "camera_trap" / "caltech_subset",
        "serengeti": DATA_DIR / "camera_trap" / "serengeti_subset",
    }

    records = []
    for _source, _img_dir in _sources.items():
        if not _img_dir.exists():
            continue
        _files = sorted(
            p for p in _img_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg"}
        )
        for _p in _files:
            _arr = np.array(Image.open(_p).convert("RGB"))
            _res = detector.generate_detections_one_image(
                _arr, image_id=_p.name, detection_threshold=_CONF_THRESHOLD
            )
            for _d in _res["detections"]:
                records.append({
                    "source":     _source,
                    "filename":   _p.name,
                    "filepath":   str(_p),
                    "category":   _LABELS.get(_d["category"], "unknown"),
                    "confidence": round(_d["conf"], 4),
                    "bbox_x":     round(_d["bbox"][0], 4),
                    "bbox_y":     round(_d["bbox"][1], 4),
                    "bbox_w":     round(_d["bbox"][2], 4),
                    "bbox_h":     round(_d["bbox"][3], 4),
                })

    detections_df = pd.DataFrame(records)
    _out = DATA_DIR / "camera_trap_detections.csv"
    detections_df.to_csv(_out, index=False)

    print(f"Images processed : {detections_df['filename'].nunique()}")
    print(f"Total detections : {len(detections_df)}")
    print()
    print("Category breakdown:")
    print(detections_df.groupby(["source", "category"]).size().to_string())
    print(f"\nSaved → {_out}")

    return (detections_df,)


@app.cell(hide_code=True)
def _step4_md(mo):
    mo.md(r"""
    ### Step 4 — Confidence threshold and empty-frame filtering

    The confidence threshold is the most important tuning parameter.
    Too low → many false positives (wind, leaves). Too high → missed animals.

    A common field deployment strategy: set threshold at 0.2, send all
    detections above it to a reviewer, skip everything below.
    """)
    return


@app.cell
def _threshold_analysis(detections_df, np, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))

    _axes[0].hist(
        detections_df["confidence"], bins=30,
        color="steelblue", edgecolor="white",
    )
    _axes[0].set_xlabel("Confidence score")
    _axes[0].set_ylabel("Number of detections")
    _axes[0].set_title("Confidence distribution — all detections")
    for _t in [0.2, 0.5, 0.8]:
        _axes[0].axvline(_t, color="red", linestyle="--", linewidth=1, alpha=0.6,
                         label=f"t={_t}")
    _axes[0].legend(fontsize=8)

    _thresholds = np.arange(0.05, 0.99, 0.05)
    _counts = [len(detections_df[detections_df["confidence"] >= t]) for t in _thresholds]
    _axes[1].plot(_thresholds, _counts, "o-", color="darkorange")
    _axes[1].set_xlabel("Confidence threshold")
    _axes[1].set_ylabel("Detections retained")
    _axes[1].set_title("Detections retained vs. threshold")
    _axes[1].grid(alpha=0.3)

    plt.suptitle("Choosing a confidence threshold", fontsize=12)
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _step5_md(mo):
    mo.md(r"""
    ### Step 5 — Extract animal crops

    Save each animal detection as a JPEG crop.
    These crops are the input to the species classifier in P5.
    """)
    return


@app.cell
def _crops(DATA_DIR, Image, detections_df):
    CROPS_DIR = DATA_DIR / "camera_trap_crops"
    CROPS_DIR.mkdir(exist_ok=True)

    _animals = detections_df[
        (detections_df["category"] == "animal") &
        (detections_df["confidence"] >= 0.2)
    ].copy()

    _saved = 0
    for _, _row in _animals.iterrows():
        _img = Image.open(_row["filepath"]).convert("RGB")
        _W, _H = _img.size
        _x1 = max(0, int(_row["bbox_x"] * _W))
        _y1 = max(0, int(_row["bbox_y"] * _H))
        _x2 = min(_W, int((_row["bbox_x"] + _row["bbox_w"]) * _W))
        _y2 = min(_H, int((_row["bbox_y"] + _row["bbox_h"]) * _H))
        _crop = _img.crop((_x1, _y1, _x2, _y2))
        _out  = CROPS_DIR / f"{_row['source']}_{_row['filename'].rsplit('.', 1)[0]}_crop{_saved:04d}.jpg"
        _crop.save(_out, quality=90)
        _saved += 1

    print(f"Saved {_saved} animal crops → {CROPS_DIR}")
    return (CROPS_DIR,)


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — SAHI: TILED INFERENCE FOR LARGE IMAGES
# ═══════════════════════════════════════════════════════════════════════════


@app.cell(hide_code=True)
def _part2_header(mo):
    mo.md(r"""
    ## Part 2 — SAHI: Tiled Inference for Large Images

    MegaDetector was trained on camera trap images (~1920×1080 px).
    Drone orthomosaics can be 10,000×10,000 px or larger, with animals
    appearing as tiny blobs of 20–50 pixels.

    **SAHI** (Slicing Aided Hyper Inference) solves this by:

    1. Slicing the image into overlapping tiles (e.g. 640×640)
    2. Running detection on each tile independently
    3. Merging results with NMS to remove duplicates at tile borders

    This lets a standard detector — trained on 640 px inputs — find small
    objects in arbitrarily large images.

    ```
    ┌──────────────────────────┐
    │  ┌───────┬───┬───────┐   │
    │  │ tile1 │ ↔ │ tile2 │   │   ↔ = overlap zone
    │  ├───────┼───┼───────┤   │
    │  │   ↕   │   │   ↕   │   │   ↕ = overlap zone
    │  ├───────┼───┼───────┤   │
    │  │ tile3 │   │ tile4 │   │
    │  └───────┴───┴───────┘   │
    └──────────────────────────┘
    ```
    """)
    return


@app.cell(hide_code=True)
def _step6_md(mo):
    mo.md(r"""
    ### Step 6 — Load a SAHI detection model

    SAHI wraps any detection model via `AutoDetectionModel`. Supported types
    include `yolov8`, `yolov5`, `detectron2`, `mmdet`, and more.

    We use YOLOv8-nano pre-trained on COCO (80 object classes including
    several animal categories). After Part 3 you can swap in your own
    fine-tuned weights.
    """)
    return


@app.cell
def _sahi_model_setup():
    from sahi import AutoDetectionModel

    sahi_det_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path="yolov8n.pt",
        confidence_threshold=0.2,
        device="cpu",
    )
    print("SAHI detection model loaded.")
    print(f"  Backbone         : YOLOv8-nano (COCO pre-trained, 80 classes)")
    print(f"  Conf. threshold  : {sahi_det_model.confidence_threshold}")
    return (sahi_det_model,)


@app.cell(hide_code=True)
def _step7_md(mo):
    mo.md(r"""
    ### Step 7 — Full-image vs. sliced inference

    We compare standard full-image detection with SAHI's sliced approach on
    a drone tile from the HerdNet test sample.

    On large images with small targets, sliced inference typically finds
    significantly more objects because each tile is resized to the model's
    native resolution (640 px), making small objects appear larger.
    """)
    return


@app.cell
def _sahi_compare(DATA_DIR, Image, mpatches, np, plt, sahi_det_model):
    from sahi.predict import get_prediction, get_sliced_prediction

    _drone_dir = DATA_DIR / "general_dataset" / "test_sample"
    _drone_imgs = sorted(_drone_dir.glob("*.jpeg"))[:1]

    if not _drone_imgs:
        print("No drone images found in data/general_dataset/test_sample/")
        print("Run: python week1/data/download_data.py --sample")
    else:
        _img_path = str(_drone_imgs[0])
        _pil = Image.open(_img_path).convert("RGB")
        _arr = np.array(_pil)

        # Full-image inference (no slicing)
        _full = get_prediction(
            image=_img_path,
            detection_model=sahi_det_model,
        )

        # Sliced inference
        _sliced = get_sliced_prediction(
            image=_img_path,
            detection_model=sahi_det_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            postprocess_type="NMS",
            postprocess_match_threshold=0.5,
        )

        print(f"Image: {_drone_imgs[0].name} ({_pil.width}×{_pil.height} px)")
        print(f"Full-image detections : {len(_full.object_prediction_list)}")
        print(f"Sliced detections     : {len(_sliced.object_prediction_list)}")

        _fig, _axes = plt.subplots(1, 2, figsize=(14, 6))

        for _ax, _res, _title in [
            (_axes[0], _full, "Full-image inference"),
            (_axes[1], _sliced, "SAHI sliced (640×640, 20% overlap)"),
        ]:
            _ax.imshow(_arr)
            for _pred in _res.object_prediction_list:
                _bb = _pred.bbox
                _x1, _y1 = _bb.minx, _bb.miny
                _x2, _y2 = _bb.maxx, _bb.maxy
                _col = "#E74C3C"
                _ax.add_patch(mpatches.Rectangle(
                    (_x1, _y1), _x2 - _x1, _y2 - _y1,
                    linewidth=1.5, edgecolor=_col, facecolor="none",
                ))
                _ax.text(
                    _x1, _y1 - 3,
                    f"{_pred.category.name} {_pred.score.value:.2f}",
                    fontsize=6, color=_col, fontweight="bold",
                )
            _n = len(_res.object_prediction_list)
            _ax.set_title(f"{_title} ({_n} hits)", fontsize=9)
            _ax.axis("off")

        plt.suptitle(
            "Full-image vs. SAHI tiled inference on drone imagery", fontsize=11
        )
        plt.tight_layout()

    plt.gca()
    return


@app.cell(hide_code=True)
def _step8_md(mo):
    mo.md(r"""
    ### Step 8 — Interactive SAHI parameter tuning

    Use the sliders below to explore how slice size, overlap, and confidence
    threshold affect detection results.

    | Parameter | ↑ Higher | ↓ Lower |
    |-----------|----------|---------|
    | **Slice size** | Faster, misses small objects | Slower, catches tiny targets |
    | **Overlap** | Better border coverage, slower | Faster, may miss border objects |
    | **Confidence** | Fewer false positives | More detections, more noise |

    **Postprocess strategies:**
    - **NMS** (Non-Maximum Suppression) — removes the lower-confidence box when two overlap
    - **NMM** (Non-Maximum Merging) — merges overlapping boxes into one larger box
    """)
    return


@app.cell
def _sahi_sliders(mo):
    slice_size = mo.ui.slider(
        128, 1024, step=64, value=640, label="Slice size (px)"
    )
    overlap = mo.ui.slider(
        0.0, 0.5, step=0.05, value=0.2, label="Overlap ratio"
    )
    sahi_conf = mo.ui.slider(
        0.05, 0.9, step=0.05, value=0.2, label="Confidence threshold"
    )
    postprocess = mo.ui.dropdown(
        options={"NMS (suppress duplicates)": "NMS", "NMM (merge duplicates)": "NMM"},
        value="NMS",
        label="Postprocess",
    )
    mo.hstack([slice_size, overlap, sahi_conf, postprocess], justify="start", gap=2)
    return overlap, postprocess, sahi_conf, slice_size


@app.cell
def _sahi_interactive(
    DATA_DIR, Image, mpatches, np, overlap, plt, postprocess,
    sahi_conf, sahi_det_model, slice_size,
):
    from sahi.predict import get_sliced_prediction

    _drone_dir = DATA_DIR / "general_dataset" / "test_sample"
    _drone_imgs = sorted(_drone_dir.glob("*.jpeg"))[:1]

    if not _drone_imgs:
        print("No drone images found.")
    else:
        _img_path = str(_drone_imgs[0])
        _pil = Image.open(_img_path).convert("RGB")
        _arr = np.array(_pil)

        sahi_det_model.confidence_threshold = sahi_conf.value

        _result = get_sliced_prediction(
            image=_img_path,
            detection_model=sahi_det_model,
            slice_height=slice_size.value,
            slice_width=slice_size.value,
            overlap_height_ratio=overlap.value,
            overlap_width_ratio=overlap.value,
            postprocess_type=postprocess.value,
            postprocess_match_threshold=0.5,
        )

        _n_tiles = (
            ((_pil.width - 1) // int(slice_size.value * (1 - overlap.value)) + 1) *
            ((_pil.height - 1) // int(slice_size.value * (1 - overlap.value)) + 1)
        )

        print(f"Image : {_drone_imgs[0].name} ({_pil.width}×{_pil.height} px)")
        print(f"Tiles : ~{_n_tiles}  (slice={slice_size.value}, overlap={overlap.value:.0%})")
        print(f"Hits  : {len(_result.object_prediction_list)}")

        _fig, _ax = plt.subplots(figsize=(10, 8))
        _ax.imshow(_arr)

        for _pred in _result.object_prediction_list:
            _bb = _pred.bbox
            _x1, _y1 = _bb.minx, _bb.miny
            _x2, _y2 = _bb.maxx, _bb.maxy
            _ax.add_patch(mpatches.Rectangle(
                (_x1, _y1), _x2 - _x1, _y2 - _y1,
                linewidth=1.5, edgecolor="#E74C3C", facecolor="none",
            ))
            _ax.text(
                _x1, _y1 - 3,
                f"{_pred.category.name} {_pred.score.value:.2f}",
                fontsize=6, color="#E74C3C", fontweight="bold",
            )

        _n = len(_result.object_prediction_list)
        _ax.set_title(
            f"SAHI — slice={slice_size.value}px, overlap={overlap.value:.0%}, "
            f"conf≥{sahi_conf.value:.2f}, {postprocess.value} → {_n} detections",
            fontsize=10,
        )
        _ax.axis("off")
        plt.tight_layout()

    plt.gca()
    return


@app.cell(hide_code=True)
def _sahi_batch_md(mo):
    mo.md(r"""
    ### Step 9 — SAHI batch inference on multiple drone tiles

    Run SAHI on all test sample tiles and collect results into a DataFrame.
    This is the workflow you would use for a full orthomosaic survey.
    """)
    return


@app.cell
def _sahi_batch(DATA_DIR, pd, sahi_det_model):
    from sahi.predict import get_sliced_prediction

    _drone_dir = DATA_DIR / "general_dataset" / "test_sample"
    _drone_imgs = sorted(_drone_dir.glob("*.jpeg"))

    if not _drone_imgs:
        print("No drone images found.")
        sahi_batch_df = pd.DataFrame()
    else:
        sahi_det_model.confidence_threshold = 0.2
        _records = []

        for _p in _drone_imgs:
            _result = get_sliced_prediction(
                image=str(_p),
                detection_model=sahi_det_model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                postprocess_type="NMS",
                postprocess_match_threshold=0.5,
            )
            for _pred in _result.object_prediction_list:
                _bb = _pred.bbox
                _records.append({
                    "filename":   _p.name,
                    "filepath":   str(_p),
                    "category":   _pred.category.name,
                    "confidence": round(_pred.score.value, 4),
                    "bbox_x1":    _bb.minx,
                    "bbox_y1":    _bb.miny,
                    "bbox_x2":    _bb.maxx,
                    "bbox_y2":    _bb.maxy,
                })

        sahi_batch_df = pd.DataFrame(_records)
        _out = DATA_DIR / "sahi_drone_detections.csv"
        sahi_batch_df.to_csv(_out, index=False)

        print(f"Drone tiles processed : {len(_drone_imgs)}")
        print(f"Total SAHI detections : {len(sahi_batch_df)}")
        if len(sahi_batch_df) > 0:
            print(f"\nCategory breakdown:")
            print(sahi_batch_df["category"].value_counts().to_string())
        print(f"\nSaved → {_out}")

    return (sahi_batch_df,)


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — FINE-TUNING A CUSTOM YOLOV8 DETECTOR
# ═══════════════════════════════════════════════════════════════════════════


@app.cell(hide_code=True)
def _part3_header(mo):
    mo.md(r"""
    ## Part 3 — Fine-Tuning a Custom YOLOv8 Detector

    MegaDetector was trained on millions of camera trap images from around the world.
    But what if your images come from a new context — drone aerial view, unusual species,
    or a setting very different from the training data?

    Here you fine-tune a lightweight **YOLOv8** detector on the bounding-box annotations
    you created in P2. The workflow:

    ```
    Label Studio export (COCO JSON)
        → convert to YOLO .txt format
        → create dataset.yaml
        → yolo train
        → evaluate
        → run SAHI with your trained model
    ```

    **Note:** If you haven't annotated enough images in P2 yet, the Caltech bounding boxes
    from `camera_trap_labels.csv` are used automatically as a fallback.
    """)
    return


@app.cell(hide_code=True)
def _step10_md(mo):
    mo.md(r"""
    ### Step 10 — Prepare training data

    YOLO expects one `.txt` file per image, same stem as the image file.
    Each line: `class_id  cx  cy  w  h` — all values normalised 0–1.

    ```
    0  0.512  0.344  0.231  0.189     ← class 0 (animal), centre x/y, width, height
    ```

    We convert from either:
    - **COCO JSON** (Label Studio export): `[x, y, w, h]` pixel → normalised centre
    - **Caltech CSV fallback**: same conversion from `bbox_x/y/w/h` columns
    """)
    return


@app.cell
def _prepare_yolo(DATA_DIR, Image):
    import shutil as _shutil
    import json as _json

    YOLO_DIR  = DATA_DIR / "yolo_dataset"
    TRAIN_DIR = YOLO_DIR / "images" / "train"
    VAL_DIR   = YOLO_DIR / "images" / "val"
    LABEL_TRAIN = YOLO_DIR / "labels" / "train"
    LABEL_VAL   = YOLO_DIR / "labels" / "val"

    for _d in [TRAIN_DIR, VAL_DIR, LABEL_TRAIN, LABEL_VAL]:
        _d.mkdir(parents=True, exist_ok=True)

    CLASSES = ["animal", "person", "vehicle"]

    def _coco_to_yolo_line(x, y, w, h, img_w, img_h, class_id):
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        wn = w / img_w
        hn = h / img_h
        return f"{class_id} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}"

    # ── Source 1: Label Studio COCO JSON export from p2 ──────────────────
    _ls_path = DATA_DIR / "my_serengeti_bboxes.json"
    annotations_by_file: dict = {}

    if _ls_path.exists():
        print(f"Loading Label Studio export: {_ls_path.name}")
        with open(_ls_path) as _f:
            _coco = _json.load(_f)
        _id_to_img = {img["id"]: img for img in _coco["images"]}
        _id_to_cat = {c["id"]: c["name"].lower() for c in _coco["categories"]}
        for _ann in _coco["annotations"]:
            if not _ann.get("bbox"):
                continue
            _img  = _id_to_img[_ann["image_id"]]
            _fname = _img["file_name"].split("/")[-1]
            _iw, _ih = _img["width"], _img["height"]
            _cls  = CLASSES.index(_id_to_cat.get(_ann["category_id"], "animal"))
            _x, _y, _w, _h = _ann["bbox"]
            annotations_by_file.setdefault(_fname, []).append(
                _coco_to_yolo_line(_x, _y, _w, _h, _iw, _ih, _cls)
            )
        print(f"  {len(annotations_by_file)} images with bbox annotations")

    # ── Source 2: Caltech CSV fallback ───────────────────────────────────
    _caltech_csv = DATA_DIR / "camera_trap_labels.csv"
    _caltech_dir = DATA_DIR / "camera_trap" / "caltech_subset"

    if not annotations_by_file and _caltech_csv.exists():
        print("No Label Studio export found — using Caltech CSV as fallback.")
        import pandas as _pd
        _df = _pd.read_csv(_caltech_csv).dropna(subset=["bbox_x"])
        for _, _row in _df.iterrows():
            _fname = _row["crop"]
            _src   = _caltech_dir / _fname
            if not _src.exists():
                continue
            with Image.open(_src) as _im:
                _iw, _ih = _im.size
            _cls = 0  # all "animal"
            _line = _coco_to_yolo_line(
                _row["bbox_x"], _row["bbox_y"], _row["bbox_w"], _row["bbox_h"],
                _iw, _ih, _cls,
            )
            annotations_by_file.setdefault(_fname, []).append(_line)
        print(f"  {len(annotations_by_file)} images with bbox annotations (Caltech fallback)")

    # ── Split and copy ───────────────────────────────────────────────────
    _fnames = sorted(annotations_by_file.keys())
    _split  = int(len(_fnames) * 0.8)
    _train_fnames = _fnames[:_split]
    _val_fnames   = _fnames[_split:]

    def _write_split(fnames, img_dst, lbl_dst, source_dirs):
        _written = 0
        for _fn in fnames:
            for _src_dir in source_dirs:
                _src = _src_dir / _fn
                if _src.exists():
                    _shutil.copy(_src, img_dst / _fn)
                    (lbl_dst / (_src.stem + ".txt")).write_text(
                        "\n".join(annotations_by_file[_fn])
                    )
                    _written += 1
                    break
        return _written

    _src_dirs = [
        DATA_DIR / "camera_trap" / "serengeti_subset",
        DATA_DIR / "camera_trap" / "caltech_subset",
        DATA_DIR / "ls_serengeti",
        DATA_DIR / "ls_caltech",
    ]

    _n_train = _write_split(_train_fnames, TRAIN_DIR, LABEL_TRAIN, _src_dirs)
    _n_val   = _write_split(_val_fnames,   VAL_DIR,   LABEL_VAL,   _src_dirs)

    # ── dataset.yaml ─────────────────────────────────────────────────────
    _yaml = f"""path: {YOLO_DIR}
train: images/train
val:   images/val

nc: {len(CLASSES)}
names: {CLASSES}
"""
    (YOLO_DIR / "dataset.yaml").write_text(_yaml)

    print(f"\nYOLO dataset ready:")
    print(f"  Train : {_n_train} images  ({TRAIN_DIR})")
    print(f"  Val   : {_n_val}   images  ({VAL_DIR})")
    print(f"  YAML  : {YOLO_DIR / 'dataset.yaml'}")

    return CLASSES, YOLO_DIR, annotations_by_file


@app.cell(hide_code=True)
def _step11_md(mo):
    mo.md(r"""
    ### Step 11 — Choose model variant and train

    Select a YOLOv8 variant and number of epochs. The model is initialised from
    COCO pre-trained weights (transfer learning), then fine-tuned on your data.

    | Variant | Size | Speed | Accuracy |
    |---------|------|-------|----------|
    | `yolov8n` (nano) | 3 MB | Fastest | Baseline |
    | `yolov8s` (small) | 11 MB | Fast | Better |
    | `yolov8m` (medium) | 26 MB | Moderate | Best |

    With 20 epochs and a small dataset you won't reach production accuracy —
    but you will see the model learning. Production training typically uses 100–300 epochs.
    """)
    return


@app.cell
def _training_config(mo):
    yolo_variant = mo.ui.dropdown(
        options={
            "YOLOv8-nano (3 MB, fastest)": "yolov8n.pt",
            "YOLOv8-small (11 MB, better)": "yolov8s.pt",
            "YOLOv8-medium (26 MB, best)": "yolov8m.pt",
        },
        value="yolov8n.pt",
        label="Base model",
    )
    epochs_slider = mo.ui.slider(
        5, 100, step=5, value=20, label="Training epochs"
    )
    mo.hstack([yolo_variant, epochs_slider], justify="start", gap=2)
    return epochs_slider, yolo_variant


@app.cell
def _train(YOLO_DIR, epochs_slider, yolo_variant):
    from ultralytics import YOLO as _YOLO

    _yaml_path = YOLO_DIR / "dataset.yaml"

    yolo_model = _YOLO(yolo_variant.value)

    _results = yolo_model.train(
        data=str(_yaml_path),
        epochs=epochs_slider.value,
        imgsz=640,
        batch=8,
        project=str(YOLO_DIR / "runs"),
        name="train",
        exist_ok=True,
        verbose=False,
    )

    print(f"Training complete ({epochs_slider.value} epochs, {yolo_variant.value}).")
    print(f"  Best weights : {YOLO_DIR / 'runs' / 'train' / 'weights' / 'best.pt'}")
    _map50 = _results.results_dict.get("metrics/mAP50(B)", None)
    if _map50 is not None:
        print(f"  mAP50        : {_map50:.3f}")

    return (yolo_model,)


@app.cell(hide_code=True)
def _step12_md(mo):
    mo.md(r"""
    ### Step 12 — Evaluate on validation images

    Compare your trained detector against MegaDetector on the same images.
    At this annotation scale (tens of images) the custom detector will likely
    underperform MegaDetector — but you can see it learning.
    """)
    return


@app.cell
def _evaluate(Image, YOLO_DIR, mpatches, np, plt, yolo_model):
    _val_dir = YOLO_DIR / "images" / "val"
    _val_imgs = sorted(_val_dir.iterdir())[:6]

    if not _val_imgs:
        print("No validation images found.")
    else:
        _ncols = 3
        _nrows = (len(_val_imgs) + _ncols - 1) // _ncols
        _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(_ncols * 5, _nrows * 4))
        _axes_flat = np.array(_axes).flatten()

        for _i, _p in enumerate(_val_imgs):
            _arr  = np.array(Image.open(_p).convert("RGB"))
            _preds = yolo_model.predict(_arr, conf=0.2, verbose=False)[0]

            _ax = _axes_flat[_i]
            _ax.imshow(_arr)

            for _box in _preds.boxes:
                _x1, _y1, _x2, _y2 = _box.xyxy[0].tolist()
                _conf = float(_box.conf[0])
                _cls  = int(_box.cls[0])
                _col  = ["#E74C3C", "#3498DB", "#F39C12"][_cls % 3]
                _ax.add_patch(mpatches.Rectangle(
                    (_x1, _y1), _x2 - _x1, _y2 - _y1,
                    linewidth=2, edgecolor=_col, facecolor="none",
                ))
                _ax.text(_x1, _y1 - 4, f"{_conf:.2f}", fontsize=7, color=_col)

            _ax.set_title(_p.name, fontsize=7)
            _ax.axis("off")

        for _ax in _axes_flat[len(_val_imgs):]:
            _ax.axis("off")

        plt.suptitle("Custom YOLOv8 detector — validation images", fontsize=11)
        plt.tight_layout()

    plt.gca()
    return


@app.cell(hide_code=True)
def _step13_md(mo):
    mo.md(r"""
    ### Step 13 — SAHI with your fine-tuned model

    The real power of SAHI comes when you combine it with a domain-specific
    detector. Your fine-tuned YOLOv8 knows your target classes. SAHI lets
    it work on images of any size.

    This is the full pipeline for production drone surveys:
    ```
    annotate images → fine-tune YOLOv8 → SAHI tiled inference on orthomosaic
    ```
    """)
    return


@app.cell
def _sahi_finetuned(DATA_DIR, Image, YOLO_DIR, mpatches, np, plt):
    from sahi import AutoDetectionModel
    from sahi.predict import get_prediction, get_sliced_prediction

    _best_weights = YOLO_DIR / "runs" / "train" / "weights" / "best.pt"

    if not _best_weights.exists():
        print("No trained weights found — run Step 11 first.")
    else:
        _custom_sahi = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=str(_best_weights),
            confidence_threshold=0.2,
            device="cpu",
        )

        _drone_dir = DATA_DIR / "general_dataset" / "test_sample"
        _drone_imgs = sorted(_drone_dir.glob("*.jpeg"))[:1]

        if not _drone_imgs:
            print("No drone images found.")
        else:
            _img_path = str(_drone_imgs[0])
            _pil = Image.open(_img_path).convert("RGB")
            _arr = np.array(_pil)

            _full = get_prediction(
                image=_img_path,
                detection_model=_custom_sahi,
            )

            _sliced = get_sliced_prediction(
                image=_img_path,
                detection_model=_custom_sahi,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                postprocess_type="NMS",
                postprocess_match_threshold=0.5,
            )

            print(f"Fine-tuned model on drone tile: {_drone_imgs[0].name}")
            print(f"  Full-image : {len(_full.object_prediction_list)} detections")
            print(f"  SAHI sliced: {len(_sliced.object_prediction_list)} detections")

            _fig, _axes = plt.subplots(1, 2, figsize=(14, 6))

            for _ax, _res, _title in [
                (_axes[0], _full, "Fine-tuned — full image"),
                (_axes[1], _sliced, "Fine-tuned — SAHI sliced"),
            ]:
                _ax.imshow(_arr)
                for _pred in _res.object_prediction_list:
                    _bb = _pred.bbox
                    _x1, _y1 = _bb.minx, _bb.miny
                    _x2, _y2 = _bb.maxx, _bb.maxy
                    _ax.add_patch(mpatches.Rectangle(
                        (_x1, _y1), _x2 - _x1, _y2 - _y1,
                        linewidth=1.5, edgecolor="#2ECC71", facecolor="none",
                    ))
                    _ax.text(
                        _x1, _y1 - 3,
                        f"{_pred.category.name} {_pred.score.value:.2f}",
                        fontsize=6, color="#2ECC71", fontweight="bold",
                    )
                _n = len(_res.object_prediction_list)
                _ax.set_title(f"{_title} ({_n} hits)", fontsize=9)
                _ax.axis("off")

            plt.suptitle(
                "Fine-tuned YOLOv8 + SAHI on drone imagery", fontsize=11
            )
            plt.tight_layout()

    plt.gca()
    return


# ═══════════════════════════════════════════════════════════════════════════
# EXERCISES & REFLECTION
# ═══════════════════════════════════════════════════════════════════════════


@app.cell(hide_code=True)
def _exercise(mo):
    mo.md(r"""
    ## Exercise

    1. **Threshold sweep** — Change `_CONF_THRESHOLD` in Step 3 to 0.05, 0.2, and 0.5.
       At each threshold, how many images are classified as "empty" (zero detections)?
       What is the false-positive rate at 0.05?

    2. **Failure modes** — Browse your crops in `data/camera_trap_crops/`.
       Find one of each:
       - A confident correct detection (conf > 0.8)
       - A confident false positive (conf > 0.5 but clearly not an animal)
       - A missed animal (visible in the original image, no detection)

    3. **SAHI parameter exploration** — Using the sliders in Step 8:
       - Set slice size to 256 and then 1024. How does detection count change?
       - Set overlap to 0 vs 0.4. Do you see missed objects at tile borders with 0 overlap?
       - Switch postprocess from NMS to NMM. What happens to overlapping detections?

    4. **Training data size** — Your custom model was trained on very few images.
       After adding more annotations from p2, re-run Step 11. Does mAP improve?
       How many images do you need before it approaches MegaDetector performance?

    5. **Fine-tuned SAHI** — Compare the COCO-pretrained SAHI results (Step 7)
       with your fine-tuned SAHI results (Step 13). Does domain-specific training
       help even with SAHI's tiling strategy?

    6. **MegaDetector on aerial imagery** — Run MegaDetector on one HerdNet tile
       (from `data/general_dataset/test_sample/`). What does it detect?
       Is a camera-trap detector appropriate for drone data?
    """)
    return


@app.cell(hide_code=True)
def _reflection(mo):
    mo.md(r"""
    ## Reflection

    - MegaDetector was trained on 4.5 million images from 89 sources.
      Your custom detector was trained on tens of images.
      What does this say about the role of pre-trained foundation models in ecology?

    - MegaDetector detects *animal* as one class. It does not distinguish
      "zebra" from "elephant". What are the implications for a species monitoring pipeline?

    - SAHI adds a tiling layer on top of any detector. When is this essential
      (drone surveys, satellite imagery) vs unnecessary (camera traps at 1080p)?
      What is the computational cost of slicing a 10,000×10,000 px image into 640 px tiles?

    - You used citizen-science species labels (Serengeti) to draw bounding boxes.
      If those species labels are wrong, how does that affect:
      (a) the bounding box quality?
      (b) the downstream classifier trained on those crops?

    - YOLOv8n achieves competitive accuracy on standard benchmarks with a 3 MB model.
      MegaDetector v5 is ~600 MB. When would you prefer the smaller model?

    - Compare two approaches for detecting small animals in large drone images:
      (a) SAHI + YOLOv8 (box-based, tile-and-merge)
      (b) HerdNet (density-map-based, with Stitcher)
      What are the trade-offs?
    """)
    return


if __name__ == "__main__":
    app.run()
