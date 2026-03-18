import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _imports():
    import json
    from pathlib import Path

    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    DATA_DIR = Path(__file__).parent.parent / "data"
    return DATA_DIR, Image, Path, json, mpatches, np, plt


@app.cell(hide_code=True)
def _part1(mo):
    mo.md(r"""
    # Practical 2 — Annotating Images with Label Studio

    **Context:** A model can only learn to find what humans have labelled.
    Before training anything, someone must draw boxes or outlines on every animal.
    This practical teaches you how to do that with **Label Studio** — a free,
    open-source tool that runs entirely on your laptop with no cloud account required.

    We work with two real datasets that represent two common annotation gaps:

    | Dataset | What exists | What you will add |
    |---------|-------------|-------------------|
    | **Snapshot Serengeti** | Species label per image — no locations | Bounding boxes |
    | **Caltech Camera Traps** | Bounding boxes | Polygon outlines |

    This mirrors real project work: you often receive image-level labels from a
    citizen science platform and must add spatial annotations before training a detector.

    **Installation (run once):**
    ```bash
    pip install label-studio
    ```
    """)
    return


@app.cell
def _viz_polygons(mo):
    mo.md(r"""
    ## Part 1 — Setup

    ### Step 1 — Start Label Studio

    ```bash
    label-studio start
    ```

    Your browser will open at **http://localhost:8080**.
    Create a free local account (no data leaves your machine).

    **Troubleshooting:**
    - Port busy → `label-studio start --port 8081`
    - Windows → run in a standard terminal, not WSL
    - Browser didn't open → navigate to http://localhost:8080 manually
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Step 2 — Programmatic sync (optional but recommended)

    Instead of uploading images by hand, `label_studio_sync.py` does it from the terminal.
    It also pre-fills bounding boxes from existing annotation files so annotators only
    need to correct rather than draw from scratch.

    **Get your API token first:**
    `http://localhost:8080/user/account` → copy the **Access Token**

    ```bash
    export LS_TOKEN=your_token_here   # Windows: set LS_TOKEN=your_token_here
    ```

    **Upload Serengeti images** (no pre-annotations — you draw the boxes):
    ```bash
    python week1/data/label_studio_sync.py upload \
        --images week1/data/ls_serengeti \
        --project "Serengeti Boxes" \
        --config bbox
    ```

    **Upload Caltech images with existing bounding boxes pre-filled:**
    ```bash
    python week1/data/label_studio_sync.py upload \
        --images week1/data/ls_caltech \
        --annotations week1/data/camera_trap_labels.csv \
        --annotation-format caltech-csv \
        --project "Caltech Polygons" \
        --config polygon
    ```

    **Upload Eikelboom dataset** (Pascal VOC CSV, no header):
    ```bash
    python week1/data/label_studio_sync.py upload \
        --images /path/to/eikelboom/train \
        --annotations /path/to/eikelboom/annotations_train.csv \
        --annotation-format eikelboom-csv \
        --project "Eikelboom" \
        --config bbox
    ```

    **Export completed annotations** as COCO JSON:
    ```bash
    python week1/data/label_studio_sync.py export \
        --project "Serengeti Boxes" \
        --output week1/data/my_serengeti_bboxes.json

    python week1/data/label_studio_sync.py export \
        --project "Caltech Polygons" \
        --output week1/data/my_caltech_polygons.json
    ```

    Pre-annotations appear in Label Studio as **model suggestions** (dashed outlines).
    Annotators can accept, adjust, or reject each one before submitting.
    Supported formats: COCO JSON, Caltech CSV, Eikelboom Pascal VOC CSV.
    """)
    return


@app.cell(hide_code=True)
def _part2(mo):
    mo.md(r"""
    ## Part 2 — Bounding Box Annotation (Snapshot Serengeti)

    The Serengeti dataset was labelled by citizen scientists on the Zooniverse platform.
    Each image has a **species label** — but no location. Volunteers typed "zebra" but
    never drew a box. Your job is to add the boxes.

    This is the annotation gap that MegaDetector was trained to close:
    detect *where* an animal is, then classify *what* it is.
    """)
    return


@app.cell(hide_code=True)
def _step3(DATA_DIR, Image, Path, json, np, plt):
    """Show Serengeti images with their image-level species labels."""
    import shutil as _shutil

    _meta_path = DATA_DIR / "camera_trap" / "serengeti_meta.json"
    _img_dir   = DATA_DIR / "camera_trap" / "serengeti_subset"

    SERENGETI_UPLOAD = DATA_DIR / "ls_serengeti"
    SERENGETI_UPLOAD.mkdir(exist_ok=True)

    # Always defined at outer scope so marimo can export it
    serengeti_labels: dict = {}

    if not _meta_path.exists():
        print("Serengeti data not found — run the download cell in p1 first.")
    else:
        with open(_meta_path) as _f:
            _meta = json.load(_f)

        _cat_map = {c["id"]: c["name"] for c in _meta["categories"]}
        _seq_labels: dict = {}
        for _a in _meta["annotations"]:
            _seq_labels.setdefault(_a["seq_id"], set()).add(
                _cat_map.get(_a["category_id"], "unknown")
            )

        _copied = 0
        for _img in _meta["images"]:
            _basename = Path(_img["file_name"]).name
            _src = _img_dir / _basename
            if not _src.exists():
                continue
            _dst = SERENGETI_UPLOAD / _basename
            if not _dst.exists():
                _shutil.copy(_src, _dst)
            _lbl_set = _seq_labels.get(_img["seq_id"], {"empty"})
            serengeti_labels[_basename] = ", ".join(sorted(_lbl_set))
            _copied += 1

        print(f"Serengeti upload folder : {SERENGETI_UPLOAD}")
        print(f"Images copied           : {_copied}")
        print()
        print("Image-level labels (no spatial information):")
        for _name, _lbl in list(serengeti_labels.items())[:10]:
            print(f"  {_name:<35s}  {_lbl}")

        _items = [(k, v) for k, v in serengeti_labels.items() if v != "empty"][:10]
        if _items:
            _fig, _axes = plt.subplots(2, 5, figsize=(15, 6))
            _axes_flat = np.array(_axes).flatten()
            for _i, (_fname, _lbl) in enumerate(_items):
                _arr = np.array(Image.open(SERENGETI_UPLOAD / _fname).convert("RGB"))
                _axes_flat[_i].imshow(_arr)
                _axes_flat[_i].set_title(_lbl, fontsize=8, color="steelblue")
                _axes_flat[_i].axis("off")
            for _ax in _axes_flat[len(_items):]:
                _ax.axis("off")
            plt.suptitle(
                "Snapshot Serengeti — species label only, no bounding boxes",
                fontsize=11,
            )
            plt.tight_layout()

    plt.gca()
    return SERENGETI_UPLOAD, serengeti_labels


@app.cell(hide_code=True)
def _step5(SERENGETI_UPLOAD, mo):
    mo.md(f"""
    ### Step 2 — Create a bounding box project in Label Studio

    1. Click **Create Project** → name it `"Serengeti Boxes"`
    2. **Data Import** tab → drag this folder:
       `{SERENGETI_UPLOAD}`
    3. **Labelling Setup** tab → choose **Custom template** and paste:

    ```xml
    <View>
      <Image name="image" value="$image" zoom="true" zoomControl="true"/>
      <RectangleLabels name="label" toName="image">
        <Label value="animal"  background="#E74C3C"/>
        <Label value="vehicle" background="#3498DB"/>
        <Label value="person"  background="#2ECC71"/>
      </RectangleLabels>
    </View>
    ```

    **Keyboard shortcuts:**

    | Key | Action |
    |-----|--------|
    | `1` / `2` / `3` | Select label |
    | drag | Draw a box |
    | `Backspace` | Delete selected box |
    | `Ctrl+Z` | Undo |
    | `D` / `A` | Next / previous image |
    | `Ctrl+Enter` | Submit and advance |

    **Goal:** Annotate all animals in at least 5 images. The species label shown in
    the grid above tells you what to look for — your box should show *where* it is.
    """)
    return


@app.cell(hide_code=True)
def _step8(mo):
    mo.md(r"""
    ### Step 3 — Export and load your bounding boxes

    1. Go back to the project list (click the project name)
    2. Click **Export** → choose **COCO JSON**
    3. Save the file to: `week1/data/my_serengeti_bboxes.json`
    """)
    return


@app.cell
def _load_polygons(DATA_DIR, json):
    _path = DATA_DIR / "my_serengeti_bboxes.json"

    if not _path.exists():
        bbox_coco = None
        print(f"Not found: {_path}")
        print("Annotate images in Label Studio, export as COCO JSON, and save to that path.")
    else:
        with open(_path) as _f:
            bbox_coco = json.load(_f)

        _n_img = len(bbox_coco["images"])
        _n_ann = len(bbox_coco["annotations"])
        _cats  = [c["name"] for c in bbox_coco["categories"]]

        print(f"Loaded  : {_path.name}")
        print(f"Images  : {_n_img}  |  Annotations: {_n_ann}  |  Classes: {_cats}")
        print()
        print("COCO structure:")
        print(f"  images[0]       = {bbox_coco['images'][0]}")
        print(f"  annotations[0]  = {bbox_coco['annotations'][0]}")
        print()
        print("bbox format: [x, y, w, h]  (COCO — top-left corner, pixel units)")
    return (bbox_coco,)


@app.cell(hide_code=True)
def _step6(
    Image,
    Path,
    SERENGETI_UPLOAD,
    bbox_coco,
    mpatches,
    np,
    plt,
    serengeti_labels: dict,
):
    """Bounding boxes overlaid on images, with the original image-level label shown."""
    if bbox_coco is None:
        print("No annotations loaded — complete Step 3 first.")
    else:
        _id_to_img = {img["id"]: img for img in bbox_coco["images"]}
        _id_to_cat = {c["id"]: c["name"] for c in bbox_coco["categories"]}

        _img_anns: dict = {}
        for _a in bbox_coco["annotations"]:
            _img_anns.setdefault(_a["image_id"], []).append(_a)

        _img_ids = list(_img_anns.keys())[:6]
        _ncols = min(3, len(_img_ids))
        _nrows = (len(_img_ids) + _ncols - 1) // _ncols

        _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(_ncols * 5, _nrows * 4))
        _axes_flat = np.array(_axes).flatten()

        _col_map = {"animal": "#E74C3C", "vehicle": "#3498DB", "person": "#2ECC71"}

        for _i, _img_id in enumerate(_img_ids):
            _meta  = _id_to_img[_img_id]
            _fname = Path(_meta["file_name"]).name
            _path  = SERENGETI_UPLOAD / _fname
            if not _path.exists():
                _axes_flat[_i].axis("off")
                continue

            _arr = np.array(Image.open(_path).convert("RGB"))
            _ax  = _axes_flat[_i]
            _ax.imshow(_arr)

            for _ann in _img_anns[_img_id]:
                _x, _y, _w, _h = _ann["bbox"]
                _cat = _id_to_cat.get(_ann["category_id"], "?")
                _col = _col_map.get(_cat, "#F39C12")
                _ax.add_patch(mpatches.Rectangle(
                    (_x, _y), _w, _h,
                    linewidth=2, edgecolor=_col, facecolor="none",
                ))
                _ax.text(_x, _y - 4, _cat, fontsize=7, color=_col, fontweight="bold")

            # Show the original image-level label in the title
            _orig_label = serengeti_labels.get(_fname, "?")
            _ax.set_title(f"{_fname}\nlabel: {_orig_label}", fontsize=7)
            _ax.axis("off")

        for _ax in _axes_flat[len(_img_ids):]:
            _ax.axis("off")

        plt.suptitle(
            "Your boxes (spatial) vs. original labels (image-level) — Snapshot Serengeti",
            fontsize=11,
        )
        plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _part3(mo):
    mo.md(r"""
    ## Part 3 — Polygon / Segmentation Annotation (Caltech Camera Traps)

    Caltech Camera Traps already has bounding boxes — but no outlines.
    For segmentation models (SAM, U-Net) you need polygon masks that trace the
    exact animal shape. Camera trap images work well here because the animal
    is large in the frame and easy to outline.
    """)
    return


@app.cell(hide_code=True)
def _step9(DATA_DIR, Image, np, plt):
    """Show Caltech images with their ground-truth bounding boxes."""
    import shutil as _shutil
    import pandas as _pd

    _caltech_dir   = DATA_DIR / "camera_trap" / "caltech_subset"
    _labels_path   = DATA_DIR / "camera_trap_labels.csv"

    CALTECH_UPLOAD = DATA_DIR / "ls_caltech"
    CALTECH_UPLOAD.mkdir(exist_ok=True)

    # Always defined at outer scope so marimo can export it
    caltech_df = None

    if not _labels_path.exists() or not _caltech_dir.exists():
        print("Caltech data not found — run the download cell in p1 first.")
    else:
        caltech_df = _pd.read_csv(_labels_path)
        caltech_df = caltech_df.dropna(subset=["bbox_x"]).head(12)

        _copied = 0
        for _, _row in caltech_df.iterrows():
            _src = _caltech_dir / _row["crop"]
            _dst = CALTECH_UPLOAD / _row["crop"]
            if _src.exists() and not _dst.exists():
                _shutil.copy(_src, _dst)
            if _dst.exists():
                _copied += 1

        print(f"Caltech upload folder : {CALTECH_UPLOAD}")
        print(f"Images copied         : {_copied}")
        print()
        print("These images already have bounding boxes — you will add polygons:")
        print(caltech_df[["crop", "true_label", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]]
              .head(8).to_string(index=False))

        import matplotlib.patches as _patches
        _n = min(6, len(caltech_df))
        _ncols = 3
        _nrows = (_n + _ncols - 1) // _ncols
        _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(_ncols * 4, _nrows * 3))
        _axes_flat = np.array(_axes).flatten()

        for _i, (_, _row) in enumerate(list(caltech_df.iterrows())[:_n]):
            _p = CALTECH_UPLOAD / _row["crop"]
            if not _p.exists():
                _axes_flat[_i].axis("off")
                continue
            _arr = np.array(Image.open(_p).convert("RGB"))
            _axes_flat[_i].imshow(_arr)
            _axes_flat[_i].add_patch(_patches.Rectangle(
                (_row["bbox_x"], _row["bbox_y"]), _row["bbox_w"], _row["bbox_h"],
                linewidth=2, edgecolor="lime", facecolor="none",
            ))
            _axes_flat[_i].set_title(_row["true_label"], fontsize=8)
            _axes_flat[_i].axis("off")
        for _ax in _axes_flat[_n:]:
            _ax.axis("off")
        plt.suptitle("Caltech Camera Traps — bounding boxes exist, polygons needed", fontsize=11)
        plt.tight_layout()

    plt.gca()
    return CALTECH_UPLOAD, caltech_df


@app.cell(hide_code=True)
def _step4(CALTECH_UPLOAD, mo):
    mo.md(f"""
    ### Step 4 — Create a polygon project in Label Studio

    1. Click **Create Project** → name it `"Caltech Polygons"`
    2. **Data Import** tab → drag this folder:
       `{CALTECH_UPLOAD}`
    3. **Labelling Setup** → **Custom template**:

    ```xml
    <View>
      <Image name="image" value="$image" zoom="true" zoomControl="true"/>
      <PolygonLabels name="label" toName="image" strokeWidth="2">
        <Label value="animal"  background="#8E44AD"/>
        <Label value="habitat" background="#27AE60"/>
      </PolygonLabels>
    </View>
    ```

    **Drawing a polygon:**
    1. Select a label (`1` for animal)
    2. Click each vertex around the animal's outline
    3. Click the **first point again** to close the polygon
    4. `Backspace` removes the last point while drawing

    **Goal:** Annotate 5 images. Trace around the animal body — compare how much
    tighter your outline is compared to the bounding box shown in the preview above.
    """)
    return


@app.cell(hide_code=True)
def _step7(mo):
    mo.md(r"""
    ### Step 5 — Export and load your polygons

    1. Go back to the Caltech Polygons project
    2. **Export** → **COCO JSON**
    3. Save to: `week1/data/my_caltech_polygons.json`

    COCO stores polygon segmentation as a flat list of alternating coordinates:
    `[x1, y1, x2, y2, x3, y3, ...]`
    """)
    return


@app.cell
def _load_bboxes(DATA_DIR, json):
    _path = DATA_DIR / "my_caltech_polygons.json"

    if not _path.exists():
        poly_coco = None
        print(f"Not found: {_path}")
        print("Annotate images in the polygon project, export as COCO JSON, save to that path.")
    else:
        with open(_path) as _f:
            poly_coco = json.load(_f)

        _n_img = len(poly_coco["images"])
        _n_ann = len(poly_coco["annotations"])

        print(f"Loaded  : {_path.name}")
        print(f"Images  : {_n_img}  |  Annotations: {_n_ann}")

        _with_seg = [a for a in poly_coco["annotations"] if a.get("segmentation")]
        if _with_seg:
            _a   = _with_seg[0]
            _pts = _a["segmentation"][0]
            print()
            print(f"First polygon — {len(_pts) // 2} vertices")
            print(f"  segmentation[0][:8] = {_pts[:8]} ...")
            print(f"  bbox (polygon bounds) = {_a['bbox']}")
    return (poly_coco,)


@app.cell
def _viz_bboxes(
    CALTECH_UPLOAD,
    Image,
    Path,
    caltech_df,
    mpatches,
    np,
    plt,
    poly_coco,
):
    """Polygon masks overlaid on images, with original bounding boxes for comparison."""
    if poly_coco is None:
        print("No annotations loaded — complete Step 5 first.")
    else:
        _id_to_img = {img["id"]: img for img in poly_coco["images"]}
        _id_to_cat = {c["id"]: c["name"] for c in poly_coco["categories"]}

        _img_anns: dict = {}
        for _a in poly_coco["annotations"]:
            if _a.get("segmentation"):
                _img_anns.setdefault(_a["image_id"], []).append(_a)

        _img_ids = list(_img_anns.keys())[:4]
        _ncols   = min(2, len(_img_ids))
        _nrows   = (len(_img_ids) + _ncols - 1) // _ncols
        _poly_colors = ["#8E44AD", "#27AE60", "#E67E22", "#2980B9"]

        # Build lookup: filename → existing bbox row (for comparison overlay)
        _bbox_lookup: dict = {}
        if caltech_df is not None:
            for _, _row in caltech_df.iterrows():
                _bbox_lookup[_row["crop"]] = _row

        _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(_ncols * 6, _nrows * 5))
        _axes_flat  = np.array(_axes).flatten()

        for _i, _img_id in enumerate(_img_ids):
            _meta  = _id_to_img[_img_id]
            _fname = Path(_meta["file_name"]).name
            _path  = CALTECH_UPLOAD / _fname
            if not _path.exists():
                _axes_flat[_i].axis("off")
                continue

            _arr = np.array(Image.open(_path).convert("RGB"))
            _ax  = _axes_flat[_i]
            _ax.imshow(_arr)

            # Draw original bounding box (ground truth) in lime
            if _fname in _bbox_lookup:
                _r = _bbox_lookup[_fname]
                _ax.add_patch(mpatches.Rectangle(
                    (_r["bbox_x"], _r["bbox_y"]), _r["bbox_w"], _r["bbox_h"],
                    linewidth=1.5, edgecolor="lime", facecolor="none",
                    linestyle="--", label="original bbox",
                ))

            # Draw your polygons on top
            for _j, _ann in enumerate(_img_anns[_img_id]):
                _col = _poly_colors[_j % len(_poly_colors)]
                _cat = _id_to_cat.get(_ann["category_id"], "?")
                for _seg in _ann["segmentation"]:
                    _pts = np.array(_seg).reshape(-1, 2)
                    _ax.add_patch(mpatches.Polygon(
                        _pts, closed=True,
                        linewidth=2, edgecolor=_col,
                        facecolor=_col, alpha=0.25,
                    ))
                    _ax.text(
                        _pts[:, 0].mean(), _pts[:, 1].mean(), _cat,
                        fontsize=8, color=_col, ha="center", va="center",
                        fontweight="bold",
                    )

            _ax.set_title(_fname, fontsize=7)
            _ax.axis("off")

        for _ax in _axes_flat[len(_img_ids):]:
            _ax.axis("off")

        plt.suptitle(
            "Your polygons (coloured fill) vs. ground-truth bounding boxes (dashed lime)",
            fontsize=11,
        )
        plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _part4(mo):
    mo.md(r"""
    ## Part 4 — Annotation Effort and Format Summary

    ### Effort vs. information trade-off

    | Type | Time per object | What the model gets | Used for |
    |------|----------------|---------------------|----------|
    | **Image label** | ~0.5 s | Species only — no location | Classification only |
    | **Bounding box** | ~3–5 s | Location + approximate size | MegaDetector, YOLO |
    | **Polygon** | ~15–60 s | Exact outline / shape | SAM, U-Net, habitat mapping |

    For the same 8-hour labelling budget: ~55 000 image labels, ~5 000 boxes, or ~500 polygons.

    ### COCO JSON structure (both exports share this layout)

    ```
    {
      "images":      [{"id": …, "file_name": "…", "width": …, "height": …}],
      "annotations": [{"id": …, "image_id": …, "category_id": …,
                       "bbox": [x, y, w, h],         ← boxes
                       "segmentation": [[x1,y1,…]]}, ← polygons
                     …],
      "categories":  [{"id": …, "name": "animal"}, …]
    }
    ```

    The `image_id` / `category_id` foreign keys join the three lists — the same
    pattern you will see in MegaDetector output (P3) and SAM masks (P7).
    """)
    return


@app.cell(hide_code=True)
def _exercise(mo):
    mo.md(r"""
    ## Exercise

    1. **Serengeti boxes** — Pick one image where you disagree with the citizen-science
       species label. What would you have labelled it? What does this say about using
       crowd-sourced labels as ground truth?

    2. **Caltech polygons** — Measure the area of your polygon vs. the bounding box area
       for the same object. What fraction of the box is actually animal?
       ```python
       import numpy as np
       # polygon area via shoelace formula
       pts = np.array(segmentation[0]).reshape(-1, 2)
       area_poly = 0.5 * abs(np.dot(pts[:,0], np.roll(pts[:,1], 1))
                            - np.dot(pts[:,1], np.roll(pts[:,0], 1)))
       area_box  = bbox_w * bbox_h
       print(f"Polygon / box area ratio: {area_poly / area_box:.2f}")
       ```

    3. **Pre-labelling** — Label Studio can import existing predictions as draft
       annotations for you to correct. If you imported MegaDetector detections as
       starting boxes, how would that change the effort for the Serengeti task?

    4. **Format round-trip** — Your COCO export has `bbox: [x, y, w, h]`.
       Convert to YOLO format (`[cx, cy, w, h]` normalised) for one annotation.
    """)
    return


@app.cell(hide_code=True)
def _reflection(mo):
    mo.md(r"""
    ## Reflection

    - The Serengeti labels came from volunteers who never drew a box.
      A detector trained only on those labels cannot learn *where* to look.
      What is the minimum annotation effort needed to train a usable detector?

    - You drew polygons on top of existing bounding boxes. If a segmentation model
      is only 5 % more accurate than a box-based model, was the extra annotation
      effort worth it?

    - Label Studio supports **pre-labelling**: import model predictions as draft
      annotations, then correct them. For which dataset would pre-labelling save
      more time — Serengeti (boxes) or Caltech (polygons)?
    """)
    return


if __name__ == "__main__":
    app.run()
