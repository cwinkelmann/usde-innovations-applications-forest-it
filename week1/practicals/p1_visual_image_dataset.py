import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _species_gallery(mo):
    mo.md(r"""
    # Practical 1 — Visual Wildlife Datasets

    Before building any model, you need to understand the data. Wildlife
    imagery comes from two main platforms — **camera traps** (ground-level)
    and **drones** (aerial) — and each platform uses different annotation
    formats to label what is in the images.

    This practical walks through **four annotation types**, from simplest to
    most detailed, showing real examples from public datasets at each level:

    1. **Image classification** — one species label per photo
    2. **Bounding boxes** — a rectangle around each animal
    3. **Point annotations** — a single (x, y) coordinate per animal
    4. **Segmentation masks** — pixel-level outlines (preview for Practical 7)

    Each type trades off **annotation effort** against **spatial detail**.
    By the end you will know which format each tool uses (MegaDetector → boxes,
    HerdNet → points, SAM → masks) and when each is the right choice.
    """)
    return


@app.cell
def _format_detail():
    """Resolve week1/data/ so download_data is importable."""
    import sys as _sys
    from pathlib import Path as _Path

    DATA_BASE = _Path(__file__).parent.parent / "data"
    if str(DATA_BASE) not in _sys.path:
        _sys.path.insert(0, str(DATA_BASE))
    return (DATA_BASE,)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 1. Image Classification

    The simplest annotation type: **one label per image**. The entire photo
    is tagged with a species name — no information about where the animal is
    in the frame.

    This is the format used by camera trap management platforms (Wildlife
    Insights, Agouti, Trapper) and by species classifiers downstream of
    MegaDetector.

    **Snapshot Serengeti** — 7.1 million camera trap images from Tanzania,
    40+ mammalian species, stored as COCO JSON with three top-level lists
    (`images`, `annotations`, `categories`) linked by ID fields.
    """)
    return


@app.cell
def _(DATA_BASE):
    """Download Snapshot Serengeti (50 images + bounding box annotations)."""
    from download_data import download_serengeti
    download_serengeti(n_images=50, output_dir=DATA_BASE)
    return


@app.cell(hide_code=True)
def _(DATA_BASE):
    """Species distribution — which species appear and how often?"""
    from download_data import show_class_distribution
    show_class_distribution("serengeti", output_dir=DATA_BASE)
    return


@app.cell
def _(DATA_BASE):
    """Sample camera trap images — labelled by species."""
    from download_data import show_samples
    show_samples("serengeti", n=6, output_dir=DATA_BASE)
    return


@app.cell
def _tile_inspect(mo):
    mo.md(r"""
    ---

    ## 2. Bounding Boxes

    A bounding box adds **spatial information**: a rectangle around each animal
    tells you not just *what* is there, but *where* it is. This is the format
    used by most object detectors: **MegaDetector**, **YOLOv8**, **Faster R-CNN**.

    Three common bounding box formats exist — you will convert between them
    in Practical 3:

    | Format | Coordinates | Used by |
    |--------|-------------|---------|
    | **COCO** | `[x, y, w, h]` — top-left + size, pixels | COCO JSON, MegaDetector |
    | **YOLO** | `[cx, cy, w, h]` — centre + size, normalised 0–1 | Ultralytics, training |
    | **Pascal VOC** | `[x_min, y_min, x_max, y_max]` — corners, pixels | VOC XML |

    We show bounding boxes from two perspectives:
    - **Camera trap** (Serengeti) — ground-level, COCO format `[x, y, w, h]`
    - **Aerial** (Eikelboom 2019) — nadir view, YOLO format `[cx, cy, w, h]`
    """)
    return


@app.cell
def _(DATA_BASE):
    """Serengeti camera trap images with COCO bounding boxes drawn."""
    from download_data import show_bboxes
    show_bboxes("serengeti", n=6, output_dir=DATA_BASE)
    return


@app.cell
def _tile_dist(DATA_BASE):
    """Download Eikelboom 2019 aerial dataset (50 images per split)."""
    from download_data import download_eikelboom
    download_eikelboom(n_images=50, output_dir=DATA_BASE)
    return


@app.cell
def _imports(DATA_BASE):
    """Eikelboom aerial tiles with YOLO bounding boxes drawn."""
    from download_data import show_bboxes
    show_bboxes("eikelboom", n=6, output_dir=DATA_BASE)
    return


@app.cell
def _gallery(DATA_BASE):
    """Eikelboom class distribution from YOLO label files."""
    from download_data import show_class_distribution
    show_class_distribution("eikelboom", output_dir=DATA_BASE)
    return


@app.cell
def _load(mo):
    mo.md(r"""
    ---

    ## 3. Point Annotations

    A point annotation is the **lightest spatial label**: a single `(x, y)`
    coordinate marking the centre of each animal. No box dimensions needed.

    This dramatically reduces annotation effort — especially for **dense
    colonies** where drawing tight boxes around overlapping animals is
    impractical. It is the format used by **HerdNet** (Delplanque et al., 2023)
    and the **Iguanas From Above** project throughout this course.

    The HerdNet General Dataset stores annotations as a CSV:

    | Column | Meaning |
    |--------|---------|
    | `images` | Tile filename |
    | `x`, `y` | Pixel coordinate within the tile |
    | `labels` | Species class |

    **One row = one animal.** Tiles with no animals have no rows in the CSV.
    """)
    return


@app.cell
def _(DATA_BASE):
    """Download HerdNet General Dataset (50 images)."""
    from download_data import download_general_dataset
    download_general_dataset(n_images=50, output_dir=DATA_BASE)
    return


@app.cell
def _step6(DATA_BASE):
    """HerdNet species distribution."""
    from download_data import show_class_distribution
    show_class_distribution("general_dataset", output_dir=DATA_BASE)
    return


@app.cell
def _classes(DATA_BASE):
    """Densest tiles with point annotations overlaid (red + = one animal)."""
    from download_data import show_annotated_tiles
    show_annotated_tiles(n=3, output_dir=DATA_BASE)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 4. Segmentation Masks (Preview)

    The fourth and most detailed annotation type: **every pixel** is assigned
    a class label. This is the most expensive to annotate but gives precise
    outlines. You will work with this in **Practical 7** using Segment
    Anything (SAM).

    **When is segmentation useful in ecology?**
    - Habitat mapping (land cover from satellite/drone imagery)
    - Canopy delineation (individual tree crowns)
    - Precise area estimation (coral coverage, burn scars)
    - Instance segmentation when animals overlap heavily

    ---

    ## Summary — Annotation Types at a Glance

    | Type | Spatial detail | Annotation effort | Example tools | Best for |
    |------|---------------|-------------------|---------------|----------|
    | **Image label** | None | Lowest | DeepFaune, SpeciesNet | Species ID after detection |
    | **Bounding box** | Rectangle | Medium | MegaDetector, YOLOv8 | Counting, localisation |
    | **Point** | Centre (x, y) | Low | HerdNet | Dense colonies, aerial |
    | **Segmentation mask** | Pixel outline | Highest | SAM, U-Net | Habitat mapping, area |

    More spatial detail is **not always better** — the right choice depends
    on your ecological question and how much annotation time you can afford.
    """)
    return


@app.cell
def _step2(mo):
    mo.md(r"""
    ## Exercise

    1. Compare a camera trap image with an aerial tile side by side.
       List three visual differences that would affect a detection model.
    2. The Serengeti dataset uses bounding boxes; HerdNet uses points.
       Which annotation type would you choose for counting iguanas on a
       rocky beach? Why?
    3. Look at the Eikelboom aerial images. How do the animals compare in
       size and visibility to the HerdNet tiles?
    4. Point annotations are stored as pixel coordinates. What extra
       information would you need to convert them to GPS positions?
    """)
    return


@app.cell
def _step3(mo):
    mo.md(r"""
    ## Reflection

    - Camera traps and drones produce very different data. How should this
      affect your choice of detection model?
    - Empty tiles have no rows in the CSV. What does this mean for training
      a model that must also learn to say "nothing here"?
    - The four annotation types form a hierarchy of increasing spatial detail.
      Is more detail always better, or is there a cost?
    """)
    return


if __name__ == "__main__":
    app.run()
