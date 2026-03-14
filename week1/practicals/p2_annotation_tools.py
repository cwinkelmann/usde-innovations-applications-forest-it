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
    # Practical 2 — Annotation Formats & Tools

    **Context:** Before a model can learn to find animals, humans must mark where they are.
    Those marks come in three flavours — *points*, *bounding boxes*, and *polygons* — and
    each trades off annotation effort against the information given to the model.

    In this practical you will:
    - Load the HerdNet CSV annotation format (point annotations)
    - Synthesise bounding box and polygon annotations from point data
    - Visualise all three annotation types on the same tile
    - Understand why point annotations are preferred for aerial wildlife counting

    **Reference tools:**
    - [CVAT](https://cvat.ai) — open-source, multi-task annotation
    - [Label Studio](https://labelstud.io) — flexible, supports audio/text/image
    - Both can export to CSV, COCO JSON, or YOLO format
    """)


@app.cell
def _imports():
    from pathlib import Path

    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from PIL import Image

    return Image, Path, mpatches, np, pd, plt


@app.cell
def _step1(mo):
    mo.md(r"""
    ## Step 1 — Load point annotations (CSV format)

    The HerdNet dataset stores annotations in a plain CSV with four columns:
    `images` (filename), `x`, `y` (pixel coords), `labels` (species class).

    This is the simplest possible annotation format — one row per animal.
    Most tools (CVAT, Label Studio, Roboflow) can export in this format.
    """)


@app.cell
def _load_data(Path, pd):
    DATA_DIR = Path("../data/general_dataset/test_sample")
    CSV_PATH = Path("../data/general_dataset/test_sample.csv")

    if not CSV_PATH.exists():
        print("Data not found — run:  python week1/data/download_data.py --sample")
        df = None
        tile_files = []
    else:
        df = pd.read_csv(CSV_PATH)
        tile_files = sorted(DATA_DIR.glob("*.jpg"))
        print(f"Loaded {len(df)} point annotations across {df['images'].nunique()} tiles")
        print(f"\nColumn names: {list(df.columns)}")
        print(f"\nFirst 5 rows:")
        print(df.head().to_string(index=False))

    return DATA_DIR, CSV_PATH, df, tile_files


@app.cell
def _step2(mo):
    mo.md(r"""
    ## Step 2 — Synthesise bounding boxes from points

    We don't have real bounding box annotations in this dataset — only points.
    For the visualisation below we synthesise boxes by placing a square of fixed
    radius around each point. This illustrates the format without needing a
    separate annotation round.

    In a real project, box annotations come from a tool like CVAT and are stored
    as a COCO JSON or YOLO `.txt` file.
    """)


@app.cell
def _synthesise_boxes(DATA_DIR, df, pd, tile_files):

    if df is None or not tile_files:
        boxes_df = None
    else:
        from PIL import Image as _Image

        # Use the tile with the most annotations
        _counts = df.groupby("images").size()
        rich_tile = _counts.idxmax()

        img = _Image.open(DATA_DIR / rich_tile)
        W, H = img.size

        _tile_df = df[df["images"] == rich_tile].copy()

        BOX_RADIUS = 20  # pixels — approximate animal half-size at this resolution

        boxes_df = _tile_df.assign(
            x1=(_tile_df["x"] - BOX_RADIUS).clip(0, W),
            y1=(_tile_df["y"] - BOX_RADIUS).clip(0, H),
            x2=(_tile_df["x"] + BOX_RADIUS).clip(0, W),
            y2=(_tile_df["y"] + BOX_RADIUS).clip(0, H),
        )

        print(f"Tile: {rich_tile}  ({len(_tile_df)} animals)")
        print(f"Image size: {W} × {H} px")
        print(f"Synthetic box radius: {BOX_RADIUS} px")
        print(f"\nBox DataFrame (first 5):")
        print(boxes_df[["images", "x", "y", "x1", "y1", "x2", "y2"]].head().to_string(index=False))

    return boxes_df, rich_tile


@app.cell
def _step3(mo):
    mo.md(r"""
    ## Step 3 — Visualise annotation types side by side

    The same tile annotated three ways:
    - **Points** — (x, y) per animal, ~1 second per annotation
    - **Bounding boxes** — (x1, y1, x2, y2) per animal, ~3–5 seconds per annotation
    - **Simplified polygons** — diamond shape around point, illustrates the format

    Notice how much more work each step adds — and ask: does the model actually need it?
    """)


@app.cell
def _visualise(DATA_DIR, Image, boxes_df, df, mpatches, np, plt, rich_tile):
    if boxes_df is None:
        print("No data loaded.")
    else:
        img_arr = np.array(Image.open(DATA_DIR / rich_tile))
        _tile_df = df[df["images"] == rich_tile]

        _fig, _axes = plt.subplots(1, 3, figsize=(15, 5))
        titles = [
            f"Points  ({len(_tile_df)} annotations)",
            f"Bounding boxes  ({len(boxes_df)})",
            f"Polygons  ({len(boxes_df)})",
        ]

        for _ax, title in zip(_axes, titles):
            _ax.imshow(img_arr)
            _ax.set_title(title, fontsize=11)
            _ax.axis("off")

        # Points — scatter
        _axes[0].scatter(
            tile_df["x"], tile_df["y"],
            c="red", s=30, marker="+", linewidths=1.5,
        )

        # Bounding boxes — rectangles
        for _, row in boxes_df.iterrows():
            rect = mpatches.Rectangle(
                (row["x1"], row["y1"]),
                row["x2"] - row["x1"],
                row["y2"] - row["y1"],
                linewidth=1.5, edgecolor="yellow", facecolor="none",
            )
            _axes[1].add_patch(rect)

        # Polygons — diamond around centre point
        for _, row in boxes_df.iterrows():
            cx, cy = row["x"], row["y"]
            r = (row["x2"] - row["x1"]) / 2
            diamond = mpatches.Polygon(
                [[cx, cy - r], [cx + r, cy], [cx, cy + r], [cx - r, cy]],
                closed=True, linewidth=1.2, edgecolor="cyan", facecolor="cyan", alpha=0.25,
            )
            _axes[2].add_patch(diamond)

        plt.suptitle(f"Annotation format comparison — {rich_tile}", fontsize=12)
        plt.tight_layout()
        plt.show()

    return


@app.cell
def _step4(mo):
    mo.md(r"""
    ## Step 4 — Annotation effort vs. model information

    How long does each annotation type take? A rough estimate from iguana dataset work:

    | Type | Time per animal | Information given to model |
    |------|----------------|---------------------------|
    | Point | ~1 s | Location only |
    | Bounding box | ~3–5 s | Location + approximate size |
    | Polygon | ~15–30 s | Exact shape |

    HerdNet uses **point annotations** — the cheapest type. It learns to predict a
    density map from those sparse labels. You can annotate 3–5× more images for the
    same labelling budget.
    """)


@app.cell
def _exercise(mo):
    mo.md(r"""
    ## Exercise

    > **Open CVAT or Label Studio (web-based — no install needed) and annotate
    > 20–30 animals in the provided sample image using point annotations.**

    Things to pay attention to:
    - What counts as one animal vs. a group?
    - How do you handle partially visible animals at the image edge?
    - What would you do with a blurry or ambiguous blob?

    Discuss with your neighbour: did you mark the same animals?
    """)


@app.cell
def _annotation_stats(df):
    if df is not None:
        # Distribution of annotations per tile
        _counts2 = df.groupby("images").size().sort_values(ascending=False)
        print("Annotation count distribution across tiles:")
        print(f"  Max per tile   : {_counts2.max()}")
        print(f"  Median per tile: {_counts2.median():.0f}")
        print(f"  Min per tile   : {_counts2.min()}")
        print(f"  Empty tiles    : {(_counts2 == 0).sum()} (tiles in CSV with 0 entries are just absent)")
        print(f"\nTop 5 busiest tiles:")
        print(_counts2.head().to_string())

    return


@app.cell
def _reflection(mo):
    mo.md(r"""
    ## Reflection

    - Why does annotation disagreement matter for model training?
    - If two annotators systematically disagree, how would that affect model evaluation?
    - The karisu/General_Dataset CSV uses the `labels` column for species class.
      If you had multiple annotators, what fields would you add to track inter-annotator agreement?
    - Point annotations give location but not size. What implicit assumption does HerdNet
      make about animal size in the FIDT map construction?
    """)


if __name__ == "__main__":
    app.run()
