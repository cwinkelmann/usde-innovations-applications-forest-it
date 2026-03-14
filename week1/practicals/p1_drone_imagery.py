import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _context(mo):
    mo.md(r"""
    # Practical 1 — Working with Drone Imagery Tiles

    **Context:** A drone survey over the Galápagos produces a large orthomosaic —
    a stitched image that can be tens of thousands of pixels across. Neural networks
    can't process an image that large at once. Instead, we slide a window across it
    and feed each *tile* to the model separately.

    In this practical we work with the **HerdNet General Dataset** — pre-tiled aerial
    images of African mammals with point annotations. This is the same format your
    own survey data would arrive in after the tiling step.

    By the end you will:
    - Load and inspect image tiles using PIL
    - Visualise point annotations on a tile
    - Generate a sliding-window tile grid and understand size/overlap trade-offs
    - See how annotation counts vary across tiles (empty vs. dense)

    **Key question:** if an iguana is ~30 cm long and your drone flies at 30 m altitude
    with a 20 MP camera, roughly how many pixels wide is the iguana?
    """)
    return


@app.cell
def _(args):


    from week1.data.download_data import download_herdnet_weights, download_general_dataset, download_caltech, download_serengeti

    download_general_dataset(full=args.full)

    return download_caltech, download_herdnet_weights, download_serengeti


@app.cell
def _(args, download_caltech, download_serengeti):
    # Amount of images to download
    n = 500 if args.full else 500

    download_serengeti(n_images=n)
    download_caltech(n_images=n)

    print("\n" + "=" * 50)
    print("Week 1 datasets ready.")
    print(f"  general_dataset/    ← HerdNet aerial data")
    print(f"  camera_trap/serengeti_subset/  ← {n} Serengeti images")
    print(f"  camera_trap/caltech_subset/    ← {n} Caltech images")
    print(f"  camera_trap_labels.csv         ← P6 reference labels")
    print()
    print("Still needed (manual copy from C. Winkelmann):")
    print("  iguanas/tiles/       ← ~500 iguana tile JPEGs")
    print("  iguanas/iguana_counts.csv")
    print("See DATASETS.md section 2 for details.")
    return


@app.cell
def _(download_herdnet_weights):
    download_herdnet_weights()
    return


@app.cell
def _imports():
    from pathlib import Path

    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from PIL import Image

    from wildlife_detection.tiling.utils import generate_tile_windows, read_tile

    return Image, Path, generate_tile_windows, mpatches, np, pd, plt


@app.cell
def _step1(mo):
    mo.md(r"""
    ## Step 1 — Load the dataset

    The HerdNet test sample lives in `week1/data/general_dataset/test_sample/`.
    Run `week1/data/download_data.py --sample` to download it if you haven't already.

    The CSV has four columns: `images` (filename), `x`, `y` (pixel coords), `labels`.
    """)
    return


@app.cell
def _load_data(Path, pd):
    DATA_DIR = Path("../data/general_dataset/test_sample")
    CSV_PATH = Path("../data/general_dataset/test_sample.csv")

    if not DATA_DIR.exists() or not CSV_PATH.exists():
        print("Data not found — run:  python week1/data/download_data.py --sample")
        df = None
        tile_files = []
    else:
        df = pd.read_csv(CSV_PATH)
        tile_files = sorted(DATA_DIR.glob("*.jpg"))
        print(f"Tiles     : {len(tile_files)}")
        print(f"Annotations: {len(df)} points across {df['images'].nunique()} tiles")
        print(f"\nClass distribution:")
        print(df["labels"].value_counts().to_string())
    return DATA_DIR, df, tile_files


@app.cell
def _step2(mo):
    mo.md(r"""
    ## Step 2 — Inspect a single tile

    Each tile is a plain JPEG. PIL gives us width, height, and mode (RGB / L).
    """)
    return


@app.cell
def _inspect_tile(Image, np, tile_files):
    if not tile_files:
        print("No tiles loaded.")
    else:
        _sample_path = tile_files[0]
        _img = Image.open(_sample_path)
        _arr = np.array(_img)

        print(f"File       : {_sample_path.name}")
        print(f"Size       : {_img.width} × {_img.height} px")
        print(f"Mode       : {_img.mode}")
        print(f"dtype      : {_arr.dtype}  shape: {_arr.shape}")

        for _i, _ch in enumerate(["R", "G", "B"]):
            print(f"  {_ch}: min={_arr[:,:,_i].min()}  max={_arr[:,:,_i].max()}  mean={_arr[:,:,_i].mean():.1f}")
    return


@app.cell
def _step3(mo):
    mo.md(r"""
    ## Step 3 — Visualise annotations on a tile

    Each row in the CSV corresponds to one animal. `x` and `y` are pixel coordinates
    relative to the tile — not geographic coordinates.
    """)
    return


@app.cell
def _annotated_tile(DATA_DIR, Image, df, np, plt, tile_files):
    if not tile_files or df is None:
        print("No data loaded.")
    else:
        _counts = df.groupby("images").size()
        _rich_tile = _counts.idxmax()
        _img = np.array(Image.open(DATA_DIR / _rich_tile))
        _tile_df = df[df["images"] == _rich_tile]

        _fig, _ax = plt.subplots(figsize=(7, 7))
        _ax.imshow(_img)
        _ax.scatter(_tile_df["x"], _tile_df["y"], c="red", s=20, marker="+",
                    linewidths=1.5, label=f"{len(_tile_df)} annotations")
        _ax.set_title(f"{_rich_tile}  ({len(_tile_df)} animals)")
        _ax.legend(fontsize=9)
        _ax.axis("off")
        plt.tight_layout()
    plt.gca()
    return


@app.cell
def _step4(mo):
    mo.md(r"""
    ## Step 4 — Sliding-window tiling

    To process a *larger* image we slide a fixed-size window across it with overlap.
    Here we use one of the sample tiles as a stand-in for a full orthomosaic —
    the logic is identical regardless of image size.
    """)
    return


@app.cell
def _tiling(Image, generate_tile_windows, np, tile_files):
    if not tile_files:
        print("No tiles loaded.")
        TILE_SIZE = OVERLAP = width = height = n_tiles = step = 0
        big_img = np.zeros((1, 1, 3), dtype=np.uint8)
        windows = []
    else:
        big_img = np.array(Image.open(tile_files[0]))
        height, width = big_img.shape[:2]

        TILE_SIZE = 256
        OVERLAP = 64
        step = TILE_SIZE - OVERLAP
        windows = list(generate_tile_windows(width, height, TILE_SIZE, OVERLAP))
        n_tiles = len(windows)

        print(f"Image size : {width} × {height} px")
        print(f"Tile size  : {TILE_SIZE} px")
        print(f"Overlap    : {OVERLAP} px  (step = {step} px)")
        print(f"Tiles      : {n_tiles}")
    return OVERLAP, TILE_SIZE, big_img, windows


@app.cell
def _step5(mo):
    mo.md(r"""
    ## Step 5 — Visualise the tile grid

    Each green rectangle is one tile window. Overlapping tiles share a strip of pixels
    along each edge — this ensures animals near a boundary appear fully in at least one tile.
    """)
    return


@app.cell
def _visualise_grid(OVERLAP, TILE_SIZE, big_img, mpatches, plt, windows):
    if not windows:
        print("No windows to visualise.")
    else:
        _fig, _ax = plt.subplots(figsize=(7, 7))
        _ax.imshow(big_img)

        for _win in windows:
            _rect = mpatches.Rectangle(
                (_win.col_off, _win.row_off),
                _win.width,
                _win.height,
                linewidth=0.8,
                edgecolor="lime",
                facecolor="none",
            )
            _ax.add_patch(_rect)

        _ax.set_title(
            f"Tile grid — {TILE_SIZE}px tiles, {OVERLAP}px overlap  ({len(windows)} tiles)",
            fontsize=11,
        )
        _ax.axis("off")
        plt.tight_layout()
    plt.gca()
    return


@app.cell
def _exercise(mo):
    mo.md(r"""
    ## Exercise

    > **In the `_tiling` cell above, change `TILE_SIZE` to 128 and `OVERLAP` to 0.
    > How does the tile count change? Re-run and look at the grid.**

    Then try `OVERLAP = 64`. With a 512×512 tile and 64px overlap, tiles at column
    offsets 0 and 448 share a 64-pixel strip. Why does that matter for an animal
    sitting right on the boundary?

    | Setting | Tile count | Edge coverage |
    |---------|-----------|---------------|
    | 256 px, 0 overlap | ? | Poor — edge animals split |
    | 256 px, 64 overlap | ? | Good — edge animals duplicated |
    | 128 px, 32 overlap | ? | Best coverage, most tiles |

    **Bonus:** look at the annotation CSV. Which tile has the most animals?
    Which has zero? What does an empty tile mean for model training?
    """)
    return


@app.cell
def _reflection(mo):
    mo.md(r"""
    ## Reflection

    - The tile filename convention in this project is `{source_stem}_{col_offset}_{row_offset}.jpg`.
      Given a tile named `survey_512_1024.jpg`, what does that tell you about where it came from?
    - If your tile is 512 × 512 px and the original image is 10,000 × 8,000 px with no overlap,
      how many tiles do you get? What fraction of the image area is covered by the edge tiles?
    - The CSV uses pixel coordinates, not geographic coordinates. What would you need to
      convert back to GPS positions — and why might that matter for a field team?
    """)
    return


if __name__ == "__main__":
    app.run()
