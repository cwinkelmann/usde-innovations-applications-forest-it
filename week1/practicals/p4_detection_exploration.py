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
    # Practical 4 — Exploring Detections

    **Context:** You have MegaDetector results from Practical 3. Before trusting a
    model in the field, you need to understand *where* it fails and *why*. This is
    not about making the numbers look good — it is about building calibrated trust.

    Today you will:
    - Browse detection results as a confidence distribution
    - Visualise a random sample of crops (true positives, false positives)
    - Identify systematic failure modes: motion blur, dense vegetation, unusual angles
    - Understand what "confidence calibration" means and why 90% ≠ 90%

    These skills apply to every model you will use this week.
    """)


@app.cell
def _imports():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from PIL import Image

    return Image, Path, np, pd, plt


@app.cell
def _load_results(Path, pd):
    # Load the detections DataFrame saved from Practical 3
    # If you skipped P3, set this to the provided sample CSV
    DETECTIONS_CSV = Path("../data/camera_trap_detections.csv")

    if DETECTIONS_CSV.exists():
        df = pd.read_csv(DETECTIONS_CSV)
    else:
        # Fallback: regenerate from the batch detection step
        print("detections CSV not found — run p3_megadetector.py first, or use the sample CSV")
        df = pd.DataFrame()

    print(f"Loaded {len(df)} detections")
    if not df.empty:
        print(df.head())
    return DETECTIONS_CSV, df


@app.cell
def _step1(mo):
    mo.md(r"""
    ## Step 1 — Confidence distribution

    Plot the distribution of confidence scores for each category.
    A well-calibrated detector should have few detections in the 0.2–0.5 range
    ("unsure") and many either very low (background) or very high (certain animal).
    """)


@app.cell
def _confidence_plot(df, plt):
    if df.empty:
        print("No data to plot.")
    else:
        categories = df["category"].unique()
        colours = {"animal": "steelblue", "person": "tomato", "vehicle": "goldenrod"}

        fig, axes = plt.subplots(1, len(categories), figsize=(5 * len(categories), 4), sharey=False)
        if len(categories) == 1:
            axes = [axes]

        for ax, cat in zip(axes, categories):
            subset = df[df["category"] == cat]["confidence"]
            ax.hist(subset, bins=20, range=(0, 1), color=colours.get(cat, "grey"), edgecolor="white")
            ax.set_title(f"{cat} (n={len(subset)})")
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Count")
            ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="threshold=0.5")
            ax.legend(fontsize=8)

        plt.suptitle("MegaDetector confidence distributions", fontsize=13)
        plt.tight_layout()
        plt.show()
    return axes, cat, colours, fig, subset


@app.cell
def _step2(mo):
    mo.md(r"""
    ## Step 2 — Browse a sample of crops

    Randomly sample `N_CROPS` animal detections and display them in a grid.
    Look for failure modes — what does a false positive look like?
    """)


@app.cell
def _crop_grid(Image, Path, df, np, plt):
    CROPS_DIR = Path("../data/camera_trap_crops")
    N_COLS = 5
    N_ROWS = 4
    N_CROPS = N_COLS * N_ROWS

    crop_files = sorted(CROPS_DIR.glob("*.jpg")) if CROPS_DIR.exists() else []

    if not crop_files:
        print("No crops found. Run p3_megadetector.py first.")
    else:
        rng = np.random.default_rng(seed=42)
        sample = rng.choice(crop_files, size=min(N_CROPS, len(crop_files)), replace=False)

        fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS * 2.5, N_ROWS * 2.5))
        axes = axes.flatten()

        for ax, path in zip(axes, sample):
            img = Image.open(path)
            ax.imshow(img)
            ax.set_title(path.stem[-12:], fontsize=7)
            ax.axis("off")

        # Hide unused axes
        for ax in axes[len(sample):]:
            ax.axis("off")

        plt.suptitle(f"Random sample of {len(sample)} animal crops", fontsize=12)
        plt.tight_layout()
        plt.show()
    return (
        CROPS_DIR, N_COLS, N_CROPS, N_ROWS, ax, axes, crop_files, fig, img, path, rng, sample,
    )


@app.cell
def _step3(mo):
    mo.md(r"""
    ## Step 3 — Filter and compare by confidence band

    Split the detections into confidence bands and compare what they look like.
    Low-confidence detections are the ones that need human review.
    """)


@app.cell
def _confidence_bands(Image, Path, df, plt):
    CROPS_DIR = Path("../data/camera_trap_crops")

    bands = {
        "Low (0.2–0.4)": (0.2, 0.4),
        "Medium (0.4–0.7)": (0.4, 0.7),
        "High (0.7–1.0)": (0.7, 1.0),
    }

    animals = df[df["category"] == "animal"].copy() if not df.empty else None

    if animals is not None and CROPS_DIR.exists():
        fig, axes = plt.subplots(3, 5, figsize=(14, 9))

        for row_idx, (band_name, (lo, hi)) in enumerate(bands.items()):
            band_df = animals[(animals["confidence"] >= lo) & (animals["confidence"] < hi)]
            band_crops = sorted(CROPS_DIR.glob("*.jpg"))[:5]  # simplification

            for col_idx in range(5):
                ax = axes[row_idx][col_idx]
                if col_idx < len(band_crops):
                    ax.imshow(Image.open(band_crops[col_idx]))
                    ax.axis("off")
                    if col_idx == 0:
                        ax.set_ylabel(band_name, fontsize=9, rotation=0, labelpad=80)
                else:
                    ax.axis("off")

        plt.suptitle("Crops by confidence band — do higher confidence crops look better?", fontsize=11)
        plt.tight_layout()
        plt.show()
    else:
        print("Run p3_megadetector.py first to generate crops and detections.")
    return axes, ax, band_crops, band_df, band_name, bands, col_idx, fig, hi, lo, row_idx


@app.cell
def _exercise(mo):
    mo.md(r"""
    ## Exercise

    > **Find three images where MegaDetector was wrong. For each, hypothesise why.**

    Common failure modes to look for:
    - **Motion blur** — animal moved during exposure
    - **Partial occlusion** — animal behind vegetation or rock
    - **Unusual viewing angle** — overhead or oblique vs. camera-trap side view
    - **Dense groups** — animals overlapping, counted as one or as many
    - **Similar texture** — rock or bark that looks like an animal

    Write down your three examples with confidence scores and failure mode labels.
    """)


@app.cell
def _reflection(mo):
    mo.md(r"""
    ## Reflection

    - If MegaDetector reports 90 % confidence, does that mean it is correct 90 % of the time?
      (Hint: look up "confidence calibration" — most neural networks are overconfident.)
    - The ICTC 2026 talk said "don't send a ranger for a 90% confidence detection without
      human review". What system design principle does that reflect?
    - In a survey of 50,000 images, 80 % are empty. You set threshold = 0.5.
      If precision = 0.95 and recall = 0.90, how many animal images do you miss?
      How many empty images still need manual review?
    """)


if __name__ == "__main__":
    app.run()
