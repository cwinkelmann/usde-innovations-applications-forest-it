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
    # Practical 8 — Week 1 Wrap-up

    **Context:** You have now run through the complete Week 1 pipeline:

    ```
    Drone survey → Tiles (P1: explore dataset)
    Annotation tools (P2)
    MegaDetector on camera traps (P3) → Detection exploration (P4)
    Species classification (P5) → Evaluation (P6)
    Segmentation intro (P7)
    ```

    This session is for consolidation, not new content. You will:
    - Export a summary of your results from the week
    - Identify one thing that surprised you and one open question
    - Preview the Week 2 jump from RGB imagery to radar backscatter

    There is no new code to write. Use the cells below as a structured reflection.
    """)


@app.cell
def _pipeline_summary(mo):
    mo.md(r"""
    ## The pipeline you ran this week

    ```
                    ┌─────────────────────────────────────────┐
                    │          AI Wildlife Monitoring          │
                    └─────────────────────────────────────────┘

    Drone survey
        │
        ▼
    JPEG tiles (PIL/numpy) ────────────────► Dataset inspection (P1)
                                                    │
                                  Point annotations ┘ (CVAT / Label Studio)
                                                    │
                                                    ▼
                                          Bounding box detector
                                          (MegaDetector / YOLO)
                                                    │
                                          Animal crops │
                                                    ▼
                                          Species classifier
                                          (EfficientNet / timm)
                                                    │
                                          Results table │
                                          (P6 evaluation)
                                                    │
                                                    ▼
                                         Habitat map (SAM / U-Net)
    ```
    """)


@app.cell
def _export_results(mo):
    mo.md(r"""
    ## Step 1 — Export your results

    Run the cell below to collect outputs from the week into a single summary folder.
    """)


@app.cell
def _export(mo):
    from pathlib import Path
    import shutil
    import datetime

    EXPORT_DIR = Path("../data/week1_export")
    EXPORT_DIR.mkdir(exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    summary_path = EXPORT_DIR / f"week1_summary_{timestamp}.txt"

    files_to_check = {
        "Detections (P3)": Path("../data/camera_trap_detections.csv"),
        "Classifier predictions (P5)": Path("../data/classifier_predictions.csv"),
        "Evaluation report (P6)": Path("../data/evaluation_report.txt"),
    }

    lines = [f"Week 1 Export — {timestamp}\n", "=" * 40 + "\n"]
    for label, path in files_to_check.items():
        status = "✓ found" if path.exists() else "✗ not found"
        lines.append(f"{label:35s} {status}\n")
        if path.exists() and path.suffix == ".csv":
            import pandas as pd
            df = pd.read_csv(path)
            lines.append(f"  → {len(df)} rows, columns: {list(df.columns)}\n")

    summary_path.write_text("".join(lines))
    print("".join(lines))
    print(f"Summary saved to {summary_path}")
    return EXPORT_DIR, datetime, files_to_check, label, lines, path, shutil, status, summary_path, timestamp


@app.cell
def _reflection_prompts(mo):
    mo.md(r"""
    ## Step 2 — Structured reflection

    Take 5 minutes and write answers to these questions in the cell below.
    You do not need to write code — use it as a text cell.
    """)


@app.cell
def _your_reflection(mo):
    mo.md(r"""
    ### Your reflection (edit this cell)

    **One thing that surprised me this week:**
    > *Write here...*

    **One thing I expected to work better than it did:**
    > *Write here...*

    **One open question I am carrying into the break:**
    > *Write here...*

    **Which part of the pipeline would I want to improve most, and why:**
    > *Write here...*
    """)


@app.cell
def _week2_preview(mo):
    mo.md(r"""
    ## Preview — Week 2: From RGB to Radar

    In Week 2 you move from drone RGB imagery to **Sentinel-1 SAR (Synthetic Aperture Radar)**.

    | | Week 1 | Week 2 |
    |-|--------|--------|
    | Sensor | Drone RGB camera | Sentinel-1 SAR satellite |
    | Output | JPEG / GeoTIFF (3 bands: R, G, B) | GeoTIFF (2 bands: VV, VH backscatter) |
    | Spatial res | 2–20 cm/px (drone) | ~10 m/px |
    | Objects | Animals (0.3–2 m) | Mining ponds, bare ground (10–100 m) |
    | ML task | Detection + classification | Change detection + segmentation |
    | Case study | Marine iguanas, Galápagos | Galamsey, Ghana Biosphere Reserves |

    The segmentation concepts from P7 transfer directly — you will classify pixels,
    but the features are radar backscatter intensities, not RGB colours.

    **Lead:** N. Voss & A. Bosu (radar remote sensing)
    """)


@app.cell
def _open_questions(mo):
    mo.md(r"""
    ## Open questions from the course

    Some threads left deliberately open — worth following up if you are curious:

    - **HerdNet** — the model behind the Galápagos iguana count. It uses point
      annotations and FIDT density maps, not bounding boxes. See `notebooks/01_herdnet_pipeline.py`.
    - **Active learning** — how to choose which images to annotate next to improve
      the model fastest. See the `active-learning-wildlife` skill.
    - **Photogrammetric deduplication** — when two overlapping drone frames both detect
      the same iguana, how do you count it only once? Uses DEM + camera pose.
    - **Individual re-identification** — can you recognise the same iguana across visits?
      See PyTorch Wildlife's ReID modules and the BearID project.

    See `README.md` → Further Reading for papers and tools on all of the above.
    """)


if __name__ == "__main__":
    app.run()
