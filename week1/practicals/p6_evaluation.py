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
    # Practical 6 — Evaluating Classifier Results

    **Context:** A model that reports "92 % accuracy" sounds good — but on a dataset
    where 92 % of images are one class, a model that always predicts that class
    also gets 92 %. Accuracy alone is not enough.

    Today you will:
    - Compare predictions against a small labelled reference set
    - Compute precision, recall, and F1 for each class
    - Build and interpret a confusion matrix
    - Understand when precision matters more than recall (and vice versa)

    **Uses crops from P3 and predictions from P5.**
    """)


@app.cell
def _imports():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        classification_report,
        confusion_matrix,
    )

    return ConfusionMatrixDisplay, Path, classification_report, confusion_matrix, np, pd, plt


@app.cell
def _load_data(Path, np, pd):
    GT_CSV = Path("../data/camera_trap_labels.csv")
    PRED_CSV = Path("../data/classifier_predictions.csv")

    if GT_CSV.exists() and PRED_CSV.exists():
        eval_df = (
            pd.read_csv(GT_CSV)
            .merge(pd.read_csv(PRED_CSV), on="crop", how="inner")
        )
        print(f"Evaluation set: {len(eval_df)} labelled crops")
    else:
        print("Reference CSVs not found — using synthetic demo data.")
        rng = np.random.default_rng(42)
        classes = ["lion", "elephant", "zebra", "empty"]
        n = 120
        true_labels = rng.choice(classes, n, p=[0.3, 0.2, 0.3, 0.2])
        pred_labels = true_labels.copy()
        # Introduce 20 random errors to simulate a realistic classifier
        noise_idx = rng.choice(n, size=20, replace=False)
        pred_labels[noise_idx] = rng.choice(classes, size=20)
        eval_df = pd.DataFrame({
            "crop": [f"crop_{i:04d}.jpg" for i in range(n)],
            "true_label": true_labels,
            "top1_label": pred_labels,
        })

    print(eval_df["true_label"].value_counts().rename("n_true").to_string())
    return GT_CSV, PRED_CSV, eval_df


@app.cell
def _step1(mo):
    mo.md(r"""
    ## Step 1 — Classification report

    `sklearn.metrics.classification_report` gives per-class precision, recall, and F1.

    | Metric | Formula | When it matters |
    |--------|---------|----------------|
    | Precision | TP / (TP + FP) | Cost of false alarms is high |
    | Recall | TP / (TP + FN) | Cost of missing cases is high |
    | F1 | 2 × P × R / (P + R) | Balance of both |
    """)


@app.cell
def _report(classification_report, eval_df):
    y_true = eval_df["true_label"].tolist()
    y_pred = eval_df["top1_label"].tolist()

    report = classification_report(y_true, y_pred, digits=3)
    print(report)
    return report, y_pred, y_true


@app.cell
def _step2(mo):
    mo.md(r"""
    ## Step 2 — Confusion matrix

    Rows = true class, columns = predicted class. Off-diagonal entries are errors.
    Large off-diagonal values tell you which pairs of classes the model confuses most.
    """)


@app.cell
def _confusion_plot(ConfusionMatrixDisplay, confusion_matrix, plt, y_pred, y_true):
    labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion matrix — EfficientNet on camera trap crops")
    plt.tight_layout()
    plt.show()
    return ax, cm, disp, fig, labels


@app.cell
def _step3(mo):
    mo.md(r"""
    ## Step 3 — Accuracy vs. class imbalance

    If the dataset is imbalanced, a model that always predicts the majority class
    will look good on accuracy but terrible on per-class recall.
    """)


@app.cell
def _imbalance_demo(eval_df, np, pd):
    class_counts = eval_df["true_label"].value_counts()
    majority_class = class_counts.idxmax()
    majority_accuracy = class_counts.max() / len(eval_df)

    print(f"Class distribution:")
    print(class_counts.to_string())
    print(f"\nA model that always predicts '{majority_class}' achieves:")
    print(f"  Accuracy = {majority_accuracy:.1%}")
    print(f"  Recall for other classes = 0 %")
    print(f"\nThis is why F1 and per-class recall matter.")
    return class_counts, majority_accuracy, majority_class


@app.cell
def _exercise(mo):
    mo.md(r"""
    ## Exercise

    > **Look at the confusion matrix. Which two classes are most often confused?**
    > Can you explain why from an ecological / visual perspective?

    Now compute accuracy by hand:
    - How many predictions are correct? (diagonal sum of confusion matrix)
    - Divide by total predictions
    - Does this match the "accuracy" row in the classification report?

    **Bonus:** If the field team can only manually review 100 images per day, and you have
    10,000 detections, which class would you prioritise for human review — high-precision
    or high-recall? Why?
    """)


@app.cell
def _reflection(mo):
    mo.md(r"""
    ## Reflection

    - In conservation, missing a real animal (false negative) costs differently from
      flagging an empty image (false positive). Which error matters more depends on
      the application. Give one example where precision matters more, and one where recall matters more.
    - Spatial autocorrelation (images from the same site) can inflate evaluation metrics.
      How would you structure a train/test split to avoid this?
    - The model was trained on ImageNet. If you apply it to a new survey site with a
      species it has never seen, what metrics would you expect to change most?
    """)


if __name__ == "__main__":
    app.run()
