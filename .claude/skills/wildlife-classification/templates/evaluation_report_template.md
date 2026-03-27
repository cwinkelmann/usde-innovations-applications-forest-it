# Evaluation Report Template

Use this template for the `evaluation_agent` output. Fill in all bracketed values with actual results.

---

## Evaluation Report: [Model Name] on [Dataset Name]

**Date:** [YYYY-MM-DD]
**Checkpoint:** [path/to/checkpoint.pth]
**Test set:** [path/to/test/data]

---

### Summary

| Metric | Value |
|--------|-------|
| Model | [backbone name, e.g., vit_base_patch14_dinov2.lvd142m] |
| Input size | [NxN, e.g., 518x518] |
| Total samples | [N] |
| Classes | [C] |
| Overall accuracy | [X.XXX] |
| Balanced accuracy | [X.XXX] |
| Macro precision | [X.XXX] |
| Macro recall | [X.XXX] |
| Macro F1 | [X.XXX] |
| Expected Calibration Error | [X.XXXX] |

---

### Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| [class_1] | [X.XXX] | [X.XXX] | [X.XXX] | [N] |
| [class_2] | [X.XXX] | [X.XXX] | [X.XXX] | [N] |
| [class_3] | [X.XXX] | [X.XXX] | [X.XXX] | [N] |
| ... | ... | ... | ... | ... |
| **Macro avg** | **[X.XXX]** | **[X.XXX]** | **[X.XXX]** | **[N]** |
| **Weighted avg** | **[X.XXX]** | **[X.XXX]** | **[X.XXX]** | **[N]** |

---

### Confusion Matrix

```
              [cls_1]    [cls_2]    [cls_3]    ...
[cls_1]        [N]        [N]        [N]
[cls_2]        [N]        [N]        [N]
[cls_3]        [N]        [N]        [N]
```

**Top confusion pairs:**
1. [true_class] misclassified as [pred_class]: [N] times ([X.X%] of [true_class])
2. [true_class] misclassified as [pred_class]: [N] times ([X.X%] of [true_class])
3. [true_class] misclassified as [pred_class]: [N] times ([X.X%] of [true_class])

---

### Confidence Analysis

| Subset | Mean Confidence | Median | Min |
|--------|----------------|--------|-----|
| All predictions | [X.XXX] | [X.XXX] | [X.XXX] |
| Correct predictions | [X.XXX] | [X.XXX] | [X.XXX] |
| Incorrect predictions | [X.XXX] | [X.XXX] | [X.XXX] |

**Calibration assessment:** [Well-calibrated / Overconfident / Underconfident]

[If ECE > 0.10: "The model is poorly calibrated. Consider applying temperature scaling (recommended T=[X.X]) before deployment."]

---

### Key Findings

1. **Best performing class:** [class_name] (F1=[X.XXX]) -- [brief explanation why]
2. **Worst performing class:** [class_name] (F1=[X.XXX]) -- [brief explanation why]
3. **Notable confusion:** [class_A] and [class_B] are frequently confused -- [ecological explanation if applicable]
4. **Calibration:** [Assessment of model confidence reliability]

---

### Recommendations

1. **Data collection:** [If a class underperforms due to low support, recommend collecting more images]
2. **Augmentation:** [If domain-specific issues are identified, recommend targeted augmentation]
3. **Model architecture:** [If the model struggles with fine-grained differences, recommend larger backbone or higher resolution]
4. **Threshold tuning:** [If binary deployment, recommend optimal threshold from sweep]
5. **Calibration:** [If poorly calibrated, recommend temperature scaling]

---

### Reproduction

```bash
python evaluation_template.py \
    --checkpoint [path] \
    --data-dir [path] \
    --model-name [model] \
    --output-dir [path]
```
