# Active Learning Sampling Strategies — Theory and Practice

## Taxonomy of Sampling Strategies

Active learning strategies fall into three families:

### 1. Uncertainty-Based (Query-by-Uncertainty)

Select samples where the model is least confident.

| Metric | Formula | Best for |
|--------|---------|----------|
| Least Confidence | `1 - max(p(y|x))` | Simple, any number of classes |
| Margin Sampling | `p(y₁|x) - p(y₂|x)` | When top-2 classes are competitive |
| Entropy | `-Σ p(y|x) log p(y|x)` | Many classes, spread uncertainty |
| MC Dropout | Variance over T stochastic passes | Better calibrated uncertainty |
| Deep Ensembles | Disagreement across N models | Most accurate, but expensive |

**For detection models:**
Uncertainty is typically computed per-image by aggregating detection-level
scores. Common approaches:
- Mean detection confidence across all detections in the image
- Entropy of the detection confidence distribution
- Number of detections near the confidence threshold ("borderline" detections)
- For images with no detections: assign maximum uncertainty

### 2. Diversity-Based (Query-by-Committee / Representativeness)

Select samples that are maximally diverse or representative of the unlabeled pool.

| Method | Mechanism | Requirements |
|--------|-----------|--------------|
| Random | Uniform sampling | None |
| RGB Contrast | Color histogram diversity | None |
| Coreset | Greedy farthest-point in feature space | Feature extractor |
| K-Center | Minimize max distance from any point to nearest selected point | Feature extractor |
| Cluster-based | Select cluster representatives | Feature extractor |

### 3. Hybrid (Query-by-Committee + Uncertainty)

Combine uncertainty and diversity to avoid "uncertainty tunnel vision."

| Method | Mechanism |
|--------|-----------|
| BADGE | Gradient embeddings clustered via K-Means++ |
| BatchBALD | Mutual information across a batch |
| Weighted Hybrid | α × uncertainty + (1-α) × diversity score |
| Two-stage | Uncertainty pre-filter → diversity selection |

---

## Strategy Selection for Wildlife Detection

### Decision Framework

```
Round 1 (Cold Start)?
├── YES → RGB Contrast or Embedding Clustering
│         (no model available for uncertainty)
│
└── NO → Is the model well-calibrated?
    ├── YES → Logit Uncertainty
    │         (entropy or margin sampling)
    │
    └── NO → Embedding Clustering + retrain
              (poorly calibrated = unreliable uncertainty)

Later rounds (3+)?
├── Still improving → Continue Uncertainty
├── Plateau detected → Switch to Hybrid
│                      (70% uncertainty + 30% diversity)
└── Coverage gap → Embedding Clustering
                   (underrepresented visual domains)
```

### Wildlife-Specific Considerations

**Dense colonies (iguanas, seabirds, seals):**
- Uncertainty sampling works well — model struggles most with overlapping animals
- High uncertainty images are often the most ecologically important (dense groups)
- Caution: uncertainty may focus on the same colony repeatedly
  → Add diversity component after round 2

**Sparse wildlife (large mammals, solitary predators):**
- Embedding clustering ensures coverage of different habitats and conditions
- Uncertainty may over-sample "empty" images where the model is unsure
  → Filter: only consider images where the model predicts ≥1 detection

**Camera traps (mixed species):**
- Uncertainty targets confused species pairs (e.g., deer vs. elk)
- Valuable for improving classification accuracy on similar-looking species
- Temporal bias: many camera trap images are of the same scene at different times
  → Use session-aware sampling to avoid selecting redundant images from the same burst

---

## Annotation Budget Guidelines

### How Many Images Per Round?

| Pool Size | Budget | Rounds | Per Round | Rationale |
|-----------|--------|--------|-----------|-----------|
| 1,000 | 200 | 4 | 50 | Small dataset, conservative |
| 5,000 | 1,000 | 5 | 200 | Medium dataset, standard |
| 10,000 | 2,000 | 5 | 400 | Large dataset, standard |
| 50,000+ | 5,000 | 5 | 1,000 | Very large, aggressive sampling |

**Rule of thumb:** Total annotation budget = 10-20% of the unlabeled pool.
Active learning aims to achieve 90%+ of "annotate everything" performance
with 10-20% of the labels.

### Stopping Criteria

1. **Budget exhausted**: Total annotations ≥ budget
2. **Performance plateau**: F1 improvement < 0.5% for 2+ consecutive rounds
3. **Target metric**: F1 ≥ target threshold (domain-specific)
4. **Marginal cost**: Cost per F1 point > acceptable threshold
5. **Expert judgment**: Domain expert considers the model "good enough"

---

## Measuring Strategy Effectiveness

### Learning Curve Analysis

The primary diagnostic tool. Plot model performance (F1, recall, mAP) vs.
cumulative annotations for different strategies.

**What to look for:**
- **Steep initial slope**: Strategy is selecting informative samples early
- **Higher plateau**: Strategy leads to better final performance
- **Faster convergence**: Same performance with fewer annotations
- **Area under the curve (AUC)**: Higher AUC = better overall efficiency

### Efficiency Metrics

| Metric | Formula | Measures |
|--------|---------|----------|
| Annotation Efficiency | ΔF1 / N_new_annotations | Information gain per label |
| Speedup Ratio | N_random / N_active to reach target F1 | Labeling savings |
| Budget Savings | 1 - (N_active / N_random) at equal F1 | Cost reduction |

### Diversity Diagnostics

After each round, check whether the selected samples cover the visual domain:

1. **t-SNE/UMAP visualization** of selected vs. remaining in embedding space
2. **Class distribution** of selected samples (is the strategy biased?)
3. **Spatial distribution** of detections (for georeferenced data)
4. **Temporal distribution** (for camera trap data with timestamps)
