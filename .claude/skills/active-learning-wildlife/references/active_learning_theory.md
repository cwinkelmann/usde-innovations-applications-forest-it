# Active Learning Theory for Wildlife Detection

## Foundational Concepts

### What is Active Learning?

Active learning is a machine learning paradigm where the learning algorithm
interactively queries a human annotator (the "oracle") to label data points
it finds most informative. Instead of learning from a large randomly-sampled
labeled dataset, the model strategically selects which data to learn from.

**Key terminology:**
- **Oracle**: The human expert who provides labels (annotations)
- **Query strategy**: The algorithm that selects which samples to label next
- **Unlabeled pool**: The set of available but unlabeled data
- **Annotation budget**: The total number of labels the oracle can provide
- **Learning curve**: Model performance plotted against cumulative labels

### Why Active Learning for Wildlife?

Wildlife annotation is expensive because:
1. **Expert knowledge required**: Species identification needs domain expertise
2. **Dense scenes**: Counting animals in colonies is labor-intensive
3. **Large images**: Drone orthomosaics contain thousands of potential targets
4. **Diminishing returns**: Most images are similar; random selection wastes budget

Active learning addresses this by:
- Selecting the most informative images first
- Reducing total annotation cost by 30-60% vs. random sampling
- Focusing expert time on genuinely difficult cases
- Providing pre-annotations to speed up per-image annotation time

---

## The Active Learning Loop

### Formal Definition

Given:
- Unlabeled pool U = {x₁, x₂, ..., xₙ}
- Small initial labeled set L (possibly empty)
- Oracle O: x → y (provides labels)
- Query budget B
- Query strategy Q: (model, U) → x* (selects next sample)
- Model M: x → ŷ (trained on L)

The loop:
```
for round = 1 to B/k:
    Train M on L
    x* = Q(M, U)          # Select k most informative samples
    y* = O(x*)             # Get labels from oracle
    L = L ∪ {(x*, y*)}     # Add to labeled set
    U = U \ {x*}           # Remove from pool
```

### Pool-Based vs. Stream-Based vs. Membership Query

| Paradigm | Access | Wildlife Use |
|----------|--------|--------------|
| **Pool-based** | Full access to U; select from pool | Standard for batch annotation |
| **Stream-based** | See one sample at a time; decide to query or skip | Camera trap real-time filtering |
| **Membership query** | Generate synthetic samples to query | Not practical for wildlife |

**Wildlife detection uses pool-based active learning** almost exclusively,
because we have a finite set of images and can select batches for annotation.

---

## Query Strategies in Detail

### Uncertainty Sampling

**Idea:** Select samples where the model is most uncertain.

**For classification (single-label):**
```
Least Confidence:   x* = argmax_x (1 - P(ŷ|x))
Margin:             x* = argmin_x (P(ŷ₁|x) - P(ŷ₂|x))
Entropy:            x* = argmax_x (-Σᵢ P(yᵢ|x) log P(yᵢ|x))
```

**For detection (multi-object):**
Per-image uncertainty is typically an aggregation:
- Mean detection confidence
- Entropy of detection confidences
- Number of detections with confidence in [0.3, 0.7] ("uncertain zone")
- Maximum per-class entropy

**Limitations:**
- Biased toward outliers and noisy samples
- Does not ensure diversity
- Requires a well-calibrated model (poorly calibrated → wrong uncertainties)
- Can get stuck in "uncertainty bubbles" (same hard region repeatedly)

### Diversity-Based Strategies

**Idea:** Select samples that are maximally representative of the data distribution.

**Coreset/Greedy K-Center:**
Select samples that minimize the maximum distance from any unlabeled point
to its nearest selected point. Equivalent to solving a K-center problem.

```
x* = argmax_x min_{l∈L} d(x, l)    # Farthest point from labeled set
```

**Cluster-Based:**
Cluster the unlabeled pool and select representatives from each cluster.
Ensures coverage of all visual domains.

**Advantages:**
- No model needed (can use pre-trained features)
- Good for cold start
- Ensures broad coverage

**Limitations:**
- Does not target model weaknesses
- May select "easy" representative samples
- Requires feature extraction (compute cost)

### Hybrid Strategies

**BADGE (Batch Active learning by Diverse Gradient Embeddings):**
Combines uncertainty (gradient magnitude) with diversity (gradient direction).
Selects a batch using K-Means++ on gradient embeddings.

**Practical hybrid for wildlife:**
```python
# Two-stage selection
n_total = 200
n_uncertain = 140  # 70%
n_diverse = 60     # 30%

# Stage 1: Pre-filter by uncertainty (top 500)
uncertain_pool = uncertainty_rank(model, unlabeled_pool, top_k=500)

# Stage 2: From uncertain pool, select by diversity
selected_uncertain = uncertainty_rank(model, uncertain_pool, top_k=n_uncertain)
remaining = [x for x in uncertain_pool if x not in selected_uncertain]
selected_diverse = diversity_select(remaining, n_diverse)

batch = selected_uncertain + selected_diverse
```

---

## Cold Start Problem

When no labeled data exists, uncertainty-based strategies fail because there
is no model to compute uncertainty from.

### Solutions

| Approach | Method | Effectiveness |
|----------|--------|---------------|
| Random | Uniform sampling from pool | Baseline |
| Diversity | Feature-based selection (RGB, embeddings) | Better coverage |
| Transfer | Use pre-trained model (MegaDetector) as proxy | Often best |
| Self-supervised | Train on unlabeled data first, then select | Expensive but effective |

**Recommended cold start for wildlife:**
1. **Round 1**: RGB Contrast or Embedding Clustering (diversity-based)
2. **Round 2**: Train initial model → switch to uncertainty-based
3. **Round 3+**: Uncertainty with diversity mixing

---

## Batch Mode Active Learning

### Why Batches?

In practice, we don't annotate one image at a time. We select a batch of K images,
send them all for annotation, then retrain. This introduces the **batch diversity**
problem: if we select K individually-most-uncertain images, they may all be similar.

### Batch Selection Methods

1. **Naive top-K**: Select K most uncertain. Simple but redundant.
2. **Diverse top-K**: Select K uncertain images that are also diverse (e.g., via
   K-Means++ on uncertain candidates).
3. **Submodular optimization**: Maximize an acquisition function subject to batch
   diversity constraints.

---

## Evaluating Active Learning

### Proper Experimental Setup

1. **Fixed test set**: Never include test images in the unlabeled pool
2. **Same initial model**: All strategies start from the same (or no) checkpoint
3. **Same budget**: Compare at equal annotation counts, not equal rounds
4. **Multiple seeds**: Run 3-5 times with different random initializations
5. **Baseline**: Always include random sampling as a baseline

### Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **ALC** (Area under Learning Curve) | Integral of performance vs. annotations | Overall efficiency |
| **Speedup** | N_random / N_active at target metric | Annotation savings |
| **Final performance** | Metric at budget exhaustion | Maximum achievable |
| **Annotations to target** | N at first reaching target F1 | Time to deployment |

---

## Key References

- Settles (2009). "Active Learning Literature Survey." Technical Report, UW-Madison.
  — The foundational survey, covers all classical strategies.

- Ren et al. (2021). "A Survey of Deep Active Learning." ACM Computing Surveys.
  — Modern deep learning extensions.

- Sener & Savarese (2018). "Active Learning for Convolutional Neural Networks:
  A Core-Set Approach." ICLR 2018.
  — Core-set approach with theoretical guarantees.

- Ash et al. (2020). "Deep Batch Active Learning by Diverse, Uncertain Gradient
  Lower Bounds." ICLR 2020.
  — BADGE algorithm combining uncertainty and diversity.

- Kellenberger et al. (2019). "Half a Percent of Labels is Enough: Efficient
  Animal Detection in UAV Imagery." CVPR Workshops 2019.
  — Active learning applied to wildlife drone imagery.
