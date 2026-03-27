#!/usr/bin/env python3
"""
Sampling Strategy Comparison Template
Simulates active learning with different sampling strategies on a dataset
with known ground truth, producing learning curves for comparison.

This is a simulation tool for evaluating sampling strategies without requiring
actual annotation — it uses existing ground truth labels as the "oracle."

Usage:
    python sampling_comparison_template.py \
        --data-dir path/to/imagefolder \
        --strategies rgb_contrast random embedding_clustering \
        --budget 500 \
        --images-per-round 100 \
        --output-dir comparison_results/ \
        --n-runs 3
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================================
# Sampling strategies (simplified for simulation)
# ============================================================================

def compute_histograms(image_paths):
    """Compute color histograms for all images."""
    features = []
    for p in image_paths:
        img = Image.open(p).convert('RGB').resize((128, 128))
        arr = np.array(img)
        hist, _ = np.histogramdd(
            arr.reshape(-1, 3).astype(float),
            bins=(8, 8, 8),
            range=((0, 256), (0, 256), (0, 256)),
        )
        hist = hist.flatten()
        features.append(hist / (hist.sum() + 1e-8))
    return np.array(features)


def sample_random(n_available, n_select, already_selected, features=None):
    """Random sampling."""
    available = list(set(range(n_available)) - already_selected)
    n = min(n_select, len(available))
    return set(np.random.choice(available, size=n, replace=False))


def sample_rgb_contrast(n_available, n_select, already_selected, features):
    """RGB contrast (farthest-point sampling on color histograms)."""
    available = list(set(range(n_available)) - already_selected)
    if len(available) <= n_select:
        return set(available)

    avail_features = features[available]
    selected_indices = []

    # First: most different from mean
    mean_feat = avail_features.mean(axis=0)
    dists = np.linalg.norm(avail_features - mean_feat, axis=1)
    first = np.argmax(dists)
    selected_indices.append(first)

    for _ in range(n_select - 1):
        sel_feats = avail_features[selected_indices]
        min_dists = np.min(
            np.linalg.norm(avail_features[:, None] - sel_feats[None, :], axis=2),
            axis=1,
        )
        min_dists[selected_indices] = -1
        selected_indices.append(np.argmax(min_dists))

    return set(available[i] for i in selected_indices)


def sample_embedding_clustering(n_available, n_select, already_selected, features):
    """Embedding clustering (using color features as proxy for embeddings)."""
    from sklearn.cluster import KMeans

    available = list(set(range(n_available)) - already_selected)
    if len(available) <= n_select:
        return set(available)

    avail_features = features[available]
    n_clusters = max(n_select // 5, 2)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(avail_features)

    selected_indices = []
    per_cluster = max(n_select // n_clusters, 1)

    for cid in range(n_clusters):
        mask = labels == cid
        indices = np.where(mask)[0]
        centroid = kmeans.cluster_centers_[cid]
        dists = np.linalg.norm(avail_features[indices] - centroid, axis=1)
        order = np.argsort(dists)
        n_sel = min(per_cluster, len(order))
        selected_indices.extend(indices[order[:n_sel]])

    # Fill remaining
    while len(selected_indices) < n_select and len(selected_indices) < len(available):
        remaining = list(set(range(len(available))) - set(selected_indices))
        if not remaining:
            break
        sel_emb = avail_features[selected_indices]
        rem_emb = avail_features[remaining]
        min_d = np.min(
            np.linalg.norm(rem_emb[:, None] - sel_emb[None, :], axis=2),
            axis=1,
        )
        selected_indices.append(remaining[np.argmax(min_d)])

    return set(available[i] for i in selected_indices[:n_select])


STRATEGY_FUNCTIONS = {
    'random': sample_random,
    'rgb_contrast': sample_rgb_contrast,
    'embedding_clustering': sample_embedding_clustering,
}


# ============================================================================
# Simulated training and evaluation
# ============================================================================

def simulate_model_performance(selected_indices, all_labels, class_names):
    """Simulate model performance based on selected training set.

    This is a simplified simulation. In practice, you would train an actual
    model. Here we approximate performance based on class coverage and
    sample count.

    The simulation uses class-balanced coverage as a proxy for model quality:
    - More samples = better performance (log scaling)
    - Better class balance = better performance
    - More diverse samples = slight bonus (approximated)
    """
    if not selected_indices:
        return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}

    selected_labels = [all_labels[i] for i in selected_indices]
    n_samples = len(selected_labels)
    n_classes = len(class_names)

    # Class coverage
    unique_classes = set(selected_labels)
    class_coverage = len(unique_classes) / n_classes

    # Class balance (entropy-based)
    counts = np.array([selected_labels.count(c) for c in range(n_classes)])
    counts = counts + 1e-8  # Avoid log(0)
    probs = counts / counts.sum()
    balance = -np.sum(probs * np.log(probs)) / np.log(n_classes)

    # Sample quantity effect (diminishing returns)
    quantity_factor = min(1.0, np.log(1 + n_samples) / np.log(1 + len(all_labels)))

    # Combined metric (weighted)
    raw_f1 = 0.3 * class_coverage + 0.3 * balance + 0.4 * quantity_factor

    # Add noise to simulate training variance
    noise = np.random.normal(0, 0.02)
    f1 = np.clip(raw_f1 + noise, 0.0, 1.0)

    precision = np.clip(f1 + np.random.normal(0, 0.03), 0.0, 1.0)
    recall = np.clip(f1 + np.random.normal(0, 0.03), 0.0, 1.0)

    return {'f1': float(f1), 'precision': float(precision), 'recall': float(recall)}


# ============================================================================
# Simulation runner
# ============================================================================

def run_simulation(strategy_name, image_paths, labels, features,
                    budget, images_per_round, seed=42):
    """Run one active learning simulation with a given strategy."""
    np.random.seed(seed)

    n_images = len(image_paths)
    already_selected = set()
    learning_curve = []

    strategy_fn = STRATEGY_FUNCTIONS[strategy_name]
    n_rounds = budget // images_per_round

    for round_num in range(1, n_rounds + 1):
        remaining_budget = budget - len(already_selected)
        n_this_round = min(images_per_round, remaining_budget)

        if n_this_round <= 0:
            break

        # Select samples
        new_indices = strategy_fn(n_images, n_this_round, already_selected, features)
        already_selected.update(new_indices)

        # Simulate model performance
        class_names = list(set(labels))
        metrics = simulate_model_performance(already_selected, labels, class_names)

        learning_curve.append({
            'round': round_num,
            'n_annotations': len(already_selected),
            'f1': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'strategy': strategy_name,
        })

    return learning_curve


# ============================================================================
# Plotting
# ============================================================================

def plot_learning_curves(all_curves, output_path):
    """Plot learning curves for all strategies."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = {'random': '#999999', 'rgb_contrast': '#2196F3',
              'embedding_clustering': '#4CAF50', 'logit_uncertainty': '#FF9800'}

    for strategy_name, curves in all_curves.items():
        # Aggregate across runs
        max_rounds = max(len(c) for c in curves)
        f1_by_round = []

        for r in range(max_rounds):
            f1s = [c[r]['f1'] for c in curves if r < len(c)]
            f1_by_round.append(f1s)

        n_annotations = [curves[0][r]['n_annotations'] for r in range(max_rounds)]
        means = [np.mean(f1s) for f1s in f1_by_round]
        stds = [np.std(f1s) for f1s in f1_by_round]

        color = colors.get(strategy_name, '#000000')
        ax.plot(n_annotations, means, '-o', label=strategy_name,
                color=color, linewidth=2, markersize=6)
        ax.fill_between(
            n_annotations,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.15, color=color,
        )

    ax.set_xlabel('Number of Annotations', fontsize=12)
    ax.set_ylabel('Simulated F1 Score', fontsize=12)
    ax.set_title('Active Learning Strategy Comparison', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Learning curve plot saved to {output_path}")


def plot_efficiency_comparison(all_curves, output_path):
    """Plot annotation efficiency (F1 gain per annotation) for each strategy."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = {'random': '#999999', 'rgb_contrast': '#2196F3',
              'embedding_clustering': '#4CAF50', 'logit_uncertainty': '#FF9800'}

    for strategy_name, curves in all_curves.items():
        # Average F1 across runs
        max_rounds = max(len(c) for c in curves)
        avg_f1 = [np.mean([c[r]['f1'] for c in curves if r < len(c)])
                  for r in range(max_rounds)]

        # Compute marginal efficiency
        n_per_round = curves[0][0]['n_annotations']
        efficiency = [avg_f1[0] / n_per_round]  # First round
        for i in range(1, len(avg_f1)):
            gain = avg_f1[i] - avg_f1[i-1]
            efficiency.append(gain / n_per_round)

        rounds = list(range(1, len(efficiency) + 1))
        color = colors.get(strategy_name, '#000000')
        ax.bar([r + list(all_curves.keys()).index(strategy_name) * 0.2 for r in rounds],
               efficiency, width=0.2, label=strategy_name, color=color, alpha=0.7)

    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('F1 Gain per Annotation', fontsize=12)
    ax.set_title('Annotation Efficiency by Round', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Efficiency plot saved to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    data_dir = Path(args.data_dir)
    image_paths = []
    labels = []

    # Assume ImageFolder structure: data_dir/class_name/image.jpg
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]

    for class_idx, class_dir in enumerate(class_dirs):
        for img_path in sorted(class_dir.glob('*')):
            if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                image_paths.append(img_path)
                labels.append(class_idx)

    print(f"Dataset: {len(image_paths)} images, {len(class_names)} classes")
    print(f"Classes: {class_names}")

    # Compute features once (shared across strategies)
    print("Computing color features...")
    features = compute_histograms(image_paths)

    # Run simulations
    all_curves = {}
    for strategy in args.strategies:
        print(f"\n{'='*40}")
        print(f"Strategy: {strategy}")
        print(f"{'='*40}")

        curves = []
        for run in range(args.n_runs):
            print(f"  Run {run+1}/{args.n_runs}...", end=' ')
            t0 = time.time()
            curve = run_simulation(
                strategy, image_paths, labels, features,
                args.budget, args.images_per_round, seed=42 + run,
            )
            elapsed = time.time() - t0
            final_f1 = curve[-1]['f1'] if curve else 0
            print(f"final F1={final_f1:.3f} ({elapsed:.1f}s)")
            curves.append(curve)

        all_curves[strategy] = curves

    # Save raw results
    results = {}
    for strategy, curves in all_curves.items():
        results[strategy] = []
        for run_idx, curve in enumerate(curves):
            results[strategy].append({
                'run': run_idx,
                'learning_curve': curve,
            })

    with open(output_dir / 'raw_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    plot_learning_curves(all_curves, output_dir / 'learning_curves.png')
    plot_efficiency_comparison(all_curves, output_dir / 'efficiency.png')

    # Summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"{'Strategy':<25} {'Final F1 (mean±std)':<25} {'AUC':<10}")
    print(f"{'-'*60}")

    for strategy, curves in all_curves.items():
        final_f1s = [c[-1]['f1'] for c in curves if c]
        mean_f1 = np.mean(final_f1s)
        std_f1 = np.std(final_f1s)

        # AUC (area under learning curve)
        mean_curve = [np.mean([c[r]['f1'] for c in curves if r < len(c)])
                      for r in range(max(len(c) for c in curves))]
        auc = np.trapz(mean_curve)

        print(f"{strategy:<25} {mean_f1:.3f} ± {std_f1:.3f}             {auc:.3f}")

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Active Learning Strategy Comparison')

    parser.add_argument('--data-dir', type=str, required=True,
                        help='ImageFolder dataset (class_name/image.jpg)')
    parser.add_argument('--strategies', nargs='+',
                        default=['random', 'rgb_contrast', 'embedding_clustering'],
                        choices=['random', 'rgb_contrast', 'embedding_clustering'],
                        help='Strategies to compare')
    parser.add_argument('--budget', type=int, default=500,
                        help='Total annotation budget')
    parser.add_argument('--images-per-round', type=int, default=100,
                        help='Images per round')
    parser.add_argument('--n-runs', type=int, default=3,
                        help='Number of simulation runs per strategy')
    parser.add_argument('--output-dir', type=str, default='comparison_results/',
                        help='Output directory')

    main(parser.parse_args())
