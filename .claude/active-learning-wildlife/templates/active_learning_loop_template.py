#!/usr/bin/env python3
"""
Active Learning Loop Template — Wildlife Detection
Runs an active learning loop with configurable sampling strategy,
annotation tool integration, and learning curve logging.

Usage:
    python active_learning_loop_template.py \
        --unlabeled-dir /path/to/images \
        --annotation-budget 1000 \
        --images-per-round 200 \
        --num-rounds 5 \
        --strategy rgb_contrast \
        --output-dir ./al_output

    # With existing model (skip cold start):
    python active_learning_loop_template.py \
        --unlabeled-dir /path/to/images \
        --model-path checkpoints/best.pth \
        --strategy logit_uncertainty \
        --output-dir ./al_output
"""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


# ============================================================================
# Device selection
# ============================================================================

def get_device(preferred=None):
    """Select best available device."""
    if preferred:
        return preferred
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
    except ImportError:
        pass
    return 'cpu'


# ============================================================================
# Image collection
# ============================================================================

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def collect_images(directory):
    """Collect all image files from a directory."""
    directory = Path(directory)
    images = sorted([
        p for p in directory.rglob('*')
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])
    return images


# ============================================================================
# Sampling strategies
# ============================================================================

def compute_color_histogram(image_path, bins=(8, 8, 8)):
    """Compute normalized color histogram for an image."""
    img = Image.open(image_path).convert('RGB').resize((256, 256))
    arr = np.array(img)
    hist, _ = np.histogramdd(
        arr.reshape(-1, 3).astype(float),
        bins=bins,
        range=((0, 256), (0, 256), (0, 256)),
    )
    hist = hist.flatten()
    return hist / (hist.sum() + 1e-8)


def sample_rgb_contrast(image_paths, n_samples, already_selected=None):
    """Select diverse images via RGB contrast (farthest-point sampling)."""
    if already_selected is None:
        already_selected = set()

    available = [p for p in image_paths if str(p) not in already_selected]
    if len(available) <= n_samples:
        return available

    print(f"  Computing color histograms for {len(available)} images...")
    features = np.array([compute_color_histogram(p) for p in available])

    selected = []
    mean_feat = features.mean(axis=0)

    # First: most different from mean
    distances = np.linalg.norm(features - mean_feat, axis=1)
    first = np.argmax(distances)
    selected.append(first)

    # Greedy farthest-point
    for _ in range(n_samples - 1):
        sel_feats = features[selected]
        min_dists = np.min(
            np.linalg.norm(features[:, None] - sel_feats[None, :], axis=2),
            axis=1,
        )
        min_dists[selected] = -1
        selected.append(np.argmax(min_dists))

    return [available[i] for i in selected]


def sample_random(image_paths, n_samples, already_selected=None):
    """Random sampling baseline."""
    if already_selected is None:
        already_selected = set()

    available = [p for p in image_paths if str(p) not in already_selected]
    n = min(n_samples, len(available))
    indices = np.random.choice(len(available), size=n, replace=False)
    return [available[i] for i in indices]


def sample_embedding_clustering(image_paths, n_samples, n_clusters=None,
                                  device='cuda', already_selected=None):
    """Select diverse images via embedding clustering (DINOv2)."""
    import torch
    from sklearn.cluster import KMeans
    from torchvision import transforms

    if already_selected is None:
        already_selected = set()

    available = [p for p in image_paths if str(p) not in already_selected]
    if len(available) <= n_samples:
        return available

    if n_clusters is None:
        n_clusters = max(n_samples // 5, 2)

    # Extract embeddings
    print(f"  Extracting DINOv2 embeddings for {len(available)} images...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    embeddings = []
    with torch.no_grad():
        for img_path in available:
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            emb = model(tensor).cpu().numpy().flatten()
            embeddings.append(emb)

    embeddings = np.array(embeddings)

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Select representatives
    selected = []
    per_cluster = max(n_samples // n_clusters, 1)

    for cid in range(n_clusters):
        mask = labels == cid
        indices = np.where(mask)[0]
        centroid = kmeans.cluster_centers_[cid]
        dists = np.linalg.norm(embeddings[indices] - centroid, axis=1)
        order = np.argsort(dists)
        n_sel = min(per_cluster, len(order))
        selected.extend(indices[order[:n_sel]])

    # Fill remaining with farthest-point
    while len(selected) < n_samples and len(selected) < len(available):
        remaining = list(set(range(len(available))) - set(selected))
        if not remaining:
            break
        sel_emb = embeddings[selected]
        rem_emb = embeddings[remaining]
        min_d = np.min(
            np.linalg.norm(rem_emb[:, None] - sel_emb[None, :], axis=2),
            axis=1,
        )
        selected.append(remaining[np.argmax(min_d)])

    return [available[i] for i in selected[:n_samples]]


def sample_uncertainty(image_paths, n_samples, model, device='cuda',
                        metric='entropy', already_selected=None):
    """Select most uncertain images using model predictions.

    This is a placeholder — adapt to your specific model's prediction API.
    """
    if already_selected is None:
        already_selected = set()

    available = [p for p in image_paths if str(p) not in already_selected]
    if len(available) <= n_samples:
        return available

    print(f"  Computing uncertainty for {len(available)} images...")
    uncertainties = []

    for img_path in available:
        # TODO: Replace with your model's prediction API
        # predictions = model.predict(img_path, device=device)
        # scores = predictions.scores
        #
        # if len(scores) == 0:
        #     uncertainty = 1.0  # No detections = high uncertainty
        # elif metric == 'entropy':
        #     p = np.clip(scores, 1e-8, 1.0)
        #     uncertainty = float(np.mean(-(p * np.log(p) + (1-p) * np.log(1-p))))
        # elif metric == 'least_confidence':
        #     uncertainty = float(1.0 - np.mean(scores))
        # else:
        #     uncertainty = float(np.mean(np.abs(scores - 0.5)))

        uncertainty = np.random.random()  # Placeholder
        uncertainties.append(uncertainty)

    uncertainties = np.array(uncertainties)
    top_indices = np.argsort(uncertainties)[-n_samples:][::-1]
    return [available[i] for i in top_indices]


def select_samples(strategy, image_paths, n_samples, already_selected=None,
                    model=None, device='cuda'):
    """Dispatch to the appropriate sampling strategy."""
    if strategy == 'rgb_contrast':
        return sample_rgb_contrast(image_paths, n_samples, already_selected)
    elif strategy == 'random':
        return sample_random(image_paths, n_samples, already_selected)
    elif strategy == 'embedding_clustering':
        return sample_embedding_clustering(image_paths, n_samples,
                                            device=device,
                                            already_selected=already_selected)
    elif strategy == 'logit_uncertainty':
        return sample_uncertainty(image_paths, n_samples, model, device,
                                   already_selected=already_selected)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ============================================================================
# Pre-annotation export
# ============================================================================

def export_for_annotation(selected_images, predictions, output_dir, round_num):
    """Export selected images and pre-annotations for manual review.

    Creates a folder structure for annotation and a COCO JSON with predictions.
    """
    round_dir = output_dir / f'round_{round_num}'
    images_dir = round_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    # Symlink or copy images
    for img_path in selected_images:
        dest = images_dir / img_path.name
        if not dest.exists():
            dest.symlink_to(img_path.resolve())

    # Write pre-annotations as COCO JSON
    coco = {
        'images': [],
        'annotations': [],
        'categories': [{'id': 0, 'name': 'animal'}],
    }

    ann_id = 0
    for img_id, img_path in enumerate(selected_images):
        coco['images'].append({
            'id': img_id,
            'file_name': img_path.name,
        })

        img_preds = predictions.get(str(img_path), [])
        for pred in img_preds:
            x1, y1, x2, y2 = pred.get('bbox', [0, 0, 0, 0])
            w, h = x2 - x1, y2 - y1
            coco['annotations'].append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': 0,
                'bbox': [x1, y1, w, h],
                'area': w * h,
                'iscrowd': 0,
                'score': pred.get('score', 0.5),
            })
            ann_id += 1

    coco_path = round_dir / 'pre_annotations.json'
    with open(coco_path, 'w') as f:
        json.dump(coco, f, indent=2)

    print(f"  Exported {len(selected_images)} images + {ann_id} pre-annotations")
    print(f"  Images: {images_dir}")
    print(f"  Pre-annotations: {coco_path}")

    return round_dir


# ============================================================================
# Annotation merging
# ============================================================================

def merge_annotations(existing_coco, new_coco_path):
    """Merge new round annotations into the existing training set."""
    with open(new_coco_path) as f:
        new_coco = json.load(f)

    if existing_coco is None:
        return new_coco

    # Offset IDs to avoid conflicts
    max_img_id = max((img['id'] for img in existing_coco['images']), default=-1) + 1
    max_ann_id = max((ann['id'] for ann in existing_coco['annotations']), default=-1) + 1

    for img in new_coco['images']:
        old_id = img['id']
        img['id'] = max_img_id
        # Update annotations to match new image ID
        for ann in new_coco['annotations']:
            if ann['image_id'] == old_id:
                ann['image_id'] = max_img_id
                ann['id'] = max_ann_id
                max_ann_id += 1
        max_img_id += 1

    existing_coco['images'].extend(new_coco['images'])
    existing_coco['annotations'].extend(new_coco['annotations'])

    return existing_coco


# ============================================================================
# Learning curve logging
# ============================================================================

def log_round(learning_curve_path, round_num, n_annotations, metrics, strategy):
    """Log one round to the learning curve CSV."""
    file_exists = learning_curve_path.exists()

    with open(learning_curve_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'round', 'n_annotations', 'val_f1', 'val_precision',
                'val_recall', 'strategy', 'timestamp',
            ])

        writer.writerow([
            round_num,
            n_annotations,
            f"{metrics.get('f1', 0):.4f}",
            f"{metrics.get('precision', 0):.4f}",
            f"{metrics.get('recall', 0):.4f}",
            strategy,
            datetime.now().isoformat(),
        ])


def check_plateau(learning_curve_path, threshold=0.005, window=3):
    """Check if the learning curve has plateaued."""
    if not learning_curve_path.exists():
        return False

    with open(learning_curve_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) < window:
        return False

    recent = rows[-window:]
    f1_values = [float(r['val_f1']) for r in recent]
    improvements = [f1_values[i+1] - f1_values[i] for i in range(len(f1_values)-1)]

    if all(imp < threshold for imp in improvements):
        print(f"  Performance plateau detected (improvements: {improvements})")
        return True

    return False


# ============================================================================
# Main loop
# ============================================================================

def run_active_learning(args):
    """Run the complete active learning loop."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    print(f"Device: {device}")

    # Collect unlabeled images
    unlabeled_pool = collect_images(args.unlabeled_dir)
    print(f"Unlabeled pool: {len(unlabeled_pool)} images")

    if not unlabeled_pool:
        print("ERROR: No images found in unlabeled directory")
        return

    # Load existing model if provided
    model = None
    if args.model_path:
        # TODO: Load your model here
        # model = load_model(args.model_path, device)
        print(f"Model loaded from {args.model_path}")

    # Initialize tracking
    already_selected = set()
    training_coco = None
    total_annotated = 0
    learning_curve_path = output_dir / 'learning_curve.csv'

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Active learning loop
    for round_num in range(1, args.num_rounds + 1):
        print(f"\n{'='*60}")
        print(f"ACTIVE LEARNING ROUND {round_num}/{args.num_rounds}")
        print(f"{'='*60}")

        # Check budget
        remaining_budget = args.annotation_budget - total_annotated
        if remaining_budget <= 0:
            print("Annotation budget exhausted. Stopping.")
            break

        n_this_round = min(args.images_per_round, remaining_budget)

        # Determine strategy
        strategy = args.strategy
        if strategy == 'logit_uncertainty' and model is None:
            print("  No model available — falling back to rgb_contrast for cold start")
            strategy = 'rgb_contrast'

        # Step 1: Select samples
        print(f"\nStep 1: Selecting {n_this_round} images via {strategy}...")
        selected = select_samples(
            strategy, unlabeled_pool, n_this_round,
            already_selected=already_selected,
            model=model, device=device,
        )
        print(f"  Selected {len(selected)} images")

        # Step 2: Generate pre-annotations (if model exists)
        predictions = {}
        if model is not None:
            print("\nStep 2: Generating pre-annotations...")
            # TODO: Run model inference on selected images
            # for img_path in selected:
            #     preds = model.predict(str(img_path), device=device)
            #     predictions[str(img_path)] = [...]
            print(f"  Generated predictions for {len(selected)} images")
        else:
            print("\nStep 2: No model — skipping pre-annotations")

        # Step 3: Export for annotation
        print(f"\nStep 3: Exporting for annotation...")
        round_dir = export_for_annotation(selected, predictions, output_dir, round_num)

        # Step 4: Wait for human annotation
        print(f"\n--- HUMAN ANNOTATION REQUIRED ---")
        print(f"Please annotate images in: {round_dir / 'images'}")
        print(f"Pre-annotations available at: {round_dir / 'pre_annotations.json'}")
        print(f"Save corrected annotations as: {round_dir / 'corrected_annotations.json'}")
        input("\nPress Enter when annotation is complete...")

        # Step 5: Load corrected annotations
        corrected_path = round_dir / 'corrected_annotations.json'
        if not corrected_path.exists():
            print(f"WARNING: {corrected_path} not found. Using pre-annotations.")
            corrected_path = round_dir / 'pre_annotations.json'

        # Step 6: Merge
        training_coco = merge_annotations(training_coco, corrected_path)
        total_annotated += len(selected)
        already_selected.update(str(p) for p in selected)

        # Save merged training set
        merged_path = output_dir / 'training_annotations.json'
        with open(merged_path, 'w') as f:
            json.dump(training_coco, f, indent=2)
        print(f"\nTraining set: {total_annotated} annotated images")

        # Step 7: Retrain
        print("\nStep 7: Retraining model...")
        # TODO: Train model on merged_path
        # model = train_model(merged_path, output_dir / f'round_{round_num}', device)
        print("  [Placeholder: implement model training]")

        # Step 8: Evaluate
        print("\nStep 8: Evaluating...")
        # TODO: Evaluate model on validation set
        metrics = {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}  # Placeholder
        # metrics = evaluate_model(model, val_dataset, device)

        log_round(learning_curve_path, round_num, total_annotated, metrics, strategy)
        print(f"  F1: {metrics['f1']:.3f}, P: {metrics['precision']:.3f}, R: {metrics['recall']:.3f}")

        # Step 9: Check stopping criteria
        if check_plateau(learning_curve_path):
            print("\nPerformance plateau — stopping early.")
            break

    # Final summary
    print(f"\n{'='*60}")
    print(f"ACTIVE LEARNING COMPLETE")
    print(f"{'='*60}")
    print(f"  Rounds completed: {round_num}")
    print(f"  Total annotations: {total_annotated}")
    print(f"  Learning curve: {learning_curve_path}")
    print(f"  Training set: {output_dir / 'training_annotations.json'}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Active Learning Loop for Wildlife Detection')

    # Data
    parser.add_argument('--unlabeled-dir', type=str, required=True,
                        help='Directory of unlabeled images')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to pre-trained model (optional, for warm start)')

    # Budget
    parser.add_argument('--annotation-budget', type=int, default=1000,
                        help='Total images to annotate across all rounds')
    parser.add_argument('--images-per-round', type=int, default=200,
                        help='Images to select per round')
    parser.add_argument('--num-rounds', type=int, default=5,
                        help='Maximum number of active learning rounds')

    # Strategy
    parser.add_argument('--strategy', type=str, default='rgb_contrast',
                        choices=['rgb_contrast', 'random', 'embedding_clustering',
                                 'logit_uncertainty'],
                        help='Sampling strategy')

    # Output
    parser.add_argument('--output-dir', type=str, default='./al_output',
                        help='Output directory for all results')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda, mps, cpu)')

    args = parser.parse_args()
    run_active_learning(args)
