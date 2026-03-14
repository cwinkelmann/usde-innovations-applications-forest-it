#!/usr/bin/env python3
"""
SAHI Batch Inference -- Process a folder of large images.
Runs sliced inference on each image, aggregates results to a single CSV
and per-image COCO JSON files. Supports checkpointing for long runs.

Usage:
    python sahi_batch_template.py \
        --input-dir orthomosaics/ \
        --model-path yolov8s.pt \
        --model-type yolov8 \
        --slice-size 640 \
        --overlap 0.25 \
        --output-dir batch_results/

Resume after interruption:
    python sahi_batch_template.py \
        --input-dir orthomosaics/ \
        --model-path yolov8s.pt \
        --output-dir batch_results/ \
        --resume
"""
import argparse
import csv
import json
import time
from pathlib import Path

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


# ============================================================================
# Constants
# ============================================================================

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


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
            return 'cuda:0'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
    except ImportError:
        pass
    return 'cpu'


# ============================================================================
# Checkpointing
# ============================================================================

def load_checkpoint(output_dir):
    """Load list of already-processed images from checkpoint."""
    checkpoint_path = output_dir / 'checkpoint.json'
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            return set(json.load(f).get('processed', []))
    return set()


def save_checkpoint(output_dir, processed_images):
    """Save checkpoint of processed images."""
    checkpoint_path = output_dir / 'checkpoint.json'
    with open(checkpoint_path, 'w') as f:
        json.dump({'processed': list(processed_images)}, f, indent=2)


# ============================================================================
# Batch processing
# ============================================================================

def collect_images(input_dir):
    """Collect all image files from the input directory."""
    input_path = Path(input_dir)
    images = sorted([
        p for p in input_path.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])
    return images


def process_single_image(image_path, model, args, per_image_dir):
    """Process a single image and return detections."""
    try:
        t0 = time.time()

        result = get_sliced_prediction(
            image=str(image_path),
            detection_model=model,
            slice_height=args.slice_size,
            slice_width=args.slice_size,
            overlap_height_ratio=args.overlap,
            overlap_width_ratio=args.overlap,
            postprocess_type=args.postprocess_type,
            postprocess_match_metric=args.match_metric,
            postprocess_match_threshold=args.match_threshold,
            perform_standard_pred=args.include_standard,
            verbose=0,
        )

        elapsed = time.time() - t0
        n_dets = len(result.object_prediction_list)

        # Save per-image COCO JSON
        annotations = []
        for i, pred in enumerate(result.object_prediction_list):
            x1, y1, x2, y2 = pred.bbox.to_xyxy()
            w = x2 - x1
            h = y2 - y1
            annotations.append({
                'id': i,
                'image_id': 0,
                'category_id': pred.category.id,
                'category_name': pred.category.name,
                'bbox': [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)],
                'score': round(pred.score.value, 4),
            })

        coco_path = per_image_dir / f'{image_path.stem}_coco.json'
        with open(coco_path, 'w') as f:
            json.dump({
                'images': [{'id': 0, 'file_name': image_path.name}],
                'annotations': annotations,
            }, f, indent=2)

        # Save visualization if requested
        if args.visualize:
            vis_dir = per_image_dir / 'visuals'
            vis_dir.mkdir(exist_ok=True)
            result.export_visuals(
                export_dir=str(vis_dir),
                file_name=image_path.stem,
                rect_th=2,
                text_size=0.5,
                text_th=1,
            )

        return result, elapsed, None

    except Exception as e:
        return None, 0.0, str(e)


def run_batch(args):
    """Run batch inference on all images in the input directory."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_image_dir = output_dir / 'per_image'
    per_image_dir.mkdir(exist_ok=True)

    # Collect images
    images = collect_images(input_dir)
    if not images:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(images)} images in {input_dir}")

    # Check for resume
    processed = set()
    if args.resume:
        processed = load_checkpoint(output_dir)
        if processed:
            print(f"Resuming: {len(processed)} already processed, "
                  f"{len(images) - len(processed)} remaining")

    # Load model
    device = get_device(args.device)
    print(f"Loading {args.model_type} model on {device}")
    model = AutoDetectionModel.from_pretrained(
        model_type=args.model_type,
        model_path=args.model_path,
        confidence_threshold=args.confidence,
        device=device,
    )

    # Open aggregate CSV
    csv_path = output_dir / 'all_detections.csv'
    csv_mode = 'a' if args.resume and csv_path.exists() else 'w'
    csvfile = open(csv_path, csv_mode, newline='')
    writer = csv.writer(csvfile)
    if csv_mode == 'w':
        writer.writerow(['image', 'x1', 'y1', 'x2', 'y2', 'confidence', 'class'])

    # Process images
    total_detections = 0
    total_time = 0.0
    errors = []

    for i, img_path in enumerate(images):
        if img_path.name in processed:
            continue

        print(f"[{i+1}/{len(images)}] {img_path.name}...", end=' ', flush=True)

        result, elapsed, error = process_single_image(
            img_path, model, args, per_image_dir
        )

        if error:
            print(f"ERROR: {error}")
            errors.append({'image': img_path.name, 'error': error})
            continue

        n_dets = len(result.object_prediction_list)
        total_detections += n_dets
        total_time += elapsed
        print(f"{n_dets} detections in {elapsed:.1f}s")

        # Write to aggregate CSV
        for pred in result.object_prediction_list:
            x1, y1, x2, y2 = pred.bbox.to_xyxy()
            writer.writerow([
                img_path.name,
                f"{x1:.1f}", f"{y1:.1f}",
                f"{x2:.1f}", f"{y2:.1f}",
                f"{pred.score.value:.4f}",
                pred.category.name,
            ])

        # Update checkpoint
        processed.add(img_path.name)
        save_checkpoint(output_dir, processed)

    csvfile.close()

    # Save error log
    if errors:
        with open(output_dir / 'errors.json', 'w') as f:
            json.dump(errors, f, indent=2)
        print(f"\n{len(errors)} errors logged to {output_dir / 'errors.json'}")

    # Summary
    n_processed = len(processed)
    print(f"\n{'='*60}")
    print(f"BATCH INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"  Images processed: {n_processed}/{len(images)}")
    print(f"  Total detections: {total_detections}")
    print(f"  Total time: {total_time:.1f}s")
    if n_processed > 0:
        print(f"  Avg time/image: {total_time / max(n_processed - len(processed - {img.name for img in images}), 1):.1f}s")
    print(f"  Results: {output_dir}")

    # Save summary
    summary = {
        'images_processed': n_processed,
        'total_images': len(images),
        'total_detections': total_detections,
        'total_time_seconds': round(total_time, 1),
        'errors': len(errors),
        'config': {
            'model_type': args.model_type,
            'model_path': args.model_path,
            'slice_size': args.slice_size,
            'overlap': args.overlap,
            'confidence': args.confidence,
            'postprocess_type': args.postprocess_type,
            'match_metric': args.match_metric,
            'match_threshold': args.match_threshold,
        },
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAHI Batch Inference')

    # Input/Output
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--output-dir', type=str, default='batch_results/',
                        help='Output directory')

    # Model
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--model-type', type=str, default='yolov8',
                        choices=['yolov8', 'yolov5', 'detectron2', 'mmdet',
                                 'huggingface', 'torchvision'])
    parser.add_argument('--confidence', type=float, default=0.3)

    # SAHI parameters
    parser.add_argument('--slice-size', type=int, default=640)
    parser.add_argument('--overlap', type=float, default=0.25)
    parser.add_argument('--postprocess-type', type=str, default='NMS',
                        choices=['NMS', 'NMM'])
    parser.add_argument('--match-metric', type=str, default='IOS',
                        choices=['IOU', 'IOS'])
    parser.add_argument('--match-threshold', type=float, default=0.5)
    parser.add_argument('--include-standard', action='store_true', default=False)

    # Options
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='Save annotated visualizations per image')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default=None)

    run_batch(parser.parse_args())
