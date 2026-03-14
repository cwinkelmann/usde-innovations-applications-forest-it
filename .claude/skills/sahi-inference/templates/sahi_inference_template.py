#!/usr/bin/env python3
"""
SAHI Tiled Inference -- Single Image Template
Runs sliced inference on a large image using any SAHI-supported detector.
Exports detections as CSV, COCO JSON, and optional visualization.

Usage:
    python sahi_inference_template.py \
        --image orthomosaic.tif \
        --model-path yolov8s.pt \
        --model-type yolov8 \
        --slice-size 640 \
        --overlap 0.25 \
        --output-dir results/
"""
import argparse
import csv
import json
import time
from pathlib import Path

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, get_prediction


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
# Model loading
# ============================================================================

def load_detection_model(args):
    """Load a detection model via SAHI's AutoDetectionModel."""
    device = get_device(args.device)
    print(f"Loading {args.model_type} model from {args.model_path}")
    print(f"Device: {device}")

    model = AutoDetectionModel.from_pretrained(
        model_type=args.model_type,
        model_path=args.model_path,
        confidence_threshold=args.confidence,
        device=device,
    )
    return model


# ============================================================================
# Inference
# ============================================================================

def run_sliced_inference(image_path, model, args):
    """Run SAHI sliced prediction on a single image."""
    print(f"\nProcessing: {image_path}")
    print(f"  Slice size: {args.slice_size}x{args.slice_size}")
    print(f"  Overlap: {args.overlap}")
    print(f"  Postprocess: {args.postprocess_type} / {args.match_metric} @ {args.match_threshold}")

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
    print(f"  Detections: {n_dets}")
    print(f"  Time: {elapsed:.2f}s")

    return result, elapsed


def run_standard_inference(image_path, model):
    """Run standard (non-sliced) inference for comparison."""
    print(f"\nStandard inference on: {image_path}")

    t0 = time.time()
    result = get_prediction(
        image=str(image_path),
        detection_model=model,
    )
    elapsed = time.time() - t0

    n_dets = len(result.object_prediction_list)
    print(f"  Detections: {n_dets}")
    print(f"  Time: {elapsed:.2f}s")

    return result, elapsed


# ============================================================================
# Export functions
# ============================================================================

def export_csv(result, image_name, output_path):
    """Export detections to CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'x1', 'y1', 'x2', 'y2', 'confidence', 'class'])

        for pred in result.object_prediction_list:
            x1, y1, x2, y2 = pred.bbox.to_xyxy()
            writer.writerow([
                image_name,
                f"{x1:.1f}", f"{y1:.1f}",
                f"{x2:.1f}", f"{y2:.1f}",
                f"{pred.score.value:.4f}",
                pred.category.name,
            ])

    print(f"  CSV saved: {output_path}")


def export_coco_json(result, image_path, output_path):
    """Export detections in COCO annotation format."""
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
            'bbox': [x1, y1, w, h],
            'score': pred.score.value,
            'area': w * h,
        })

    coco_output = {
        'images': [{
            'id': 0,
            'file_name': Path(image_path).name,
        }],
        'annotations': annotations,
    }

    with open(output_path, 'w') as f:
        json.dump(coco_output, f, indent=2)

    print(f"  COCO JSON saved: {output_path}")


def export_geojson(result, image_path, output_path):
    """Export detections as GeoJSON (requires rasterio for coordinate conversion)."""
    try:
        import rasterio
    except ImportError:
        print("  Skipping GeoJSON export (rasterio not installed)")
        return

    try:
        with rasterio.open(str(image_path)) as src:
            transform = src.transform
            crs = str(src.crs)
    except Exception as e:
        print(f"  Skipping GeoJSON export (not a GeoTIFF or no CRS): {e}")
        return

    features = []
    for pred in result.object_prediction_list:
        x1, y1, x2, y2 = pred.bbox.to_xyxy()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        lon, lat = rasterio.transform.xy(transform, cy, cx)

        features.append({
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [lon, lat],
            },
            'properties': {
                'confidence': round(pred.score.value, 4),
                'category': pred.category.name,
            },
        })

    geojson = {
        'type': 'FeatureCollection',
        'crs': {'type': 'name', 'properties': {'name': crs}},
        'features': features,
    }

    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    print(f"  GeoJSON saved: {output_path} ({len(features)} features)")


def export_visualization(result, output_dir):
    """Export annotated image with bounding boxes."""
    result.export_visuals(
        export_dir=str(output_dir),
        file_name='detection_result',
        rect_th=2,
        text_size=0.5,
        text_th=1,
    )
    print(f"  Visualization saved: {output_dir}/detection_result.png")


# ============================================================================
# Main
# ============================================================================

def main(args):
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_detection_model(args)

    # Run sliced inference
    result, elapsed = run_sliced_inference(image_path, model, args)

    # Optionally compare with standard inference
    if args.compare:
        std_result, std_elapsed = run_standard_inference(image_path, model)
        n_sliced = len(result.object_prediction_list)
        n_standard = len(std_result.object_prediction_list)
        print(f"\n  Comparison:")
        print(f"    Standard: {n_standard} detections in {std_elapsed:.2f}s")
        print(f"    Sliced:   {n_sliced} detections in {elapsed:.2f}s")
        print(f"    Improvement: +{n_sliced - n_standard} detections")

    # Export results
    image_name = image_path.name
    export_csv(result, image_name, output_dir / 'detections.csv')
    export_coco_json(result, image_path, output_dir / 'detections_coco.json')

    if args.geojson:
        export_geojson(result, image_path, output_dir / 'detections.geojson')

    if args.visualize:
        export_visualization(result, output_dir)

    # Summary
    print(f"\nDone. {len(result.object_prediction_list)} detections in {elapsed:.2f}s")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAHI Tiled Inference')

    # Input
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image (TIFF, JPG, PNG)')

    # Model
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to detection model weights')
    parser.add_argument('--model-type', type=str, default='yolov8',
                        choices=['yolov8', 'yolov5', 'detectron2', 'mmdet',
                                 'huggingface', 'torchvision'],
                        help='SAHI model type string')
    parser.add_argument('--confidence', type=float, default=0.3,
                        help='Confidence threshold (default: 0.3)')

    # SAHI parameters
    parser.add_argument('--slice-size', type=int, default=640,
                        help='Tile size in pixels (default: 640)')
    parser.add_argument('--overlap', type=float, default=0.25,
                        help='Overlap ratio (default: 0.25)')
    parser.add_argument('--postprocess-type', type=str, default='NMS',
                        choices=['NMS', 'NMM'],
                        help='Postprocess strategy (default: NMS)')
    parser.add_argument('--match-metric', type=str, default='IOS',
                        choices=['IOU', 'IOS'],
                        help='Match metric for postprocessing (default: IOS)')
    parser.add_argument('--match-threshold', type=float, default=0.5,
                        help='Match threshold for postprocessing (default: 0.5)')
    parser.add_argument('--include-standard', action='store_true', default=False,
                        help='Also run full-image prediction and merge results')

    # Output
    parser.add_argument('--output-dir', type=str, default='sahi_results/',
                        help='Output directory')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='Export annotated visualization')
    parser.add_argument('--geojson', action='store_true', default=False,
                        help='Export detections as GeoJSON (requires rasterio)')
    parser.add_argument('--compare', action='store_true', default=False,
                        help='Compare sliced vs. standard inference')

    # Device
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda:0, mps, cpu)')

    main(parser.parse_args())
