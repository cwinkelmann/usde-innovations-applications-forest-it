"""
MegaDetector Quickstart — Single Image + Batch Inference
Model: MDV5A (YOLOv5, auto-downloaded)
Output: Detection results as pandas DataFrame

Install: pip install megadetector
"""
from pathlib import Path
from PIL import Image
import pandas as pd
from megadetector.detection.run_detector import load_detector

# --- Config ---
MODEL = 'MDV5A'          # 'MDV5A', 'MDV5B', or path to .pt file
CONF_THRESHOLD = 0.2     # Typical for wildlife surveys (lower = more detections)

# --- Load model ---
detector = load_detector(MODEL)

# --- Single image ---
img_path = Path('path/to/image.jpg')
img = Image.open(img_path)
results = detector.generate_detections_one_batch(
    img_original=[img],
    image_id=[str(img_path)],
    detection_threshold=0.005  # Save all, filter later
)

# --- Extract results ---
# Bbox format: [x_min, y_min, width, height] NORMALIZED (0-1)
# Categories: '1'=animal, '2'=person, '3'=vehicle
rows = []
for det in results[0].get('detections', []):
    if det['conf'] >= CONF_THRESHOLD and det['category'] == '1':
        rows.append({
            'file': str(img_path),
            'confidence': det['conf'],
            'x_min': det['bbox'][0],
            'y_min': det['bbox'][1],
            'width': det['bbox'][2],
            'height': det['bbox'][3],
        })
df = pd.DataFrame(rows)
print(f"Found {len(df)} animals in {img_path.name}")
print(df)

# --- Batch processing (CLI, recommended for large jobs) ---
# python -m megadetector.detection.run_detector_batch \
#   MDV5A "image_folder/" "output.json" \
#   --output_relative_filenames --recursive --checkpoint_frequency 10000
