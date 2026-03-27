"""
MegaDetector → Species Classifier Pipeline
1. Detect animals with MegaDetector
2. Crop bounding boxes
3. Classify species with timm model
4. Merge results
"""
from pathlib import Path
from PIL import Image
import torch
import timm
from timm.data import resolve_data_config, create_transform
import pandas as pd
from megadetector.detection.run_detector import load_detector

# --- Config ---
MD_MODEL = 'MDV5A'
MD_THRESHOLD = 0.2
CLASSIFIER_MODEL = 'vit_base_patch14_dinov2.lvd142m'  # or your fine-tuned checkpoint
CLASSIFIER_CHECKPOINT = None  # Path to fine-tuned weights, or None for pretrained
NUM_CLASSES = 34  # Number of species classes
CROP_PADDING = 0.1  # 10% padding around detection box
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load models ---
detector = load_detector(MD_MODEL)

classifier = timm.create_model(CLASSIFIER_MODEL, pretrained=True, num_classes=NUM_CLASSES)
if CLASSIFIER_CHECKPOINT:
    state = torch.load(CLASSIFIER_CHECKPOINT, map_location=DEVICE, weights_only=False)
    classifier.load_state_dict(state['state_dict'] if 'state_dict' in state else state)
classifier = classifier.to(DEVICE).eval()

data_config = resolve_data_config(classifier.pretrained_cfg)
transform = create_transform(**data_config, is_training=False)

# CLASS_NAMES: list of species names matching model output indices
CLASS_NAMES = [f'species_{i}' for i in range(NUM_CLASSES)]  # Replace with actual names


def crop_detection(img: Image.Image, bbox: list, padding: float = 0.1) -> Image.Image:
    """Crop animal from MD bbox [x_min, y_min, w, h] normalized."""
    w, h = img.size
    x1 = bbox[0] * w
    y1 = bbox[1] * h
    x2 = (bbox[0] + bbox[2]) * w
    y2 = (bbox[1] + bbox[3]) * h
    pad_w = (x2 - x1) * padding
    pad_h = (y2 - y1) * padding
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)
    return img.crop((x1, y1, x2, y2))


def process_image(img_path: Path) -> list[dict]:
    """Run detection + classification on one image."""
    img = Image.open(img_path).convert('RGB')
    detections = detector.generate_detections_one_batch(
        img_original=[img], image_id=[str(img_path)], detection_threshold=0.005
    )

    results = []
    for det in detections[0].get('detections', []):
        if det['conf'] < MD_THRESHOLD or det['category'] != '1':
            continue

        crop = crop_detection(img, det['bbox'], CROP_PADDING)
        if min(crop.size) < 32:
            continue  # Skip tiny crops

        tensor = transform(crop).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = classifier(tensor)
            probs = torch.softmax(logits, dim=1)
            species_idx = probs.argmax(dim=1).item()
            species_conf = probs[0, species_idx].item()

        results.append({
            'file': str(img_path),
            'md_confidence': det['conf'],
            'bbox': det['bbox'],
            'species': CLASS_NAMES[species_idx],
            'species_confidence': species_conf,
        })
    return results


# --- Run pipeline ---
image_dir = Path('path/to/images')
all_results = []
for img_path in sorted(image_dir.glob('*.jpg')):
    all_results.extend(process_image(img_path))

df = pd.DataFrame(all_results)
df.to_csv('detection_classification_results.csv', index=False)
print(f"Processed {len(df)} detections across {df['file'].nunique()} images")
