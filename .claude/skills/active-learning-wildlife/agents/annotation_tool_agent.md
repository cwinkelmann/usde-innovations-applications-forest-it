# Annotation Tool Agent — Active Learning Wildlife

## Role

You are the **Annotation Tool Agent**. You handle all interactions with annotation
platforms (CVAT and Label Studio), including project creation, image upload,
pre-annotation import, and corrected annotation download.

## Activation

Activate this agent when the user:
- Needs to set up CVAT or Label Studio for active learning
- Wants to export predictions to an annotation tool as pre-annotations
- Needs to download corrected annotations after expert review
- Is choosing between CVAT and Label Studio

---

## Tool Comparison

| Feature | CVAT | Label Studio |
|---------|------|--------------|
| **Installation** | Docker (docker compose) | pip install / Docker |
| **Annotation types** | Boxes, polygons, points, tracks | Boxes, polygons, points, text, audio |
| **SDK** | `cvat-sdk` | `label-studio-sdk` |
| **Pre-annotation import** | CVAT XML, COCO JSON | JSON predictions format |
| **Export formats** | COCO, YOLO, Pascal VOC, CSV | COCO, YOLO, Pascal VOC, CSV |
| **Multi-user** | Yes (built-in) | Yes (with teams) |
| **Video support** | Strong (frame tracking) | Limited |
| **Best for** | Detection boxes, team annotation | Mixed annotation types, flexible workflows |

### Recommendation
- **Detection tasks (boxes/points):** CVAT — optimized for geometric annotations
- **Mixed tasks (boxes + text labels):** Label Studio — more flexible
- **Team annotation projects:** CVAT — better multi-user management
- **Quick setup (single user):** Label Studio — `pip install label-studio && label-studio start`

---

## CVAT Integration

### Setup

```bash
# Install CVAT (Docker)
git clone https://github.com/cvat-ai/cvat
cd cvat
docker compose up -d

# Install Python SDK
pip install cvat-sdk
```

### Creating a Project and Task

```python
from cvat_sdk import make_client
from cvat_sdk.api_client.model_utils import to_json


def create_cvat_project(cvat_url, credentials, project_name, labels):
    """Create a CVAT project with specified labels."""
    with make_client(cvat_url, credentials=credentials) as client:
        project = client.projects.create(
            spec={
                'name': project_name,
                'labels': [{'name': label} for label in labels],
            }
        )
        print(f"Created CVAT project: {project.name} (ID: {project.id})")
        return project.id


def create_cvat_task(cvat_url, credentials, project_id, task_name, image_paths):
    """Create a CVAT task and upload images."""
    with make_client(cvat_url, credentials=credentials) as client:
        task = client.tasks.create_from_data(
            spec={
                'name': task_name,
                'project_id': project_id,
            },
            resource_type='local',
            resources=image_paths,
        )
        print(f"Created CVAT task: {task.name} (ID: {task.id}), {len(image_paths)} images")
        return task.id
```

### Uploading Pre-Annotations

Pre-annotations show the model's predictions in CVAT so experts can correct them
rather than annotate from scratch.

```python
import json
import tempfile
from pathlib import Path


def upload_preannotations_cvat(cvat_url, credentials, task_id, predictions,
                                 image_names, labels):
    """Upload model predictions as pre-annotations to a CVAT task.

    Args:
        predictions: dict mapping image_name -> list of {bbox: [x1,y1,x2,y2], class: str, score: float}
        image_names: list of image filenames in task order
        labels: list of label names
    """
    # Build COCO-format annotations
    images = []
    annotations = []
    ann_id = 0

    label_to_id = {name: i for i, name in enumerate(labels)}

    for img_id, img_name in enumerate(image_names):
        images.append({'id': img_id, 'file_name': img_name})

        if img_name in predictions:
            for pred in predictions[img_name]:
                x1, y1, x2, y2 = pred['bbox']
                w, h = x2 - x1, y2 - y1
                annotations.append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': label_to_id.get(pred['class'], 0),
                    'bbox': [x1, y1, w, h],
                    'area': w * h,
                    'iscrowd': 0,
                })
                ann_id += 1

    coco = {
        'images': images,
        'annotations': annotations,
        'categories': [{'id': i, 'name': n} for i, n in enumerate(labels)],
    }

    # Write to temp file and upload
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(coco, f)
        coco_path = f.name

    with make_client(cvat_url, credentials=credentials) as client:
        client.tasks.retrieve(task_id).import_annotations(
            format_name='COCO 1.0',
            filename=coco_path,
        )

    print(f"Uploaded {ann_id} pre-annotations to task {task_id}")
```

### Downloading Corrected Annotations

```python
def download_cvat_annotations(cvat_url, credentials, task_id, output_path,
                                format_name='COCO 1.0'):
    """Download corrected annotations from CVAT."""
    with make_client(cvat_url, credentials=credentials) as client:
        task = client.tasks.retrieve(task_id)
        task.export_dataset(
            format_name=format_name,
            filename=str(output_path),
        )

    print(f"Downloaded annotations to {output_path}")

    # Parse COCO JSON
    with open(output_path) as f:
        coco = json.load(f)

    n_images = len(coco.get('images', []))
    n_annotations = len(coco.get('annotations', []))
    print(f"  {n_images} images, {n_annotations} annotations")

    return coco
```

---

## Label Studio Integration

### Setup

```bash
# Quick install
pip install label-studio
label-studio start  # Opens at http://localhost:8080

# Or with Docker
docker run -p 8080:8080 heartexlabs/label-studio

# Install SDK
pip install label-studio-sdk
```

### Creating a Project

```python
from label_studio_sdk import Client


def create_ls_project(ls_url, api_key, project_title, labels):
    """Create a Label Studio project for object detection."""
    client = Client(url=ls_url, api_key=api_key)

    # Build labeling config for bounding box annotation
    label_options = '\n'.join(
        f'    <RectangleLabels name="label" toName="image">\n'
        + '\n'.join(f'      <Label value="{label}"/>' for label in labels)
        + '\n    </RectangleLabels>'
    )

    labeling_config = f"""
    <View>
      <Image name="image" value="$image"/>
      {label_options}
    </View>
    """

    project = client.start_project(
        title=project_title,
        label_config=labeling_config,
    )

    print(f"Created Label Studio project: {project_title} (ID: {project.id})")
    return project


def upload_images_ls(project, image_paths):
    """Upload images to a Label Studio project."""
    tasks = []
    for img_path in image_paths:
        tasks.append({'data': {'image': f'/data/local-files/?d={img_path}'}})

    project.import_tasks(tasks)
    print(f"Uploaded {len(tasks)} images to Label Studio")
```

### Uploading Pre-Annotations

```python
def upload_preannotations_ls(project, predictions, image_width, image_height):
    """Upload model predictions as pre-annotations to Label Studio.

    Label Studio uses percentage-based coordinates (0-100).

    Args:
        predictions: list of dicts, each with 'image_path', 'detections' list
            where each detection has 'bbox' [x1,y1,x2,y2] in pixels, 'class', 'score'
    """
    tasks = []
    for pred_item in predictions:
        results = []
        for det in pred_item['detections']:
            x1, y1, x2, y2 = det['bbox']
            # Convert pixel coords to percentage (Label Studio format)
            results.append({
                'type': 'rectanglelabels',
                'value': {
                    'x': (x1 / image_width) * 100,
                    'y': (y1 / image_height) * 100,
                    'width': ((x2 - x1) / image_width) * 100,
                    'height': ((y2 - y1) / image_height) * 100,
                    'rectanglelabels': [det['class']],
                },
                'score': det['score'],
                'from_name': 'label',
                'to_name': 'image',
                'original_width': image_width,
                'original_height': image_height,
            })

        tasks.append({
            'data': {'image': pred_item['image_path']},
            'predictions': [{'result': results}],
        })

    project.import_tasks(tasks)
    print(f"Uploaded {len(tasks)} tasks with pre-annotations")
```

### Downloading Corrected Annotations

```python
def download_ls_annotations(project, output_path, export_type='COCO'):
    """Download corrected annotations from Label Studio."""
    # Export annotations
    export = project.export_tasks(export_type=export_type)

    with open(output_path, 'w') as f:
        json.dump(export, f, indent=2)

    print(f"Downloaded annotations to {output_path}")
    return export
```

---

## Format Conversion Utilities

### COCO to YOLO

```python
def coco_to_yolo(coco_json_path, output_dir, image_width, image_height):
    """Convert COCO annotations to YOLO format (one .txt per image)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json_path) as f:
        coco = json.load(f)

    # Build image_id -> filename mapping
    id_to_file = {img['id']: img['file_name'] for img in coco['images']}

    # Group annotations by image
    annotations_by_image = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    for img_id, anns in annotations_by_image.items():
        filename = id_to_file[img_id]
        txt_path = output_dir / (Path(filename).stem + '.txt')

        with open(txt_path, 'w') as f:
            for ann in anns:
                x, y, w, h = ann['bbox']
                # YOLO format: class cx cy w h (normalized 0-1)
                cx = (x + w / 2) / image_width
                cy = (y + h / 2) / image_height
                nw = w / image_width
                nh = h / image_height
                f.write(f"{ann['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    print(f"Converted {len(annotations_by_image)} images to YOLO format in {output_dir}")
```

### COCO to CSV (for HerdNet point annotations)

```python
def coco_to_csv_points(coco_json_path, output_csv_path):
    """Convert COCO bbox annotations to CSV point annotations (bbox center).

    Output format: image_name, x, y, label
    This is the format expected by HerdNet/animaloc.
    """
    with open(coco_json_path) as f:
        coco = json.load(f)

    id_to_file = {img['id']: img['file_name'] for img in coco['images']}
    id_to_cat = {cat['id']: cat['name'] for cat in coco['categories']}

    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['images', 'x', 'y', 'labels'])

        for ann in coco['annotations']:
            x, y, w, h = ann['bbox']
            cx = x + w / 2
            cy = y + h / 2
            writer.writerow([
                id_to_file[ann['image_id']],
                f"{cx:.1f}",
                f"{cy:.1f}",
                id_to_cat.get(ann['category_id'], 'animal'),
            ])

    print(f"Converted to CSV points: {output_csv_path}")
```

---

## Complete Active Learning Round: Annotation Tool Flow

```python
def annotation_round(config, selected_images, predictions, round_num):
    """Execute one complete annotation tool round."""
    tool = config['annotation_tool']

    if tool == 'cvat':
        # Create task
        task_id = create_cvat_task(
            config['cvat_url'], config['cvat_credentials'],
            config['project_id'], f'AL_round_{round_num}',
            [str(p) for p in selected_images],
        )
        # Upload pre-annotations
        upload_preannotations_cvat(
            config['cvat_url'], config['cvat_credentials'],
            task_id, predictions,
            [p.name for p in selected_images], config['labels'],
        )
        print(f"\nImages uploaded to CVAT task {task_id}.")
        print("Please review and correct annotations in CVAT.")
        input("Press Enter when annotation is complete...")

        # Download
        output_path = Path(config['output_dir']) / f'round_{round_num}_annotations.json'
        coco = download_cvat_annotations(
            config['cvat_url'], config['cvat_credentials'],
            task_id, output_path,
        )
        return coco

    elif tool == 'label_studio':
        # Similar flow with Label Studio SDK
        pass
```
