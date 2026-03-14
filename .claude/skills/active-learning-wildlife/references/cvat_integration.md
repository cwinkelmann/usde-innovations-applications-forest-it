# CVAT Integration for Active Learning

## Overview

CVAT (Computer Vision Annotation Tool) is the recommended annotation tool for
detection-focused active learning workflows. It provides robust bounding box
and point annotation capabilities with a Python SDK for automation.

**Repository:** https://github.com/cvat-ai/cvat
**License:** MIT

---

## Installation

### CVAT Server (Docker)

```bash
git clone https://github.com/cvat-ai/cvat.git
cd cvat
docker compose up -d

# Create admin user
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
# Default: http://localhost:8080
```

### Python SDK

```bash
pip install cvat-sdk
```

---

## SDK Usage

### Authentication

```python
from cvat_sdk import make_client

# Basic auth
client = make_client(
    host='http://localhost:8080',
    credentials=('admin', 'password'),
)

# API key auth
client = make_client(
    host='http://localhost:8080',
    credentials=('admin', 'password'),
)
```

### Project Management

```python
# Create a project
project = client.projects.create(spec={
    'name': 'Marine Iguana Detection',
    'labels': [
        {'name': 'iguana', 'type': 'rectangle'},
        {'name': 'ignore', 'type': 'rectangle'},
    ],
})

# List projects
projects = client.projects.list()
for p in projects:
    print(f"  {p.id}: {p.name}")

# Get project by ID
project = client.projects.retrieve(project_id)
```

### Task Management

```python
# Create a task with images
task = client.tasks.create_from_data(
    spec={
        'name': 'AL Round 1',
        'project_id': project.id,
    },
    resource_type='local',
    resources=[
        '/path/to/image1.jpg',
        '/path/to/image2.jpg',
        '/path/to/image3.jpg',
    ],
)

# List tasks in a project
tasks = client.tasks.list(project_id=project.id)

# Get task status
task = client.tasks.retrieve(task_id)
print(f"Task {task.id}: {task.status}")
```

### Importing Pre-Annotations

```python
import json
import tempfile

def import_coco_annotations(client, task_id, coco_dict):
    """Import COCO-format annotations into a CVAT task."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(coco_dict, f)
        temp_path = f.name

    task = client.tasks.retrieve(task_id)
    task.import_annotations(
        format_name='COCO 1.0',
        filename=temp_path,
    )
    print(f"Imported annotations to task {task_id}")
```

### Exporting Annotations

```python
def export_annotations(client, task_id, output_path, format_name='COCO 1.0'):
    """Export annotations from a CVAT task."""
    task = client.tasks.retrieve(task_id)
    task.export_dataset(
        format_name=format_name,
        filename=str(output_path),
    )
    print(f"Exported annotations to {output_path}")
```

**Supported export formats:**
- `COCO 1.0` — COCO JSON (recommended)
- `YOLO 1.1` — YOLO txt files
- `Pascal VOC 1.1` — Pascal VOC XML
- `CVAT for images 1.1` — CVAT native XML
- `Datumaro 1.0` — Intel Datumaro format

---

## Active Learning Workflow with CVAT

### Complete Round Function

```python
from pathlib import Path
import json
from cvat_sdk import make_client


def run_cvat_annotation_round(
    cvat_url: str,
    credentials: tuple,
    project_id: int,
    round_num: int,
    image_paths: list,
    predictions: dict,
    labels: list,
    output_dir: str,
):
    """Execute one annotation round using CVAT.

    Args:
        cvat_url: CVAT server URL
        credentials: (username, password)
        project_id: CVAT project ID
        round_num: Active learning round number
        image_paths: List of image paths to annotate
        predictions: Dict mapping image filename to list of prediction dicts
        labels: List of label names
        output_dir: Where to save exported annotations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with make_client(cvat_url, credentials=credentials) as client:
        # 1. Create task
        task = client.tasks.create_from_data(
            spec={
                'name': f'AL_round_{round_num}',
                'project_id': project_id,
            },
            resource_type='local',
            resources=[str(p) for p in image_paths],
        )
        print(f"Created task {task.id} with {len(image_paths)} images")

        # 2. Build COCO pre-annotations
        coco = build_coco_preannotations(image_paths, predictions, labels)

        # 3. Import pre-annotations
        import_coco_annotations(client, task.id, coco)

        # 4. Wait for annotation
        print(f"\nAnnotation task ready at: {cvat_url}/tasks/{task.id}")
        print("Please review and correct annotations in CVAT.")
        input("Press Enter when annotation is complete...")

        # 5. Export corrected annotations
        export_path = output_dir / f'round_{round_num}_annotations.json'
        export_annotations(client, task.id, export_path)

        # 6. Parse and return
        with open(export_path) as f:
            corrected = json.load(f)

        n_annotations = len(corrected.get('annotations', []))
        print(f"Downloaded {n_annotations} corrected annotations")

        return corrected


def build_coco_preannotations(image_paths, predictions, labels):
    """Build COCO-format JSON from model predictions."""
    label_to_id = {name: i for i, name in enumerate(labels)}

    images = []
    annotations = []
    ann_id = 0

    for img_id, img_path in enumerate(image_paths):
        img_name = Path(img_path).name
        images.append({'id': img_id, 'file_name': img_name})

        if img_name in predictions:
            for pred in predictions[img_name]:
                x1, y1, x2, y2 = pred['bbox']
                w, h = x2 - x1, y2 - y1
                annotations.append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': label_to_id.get(pred.get('class', 'animal'), 0),
                    'bbox': [x1, y1, w, h],
                    'area': w * h,
                    'iscrowd': 0,
                })
                ann_id += 1

    return {
        'images': images,
        'annotations': annotations,
        'categories': [{'id': i, 'name': n} for i, n in enumerate(labels)],
    }
```

---

## CVAT Annotation Guidelines for Wildlife

### Bounding Box Rules

1. **Tight boxes**: Draw boxes that tightly enclose the visible animal body
2. **Occluded animals**: Include the visible portion only (do not guess hidden parts)
3. **Overlapping animals**: Each animal gets its own box, even if boxes overlap
4. **Partially cut animals**: If >50% visible at image edge, annotate. Otherwise skip.
5. **Ambiguous detections**: Use the "ignore" label for uncertain cases

### Point Annotation Rules (for HerdNet)

1. **Center point**: Place the point at the animal's body center
2. **One point per animal**: Even for overlapping animals
3. **Dense clusters**: Zoom in to ensure individual counting

### Quality Control

- Review 10% of annotations for consistency
- Use CVAT's review mode for multi-annotator workflows
- Track inter-annotator agreement for quality metrics
