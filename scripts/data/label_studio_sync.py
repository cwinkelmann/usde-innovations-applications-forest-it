"""
label_studio_sync.py — programmatic upload/download for Label Studio
=====================================================================

Upload images (with optional pre-annotations) to a running Label Studio
instance, or download completed annotations as COCO JSON.

Requirements
------------
    pip install requests pillow

Label Studio must be running:
    label-studio start          # http://localhost:8080

Get your API token from:
    http://localhost:8080/user/account  →  Access Token

Usage
-----
    # Upload Serengeti images (no pre-annotations)
    python label_studio_sync.py upload \\
        --images week1/data/ls_serengeti \\
        --project "Serengeti Boxes" \\
        --config bbox

    # Upload Caltech images WITH existing COCO bounding boxes as pre-annotations
    python label_studio_sync.py upload \\
        --images week1/data/ls_caltech \\
        --annotations week1/data/camera_trap_labels.csv \\
        --annotation-format caltech-csv \\
        --project "Caltech Polygons" \\
        --config polygon

    # Upload Eikelboom images WITH Pascal VOC annotations as pre-annotations
    python label_studio_sync.py upload \\
        --images /path/to/eikelboom/train \\
        --annotations /path/to/eikelboom/annotations_train.csv \\
        --annotation-format eikelboom-csv \\
        --project "Eikelboom Train" \\
        --config bbox

    # Export completed annotations as COCO JSON
    python label_studio_sync.py export \\
        --project "Serengeti Boxes" \\
        --output week1/data/my_serengeti_bboxes.json

Authentication
--------------
Pass --token or set the LS_TOKEN environment variable.
"""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

import requests
from PIL import Image

# ── Label configs ──────────────────────────────────────────────────────────────

LABEL_CONFIGS = {
    "bbox": """<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <RectangleLabels name="label" toName="image">
    <Label value="animal"  background="#E74C3C"/>
    <Label value="vehicle" background="#3498DB"/>
    <Label value="person"  background="#2ECC71"/>
  </RectangleLabels>
</View>""",
    "polygon": """<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <PolygonLabels name="label" toName="image" strokeWidth="2">
    <Label value="animal"  background="#8E44AD"/>
    <Label value="habitat" background="#27AE60"/>
  </PolygonLabels>
</View>""",
}


# ── Coordinate conversions ─────────────────────────────────────────────────────

def coco_bbox_to_ls(x, y, w, h, img_w, img_h, label, from_name="label", to_name="image"):
    """COCO pixel [x,y,w,h] → Label Studio percentage rectangle."""
    return {
        "id": uuid.uuid4().hex[:8],
        "type": "rectanglelabels",
        "from_name": from_name,
        "to_name": to_name,
        "original_width": img_w,
        "original_height": img_h,
        "value": {
            "x":      x / img_w * 100,
            "y":      y / img_h * 100,
            "width":  w / img_w * 100,
            "height": h / img_h * 100,
            "rotation": 0,
            "rectanglelabels": [label],
        },
    }


def voc_bbox_to_ls(x1, y1, x2, y2, img_w, img_h, label, from_name="label", to_name="image"):
    """Pascal VOC pixel [x1,y1,x2,y2] → Label Studio percentage rectangle."""
    return coco_bbox_to_ls(x1, y1, x2 - x1, y2 - y1, img_w, img_h, label, from_name, to_name)


def coco_polygon_to_ls(flat_pts, img_w, img_h, label, from_name="label", to_name="image"):
    """COCO flat polygon [x1,y1,x2,y2,...] → Label Studio percentage polygon."""
    pts = list(zip(flat_pts[::2], flat_pts[1::2]))
    return {
        "id": uuid.uuid4().hex[:8],
        "type": "polygonlabels",
        "from_name": from_name,
        "to_name": to_name,
        "original_width": img_w,
        "original_height": img_h,
        "value": {
            "points": [[x / img_w * 100, y / img_h * 100] for x, y in pts],
            "polygonlabels": [label],
            "closed": True,
        },
    }


# ── Annotation loaders ─────────────────────────────────────────────────────────

def load_caltech_csv(csv_path: Path) -> dict:
    """
    Load camera_trap_labels.csv (columns: crop, true_label, bbox_x, bbox_y, bbox_w, bbox_h).
    Returns: {filename: [{"bbox": [x,y,w,h], "label": species}, ...]}
    """
    import csv
    result: dict = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if not row.get("bbox_x") or row["bbox_x"] in ("", "None"):
                continue
            fname = Path(row["crop"]).name
            result.setdefault(fname, []).append({
                "bbox":  [float(row["bbox_x"]), float(row["bbox_y"]),
                          float(row["bbox_w"]), float(row["bbox_h"])],
                "label": row["true_label"].lower(),
            })
    return result


def load_eikelboom_csv(csv_path: Path) -> dict:
    """
    Load Eikelboom annotations_*.csv (no header: filename,x1,y1,x2,y2,class).
    Returns: {filename: [{"voc": [x1,y1,x2,y2], "label": class}, ...]}
    """
    import csv
    result: dict = {}
    with open(csv_path) as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            fname = Path(row[0]).name
            result.setdefault(fname, []).append({
                "voc":   [float(row[1]), float(row[2]), float(row[3]), float(row[4])],
                "label": row[5].lower(),
            })
    return result


def load_coco_json(json_path: Path) -> dict:
    """
    Load COCO JSON.
    Returns: {filename: [{"bbox": [x,y,w,h], "label": name}, ...]}
    """
    with open(json_path) as f:
        data = json.load(f)
    id_to_file = {img["id"]: Path(img["file_name"]).name for img in data["images"]}
    id_to_cat  = {c["id"]: c["name"] for c in data["categories"]}
    result: dict = {}
    for ann in data["annotations"]:
        fname = id_to_file.get(ann["image_id"])
        if not fname:
            continue
        result.setdefault(fname, []).append({
            "bbox":        ann.get("bbox"),
            "segmentation": ann.get("segmentation"),
            "label":       id_to_cat.get(ann["category_id"], "animal"),
        })
    return result


# ── Label Studio API helpers ───────────────────────────────────────────────────

def make_session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"Authorization": f"Token {token}"})
    return s


def get_or_create_project(session: requests.Session, url: str, name: str, config: str) -> int:
    """Return project ID, creating the project if it doesn't exist."""
    r = session.get(f"{url}/api/projects")
    r.raise_for_status()
    for p in r.json().get("results", []):
        if p["title"] == name:
            print(f"  Project '{name}' already exists (id={p['id']})")
            return p["id"]

    r = session.post(f"{url}/api/projects", json={"title": name, "label_config": config})
    r.raise_for_status()
    pid = r.json()["id"]
    print(f"  Created project '{name}' (id={pid})")
    return pid


def upload_image(session: requests.Session, url: str, project_id: int, image_path: Path) -> dict:
    """
    Upload one image file to Label Studio and return the created task.
    POST /api/projects/{id}/import with multipart form.
    """
    with open(image_path, "rb") as f:
        r = session.post(
            f"{url}/api/projects/{project_id}/import",
            files={"file": (image_path.name, f, "image/jpeg")},
        )
    r.raise_for_status()
    data = r.json()
    tasks = data.get("tasks", [])
    if not tasks:
        raise RuntimeError(f"Upload returned no task for {image_path.name}: {data}")
    return tasks[0]  # {"id": ..., "data": {"image": "/data/upload/..."}}


def add_predictions(
    session: requests.Session,
    url: str,
    task_id: int,
    result: list,
    model_version: str = "pre-annotation",
) -> None:
    """POST /api/predictions — attach pre-annotations to a task."""
    r = session.post(
        f"{url}/api/predictions",
        json={"task": task_id, "result": result, "score": 1.0, "model_version": model_version},
    )
    r.raise_for_status()


def get_image_size(image_path: Path) -> tuple:
    with Image.open(image_path) as img:
        return img.width, img.height


# ── Upload command ─────────────────────────────────────────────────────────────

def cmd_upload(args):
    token = args.token or os.environ.get("LS_TOKEN")
    if not token:
        sys.exit("No token — pass --token or set LS_TOKEN environment variable.\n"
                 "Get it from http://localhost:8080/user/account")

    session = make_session(token)
    url     = args.url.rstrip("/")

    # Verify connection
    r = session.get(f"{url}/api/projects")
    if not r.ok:
        sys.exit(f"Cannot connect to Label Studio at {url} (HTTP {r.status_code}).\n"
                 "Is it running?  label-studio start")

    label_config = LABEL_CONFIGS[args.config]
    project_id   = get_or_create_project(session, url, args.project, label_config)

    # Load annotations index if provided
    ann_index: dict = {}
    if args.annotations:
        ann_path = Path(args.annotations)
        fmt      = args.annotation_format
        print(f"  Loading annotations from {ann_path.name} (format: {fmt})")
        if fmt == "caltech-csv":
            ann_index = load_caltech_csv(ann_path)
        elif fmt == "eikelboom-csv":
            ann_index = load_eikelboom_csv(ann_path)
        elif fmt == "coco-json":
            ann_index = load_coco_json(ann_path)
        else:
            sys.exit(f"Unknown annotation format: {fmt}")
        print(f"  Loaded annotations for {len(ann_index)} files")

    # Upload images
    images_dir = Path(args.images)
    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    print(f"\nUploading {len(image_files)} images to project '{args.project}'...")

    n_uploaded = 0
    n_annotated = 0

    for img_path in image_files:
        print(f"  {img_path.name}", end="", flush=True)

        task = upload_image(session, url, project_id, img_path)
        task_id = task["id"]
        n_uploaded += 1

        # Attach pre-annotations if available
        anns = ann_index.get(img_path.name, [])
        if anns:
            img_w, img_h = get_image_size(img_path)
            result = []
            for ann in anns:
                if ann.get("bbox"):
                    x, y, w, h = ann["bbox"]
                    result.append(coco_bbox_to_ls(x, y, w, h, img_w, img_h, ann["label"]))
                elif ann.get("voc"):
                    x1, y1, x2, y2 = ann["voc"]
                    result.append(voc_bbox_to_ls(x1, y1, x2, y2, img_w, img_h, ann["label"]))
                elif ann.get("segmentation"):
                    for seg in ann["segmentation"]:
                        result.append(coco_polygon_to_ls(seg, img_w, img_h, ann["label"]))

            if result:
                add_predictions(session, url, task_id, result)
                n_annotated += 1
                print(f"  → {len(result)} pre-annotations", end="")

        print()

    print(f"\nDone — {n_uploaded} tasks created, {n_annotated} with pre-annotations")
    print(f"Open: {url}/projects/{project_id}/")


# ── Export command ─────────────────────────────────────────────────────────────

def cmd_export(args):
    token = args.token or os.environ.get("LS_TOKEN")
    if not token:
        sys.exit("No token — pass --token or set LS_TOKEN environment variable.")

    session = make_session(token)
    url     = args.url.rstrip("/")

    # Find project by name
    r = session.get(f"{url}/api/projects")
    r.raise_for_status()
    project = next((p for p in r.json().get("results", []) if p["title"] == args.project), None)
    if not project:
        sys.exit(f"Project '{args.project}' not found. Available: "
                 + ", ".join(p["title"] for p in r.json().get("results", [])))

    project_id = project["id"]
    fmt        = args.format.upper()

    print(f"Exporting project '{args.project}' (id={project_id}) as {fmt}...")
    r = session.get(f"{url}/api/projects/{project_id}/export", params={"exportType": fmt})
    r.raise_for_status()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(r.content)

    if fmt == "JSON" or fmt == "COCO":
        data = json.loads(r.content)
        if isinstance(data, list):
            print(f"  {len(data)} tasks exported → {out_path}")
        elif isinstance(data, dict) and "annotations" in data:
            print(f"  {len(data['annotations'])} annotations, "
                  f"{len(data['images'])} images → {out_path}")
    else:
        print(f"  {len(r.content):,} bytes → {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Upload images + annotations to Label Studio, or export results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--url",   default="http://localhost:8080", help="Label Studio URL")
    parser.add_argument("--token", default=None, help="API token (or set LS_TOKEN env var)")

    sub = parser.add_subparsers(dest="command", required=True)

    # upload
    up = sub.add_parser("upload", help="Upload images (optionally with pre-annotations)")
    up.add_argument("--images",    required=True,   help="Folder of images to upload")
    up.add_argument("--project",   required=True,   help="Label Studio project name")
    up.add_argument("--config",    default="bbox",  choices=list(LABEL_CONFIGS),
                    help="Label config template (default: bbox)")
    up.add_argument("--annotations",        default=None, help="Annotation file (optional)")
    up.add_argument("--annotation-format",  default="coco-json",
                    choices=["coco-json", "caltech-csv", "eikelboom-csv"],
                    help="Format of --annotations file (default: coco-json)")

    # export
    ex = sub.add_parser("export", help="Export completed annotations")
    ex.add_argument("--project",  required=True, help="Label Studio project name")
    ex.add_argument("--output",   required=True, help="Output file path (.json)")
    ex.add_argument("--format",   default="COCO", choices=["COCO", "JSON", "CSV"],
                    help="Export format (default: COCO)")

    args = parser.parse_args()

    if args.command == "upload":
        cmd_upload(args)
    elif args.command == "export":
        cmd_export(args)


if __name__ == "__main__":
    main()
