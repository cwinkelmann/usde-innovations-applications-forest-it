"""
label_studio_sync.py — programmatic upload/download for Label Studio
=====================================================================

Upload images (with optional pre-annotations) to a running Label Studio
instance, or download completed annotations as COCO JSON.

Requirements
------------
    pip install label-studio-sdk pillow

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
import csv
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


def get_image_size(image_path: Path) -> tuple:
    with Image.open(image_path) as img:
        return img.width, img.height


def make_session(token: str, url: str = "http://localhost:8080") -> requests.Session:
    """Return an authenticated requests.Session for Label Studio.

    Handles both token formats:

    - Static API key (hex string) → ``Authorization: Token <key>``
    - JWT refresh token (starts with ``eyJ``, ``token_type=refresh``) →
      exchanges for a short-lived access token via ``/api/token/refresh``,
      then uses ``Authorization: Bearer <access_token>``.
      The session automatically re-exchanges on 401.
    """
    if not token.startswith("eyJ"):
        session = requests.Session()
        session.headers["Authorization"] = f"Token {token}"
        return session

    # JWT path — exchange refresh token for access token
    return _JWTSession(refresh_token=token, url=url)


class _JWTSession(requests.Session):
    """requests.Session that auto-refreshes expired JWT access tokens."""

    def __init__(self, refresh_token: str, url: str):
        super().__init__()
        self._refresh_token = refresh_token
        self._url = url.rstrip("/")
        self._exchange()

    def _exchange(self):
        r = requests.post(
            f"{self._url}/api/token/refresh",
            json={"refresh": self._refresh_token},
        )
        r.raise_for_status()
        access = r.json()["access"]
        self.headers["Authorization"] = f"Bearer {access}"

    def request(self, method, url, **kwargs):
        resp = super().request(method, url, **kwargs)
        if resp.status_code == 401:
            self._exchange()
            resp = super().request(method, url, **kwargs)
        return resp


# ── High-level project wrapper ────────────────────────────────────────────────

#: Default MegaDetector category mapping (category id → label name)
MD_CATEGORIES = {"1": "animal", "2": "person", "3": "vehicle"}


class LabelStudioProject:
    """Thin wrapper around a single Label Studio project.

    Typical notebook usage::

        from wildlife_detection.label_studio import LabelStudioProject, make_session

        session = make_session(LS_TOKEN, url=LS_URL)
        proj = LabelStudioProject.get_or_create(session, LS_URL, "Serengeti Review")

        proj.upload_with_megadetector(image_dir, md_results)
        # … annotate in the browser …
        coco = proj.export(output_path)
        print(proj.open_url())
    """

    def __init__(self, session: requests.Session, url: str, project_id: int, title: str):
        self.session = session
        self.url = url.rstrip("/")
        self.id = project_id
        self.title = title

    # ── Constructor ────────────────────────────────────────────────────────────

    @classmethod
    def get_or_create(
        cls,
        session: requests.Session,
        url: str,
        title: str,
        config: str = "bbox",
    ) -> "LabelStudioProject":
        """Return the named project, creating it if it does not exist.

        ``config`` is a key from :data:`LABEL_CONFIGS` (``"bbox"`` or ``"polygon"``).
        """
        url = url.rstrip("/")
        r = session.get(f"{url}/api/projects")
        r.raise_for_status()
        for p in r.json().get("results", []):
            if p["title"] == title:
                print(f"Using existing project '{title}' (id={p['id']})")
                return cls(session, url, p["id"], title)

        r = session.post(f"{url}/api/projects", json={
            "title": title,
            "label_config": LABEL_CONFIGS[config],
        })
        r.raise_for_status()
        pid = r.json()["id"]
        print(f"Created project '{title}' (id={pid})")
        return cls(session, url, pid, title)

    # ── Upload ─────────────────────────────────────────────────────────────────

    def upload_with_megadetector(
        self,
        image_dir: Path,
        md_results: list,
        categories: dict = None,
        conf_threshold: float = 0.1,
    ) -> None:
        """Upload images and attach MegaDetector detections as pre-annotations.

        Args:
            image_dir:       Directory that contains the image files.
            md_results:      List of MegaDetector result dicts
                             (``{"file": "name.jpg", "detections": [...]}``)
            categories:      Map from MegaDetector category id string to label name.
                             Defaults to ``{"1": "animal", "2": "person", "3": "vehicle"}``.
            conf_threshold:  Detections below this confidence are skipped.
        """
        if categories is None:
            categories = MD_CATEGORIES
        image_dir = Path(image_dir)

        n_uploaded = n_annotated = 0
        for result in md_results:
            img_path = image_dir / result["file"]
            if not img_path.exists():
                continue

            print(f"  {img_path.name}", end="", flush=True)
            task = upload_image(self.session, self.url, self.id, img_path)
            n_uploaded += 1

            detections = [
                d for d in result.get("detections", [])
                if d.get("conf", 1.0) >= conf_threshold
            ]
            if detections:
                img_w, img_h = get_image_size(img_path)
                ls_results = []
                for det in detections:
                    # MegaDetector bbox: [x, y, w, h] normalised 0–1
                    nx, ny, nw, nh = det["bbox"]
                    label = categories.get(str(det["category"]), "animal")
                    ls_results.append(
                        coco_bbox_to_ls(
                            nx * img_w, ny * img_h, nw * img_w, nh * img_h,
                            img_w, img_h, label,
                        )
                    )
                if ls_results:
                    add_predictions(self.session, self.url, task["id"], ls_results)
                    n_annotated += 1
                    print(f"  → {len(ls_results)} detections", end="")
            print()

        print(f"\nDone — {n_uploaded} uploaded, {n_annotated} with pre-annotations")

    # ── Export ─────────────────────────────────────────────────────────────────

    def export(self, output_path: Path, fmt: str = "COCO") -> dict:
        """Export annotations and save to ``output_path``.

        Label Studio wraps COCO (and some other formats) in a ZIP archive.
        This method unpacks the ZIP automatically and writes the JSON to
        ``output_path``.

        Returns the parsed JSON (COCO dict or list of tasks).
        """
        import io
        import zipfile

        r = self.session.get(
            f"{self.url}/api/projects/{self.id}/export",
            params={"exportType": fmt.upper()},
        )
        r.raise_for_status()

        # LS returns a ZIP when the export contains a single JSON file
        if r.content[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                json_names = [n for n in zf.namelist() if n.endswith(".json")]
                if not json_names:
                    raise RuntimeError(f"ZIP export contains no .json file: {zf.namelist()}")
                raw = zf.read(json_names[0])
        else:
            raw = r.content

        data = json.loads(raw)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

        n_ann = len(data.get("annotations", data if isinstance(data, list) else []))
        if n_ann == 0:
            stats = self.task_stats()
            print(
                f"Warning: export contains 0 annotations.\n"
                f"  Tasks in project : {stats['total']}\n"
                f"  Reviewed (submitted): {stats['completed']}\n"
                f"  Remaining        : {stats['total'] - stats['completed']}\n"
                f"\n"
                f"  Pre-annotations (MegaDetector predictions) are NOT exported —\n"
                f"  only tasks you have opened, checked, and submitted in the browser count.\n"
                f"  Open {self.open_url()} and submit at least one task, then re-run this cell."
            )
        else:
            if isinstance(data, list):
                print(f"Exported '{self.title}': {n_ann} tasks → {out}")
            else:
                print(f"Exported '{self.title}': {len(data['images'])} images, "
                      f"{n_ann} annotations → {out}")
        return data

    def task_stats(self) -> dict:
        """Return a dict with total/completed/remaining task counts."""
        r = self.session.get(
            f"{self.url}/api/projects/{self.id}",
        )
        r.raise_for_status()
        d = r.json()
        total     = d.get("task_number", 0)
        completed = d.get("num_tasks_with_annotations", 0)
        return {"total": total, "completed": completed, "remaining": total - completed}

    # ── Misc ───────────────────────────────────────────────────────────────────

    def open_url(self) -> str:
        """Return the browser URL for this project."""
        return f"{self.url}/projects/{self.id}/"

    def __repr__(self) -> str:
        return f"LabelStudioProject(id={self.id}, title={self.title!r})"


# ── Low-level API helpers ─────────────────────────────────────────────────────

def upload_image(session: requests.Session, url: str, project_id: int, image_path: Path) -> dict:
    """Upload one image file to Label Studio and return the created task."""
    import time
    with open(image_path, "rb") as f:
        r = session.post(
            f"{url}/api/projects/{project_id}/import",
            files={"file": (image_path.name, f, "image/jpeg")},
        )
    r.raise_for_status()
    data = r.json()

    # LS < 1.23 returns {"tasks": [...]}, LS 1.23+ returns {"file_upload_ids": [...]}
    tasks = data.get("tasks", [])
    if tasks:
        return tasks[0]

    fname = image_path.name
    for _attempt in range(5):
        r2 = session.get(f"{url}/api/tasks", params={"project": project_id, "page_size": 500})
        r2.raise_for_status()
        resp = r2.json()
        task_list = resp.get("tasks", []) if isinstance(resp, dict) else resp
        for t in task_list:
            fu = t.get("file_upload") or ""
            img_url = t.get("data", {}).get("image", "")
            if fu.endswith(fname) or img_url.endswith(fname):
                return t
        time.sleep(0.5)

    raise RuntimeError(f"Upload returned no task for {image_path.name}: {data}")


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


# ── Upload command ─────────────────────────────────────────────────────────────

def cmd_upload(args):
    token = args.token or os.environ.get("LS_TOKEN")
    if not token:
        sys.exit("No token — pass --token or set LS_TOKEN environment variable.\n"
                 "Get it from http://localhost:8080/user/account/personal-access-token")

    url     = args.url.rstrip("/")
    session = make_session(token, url)

    # Verify connection
    r = session.get(f"{url}/api/projects")
    if not r.ok:
        sys.exit(f"Cannot connect to Label Studio at {url} (HTTP {r.status_code}).\n"
                 "Is it running?  label-studio start")

    # Find or create project
    existing = {p["title"]: p for p in r.json().get("results", [])}
    if args.project in existing:
        project_id = existing[args.project]["id"]
        print(f"  Project '{args.project}' already exists (id={project_id})")
    else:
        r2 = session.post(f"{url}/api/projects", json={
            "title": args.project,
            "label_config": LABEL_CONFIGS[args.config],
        })
        r2.raise_for_status()
        project_id = r2.json()["id"]
        print(f"  Created project '{args.project}' (id={project_id})")

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
        n_uploaded += 1

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
                add_predictions(session, url, task["id"], result)
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

    url     = args.url.rstrip("/")
    session = make_session(token, url)

    r = session.get(f"{url}/api/projects")
    r.raise_for_status()
    existing = {p["title"]: p for p in r.json().get("results", [])}
    if args.project not in existing:
        sys.exit(f"Project '{args.project}' not found. Available: "
                 + ", ".join(existing.keys()))

    project_id = existing[args.project]["id"]
    fmt        = args.format.upper()

    print(f"Exporting project '{args.project}' (id={project_id}) as {fmt}...")
    r = session.get(f"{url}/api/projects/{project_id}/export", params={"exportType": fmt})
    r.raise_for_status()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(r.content)

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