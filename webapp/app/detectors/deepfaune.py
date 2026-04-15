"""DeepFaune Europe classifier wrapper.

Uses the DeepFaune v3 DINOv2 ViT-L model (`deepfaune-vit_large_patch14_dinov2.
lvd142m.v3.pt`, ~1.1 GB) to classify animal crops produced by MegaDetector.

Mirrors the SpeciesNet wrapper API: takes the webapp's md_results list,
returns the same list with a ``species`` field appended per image.
Aggregation is softmax-averaged across all animal crops in an image.

Source:   https://www.deepfaune.cnrs.fr/
License:  CC BY-NC-SA 4.0 (non-commercial / share-alike).
"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image

CROP_SIZE = 182
BACKBONE = "vit_large_patch14_dinov2.lvd142m"

# 34 European species, English names — order matches DeepFaune v3 indices.
DEEPFAUNE_SPECIES = [
    "bison", "badger", "ibex", "beaver", "red deer", "chamois", "cat",
    "goat", "roe deer", "dog", "fallow deer", "squirrel", "moose", "equid",
    "genet", "wolverine", "hedgehog", "lagomorph", "wolf", "otter", "lynx",
    "marmot", "micromammal", "mouflon", "sheep", "mustelid", "bird", "bear",
    "nutria", "raccoon", "fox", "reindeer", "wild boar", "cow",
]


def _resolve_weights(weights_path: Path, url: str) -> Path:
    """Return a local path to the weights file, downloading on first use.

    The download is ~1.1 GB and shows percentage progress to stderr so the
    worker logs make it visible. Subsequent calls return immediately.
    """
    if weights_path.exists():
        return weights_path
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[DeepFaune] downloading {url} -> {weights_path} (~1.1 GB, one-time)",
        flush=True,
    )
    last_pct = [0]

    def _hook(blocks: int, blocksize: int, total: int) -> None:
        if total <= 0:
            return
        downloaded = blocks * blocksize
        pct = int(downloaded / total * 100)
        if pct - last_pct[0] >= 5:
            mb = downloaded // (1024 * 1024)
            tot_mb = total // (1024 * 1024)
            print(
                f"[DeepFaune]   {pct}% ({mb} MB / {tot_mb} MB)",
                flush=True,
            )
            last_pct[0] = pct

    # Stream straight to a .partial sidecar so an interrupted download
    # can't be mistaken for a complete one on the next process start.
    partial = weights_path.with_suffix(weights_path.suffix + ".partial")
    urllib.request.urlretrieve(url, str(partial), reporthook=_hook)
    partial.replace(weights_path)
    print("[DeepFaune] download complete", flush=True)
    return weights_path


def _crop_square(img: Image.Image, x1: float, y1: float, x2: float, y2: float) -> Image.Image:
    """Square crop centred on the bounding box, padded to the longer side
    and clipped to image bounds. Same logic as DeepFaune's ``cropSquareCVtoPIL``.
    """
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1)
    half = side / 2
    left = max(0.0, cx - half)
    top = max(0.0, cy - half)
    right = min(float(img.width), cx + half)
    bottom = min(float(img.height), cy + half)
    return img.crop((left, top, right, bottom))


class DeepFauneClassifier:
    """Lazy-loaded DeepFaune ViT-L classifier."""

    def __init__(
        self,
        weights_path: str | Path,
        device: str | None = None,
        url: str | None = None,
    ) -> None:
        import timm
        from torch import nn
        from torchvision.transforms import InterpolationMode, transforms

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    size=(CROP_SIZE, CROP_SIZE),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor([0.4850, 0.4560, 0.4060]),
                    std=torch.tensor([0.2290, 0.2240, 0.2250]),
                ),
            ]
        )

        # The official model wraps a timm model under ``self.base_model``.
        # The state dict keys carry that prefix, so we replicate the wrapper
        # to load cleanly without renaming keys.
        class _DeepFauneModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base_model = timm.create_model(
                    BACKBONE,
                    pretrained=False,
                    num_classes=len(DEEPFAUNE_SPECIES),
                    dynamic_img_size=True,
                )

            def forward(self, x):  # noqa: ANN001 ANN201
                return self.base_model(x)

        weights_path = Path(weights_path)
        if url is None:
            # Default mirror; main.py / config.py normally pass the
            # configured DEEPFAUNE_URL, this is just a fallback.
            url = (
                "https://pbil.univ-lyon1.fr/software/download/deepfaune/v1.3/"
                "deepfaune-vit_large_patch14_dinov2.lvd142m.v3.pt"
            )
        weights_path = _resolve_weights(weights_path, url)

        self.model = _DeepFauneModel()
        # weights_only=False because the .pt also stores an 'args' dict.
        params = torch.load(
            str(weights_path), map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(params["state_dict"])
        self.model.eval()
        self.model.to(self.device)

    def predict(
        self,
        md_results: list[dict],
        image_dir: Path,
        batch_size: int = 8,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> list[dict]:
        """Classify every animal crop and attach a per-image top-5 species list.

        Per-image species comes from softmax-averaging across all of that
        image's animal crops, then sorting descending. Matches SpeciesNet's
        output shape so the gallery / Series / Map tabs consume it as-is.
        """
        image_dir = Path(image_dir)

        crops: list[Image.Image] = []
        crop_to_file: list[str] = []
        for r in md_results:
            animal_dets = [
                d for d in r.get("detections", []) if d.get("category_id") == 0
            ]
            if not animal_dets:
                continue
            img_path = image_dir / Path(r["file"]).name
            if not img_path.exists():
                continue
            try:
                with Image.open(img_path) as im:
                    im = im.convert("RGB")
                    for d in animal_dets:
                        x1, y1, x2, y2 = d["bbox_xyxy"]
                        crops.append(_crop_square(im, x1, y1, x2, y2))
                        crop_to_file.append(r["file"])
            except Exception:
                continue

        if not crops:
            if progress_cb:
                progress_cb(0, 0)
            return [{**r, "species": []} for r in md_results]

        total = len(crops)
        if progress_cb:
            progress_cb(0, total)

        per_crop_softmax: list[list[float]] = []
        for start in range(0, total, batch_size):
            batch = crops[start : start + batch_size]
            tensors = torch.stack([self.transform(c) for c in batch]).to(self.device)
            with torch.no_grad():
                out = self.model(tensors).softmax(dim=1)
            per_crop_softmax.extend(out.cpu().tolist())
            self._release_memory()
            if progress_cb:
                progress_cb(min(start + batch_size, total), total)

        # Aggregate per image: average softmax over that image's crops.
        per_file: dict[str, list[list[float]]] = {}
        for fp, sm in zip(crop_to_file, per_crop_softmax):
            per_file.setdefault(fp, []).append(sm)

        results_by_file: dict[str, list[dict[str, Any]]] = {}
        for fp, sm_list in per_file.items():
            avg = [sum(col) / len(col) for col in zip(*sm_list)]
            top5 = sorted(enumerate(avg), key=lambda kv: -kv[1])[:5]
            results_by_file[fp] = [
                {
                    "common_name": DEEPFAUNE_SPECIES[idx],
                    "species": DEEPFAUNE_SPECIES[idx],
                    "full_label": f"deepfaune;{DEEPFAUNE_SPECIES[idx]}",
                    "score": float(score),
                }
                for idx, score in top5
            ]

        return [
            {**r, "species": results_by_file.get(r["file"], [])}
            for r in md_results
        ]

    @staticmethod
    def _release_memory() -> None:
        import gc

        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if (
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ):
                torch.mps.empty_cache()
        except Exception:
            pass
