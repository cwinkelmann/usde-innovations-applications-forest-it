"""Runtime configuration read from env vars.

All paths inside the container are under /data (bind-mounted from
/Volumes/storage/_tmp/usde on the host). The host path is not referenced
here — only the container-side layout matters for the running process.
"""
from __future__ import annotations

import os
from pathlib import Path


DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUTS_DIR = DATA_DIR / "outputs"
MODEL_CACHE_DIR = DATA_DIR / "model_cache"
# Derived JPEG thumbnails mirror the uploads tree; safe to delete & regenerate.
THUMBS_DIR = DATA_DIR / "thumbs"
# Optional ground-truth CSVs (Lissl's wolf/no-wolf labels keyed by SHA-256).
# Used by the Evaluation tab and by effective_labels to override model
# predictions where ground truth exists.
LISSL_DIR = Path(os.environ.get("LISSL_DIR", str(DATA_DIR / "lissl_occurances")))

MD_WEIGHTS = os.environ.get("MD_WEIGHTS", "md_v1000.0.0-larch.pt")
SPECIESNET_MODEL = os.environ.get(
    "SPECIESNET_MODEL", "kaggle:google/speciesnet/pyTorch/v4.0.2a/1"
)
# DeepFaune ViT-L weights (~1.1 GB). Auto-downloaded from PBIL Lyon on
# first run into MODEL_CACHE_DIR/deepfaune/. Override DEEPFAUNE_URL if
# the PBIL mirror moves or you have a local mirror.
DEEPFAUNE_WEIGHTS = os.environ.get(
    "DEEPFAUNE_WEIGHTS",
    "deepfaune-vit_large_patch14_dinov2.lvd142m.v3.pt",
)
DEEPFAUNE_URL = os.environ.get(
    "DEEPFAUNE_URL",
    "https://pbil.univ-lyon1.fr/software/download/deepfaune/v1.3/"
    "deepfaune-vit_large_patch14_dinov2.lvd142m.v3.pt",
)

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8080"))

# DEVICE=auto → cuda if available, else mps (Apple Silicon), else cpu.
_device_raw = os.environ.get("DEVICE", "auto")

# NUM_WORKERS=auto → one worker per visible CUDA GPU. MPS / CPU get a single
# worker because those back-ends have one shared device.
_num_workers_raw = os.environ.get("NUM_WORKERS", "auto")


def resolve_device() -> str:
    if _device_raw != "auto":
        return _device_raw
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def resolve_num_workers() -> int:
    if _num_workers_raw != "auto":
        return max(1, int(_num_workers_raw))
    try:
        import torch

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            return n if n > 0 else 1
    except Exception:
        pass
    return 1  # MPS or CPU: single worker shares the one device


def resolve_batch_size(device: str | None = None) -> int:
    """Resolve the inference batch size for ``device``.

    Precedence (highest first):
      1. ``BATCH_SIZE`` env var — applies to every device. Useful for
         one-off overrides without editing the per-device defaults.
      2. ``BATCH_SIZE_CUDA`` / ``BATCH_SIZE_MPS`` / ``BATCH_SIZE_CPU``
         env vars — per-device knobs, the normal way to tune.
      3. Hard-coded fallbacks (cuda=16, mps=2, cpu=8) if nothing is set.
    """
    override = os.environ.get("BATCH_SIZE")
    if override:
        return max(1, int(override))
    if device is None:
        device = resolve_device()
    per_device = os.environ.get(f"BATCH_SIZE_{device.upper()}")
    if per_device:
        return max(1, int(per_device))
    if device == "cuda":
        return 16
    if device == "mps":
        return 2
    return 8


def ensure_dirs() -> None:
    for d in (
        UPLOADS_DIR,
        OUTPUTS_DIR,
        MODEL_CACHE_DIR,
        THUMBS_DIR,
        DATA_DIR / "tus",  # resumable upload temp store
    ):
        d.mkdir(parents=True, exist_ok=True)
