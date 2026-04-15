"""Canonical SpeciesNet pipeline — detector + classifier + geofence + rollup
in a single ``model.predict()`` call.

This is the API recommended by ``speciesnet/scripts/run_model.py`` and mirrors
the canonical path exercised by
``scripts/debug_sweden_speciesnet.py``. It's different from the
``SpeciesNetClassifier`` wrapper (``speciesnet.py`` in this folder): that
one runs SpeciesNet's classifier *after* our MegaDetector boxes. This one
ignores MD entirely and runs SpeciesNet's bundled pipeline end-to-end:

  1. **Detector** — SpeciesNet's packaged MDv5 finds animal / person /
     vehicle boxes.
  2. **Classifier** — per-animal species classifier runs on each box
     (or on the whole image when no box survives).
  3. **Ensemble + geofence + rollup** — the library's own post-processing
     promotes / demotes labels based on confidence, country, and the
     geofence map. Committed label lands in ``prediction`` /
     ``prediction_score`` / ``prediction_source``.

Trade-off vs. the classifier-only wrapper: we lose MDv1000's (sometimes
better) boxes and gain (a) coverage on frames MDv1000 missed, (b) the
library's confidence-threshold-driven rollup, (c) an explicit
``prediction_source`` field telling us whether the top label came from
the classifier, a rollup, or the detector's generic "animal" fallback.

Output shape matches the other detector wrappers 1:1 so the gallery
renders without caring which pipeline produced the JSON:

    {
      "file": absolute image path,
      "width": int,
      "height": int,
      "detections": [{category_id, label, conf, bbox_xyxy}, ...],
      "species": [{common_name, species, full_label, score, source}],
    }

``species[0]`` carries the ensemble-committed label. Slots 1-4 are the raw
classifier top-5 (minus duplicates) so the UI can still show runner-ups.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PIL import Image


# SpeciesNet uses 1-indexed MD category strings; our downstream (gallery,
# exports) expects the 0-indexed ids that the MegaDetector wrapper emits.
# Mapping kept tiny and local so it's obvious what the translation does.
_CATEGORY_MAP: dict[str, tuple[int, str]] = {
    "1": (0, "animal"),
    "2": (1, "person"),
    "3": (2, "vehicle"),
}


def _parse_taxonomy(label: str) -> dict[str, str]:
    """Parse a SpeciesNet taxonomy string.

    Full species labels look like
    ``<uuid>;mammalia;artiodactyla;cervidae;cervus;elaphus;red deer``.

    Rollup / fallback labels may come back with most fields empty —
    e.g. ``<uuid>;;;;;;animal`` when the ensemble couldn't commit to a
    species and fell back to the detector's label. Splitting by ``;``
    with graceful empty handling gets us the right answer for both.
    """
    parts = label.split(";")
    return {
        "full_label": label,
        "species": parts[-2] if len(parts) >= 2 and parts[-2] else "",
        "common_name": parts[-1].replace("_", " ") if parts else label,
    }


def _rel_bbox_to_xyxy(bbox_rel: list[float], w: int, h: int) -> list[float]:
    """Convert SpeciesNet's ``[x, y, w, h]`` relative bbox (0..1) to our
    ``[x1, y1, x2, y2]`` absolute-pixel form. Gallery box-drawing code
    re-normalizes back to percentages for rendering, so we could pass
    relative coordinates through too — but keeping the on-disk format
    consistent with the MegaDetector wrapper's output makes the two
    pipelines interchangeable for downstream consumers."""
    x, y, bw, bh = bbox_rel
    return [x * w, y * h, (x + bw) * w, (y + bh) * h]


class CanonicalSpeciesNet:
    """Wraps SpeciesNet's full pipeline (``components="all"``, geofence on).

    Instantiated once per worker process. Holds the model instance for
    the lifetime of the worker; subsequent jobs reuse it."""

    def __init__(self, model: str) -> None:
        from speciesnet import SpeciesNet

        self._model_name = model
        # One SpeciesNet instance serves every country — the country is
        # passed per call via ``predict(country=...)`` and applied by the
        # ensemble's geofence step.
        self.model = SpeciesNet(model, components="all", geofence=True)

    def predict(
        self,
        image_paths: list[Path | str],
        country: str | None = None,
        batch_size: int = 8,
        chunk: int = 50,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> list[dict]:
        """Run the full pipeline over ``image_paths`` and return
        gallery-compatible records.

        Chunks the underlying ``predict`` call in batches of ``chunk``
        images so we can fire progress ticks. SpeciesNet's ``predict``
        doesn't expose a per-image callback — only ``progress_bars=True``
        for stderr — so we synthesize per-chunk progress by slicing the
        input list ourselves.

        ``run_mode="single_thread"`` avoids spawning workers inside our
        worker; the multi-thread default can deadlock on MPS and wastes
        processes on CUDA where we're already one-worker-per-GPU.
        """
        filepaths = [str(p) for p in image_paths]
        total = len(filepaths)
        if progress_cb:
            progress_cb(0, total)

        out: list[dict] = []
        for start in range(0, total, chunk):
            sub = filepaths[start : start + chunk]
            raw = self.model.predict(
                filepaths=sub,
                country=country,
                run_mode="single_thread",
                batch_size=int(batch_size),
                progress_bars=False,
            )
            for pred in (raw or {}).get("predictions", []):
                record = self._translate(pred)
                if record is not None:
                    out.append(record)
            self._release_memory()
            if progress_cb:
                progress_cb(min(start + chunk, total), total)

        return out

    @staticmethod
    def _translate(pred: dict) -> dict | None:
        """Convert one SpeciesNet ``predict`` row into our record shape."""
        fp = pred.get("filepath")
        if not fp:
            return None

        # Image dimensions — needed for absolute bbox coords. PIL reads
        # only the header, so this is cheap. Missing / corrupt images
        # are silently skipped rather than failing the whole job.
        try:
            with Image.open(fp) as im:
                w, h = im.size
        except Exception:  # noqa: BLE001
            return None

        dets_out: list[dict[str, Any]] = []
        for d in pred.get("detections") or []:
            cat = d.get("category")
            cid, lbl = _CATEGORY_MAP.get(cat, (0, "animal"))
            bbox_rel = d.get("bbox") or [0.0, 0.0, 0.0, 0.0]
            dets_out.append(
                {
                    "category_id": cid,
                    "label": lbl,
                    "conf": float(d.get("conf", 0.0)),
                    "bbox_xyxy": _rel_bbox_to_xyxy(bbox_rel, w, h),
                }
            )

        # ``species[0]`` carries the ensemble's committed label — what we
        # want the gallery to show. Slots 1-4 fill from the raw classifier
        # top-5, skipping any that duplicate the ensemble top-1 so the
        # user doesn't see the same label twice.
        species: list[dict[str, Any]] = []
        ens_label = pred.get("prediction") or ""
        ens_score = float(pred.get("prediction_score") or 0.0)
        ens_source = pred.get("prediction_source") or ""
        if ens_label:
            species.append(
                {
                    **_parse_taxonomy(ens_label),
                    "score": ens_score,
                    "source": ens_source,
                }
            )

        cls = pred.get("classifications") or {}
        classes = cls.get("classes") or []
        scores = cls.get("scores") or []
        seen = {species[0]["full_label"]} if species else set()
        for lbl, sc in list(zip(classes, scores))[:5]:
            if len(species) >= 5:
                break
            if lbl in seen:
                continue
            species.append(
                {**_parse_taxonomy(lbl), "score": float(sc), "source": "classifier"}
            )
            seen.add(lbl)

        return {
            "file": fp,
            "width": w,
            "height": h,
            "detections": dets_out,
            "species": species,
        }

    @staticmethod
    def _release_memory() -> None:
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if (
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ):
                torch.mps.empty_cache()
        except Exception:  # noqa: BLE001
            pass
