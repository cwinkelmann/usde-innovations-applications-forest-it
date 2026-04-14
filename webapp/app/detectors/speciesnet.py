"""SpeciesNet classifier-only wrapper.

Mirrors week1/practicals/practical_05_species_classification.ipynb §1:

    snet = SpeciesNet("kaggle:google/speciesnet/pyTorch/v4.0.2a/1",
                      components="classifier")
    snet_results = snet.classify(
        filepaths=[...],
        detections_dict={filepath: {"detections": [{"category": "1",
                                                      "conf": 0.9,
                                                      "bbox": [x, y, w, h]}]}},
        country="TZA",
        batch_size=8,
    )

Only images with at least one animal detection (MD category "1") are sent.
Person / vehicle boxes are skipped because SpeciesNet classifies animals.
"""
from __future__ import annotations

from typing import Any, Callable


def _parse_taxonomy(label: str) -> dict[str, str]:
    parts = label.split(";")
    return {
        "full_label": label,
        "species": parts[-2] if len(parts) >= 2 else label,
        "common_name": (parts[-1].replace("_", " ") if parts else label),
    }


class SpeciesNetClassifier:
    def __init__(self, model: str) -> None:
        from speciesnet import SpeciesNet

        self.snet = SpeciesNet(model, components="classifier")

    @staticmethod
    def _build_inputs(md_results: list[dict]) -> tuple[list[str], dict[str, dict]]:
        """Return `(filepaths, detections_dict)` for images with animal boxes."""
        filepaths: list[str] = []
        detections_dict: dict[str, dict] = {}
        for r in md_results:
            w, h = r["width"], r["height"]
            animal_dets: list[dict[str, Any]] = []
            for d in r.get("detections", []):
                if d["category_id"] != 0:  # only class 0 = animal
                    continue
                x1, y1, x2, y2 = d["bbox_xyxy"]
                animal_dets.append(
                    {
                        "category": "1",
                        "conf": float(d["conf"]),
                        "bbox": [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h],
                    }
                )
            if animal_dets:
                filepaths.append(r["file"])
                detections_dict[r["file"]] = {"detections": animal_dets}
        return filepaths, detections_dict

    def predict(
        self,
        md_results: list[dict],
        country: str | None = None,
        batch_size: int = 8,
        chunk: int = 50,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> list[dict]:
        """Classify in chunks of ``chunk`` images so RAM usage stays bounded
        even when a session has thousands of images with animal detections.

        ``progress_cb(processed, total)`` fires after each chunk. ``total`` is
        the number of *images with at least one animal detection* — not the
        full session size, since images with no animals are never classified.
        """
        filepaths, detections_dict = self._build_inputs(md_results)
        total = len(filepaths)
        if not filepaths:
            if progress_cb:
                progress_cb(0, 0)
            return [{**r, "species": []} for r in md_results]

        if progress_cb:
            progress_cb(0, total)

        preds_by_file: dict[str, list[dict]] = {}

        for start in range(0, total, chunk):
            sub = filepaths[start : start + chunk]
            sub_dict = {fp: detections_dict[fp] for fp in sub}
            kwargs: dict[str, Any] = {
                "filepaths": sub,
                "detections_dict": sub_dict,
                "batch_size": int(batch_size),
                "progress_bars": False,
            }
            if country:
                kwargs["country"] = country

            raw = self.snet.classify(**kwargs)

            for pred in raw.get("predictions", []):
                fp = pred.get("filepath")
                if not fp:
                    continue
                classifications = pred.get("classifications") or {}
                classes = classifications.get("classes") or []
                scores = classifications.get("scores") or []
                top: list[dict[str, Any]] = []
                for label, score in list(zip(classes, scores))[:5]:
                    tax = _parse_taxonomy(label)
                    top.append({**tax, "score": float(score)})
                preds_by_file[fp] = top

            del raw
            self._release_memory()
            if progress_cb:
                progress_cb(min(start + chunk, total), total)

        return [{**r, "species": preds_by_file.get(r["file"], [])} for r in md_results]

    @staticmethod
    def _release_memory() -> None:
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
