"""SpeciesNet classifier + geofence wrapper.

Mirrors week1/practicals/practical_05_species_classification.ipynb (cells
§1.2 + §1.6 combined): classify with the ``classifier`` component, then
post-process with ``speciesnet.geofence_utils.geofence_animal_classification``
using taxonomy + geofence maps pulled from the ``ensemble`` component.

Why this split exists: ``SpeciesNet(..., components="classifier")`` silently
accepts ``country=...`` but never applies a geofence. The real geofencing
lives in a separate post-processing step that requires ``ensemble.taxonomy_map``
and ``ensemble.geofence_map``. Shipping the webapp with only the
classifier component meant every prediction was global — hence Swedish
camera traps being flagged as lowland tapir / canada lynx / american
black bear. See ``scripts/debug_sweden_speciesnet.py`` for the reproducer.

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
        # Remembered for the lazy ensemble load below. The ensemble is
        # fetched only on the first ``predict(country=...)`` call — if
        # every job runs global, we pay nothing extra.
        self._model_name = model
        self._geofence_maps: tuple[Any, Any] | None = None
        self._geofence_maps_tried = False

    def _load_geofence_maps(self) -> tuple[Any, Any] | None:
        """Lazily instantiate SpeciesNet's ``ensemble`` component solely
        to extract ``taxonomy_map`` and ``geofence_map``.

        We never call ``ensemble.classify`` — the forward pass stays on
        the already-loaded classifier component. Loading the ensemble
        pulls in a detector we don't need, so this roughly doubles peak
        memory during bootstrap; acceptable because (a) it only happens
        once per worker, (b) the maps are the only way to geofence
        without reimplementing SpeciesNet's region logic.

        Failure mode: we catch the load error, log it, and return
        ``None`` — callers fall back to raw (ungeofenced) top-1, which
        is strictly better than crashing the job.
        """
        if self._geofence_maps is not None:
            return self._geofence_maps
        if self._geofence_maps_tried:
            return None
        self._geofence_maps_tried = True
        try:
            from speciesnet import SpeciesNet

            ens = SpeciesNet(self._model_name, components="ensemble")
            self._geofence_maps = (
                ens.ensemble.taxonomy_map,
                ens.ensemble.geofence_map,
            )
            return self._geofence_maps
        except Exception as e:  # noqa: BLE001
            print(
                f"[SpeciesNet] ensemble load failed ({e!r}); "
                "geofence will be skipped — predictions are raw classifier output.",
                flush=True,
            )
            return None

    @staticmethod
    def _apply_geofence(
        classes: list[str],
        scores: list[float],
        country: str | None,
        maps: tuple[Any, Any] | None,
    ) -> tuple[str | None, float]:
        """Return the geofenced ``(label, score)`` or the raw top-1 if
        geofencing is disabled / unavailable. Empty predictions return
        ``(None, 0.0)`` so the caller can skip cleanly."""
        if not classes:
            return None, 0.0
        raw_top = (classes[0], float(scores[0]) if scores else 0.0)
        if not country or maps is None:
            return raw_top
        try:
            from speciesnet.geofence_utils import geofence_animal_classification

            taxonomy_map, geofence_map = maps
            label, score, _ = geofence_animal_classification(
                labels=list(classes),
                scores=list(scores),
                country=country,
                admin1_region=None,
                taxonomy_map=taxonomy_map,
                geofence_map=geofence_map,
                enable_geofence=True,
            )
            if not label:
                return raw_top
            return label, float(score)
        except Exception as e:  # noqa: BLE001
            print(
                f"[SpeciesNet] geofence call failed ({e!r}); "
                "falling back to raw top-1.",
                flush=True,
            )
            return raw_top

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

        If ``country`` is set, each prediction's ``(classes, scores)`` list
        is run through ``geofence_animal_classification``; the returned
        top-1 replaces the raw classifier's top-1 in the output. The raw
        top-5 is preserved behind it so the UI can still show runner-ups.
        """
        filepaths, detections_dict = self._build_inputs(md_results)
        total = len(filepaths)
        if not filepaths:
            if progress_cb:
                progress_cb(0, 0)
            return [{**r, "species": []} for r in md_results]

        if progress_cb:
            progress_cb(0, total)

        # Loaded once per call regardless of chunk count — the lazy
        # getter memoizes, so multi-chunk jobs don't re-load.
        maps = self._load_geofence_maps() if country else None

        preds_by_file: dict[str, list[dict]] = {}

        for start in range(0, total, chunk):
            sub = filepaths[start : start + chunk]
            sub_dict = {fp: detections_dict[fp] for fp in sub}
            # Intentionally NOT passing ``country`` to ``snet.classify``:
            # the classifier component ignores it, and keeping it out
            # makes the geofence step below the obvious single source of
            # truth when debugging region behaviour.
            raw = self.snet.classify(
                filepaths=sub,
                detections_dict=sub_dict,
                batch_size=int(batch_size),
                progress_bars=False,
            )

            for pred in raw.get("predictions", []):
                fp = pred.get("filepath")
                if not fp:
                    continue
                classifications = pred.get("classifications") or {}
                classes = classifications.get("classes") or []
                scores = classifications.get("scores") or []

                geo_label, geo_score = self._apply_geofence(
                    classes, scores, country, maps
                )

                top: list[dict[str, Any]] = []
                if geo_label:
                    top.append(
                        {**_parse_taxonomy(geo_label), "score": float(geo_score)}
                    )
                # Fill remaining slots with the raw classifier top-5,
                # skipping the one already emitted as geofenced top-1
                # so we don't repeat the same taxon. When geofencing
                # didn't change the top, this collapses to "raw top-5".
                seen = {top[0]["full_label"]} if top else set()
                for label, score in list(zip(classes, scores))[:5]:
                    if len(top) >= 5:
                        break
                    if label in seen:
                        continue
                    top.append({**_parse_taxonomy(label), "score": float(score)})
                    seen.add(label)
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
