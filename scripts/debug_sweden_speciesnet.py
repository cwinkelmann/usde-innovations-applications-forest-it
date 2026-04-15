#!/usr/bin/env python3
"""Debug the SpeciesNet geofence on Swedish camera-trap data.

Context
-------
Production predictions on trapper-06 images (Swedish forest camera traps)
include species that can't exist in Sweden:

    canada lynx, american black bear, lowland tapir

Root cause (confirmed by earlier iteration of this script): the webapp
uses ``SpeciesNet(model, components="classifier")`` and passes ``country``
to ``.classify()``. That call silently ignores the country kwarg — the
classifier component has no geofence hook. Geofencing lives either in
``model.predict()`` (full pipeline) or in ``model.ensemble_from_past_runs``
(post-hoc over pre-computed classifications).

This script uses the canonical SpeciesNet API — ``model.predict()`` with
the full ``components="all"`` pipeline and ``geofence=True`` on the
constructor — to establish what SpeciesNet's output *should* look like
when geofencing is properly applied. Three variants run on the same
image set so we can see the geofence taking effect:

  A. no country     — what SpeciesNet predicts globally (baseline)
  B. country="SWE"  — Swedish geofence (production expectation)
  C. country="TZA"  — Tanzanian geofence (sanity check; should also
                      remove tapir/lynx/black-bear but for different
                      reasons than SWE would)

If B == A at the taxon level, the geofence didn't take effect (bug).
If B != A and implausible-for-Sweden species disappear from B, it works.

Template: week1/practicals/practical_05_species_classification.ipynb §1.6
(the "Geofencing demo on HNEE German camera trap data" cell).

Usage
-----
    python scripts/debug_sweden_speciesnet.py [FOLDER]
                                              [--limit N]
                                              [--only FILE ...]
                                              [--device cuda|mps|cpu]

Default FOLDER is ``/Users/christian/Downloads/trapper_photos_6``. JSON +
CSV dumps land in ``scripts/debug_sweden_output/``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Point the webapp config at the local data dir *before* importing it —
# the default ``MODEL_CACHE_DIR`` is ``/data/model_cache`` (in-container),
# which isn't writable on the laptop.
os.environ.setdefault("DATA_DIR", str(REPO_ROOT / "webapp" / "_data"))

DEFAULT_FOLDER = Path("/Users/christian/Downloads/trapper_photos_6")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
SPECIESNET_MODEL = "kaggle:google/speciesnet/pyTorch/v4.0.2a/1"

# Common labels SpeciesNet emits that would be biological nonsense in
# Sweden. Extend this as you spot new offenders. The list isn't used to
# filter — it only flags implausible taxa in the printed summary so a
# glance at the report tells you whether geofencing is doing its job.
NOT_SWEDISH = {
    "canada lynx",
    "american black bear",
    "lowland tapir",
    "mountain tapir",
    "baird's tapir",
    "malayan tapir",
    "jaguar",
    "cougar",
    "mountain lion",
    "puma",
    "coyote",
    "white-nosed coati",
    "ring-tailed coati",
    "capybara",
    "giant anteater",
    "american bison",
    "collared peccary",
}


def _list_images(folder: Path, limit: int | None) -> list[Path]:
    imgs = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)
    return imgs[:limit] if limit else imgs


def _resolve_device(cli_value: str | None) -> str:
    if cli_value:
        return cli_value
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _parse_taxonomy(label: str) -> dict[str, str]:
    parts = label.split(";")
    return {
        "full_label": label,
        "species": parts[-2] if len(parts) >= 2 else label,
        "common_name": parts[-1].replace("_", " ") if parts else label,
    }


def _top_from_prediction(pred_obj: dict) -> dict[str, Any]:
    """Pull the geofenced top label + score out of one ``predict()`` row.

    ``model.predict()`` returns ``{"predictions": [...]}`` — a list of
    per-filepath dicts. Each entry carries:

      - ``prediction``        — final taxonomy string after ensemble
      - ``prediction_score``  — final score
      - ``prediction_source`` — "classifier" / "detector" / "ensemble" /
                                 a rollup label like "rollup_to_family"
      - ``classifications``   — raw classifier top-5 (classes, scores)
      - ``detections``        — detector boxes

    The ``prediction`` field is what the ensemble committed to after
    applying the geofence. When all classifier top-5 are geofenced out,
    the ensemble falls back to the detector's generic "animal" label
    (``prediction_source="detector"``) rather than guessing a species —
    conservative and correct.
    """
    label = pred_obj.get("prediction") or ""
    score = float(pred_obj.get("prediction_score") or 0.0)
    source = pred_obj.get("prediction_source") or "?"
    out = {**_parse_taxonomy(label), "score": score, "source": source}
    cls = pred_obj.get("classifications") or {}
    classes = cls.get("classes") or []
    scores = cls.get("scores") or []
    if classes:
        out["raw_common_name"] = _parse_taxonomy(classes[0])["common_name"]
        out["raw_score"] = float(scores[0]) if scores else 0.0
    return out


def _run_variant(
    model, filepaths: list[str], country: str | None, batch_size: int
) -> dict[str, dict[str, Any]]:
    """Invoke the canonical ``model.predict`` once for one country value.

    Returns ``{filepath: top-dict}`` where top-dict has the parsed
    taxonomy plus ``score`` and ``source`` fields.
    """
    # ``single_thread`` avoids spawning worker processes on macOS where
    # the default ``multi_thread`` mode can deadlock on MPS due to
    # torch's locking behaviour during CPU↔MPS tensor transfers.
    raw = model.predict(
        filepaths=filepaths,
        country=country,
        run_mode="single_thread",
        batch_size=batch_size,
        progress_bars=False,
    )
    predictions = (raw or {}).get("predictions") or []
    out: dict[str, dict] = {}
    for pred in predictions:
        fp = pred.get("filepath")
        if fp:
            out[fp] = _top_from_prediction(pred)
    return out


def _fmt_top(top: dict | None) -> str:
    if not top:
        return "—"
    # Show source in parens so "rollup_to_family" is visible at a glance.
    return f"{top['common_name']} ({top['score']:.2f}, {top['source']})"


def _count_species(results: dict[str, dict]) -> dict[str, int]:
    c: dict[str, int] = {}
    for top in results.values():
        if not top:
            continue
        name = top["common_name"]
        c[name] = c.get(name, 0) + 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "folder",
        nargs="?",
        default=str(DEFAULT_FOLDER),
        help="Folder of camera-trap images (default: trapper_photos_6)",
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="Process only the first N images"
    )
    ap.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="FILENAME",
        help="Restrict to specific filenames (basenames, e.g. 1031830.jpg)",
    )
    ap.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Override torch device (default: auto-detect)",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Classifier batch size (default: 2, keeps MPS happy)",
    )
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "scripts" / "debug_sweden_output"),
        help="Directory for JSON dumps",
    )
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"[error] not a directory: {folder}", file=sys.stderr)
        return 2

    device = _resolve_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = _list_images(folder, args.limit)
    if args.only:
        wanted = {name.lower() for name in args.only}
        images = [p for p in images if p.name.lower() in wanted]
        missing = wanted - {p.name.lower() for p in images}
        if missing:
            print(f"[warn] --only: no match for {sorted(missing)}", file=sys.stderr)
    if not images:
        print(f"[error] no images in {folder}", file=sys.stderr)
        return 2

    print(f"[info] folder={folder}")
    print(f"[info] {len(images)} image(s) · device={device}")
    print(f"[info] output dir={out_dir}")

    # ── Load the full SpeciesNet pipeline ─────────────────────────────────
    # ``components="all"`` (the default) gives us detector + classifier +
    # ensemble; ``geofence=True`` is what makes the ``country`` kwarg to
    # ``predict()`` actually do something. This is the canonical way to
    # run SpeciesNet — no manual geofence post-processing required.
    print("\n[step 1/4] Loading SpeciesNet (components='all', geofence=True)")
    from speciesnet import SpeciesNet

    model = SpeciesNet(SPECIESNET_MODEL, components="all", geofence=True)

    filepaths = [str(p) for p in images]
    variants = {
        "global": None,
        "SWE": "SWE",
        "TZA": "TZA",
    }

    results_by_variant: dict[str, dict[str, dict]] = {}
    for i, (name, country) in enumerate(variants.items(), start=2):
        print(
            f"\n[step {i}/4] model.predict(country={country!r})"
        )
        results_by_variant[name] = _run_variant(
            model, filepaths, country, args.batch_size
        )

    # ── Dump raw outputs for offline inspection ───────────────────────────
    dump = {
        variant: {
            fp: {
                "common_name": top["common_name"],
                "species": top["species"],
                "full_label": top["full_label"],
                "score": top["score"],
                "source": top["source"],
                "raw_common_name": top.get("raw_common_name"),
                "raw_score": top.get("raw_score"),
            }
            for fp, top in per_file.items()
        }
        for variant, per_file in results_by_variant.items()
    }
    (out_dir / "predict_by_variant.json").write_text(json.dumps(dump, indent=2))

    # ── Per-image comparison ──────────────────────────────────────────────
    print("\n──────── per-image top prediction (first 25) ────────")
    header = f"{'file':<20} {'global':<36} {'SWE':<36} {'TZA':<36}"
    print(header)
    print("-" * len(header))
    for fp in filepaths[:25]:
        print(
            "{:<20} {:<36} {:<36} {:<36}".format(
                Path(fp).name[:20],
                _fmt_top(results_by_variant["global"].get(fp))[:36],
                _fmt_top(results_by_variant["SWE"].get(fp))[:36],
                _fmt_top(results_by_variant["TZA"].get(fp))[:36],
            )
        )
    if len(filepaths) > 25:
        print(f"... ({len(filepaths) - 25} more; see {out_dir}/predict_by_variant.json)")

    # ── Species distribution per variant ──────────────────────────────────
    counts = {v: _count_species(r) for v, r in results_by_variant.items()}
    for variant, c in counts.items():
        print(f"\n──────── species distribution · variant={variant} ────────")
        for sp, n in list(c.items())[:15]:
            flag = "  ← ⚠ not plausible in Sweden" if sp in NOT_SWEDISH else ""
            print(f"  {n:4d}  {sp}{flag}")

    # ── Diagnostic verdict ────────────────────────────────────────────────
    def _implausible(variant: str) -> int:
        return sum(n for sp, n in counts[variant].items() if sp in NOT_SWEDISH)

    impl_global = _implausible("global")
    impl_swe = _implausible("SWE")
    impl_tza = _implausible("TZA")
    swe_same_as_global = counts["SWE"] == counts["global"]
    swe_same_as_tza = counts["SWE"] == counts["TZA"]

    # Count how often a variant's top species differs from the "global"
    # baseline. With the canonical pipeline, country="SWE" should differ
    # from global whenever the classifier would have emitted a
    # non-Swedish species — the geofence kicks in and rolls up.
    def _diff_from_global(variant: str) -> int:
        base = results_by_variant["global"]
        other = results_by_variant[variant]
        return sum(
            1
            for fp, top in other.items()
            if base.get(fp, {}).get("common_name") != top.get("common_name")
        )

    diff_swe = _diff_from_global("SWE")
    diff_tza = _diff_from_global("TZA")

    print("\n──────── diagnosis ────────")
    print(
        f"implausible-for-Sweden species count:  "
        f"global={impl_global}  SWE={impl_swe}  TZA={impl_tza}"
    )
    print(f"top-label differences vs global:        SWE={diff_swe}  TZA={diff_tza}")

    # With the canonical API (model.predict + components='all' + geofence=True),
    # SpeciesNet's ensemble already applies confidence thresholds and
    # rollup BEFORE the geofence step. So even without a country,
    # implausible species often never make it to the top (the ensemble
    # rolls them up to a safe ancestor or falls back to the detector's
    # "animal" label). That's why we compare top-label changes rather
    # than only looking at "implausible species" counts — the canonical
    # pipeline rarely emits implausible species anywhere.
    if impl_swe > 0:
        print(
            "[BAD] SWE variant still contains species implausible for Sweden — "
            "the geofence isn't catching everything. Check the NOT_SWEDISH list "
            "vs SpeciesNet's geofence_map and file upstream gaps."
        )
    elif swe_same_as_global and swe_same_as_tza:
        print(
            "[INCONCLUSIVE] SWE == TZA == global. The sample is too confident "
            "for the geofence to show work — classifier top-5 was already "
            "plausible everywhere, or everything rolled up to 'animal' / "
            "'mammalia' below the geofence. Rerun with more images (drop "
            "--only / --limit) to exercise the geofence harder."
        )
    elif swe_same_as_global:
        print(
            "[OK] SWE == global (no diff). Either the sample was already "
            "Sweden-safe, or SpeciesNet's built-in rollup handled it before "
            "the geofence had anything to filter. TZA differs from SWE "
            f"({diff_tza} image(s)), confirming country IS being respected."
        )
    elif diff_swe > 0:
        print(
            f"[OK] SWE changes {diff_swe}/{len(filepaths)} top labels vs "
            "global — the geofence is active. Inspect the per-image table "
            "to see which taxa got rolled up."
        )
    else:
        print("[?] Results don't fit a clean template — inspect the JSON dump.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
