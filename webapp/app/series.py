"""Group camera-trap images into time-based series and aggregate species
predictions per series.

A *series* is a maximal sequence of images whose consecutive timestamps are
within ``max_gap_seconds``. Default 60 s matches typical Reconyx HyperFire
burst behavior (3–5 frames per trigger; PIR debounce keeps re-triggers
within an event seconds apart, while cross-event gaps are minutes).
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Literal


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def group_into_series(
    records: list[dict],
    max_gap_seconds: int = 60,
) -> tuple[list[list[dict]], list[dict]]:
    """Partition ``records`` into time-based series.

    Each record must have a ``file`` key; ``datetime`` is an ISO string or
    None. Records without a parseable timestamp are returned in a separate
    bucket so the UI can still show them.

    Returns ``(series_list, no_timestamp)`` where ``series_list`` is a list
    of series, each a time-sorted list of records.
    """
    timed: list[tuple[datetime, dict]] = []
    untimed: list[dict] = []
    for r in records:
        dt = _parse_iso(r.get("datetime"))
        if dt is None:
            untimed.append(r)
        else:
            timed.append((dt, r))
    timed.sort(key=lambda x: x[0])

    series: list[list[dict]] = []
    current: list[dict] = []
    gap = timedelta(seconds=max_gap_seconds)
    last_dt: datetime | None = None
    for dt, rec in timed:
        if current and last_dt is not None and (dt - last_dt) > gap:
            series.append(current)
            current = []
        current.append(rec)
        last_dt = dt
    if current:
        series.append(current)

    return series, untimed


def series_time_range(series: list[dict]) -> tuple[str | None, str | None]:
    """Return ``(start_iso, end_iso)`` across records that have timestamps,
    or ``(None, None)`` if none do."""
    dts = [r.get("datetime") for r in series if r.get("datetime")]
    if not dts:
        return None, None
    return min(dts), max(dts)


def aggregate_species(
    series: list[dict],
    predictions_by_file: dict[str, list[dict]],
    mode: Literal["majority", "union", "max-conf"] = "majority",
) -> dict:
    """Summarize SpeciesNet predictions across one series.

    ``predictions_by_file`` maps basename → top-K prediction list (each
    item is a dict with ``common_name``, ``score``, etc. as produced by
    ``webapp/app/detectors/speciesnet.py``).

    Returns:
        image_count:       int
        top_counts:        {common_name: n images where it was top-1}
        majority:          top-1 species by vote, tie-broken by max conf
        max_conf_species:  species of the single highest-confidence frame
        max_conf:          that confidence, or 0.0
        all_species:       sorted union of top-1 species seen in this series
    """
    top_counts: Counter[str] = Counter()
    max_conf = 0.0
    max_conf_species: str | None = None
    all_species: set[str] = set()

    for rec in series:
        preds = predictions_by_file.get(rec["file"]) or []
        if not preds:
            continue
        top = preds[0]
        common = top.get("common_name") or "?"
        top_counts[common] += 1
        all_species.add(common)
        score = float(top.get("score") or 0)
        if score > max_conf:
            max_conf = score
            max_conf_species = common

    majority: str | None = None
    if top_counts:
        ranked = top_counts.most_common()
        max_count = ranked[0][1]
        tied = [sp for sp, c in ranked if c == max_count]
        if len(tied) == 1:
            majority = tied[0]
        else:
            # Tie-break by the highest single-image confidence among tied
            # species. This keeps the result deterministic and biologically
            # reasonable: the best-looking frame wins.
            by_max_conf: dict[str, float] = {}
            for rec in series:
                preds = predictions_by_file.get(rec["file"]) or []
                if not preds:
                    continue
                top = preds[0]
                sp = top.get("common_name")
                if sp in tied:
                    s = float(top.get("score") or 0)
                    by_max_conf[sp] = max(by_max_conf.get(sp, 0.0), s)
            majority = max(tied, key=lambda sp: by_max_conf.get(sp, 0.0))

    return {
        "image_count": len(series),
        "top_counts": dict(top_counts),
        "majority": majority,
        "max_conf_species": max_conf_species,
        "max_conf": max_conf,
        "all_species": sorted(all_species),
    }
