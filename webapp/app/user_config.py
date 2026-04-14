"""Persistent user config — small JSON file on the bind mount.

Kept in DATA_DIR/config.json so it survives container restarts AND rebuilds.
Used for things the user types into the UI that are worth remembering across
sessions (Label Studio token, LS URL, etc.). NOT for secrets you'd want to
keep out of the mounted volume — if that matters, inject them via env vars.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import DATA_DIR

_PATH = DATA_DIR / "config.json"


def load() -> dict:
    if not _PATH.exists():
        return {}
    try:
        return json.loads(_PATH.read_text())
    except Exception:
        return {}


def save(update: dict) -> dict:
    cur = load()
    cur.update(update)
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    _PATH.write_text(json.dumps(cur, indent=2))
    return cur


def get(key: str, default: str = "") -> str:
    return str(load().get(key, default))
