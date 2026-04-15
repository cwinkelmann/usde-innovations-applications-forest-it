"""Minimal tus.io v1.0.0 resumable-upload server.

Spec: https://tus.io/protocols/resumable-upload.html

Why we built our own instead of using a library: the protocol is small
(POST + HEAD + PATCH + OPTIONS), maintained Python tus servers are scarce
or stale, and inlining 150 lines avoids dragging in another dependency
that we'd then have to keep in sync with FastAPI / Starlette.

On-disk layout under ``DATA_DIR/tus/``:
    <upload_id>.bin       — raw bytes appended per PATCH chunk
    <upload_id>.meta.json  — ``{id, length, offset, metadata}``

When ``offset == length`` the registered ``on_complete(file_path, metadata)``
callback fires once. The callback is responsible for moving / extracting
the file and cleaning up the temp pair.
"""
from __future__ import annotations

import base64
import json
import uuid
from pathlib import Path
from typing import Callable

from fastapi import Header, Request, Response

from .config import DATA_DIR


TUS_VERSION = "1.0.0"
TUS_RESUMABLE = "1.0.0"
# We support the creation extension (POST /api/tus to create) and
# termination (DELETE). Other extensions (concatenation, expiration,
# checksum) are not implemented — not needed for our flow.
TUS_EXTENSIONS = "creation,termination"
TUS_MAX_SIZE = 50 * 1024 * 1024 * 1024  # 50 GB

_TUS_DIR = DATA_DIR / "tus"
_on_complete_cb: Callable[[str, dict], None] | None = None


def _meta_path(upload_id: str) -> Path:
    return _TUS_DIR / f"{upload_id}.meta.json"


def _data_path(upload_id: str) -> Path:
    return _TUS_DIR / f"{upload_id}.bin"


def _parse_metadata(header: str) -> dict[str, str]:
    """Parse the Upload-Metadata header: ``key1 base64,key2 base64,...``."""
    out: dict[str, str] = {}
    if not header:
        return out
    for pair in header.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if " " in pair:
            k, v = pair.split(" ", 1)
            try:
                out[k] = base64.b64decode(v).decode("utf-8", errors="replace")
            except Exception:
                pass
        else:
            out[pair] = ""
    return out


def _tus_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Headers every tus response should carry."""
    h = {
        "Tus-Resumable": TUS_RESUMABLE,
        "Access-Control-Expose-Headers": (
            "Upload-Offset, Upload-Length, Tus-Version, Tus-Resumable, "
            "Tus-Extension, Tus-Max-Size, Location"
        ),
    }
    if extra:
        h.update(extra)
    return h


def register_tus_routes(
    app, on_complete: Callable[[str, dict], None]
) -> None:
    """Mount the tus endpoints on a FastAPI app.

    ``on_complete(file_path, metadata)`` is invoked once per upload, when
    the final PATCH brings ``offset >= length``. The callback owns the
    file from that point on (move / extract / delete as needed).
    """
    global _on_complete_cb
    _on_complete_cb = on_complete
    # Defer the mkdir — at module-import time DATA_DIR may not be the
    # final runtime value yet (tests, native vs Docker). The first POST
    # / PATCH ensures the directory exists.

    @app.options("/api/tus")
    async def tus_options_root() -> Response:
        return Response(
            status_code=204,
            headers=_tus_headers(
                {
                    "Tus-Version": TUS_VERSION,
                    "Tus-Extension": TUS_EXTENSIONS,
                    "Tus-Max-Size": str(TUS_MAX_SIZE),
                }
            ),
        )

    @app.options("/api/tus/{upload_id}")
    async def tus_options_upload(upload_id: str) -> Response:
        return Response(status_code=204, headers=_tus_headers())

    @app.post("/api/tus")
    async def tus_create(
        upload_length: int | None = Header(None, alias="Upload-Length"),
        upload_metadata: str | None = Header(None, alias="Upload-Metadata"),
    ) -> Response:
        if upload_length is None or upload_length <= 0:
            return Response(status_code=400)
        if upload_length > TUS_MAX_SIZE:
            return Response(status_code=413)
        _TUS_DIR.mkdir(parents=True, exist_ok=True)
        upload_id = uuid.uuid4().hex
        meta = {
            "id": upload_id,
            "length": int(upload_length),
            "offset": 0,
            "metadata": _parse_metadata(upload_metadata or ""),
        }
        _meta_path(upload_id).write_text(json.dumps(meta))
        _data_path(upload_id).touch()
        return Response(
            status_code=201,
            headers=_tus_headers({"Location": f"/api/tus/{upload_id}"}),
        )

    @app.head("/api/tus/{upload_id}")
    async def tus_head(upload_id: str) -> Response:
        mp = _meta_path(upload_id)
        if not mp.exists():
            return Response(status_code=404, headers=_tus_headers())
        meta = json.loads(mp.read_text())
        return Response(
            status_code=200,
            headers=_tus_headers(
                {
                    "Upload-Offset": str(meta["offset"]),
                    "Upload-Length": str(meta["length"]),
                    "Cache-Control": "no-store",
                }
            ),
        )

    @app.patch("/api/tus/{upload_id}")
    async def tus_patch(
        upload_id: str,
        request: Request,
        upload_offset: int = Header(..., alias="Upload-Offset"),
        content_type: str | None = Header(None, alias="Content-Type"),
    ) -> Response:
        if content_type != "application/offset+octet-stream":
            return Response(status_code=415, headers=_tus_headers())
        mp = _meta_path(upload_id)
        if not mp.exists():
            return Response(status_code=404, headers=_tus_headers())
        meta = json.loads(mp.read_text())
        if upload_offset != meta["offset"]:
            # Client and server disagree on where to resume — tell
            # them their offset is wrong; tus-js-client will re-HEAD.
            return Response(status_code=409, headers=_tus_headers())
        # Stream chunk to disk (uvicorn already buffers a request body
        # but request.body() is fine for typical 5–8 MB chunks).
        chunk = await request.body()
        with _data_path(upload_id).open("ab") as f:
            f.write(chunk)
        new_offset = upload_offset + len(chunk)
        meta["offset"] = new_offset
        mp.write_text(json.dumps(meta))
        if new_offset >= meta["length"] and _on_complete_cb is not None:
            try:
                _on_complete_cb(str(_data_path(upload_id)), meta["metadata"])
            except Exception:  # noqa: BLE001
                # Don't let a callback failure poison the response —
                # tus-js-client treats a non-2xx as needing retry.
                pass
        return Response(
            status_code=204,
            headers=_tus_headers({"Upload-Offset": str(new_offset)}),
        )

    @app.delete("/api/tus/{upload_id}")
    async def tus_delete(upload_id: str) -> Response:
        # Spec: 204 if found, 404 otherwise. We treat both as success
        # to keep retry-after-disconnect cleanup idempotent.
        for p in (_meta_path(upload_id), _data_path(upload_id)):
            try:
                p.unlink()
            except OSError:
                pass
        return Response(status_code=204, headers=_tus_headers())
