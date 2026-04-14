"""Job queue + worker pool. One persistent worker per GPU.

Uses two plain `multiprocessing.Queue`s — jobs (main → workers) and status
(workers → main). Main drains status into a local dict on demand, so we do
not need a `multiprocessing.Manager` (which would spawn yet another process
and re-import the entry module).
"""
from __future__ import annotations

import multiprocessing as mp
import uuid
from typing import Any

from .config import resolve_device, resolve_num_workers
from .worker import worker_main


class JobManager:
    def __init__(self) -> None:
        self._ctx = mp.get_context("spawn")  # required for CUDA
        self.job_queue: mp.Queue = self._ctx.Queue()
        self.status_queue: mp.Queue = self._ctx.Queue()
        self.status: dict[str, dict] = {}
        self.workers: list[mp.Process] = []
        self.device = resolve_device()
        self.num_workers = resolve_num_workers()

    def start(self) -> None:
        for i in range(self.num_workers):
            gpu_idx = i if self.device == "cuda" else -1
            p = self._ctx.Process(
                target=worker_main,
                args=(gpu_idx, self.device, self.job_queue, self.status_queue),
                daemon=True,
            )
            p.start()
            self.workers.append(p)

    def _drain(self) -> None:
        while True:
            try:
                job_id, update = self.status_queue.get_nowait()
            except Exception:
                return
            cur = self.status.get(job_id, {})
            cur.update(update)
            self.status[job_id] = cur

    def submit(self, mode: str, session: str, **params: Any) -> str:
        job_id = uuid.uuid4().hex[:12]
        job = {"id": job_id, "mode": mode, "session": session, **params}
        self.status[job_id] = {
            "state": "queued",
            "progress": 0.0,
            "mode": mode,
            "session": session,
        }
        self.job_queue.put(job)
        return job_id

    def get_status(self, job_id: str) -> dict:
        self._drain()
        return dict(self.status.get(job_id, {"state": "unknown"}))

    def shutdown(self) -> None:
        for _ in self.workers:
            self.job_queue.put(None)
        for p in self.workers:
            p.join(timeout=5)
