"""Job queue + worker pool. One persistent worker per GPU.

Scheduling
----------
Each worker owns its own ``mp.Queue`` (not a shared one). On ``submit`` the
manager picks a target worker using live GPU introspection:

  1. Query ``nvidia-smi`` for ``memory.free`` / ``memory.total`` per GPU.
  2. Split candidates into *idle* (≥ 50 % free) and *busy* (< 50 % free).
  3. Prefer the idle pool. Within it, pick the worker with the fewest
     outstanding (queued + running) jobs; break ties by most free memory.
  4. If the idle pool is empty, fall back to the full candidate list with
     the same ordering — every GPU is busy, pick the least-bad one.

A single shared ``mp.Queue`` workers race to ``.get()`` would technically
load-balance on its own, but in practice the first-ready worker grabs
most jobs (model-load warm-up creates a large fairness gap), leaving the
other GPUs idle. Explicit per-GPU routing fixes that and lets us
deprioritize GPUs shared with external processes.
"""
from __future__ import annotations

import multiprocessing as mp
import subprocess
import uuid
from typing import Any

from .config import resolve_device, resolve_num_workers
from .worker import worker_main


# A GPU with at least this fraction of its memory free is considered
# "idle" and preferred for scheduling. GPUs below the threshold are only
# used when no idle GPU exists.
_IDLE_FREE_FRACTION = 0.5


def _query_gpu_memory() -> list[tuple[int, int]]:
    """Return ``[(free_mib, total_mib), ...]`` indexed by physical GPU.

    Shells out to ``nvidia-smi`` rather than importing ``pynvml`` so we
    avoid adding a dependency and the main process never initializes a
    CUDA context (which would interfere with CUDA_VISIBLE_DEVICES
    pinning in the spawned worker children).

    Returns ``[]`` if ``nvidia-smi`` is missing, times out, or emits
    something unparseable — callers fall back to round-robin.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    rows: list[tuple[int, int]] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                rows.append((int(parts[0]), int(parts[1])))
            except ValueError:
                continue
    return rows


class JobManager:
    def __init__(self) -> None:
        self._ctx = mp.get_context("spawn")  # required for CUDA
        self.status_queue: mp.Queue = self._ctx.Queue()
        self.job_queues: list[mp.Queue] = []
        self.status: dict[str, dict] = {}
        self.workers: list[mp.Process] = []
        self.device = resolve_device()
        self.num_workers = resolve_num_workers()
        # Outstanding (queued + running) job count per worker. Updated
        # at submit time and when status transitions to a terminal state.
        # Used as the primary load-balancing signal so freshly submitted
        # jobs don't all pile onto the same worker before any of them
        # has had a chance to allocate CUDA memory.
        self._pending: list[int] = [0] * self.num_workers
        self._rr_cursor = 0  # fallback round-robin pointer

    def start(self) -> None:
        self.job_queues = [self._ctx.Queue() for _ in range(self.num_workers)]
        for i in range(self.num_workers):
            gpu_idx = i if self.device == "cuda" else -1
            p = self._ctx.Process(
                target=worker_main,
                args=(
                    gpu_idx,
                    self.device,
                    self.job_queues[i],
                    self.status_queue,
                ),
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
            # Decrement the pending counter once — the first time we see
            # a terminal state for this job — so we don't double-count
            # across multiple status messages.
            new_state = update.get("state")
            if new_state in ("done", "error") and not cur.get("_counted_done"):
                cur["_counted_done"] = True
                w_idx = cur.get("worker_idx")
                if isinstance(w_idx, int) and 0 <= w_idx < len(self._pending):
                    self._pending[w_idx] = max(0, self._pending[w_idx] - 1)
            self.status[job_id] = cur

    def _pick_worker(self) -> int:
        """Choose the worker to receive the next job.

        On CUDA, consults ``nvidia-smi`` for per-GPU free memory and
        combines it with our local pending-job counter. On MPS / CPU the
        pool has a single worker, so this is always 0.
        """
        if self.device != "cuda" or self.num_workers <= 1:
            return 0
        # Refresh pending counts so a job that just finished frees its slot.
        self._drain()
        mem = _query_gpu_memory()
        if not mem:
            # nvidia-smi unavailable — round-robin so jobs at least spread.
            idx = self._rr_cursor % self.num_workers
            self._rr_cursor += 1
            return idx
        # Build a (worker_idx, free_frac, free_mib, pending) row per worker
        # we actually spawned. If nvidia-smi reports fewer GPUs than we
        # have workers (shouldn't happen, but defensive), we clamp.
        rows: list[tuple[int, float, int, int]] = []
        for i in range(min(self.num_workers, len(mem))):
            free, total = mem[i]
            free_frac = free / total if total else 0.0
            rows.append((i, free_frac, free, self._pending[i]))
        if not rows:
            return 0
        idle = [r for r in rows if r[1] >= _IDLE_FREE_FRACTION]
        pool = idle or rows
        # Sort by (pending asc, free desc) — "least busy, then most room".
        pool.sort(key=lambda r: (r[3], -r[2]))
        return pool[0][0]

    def submit(self, mode: str, session: str, **params: Any) -> str:
        # Drain first so picks see up-to-date pending counts and we can
        # record the eventual worker assignment in status immediately.
        self._drain()
        worker_idx = self._pick_worker()
        job_id = uuid.uuid4().hex[:12]
        job = {"id": job_id, "mode": mode, "session": session, **params}
        self.status[job_id] = {
            "state": "queued",
            "progress": 0.0,
            "mode": mode,
            "session": session,
            "worker_idx": worker_idx,
            # Display-friendly label; the worker overwrites this once it
            # starts running so the two views agree.
            "worker": (
                f"cuda:{worker_idx}" if self.device == "cuda" else self.device
            ),
        }
        self._pending[worker_idx] += 1
        self.job_queues[worker_idx].put(job)
        return job_id

    def get_status(self, job_id: str) -> dict:
        self._drain()
        return dict(self.status.get(job_id, {"state": "unknown"}))

    def gpu_snapshot(self) -> list[dict]:
        """Return a diagnostic view of the current pool: per-worker
        free/total memory (MiB), free fraction, and pending job count.

        Exposed so the Admin tab can show what the scheduler is seeing —
        otherwise a lopsided distribution (all jobs on cuda:0) looks
        like a mystery.
        """
        self._drain()
        mem = _query_gpu_memory() if self.device == "cuda" else []
        out: list[dict] = []
        for i in range(self.num_workers):
            free, total = (mem[i] if i < len(mem) else (0, 0))
            out.append(
                {
                    "worker": i,
                    "free_mib": free,
                    "total_mib": total,
                    "free_frac": (free / total) if total else None,
                    "pending": self._pending[i],
                }
            )
        return out

    def shutdown(self) -> None:
        for q in self.job_queues:
            q.put(None)
        for p in self.workers:
            p.join(timeout=5)
