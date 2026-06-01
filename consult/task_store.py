"""Per-process task store for SEP-1686 long-running MCP tool calls.

When an MCP client invokes `consult` / `refine` / `sequence` with a
`task: {ttl}` parameter, the server returns a `CreateTaskResult`
immediately and runs the tool work in a background asyncio Task. The
client then polls `tasks/get` until the task reaches a terminal status
(`completed` / `failed` / `cancelled`).

This module is the bookkeeping for that flow: an in-process registry of
`TaskRecord` instances. The actual work tracking happens in `runner` /
`refine` / `sequence`, which already write per-run artifacts to
`~/.consult/runs/<id>/`. The TaskRecord here is the thin wire-shape
layer — it carries the result for `tasks/get` and the on-disk run_id
for forensic correlation.

LIMITS:
- In-process only (no cross-process sharing). A consult-mcp server
  restart loses in-flight tasks; clients re-poll a missing taskId
  see an empty `tasks/get` and should resubmit.
- TTL is honoured loosely: completed/failed tasks are evicted when
  *another* task is created and their record is older than `ttl_ms`.
  An aggressively-low ttl with no follow-up call leaves stale records,
  which is fine — they're small (a few KB each).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# Task statuses align with MCP's TaskStatus enum (TASK_STATUS_WORKING etc.).
# Stored as plain strings here so the engine layer doesn't need to import
# `mcp.types` — only the adapter (`consult.mcp.server`) maps to the SDK
# constants when building the wire response.
STATUS_WORKING = "working"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

_TERMINAL_STATUSES = frozenset({STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED})


@dataclass
class TaskRecord:
    """One in-flight or completed task.

    `result` carries the typed tool result for the call_tool flow once
    `status == completed`. `error` carries a string when `status ==
    failed`. Both are None during execution. `cancel_event` is the
    cooperative-cancellation signal — the background task checks it
    periodically; for our long-running async tool calls the signal is
    only checked at await boundaries, which is good enough for graceful
    shutdown.
    """

    task_id: str
    status: str = STATUS_WORKING
    status_message: str | None = None
    created_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)
    ttl_ms: int | None = None
    poll_interval_ms: int = 2000
    result: Any = None
    error: str | None = None
    # `asyncio.Task` of the background work, so we can call `.cancel()`
    # on `tasks/cancel`. None for already-terminal tasks loaded from
    # disk (future feature) or synthetic records.
    bg_task: asyncio.Task[Any] | None = None
    # The run_id of the underlying consult run if the tool started one,
    # so forensic correlation is one hop instead of a search.
    run_id: str | None = None


# Process-level registry. WeakValueDictionary would invite the GC to
# evict tasks before clients can poll; a plain dict with size-bounded
# pruning is safer.
_REGISTRY: dict[str, TaskRecord] = {}
_MAX_RECORDS = 256


def _evict_stale() -> None:
    """Prune terminal records older than their ttl. Keeps the registry
    bounded over long-running server lifetimes.
    """
    if len(_REGISTRY) <= _MAX_RECORDS:
        return
    now = time.time()
    stale_keys = []
    for tid, rec in _REGISTRY.items():
        if rec.status not in _TERMINAL_STATUSES:
            continue
        if rec.ttl_ms is None:
            continue
        if (now - rec.last_updated_at) * 1000 > rec.ttl_ms:
            stale_keys.append(tid)
    for tid in stale_keys:
        _REGISTRY.pop(tid, None)
    logger.debug("task_store evicted %d stale records", len(stale_keys))


def create(*, ttl_ms: int | None = None) -> TaskRecord:
    """Create a new task in `working` status and return its record."""
    _evict_stale()
    task_id = f"task-{uuid.uuid4().hex[:16]}"
    rec = TaskRecord(task_id=task_id, ttl_ms=ttl_ms)
    _REGISTRY[task_id] = rec
    return rec


def get(task_id: str) -> TaskRecord | None:
    return _REGISTRY.get(task_id)


def attach_bg_task(task_id: str, bg_task: asyncio.Task[Any]) -> None:
    rec = _REGISTRY.get(task_id)
    if rec is None:
        return
    rec.bg_task = bg_task


def attach_run_id(task_id: str, run_id: str) -> None:
    rec = _REGISTRY.get(task_id)
    if rec is None:
        return
    rec.run_id = run_id
    rec.last_updated_at = time.time()


def complete(task_id: str, result: Any) -> None:
    rec = _REGISTRY.get(task_id)
    # Don't resurrect a terminal task. A cancel() can land while the
    # background coroutine is already past its last await and about to call
    # complete(); without this guard the cancelled task flips back to
    # completed and a polling client sees the wrong status.
    if rec is None or rec.status in _TERMINAL_STATUSES:
        return
    rec.status = STATUS_COMPLETED
    rec.result = result
    rec.last_updated_at = time.time()


def fail(task_id: str, error: str) -> None:
    rec = _REGISTRY.get(task_id)
    # Same terminal-state guard as complete(): a cancelled (or already
    # failed) task must not be overwritten by a late failure callback.
    if rec is None or rec.status in _TERMINAL_STATUSES:
        return
    rec.status = STATUS_FAILED
    rec.error = error
    rec.status_message = error[:200]
    rec.last_updated_at = time.time()


def cancel(task_id: str) -> bool:
    """Best-effort cancel. Returns True if the task existed and was
    transitioned to cancelled or was already terminal."""
    rec = _REGISTRY.get(task_id)
    if rec is None:
        return False
    if rec.status in _TERMINAL_STATUSES:
        return True
    if rec.bg_task is not None:
        rec.bg_task.cancel()
    rec.status = STATUS_CANCELLED
    rec.last_updated_at = time.time()
    return True


def list_all() -> list[TaskRecord]:
    return list(_REGISTRY.values())


def _reset_for_tests() -> None:
    """Clear the registry. Used by tests; not exposed publicly."""
    _REGISTRY.clear()
