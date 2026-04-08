from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from api.services.dagster_client import DagsterClient, PipelineStatus
from api.services import trading_service

_TERMINAL_STATUSES = {"SUCCESS", "FAILURE", "FAILED", "CANCELED", "CANCELLED"}
_POLL_INTERVAL_SECONDS = float(os.getenv("MLCOUNCIL_AUTO_EXECUTE_POLL_SECONDS", "5"))
_tasks_lock = threading.Lock()
_tasks: dict[str, "AutomationTask"] = {}


@dataclass
class AutomationTask:
    run_id: str
    partition: str | None
    status: str = "scheduled"
    created_at: str = datetime.now(timezone.utc).isoformat()
    updated_at: str = datetime.now(timezone.utc).isoformat()
    execution_result: dict | None = None
    error: str | None = None

    def touch(self, *, status: str, error: str | None = None, execution_result: dict | None = None):
        self.status = status
        self.updated_at = datetime.now(timezone.utc).isoformat()
        if error is not None:
            self.error = error
        if execution_result is not None:
            self.execution_result = execution_result


def is_auto_execute_enabled() -> bool:
    value = os.getenv("MLCOUNCIL_AUTO_EXECUTE", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def schedule_auto_execute(*, run_id: str, partition: str | None = None) -> None:
    with _tasks_lock:
        existing = _tasks.get(run_id)
        if existing and existing.status in {"scheduled", "monitoring", "executing"}:
            return
        _tasks[run_id] = AutomationTask(run_id=run_id, partition=partition)

    worker = threading.Thread(
        target=_run_monitor_loop,
        kwargs={"run_id": run_id, "partition": partition},
        daemon=True,
        name=f"mlcouncil-auto-execute-{run_id[:8]}",
    )
    worker.start()


def get_task_state(run_id: str) -> dict | None:
    with _tasks_lock:
        task = _tasks.get(run_id)
        if task is None:
            return None
        return {
            "run_id": task.run_id,
            "partition": task.partition,
            "status": task.status,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
            "execution_result": task.execution_result,
            "error": task.error,
        }


def _run_monitor_loop(*, run_id: str, partition: str | None) -> None:
    try:
        asyncio.run(_monitor_and_execute(run_id=run_id, partition=partition))
    except Exception as exc:  # noqa: BLE001
        _update_task(run_id, status="error", error=str(exc))


async def _monitor_and_execute(*, run_id: str, partition: str | None) -> None:
    client = DagsterClient()
    _update_task(run_id, status="monitoring")

    while True:
        status = await client.get_run_status(run_id)
        if status.status in _TERMINAL_STATUSES:
            break
        if status.status in {"not_found", "error", "unreachable"}:
            _update_task(
                run_id,
                status="error",
                error=f"Dagster run {run_id} entered terminal error state: {status.status}",
            )
            return
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)

    if status.status != "SUCCESS":
        _update_task(
            run_id,
            status="pipeline_failed",
            error=f"Dagster run {run_id} finished with status {status.status}",
        )
        return

    resolved_partition = partition or status.partition or trading_service.service.get_latest_order_date()
    if not resolved_partition:
        _update_task(
            run_id,
            status="error",
            error=f"Dagster run {run_id} completed but no partition could be resolved for execution.",
        )
        return

    _update_task(run_id, status="executing")
    result = trading_service.service.execute_orders(resolved_partition)
    if "error" in result:
        _update_task(
            run_id,
            status="blocked",
            error=result["error"],
            execution_result=result,
        )
        return

    _update_task(
        run_id,
        status="executed",
        execution_result=result,
    )


def _update_task(run_id: str, *, status: str, error: str | None = None, execution_result: dict | None = None) -> None:
    with _tasks_lock:
        task = _tasks.setdefault(run_id, AutomationTask(run_id=run_id, partition=None))
        task.touch(status=status, error=error, execution_result=execution_result)
