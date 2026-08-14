"""Esegue un singolo job backtest in un subprocess dedicato.

Uso: python -m api.services.experiment_worker <job_id>
Il worker carica il job dal registro, aggiorna lo stato e scrive l'esito.
"""
from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from runtime_env import load_runtime_env

load_runtime_env()

JOB_DIR = Path("data/results/experiments")


def _job_file(job_id: str) -> Path:
    return JOB_DIR / f"{job_id}.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _update(job_id: str, **fields) -> None:
    path = _job_file(job_id)
    if not path.exists():
        return
    entry = json.loads(path.read_text(encoding="utf-8"))
    entry.update(fields)
    path.write_text(json.dumps(entry, indent=2), encoding="utf-8")


def run_job(job_id: str) -> None:
    try:
        path = _job_file(job_id)
        if not path.exists():
            raise FileNotFoundError(f"job {job_id} not in registry")
        entry = json.loads(path.read_text(encoding="utf-8"))
        _update(job_id, state="running", started_at=_now())

        from backtest.playground import PlaygroundParams, run_playground_backtest

        params = PlaygroundParams.from_dict(entry["params"])
        result = run_playground_backtest(params, progress_cb=None)
        _update(
            job_id,
            state="succeeded",
            finished_at=_now(),
            snapshot_path=str(result.snapshot_path) if result.snapshot_path else None,
            elapsed_seconds=result.elapsed_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        excerpt = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        trace = traceback.format_exc()[-2000:]
        _update(job_id, state="failed", finished_at=_now(),
                error=str(exc), traceback_excerpt=excerpt, traceback=trace)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("usage: python -m api.services.experiment_worker <job_id>\n")
        sys.exit(2)
    run_job(sys.argv[1])
