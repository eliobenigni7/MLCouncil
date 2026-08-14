from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from api.errors import ApiError
from backtest.playground import (PlaygroundParams, list_snapshots,
                                 load_snapshot_equity, load_snapshot_params)

JOB_DIR = Path("data/results/experiments")
SNAPSHOTS_DIR = Path("data/results_playground")
MAX_REGISTRY_ENTRIES = 50

_lock = threading.Lock()
_procs: dict[str, subprocess.Popen] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_file(job_id: str) -> Path:
    return JOB_DIR / f"{job_id}.json"


def _read_job(job_id: str) -> dict:
    path = _job_file(job_id)
    if not path.exists():
        raise ApiError(404, "job_not_found", f"Job {job_id} not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_job(entry: dict) -> None:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    _job_file(entry["id"]).write_text(json.dumps(entry, indent=2), encoding="utf-8")


def _spawn_worker(worker_args: list[str], cwd: Path) -> None:
    proc = subprocess.Popen(worker_args, cwd=cwd, env=dict(os.environ))
    _procs[worker_args[-1]] = proc


def submit_backtest(params: dict) -> str:
    try:
        PlaygroundParams.from_dict(params)
    except Exception as exc:  # noqa: BLE001
        raise ApiError(400, "invalid_params", f"Invalid backtest params: {exc}") from exc

    job_id = f"job-{uuid.uuid4().hex[:12]}"
    _write_job({"id": job_id, "state": "queued", "params": params, "created_at": _now()})
    with _lock:
        entry = _read_job(job_id)
        entry["state"] = "running"
        entry["started_at"] = _now()
        _write_job(entry)
        _spawn_worker(
            [sys.executable, "-m", "api.services.experiment_worker", job_id],
            Path(__file__).resolve().parents[2],
        )
    return job_id


def cancel_job(job_id: str) -> dict:
    entry = _read_job(job_id)
    if entry["state"] in {"running", "queued"}:
        proc = _procs.get(job_id)
        if proc and proc.poll() is None:
            proc.terminate()
        entry["state"] = "cancelled"
        entry["finished_at"] = _now()
        _write_job(entry)
    return entry


def get_job(job_id: str) -> dict:
    return _read_job(job_id)


def list_jobs() -> list[dict]:
    if not JOB_DIR.exists():
        return []
    jobs = []
    for path in sorted(JOB_DIR.glob("job-*.json"), reverse=True):
        jobs.append(json.loads(path.read_text(encoding="utf-8")))
    return jobs


def get_job_result(job_id: str) -> dict:
    entry = _read_job(job_id)
    if entry["state"] != "succeeded" or not entry.get("snapshot_path"):
        raise ApiError(409, "job_not_finished", f"Job {job_id} has no result", entry["state"])
    snap = Path(entry["snapshot_path"])
    if not snap.exists():
        raise ApiError(404, "artifact_not_found", "Snapshot directory missing", str(snap))
    stats_file = snap / "stats.json"
    stats = json.loads(stats_file.read_text(encoding="utf-8")) if stats_file.exists() else {}
    return {"job": entry, "snapshot_dir": str(snap), "stats": stats}


def list_snapshot_records() -> list[dict]:
    df = list_snapshots(base_dir=SNAPSHOTS_DIR)
    return df.to_dict(orient="records")


def get_snapshot(snapshot_dir: str) -> dict:
    snap = Path(snapshot_dir)
    if not snap.exists():
        raise ApiError(404, "artifact_not_found", "Snapshot directory missing", snapshot_dir)
    equity = load_snapshot_equity(snap)
    params = load_snapshot_params(snap)
    return {
        "snapshot_dir": snapshot_dir,
        "equity": {"dates": [d.isoformat() for d in equity.index],
                   "values": [float(v) for v in equity.values]},
        "params": params,
    }


def boot_sweep() -> None:
    """Avvio: running->failed (orfani da restart), prune registro e snapshot."""
    if not JOB_DIR.exists():
        return
    jobs = [json.loads(p.read_text(encoding="utf-8")) for p in JOB_DIR.glob("job-*.json")]
    jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    for entry in jobs:
        if entry["state"] == "running":
            entry["state"] = "failed"
            entry["error"] = "interrupted by restart"
            entry["finished_at"] = _now()
            _write_job(entry)
    keep = jobs[:MAX_REGISTRY_ENTRIES]
    for entry in jobs[MAX_REGISTRY_ENTRIES:]:
        _job_file(entry["id"]).unlink(missing_ok=True)
    # Conservativo: pota gli snapshot solo quando il registro ha superato il
    # limite, evitando confronti timestamp fragili all'avvio.
    if len(jobs) > MAX_REGISTRY_ENTRIES and keep:
        oldest_kept = keep[-1]["created_at"]
        stamp = oldest_kept[:10].replace("-", "") + "-" + oldest_kept[11:19].replace(":", "")
        if SNAPSHOTS_DIR.exists():
            for snap in SNAPSHOTS_DIR.iterdir():
                if snap.is_dir() and snap.name < stamp:
                    shutil.rmtree(snap, ignore_errors=True)
