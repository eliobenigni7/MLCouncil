import json
from pathlib import Path


def _write_job(job_dir: Path, job_id: str, params: dict) -> Path:
    job_dir.mkdir(parents=True, exist_ok=True)
    job_file = job_dir / f"{job_id}.json"
    job_file.write_text(json.dumps({
        "id": job_id, "state": "queued", "params": params,
        "created_at": "2026-08-14T00:00:00Z",
    }), encoding="utf-8")
    return job_file


def test_worker_marks_failed_on_bad_params(tmp_path, monkeypatch):
    from api.services import experiment_worker

    job_dir = tmp_path / "experiments"
    job_file = _write_job(job_dir, "job-1", {"universe": [], "start_date": "x"})

    monkeypatch.setattr(experiment_worker, "JOB_DIR", job_dir)
    experiment_worker.run_job("job-1")

    state = json.loads(job_file.read_text(encoding="utf-8"))
    assert state["state"] == "failed"
    assert "error" in state


def test_submit_runs_worker_and_reaches_terminal_state(tmp_path, monkeypatch):
    from api.services import experiment_service, experiment_worker

    job_dir = tmp_path / "experiments"
    monkeypatch.setattr(experiment_service, "JOB_DIR", job_dir)
    monkeypatch.setattr(experiment_worker, "JOB_DIR", job_dir)

    def fake_spawn(worker_args, cwd):
        experiment_worker.run_job(worker_args[-1])
        return None

    monkeypatch.setattr(experiment_service, "_spawn_worker", fake_spawn)

    job_id = experiment_service.submit_backtest({
        "start_date": "2024-01-01", "end_date": "2024-01-10",
        "universe": ["AAPL"], "note": "test",
    })
    assert job_id
    entry = experiment_service.get_job(job_id)
    assert entry["state"] in {"succeeded", "failed"}
    assert entry["id"] == job_id


def test_boot_sweep_marks_stale_running(tmp_path, monkeypatch):
    from api.services import experiment_service

    job_dir = tmp_path / "experiments"
    job_dir.mkdir(parents=True)
    (job_dir / "job-stale.json").write_text(json.dumps({
        "id": "job-stale", "state": "running", "params": {},
        "created_at": "2026-08-01T00:00:00Z",
    }), encoding="utf-8")

    monkeypatch.setattr(experiment_service, "JOB_DIR", job_dir)
    experiment_service.boot_sweep()

    entry = json.loads((job_dir / "job-stale.json").read_text(encoding="utf-8"))
    assert entry["state"] == "failed"
    assert "interrupted by restart" in entry.get("error", "")


from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.errors import ApiError, api_error_handler
from api.routers import experiments


def _app():
    app = FastAPI()
    app.add_exception_handler(ApiError, api_error_handler)
    app.include_router(experiments.router, prefix="/api")
    return app


def test_submit_endpoint_rejects_bad_params():
    client = TestClient(_app())
    resp = client.post("/api/experiments/backtest", json={"params": {"universe": [], "start_date": "x"}})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_params"


def test_job_status_404():
    client = TestClient(_app())
    resp = client.get("/api/experiments/jobs/nope/status")
    assert resp.status_code == 404
