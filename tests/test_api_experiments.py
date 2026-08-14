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
