from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from api.services import experiment_service

router = APIRouter(prefix="/experiments", tags=["experiments"])


class BacktestRequest(BaseModel):
    params: dict


@router.post("/backtest")
def run_backtest(body: BacktestRequest):
    job_id = experiment_service.submit_backtest(body.params)
    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs")
def jobs():
    return {"jobs": experiment_service.list_jobs()}


@router.get("/jobs/{job_id}/status")
def job_status(job_id: str):
    entry = experiment_service.get_job(job_id)
    return {"id": entry["id"], "state": entry["state"], "error": entry.get("error")}


@router.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    return experiment_service.get_job_result(job_id)


@router.post("/jobs/{job_id}/cancel")
def job_cancel(job_id: str):
    return experiment_service.cancel_job(job_id)


@router.get("/snapshots")
def snapshots():
    return {"snapshots": experiment_service.list_snapshot_records()}


@router.get("/snapshots/{snapshot_dir:path}")
def snapshot(snapshot_dir: str):
    return experiment_service.get_snapshot(snapshot_dir)
