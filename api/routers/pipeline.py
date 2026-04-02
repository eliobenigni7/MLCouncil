from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.services.dagster_client import DagsterClient

router = APIRouter(prefix="/pipeline", tags=["pipeline"])
dagster_client = DagsterClient()


class TriggerRunRequest(BaseModel):
    partition: str | None = None


class TriggerRunResponse(BaseModel):
    run_id: str


class PipelineStatusResponse(BaseModel):
    run_id: str | None = None
    status: str
    start_time: str | None = None
    end_time: str | None = None
    partition: str | None = None


@router.post("/run", response_model=TriggerRunResponse)
async def trigger_run(req: TriggerRunRequest | None = None):
    partition = req.partition if req else None
    try:
        run_id = await dagster_client.trigger_run(partition=partition)
        return TriggerRunResponse(run_id=run_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Dagster error: {e}")


@router.get("/status", response_model=PipelineStatusResponse)
async def pipeline_status():
    ps = await dagster_client.get_last_status()
    return PipelineStatusResponse(
        run_id=ps.last_run_id,
        status=ps.status,
        start_time=ps.start_time,
        end_time=ps.end_time,
        partition=ps.partition,
    )
