from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.services.dagster_client import DagsterClient
from api.services.pipeline_automation import (
    get_task_state,
    is_auto_execute_enabled,
    schedule_auto_execute,
)

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


class LatestPartitionResponse(BaseModel):
    partition: str | None = None


class PipelineAutomationResponse(BaseModel):
    run_id: str
    partition: str | None = None
    status: str
    created_at: str | None = None
    updated_at: str | None = None
    execution_result: dict | None = None
    error: str | None = None


@router.post("/run", response_model=TriggerRunResponse)
async def trigger_run(req: TriggerRunRequest | None = None):
    partition = req.partition if req else None
    try:
        run_id = await dagster_client.trigger_run(partition=partition)
        if is_auto_execute_enabled():
            schedule_auto_execute(run_id=run_id, partition=partition)
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


@router.get("/latest-partition", response_model=LatestPartitionResponse)
async def latest_partition():
    try:
        partition = await dagster_client.get_latest_partition_key()
        return LatestPartitionResponse(partition=partition)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Dagster error: {e}")


@router.get("/automation/{run_id}", response_model=PipelineAutomationResponse)
async def pipeline_automation_status(run_id: str):
    state = get_task_state(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Automation task not found")
    return PipelineAutomationResponse(**state)
