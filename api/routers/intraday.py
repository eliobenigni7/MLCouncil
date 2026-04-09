from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.services import intraday_runtime_service

router = APIRouter(prefix="/intraday", tags=["intraday"])


class IntradayStatusResponse(BaseModel):
    running: bool = False
    paused: bool = False
    schedule_minutes: int = 15
    market_session: str = "closed"
    last_completed_slot: str | None = None
    latest_decision_id: str | None = None


class IntradayDecisionResponse(BaseModel):
    decision_id: str
    as_of: str
    market_session: str
    agent_trace: dict = Field(default_factory=dict)
    execution_intents: list[dict] = Field(default_factory=list)


@router.get("/status", response_model=IntradayStatusResponse)
async def intraday_status():
    return IntradayStatusResponse(**intraday_runtime_service.get_status())


@router.post("/control/start", response_model=IntradayStatusResponse)
async def intraday_start():
    return IntradayStatusResponse(**intraday_runtime_service.start())


@router.post("/control/pause", response_model=IntradayStatusResponse)
async def intraday_pause():
    return IntradayStatusResponse(**intraday_runtime_service.pause())


@router.post("/control/resume", response_model=IntradayStatusResponse)
async def intraday_resume():
    return IntradayStatusResponse(**intraday_runtime_service.resume())


@router.post("/control/stop", response_model=IntradayStatusResponse)
async def intraday_stop():
    return IntradayStatusResponse(**intraday_runtime_service.stop())


@router.post("/cycle")
async def intraday_cycle():
    return intraday_runtime_service.run_cycle()


@router.get("/decisions/latest", response_model=IntradayDecisionResponse)
async def intraday_latest_decision():
    payload = intraday_runtime_service.get_latest_decision()
    if payload is None:
        raise HTTPException(status_code=404, detail="No intraday decisions found")
    return IntradayDecisionResponse(**payload)


@router.get("/decisions/{decision_id}/explain")
async def intraday_explain_decision(decision_id: str):
    try:
        return intraday_runtime_service.explain_decision(decision_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown decision {decision_id}") from exc


@router.post("/decisions/{decision_id}/execute")
async def intraday_execute_decision(decision_id: str):
    try:
        return intraday_runtime_service.execute_decision(decision_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown decision {decision_id}") from exc
