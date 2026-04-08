"""Trading API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.services import trading_service

router = APIRouter(prefix="/trading", tags=["trading"])


class ExecuteRequest(BaseModel):
    date: str


class StatusResponse(BaseModel):
    connected: bool
    paper: bool = True
    runtime_profile: str = "local"
    paused: bool = False
    kill_switch_active: bool = False
    paper_guard_ok: bool = True
    account: dict = Field(default_factory=dict)
    positions: list = Field(default_factory=list)
    error: str | None = None


class PretradeResponse(BaseModel):
    date: str
    paper: bool = True
    paused: bool = False
    runtime_profile: str = "local"
    paper_guard_ok: bool = True
    connection_error: str | None = None
    lineage: dict = Field(default_factory=dict)
    pretrade: dict = Field(default_factory=dict)
    reconciliation: dict = Field(default_factory=dict)
    account: dict = Field(default_factory=dict)
    positions: list = Field(default_factory=list)


class ExecuteResponse(BaseModel):
    date: str
    orders_submitted: int = 0
    orders_rejected: int = 0
    liquidations: int = 0
    lineage: dict = Field(default_factory=dict)
    pretrade: dict = Field(default_factory=dict)
    reconciliation: dict = Field(default_factory=dict)
    results: list = Field(default_factory=list)
    operations_path: str | None = None
    error: str | None = None


@router.get("/status", response_model=StatusResponse)
async def trading_status():
    status = trading_service.service.get_status()
    return StatusResponse(
        connected=status.get("connected", False),
        paper=status.get("paper", True),
        runtime_profile=status.get("runtime_profile", "local"),
        paused=status.get("paused", False),
        kill_switch_active=status.get("kill_switch_active", False),
        paper_guard_ok=status.get("paper_guard_ok", True),
        account=status.get("account", {}),
        positions=status.get("positions", []),
        error=status.get("error"),
    )


@router.get("/orders/latest")
async def latest_order_date():
    date = trading_service.service.get_latest_order_date()
    if date is None:
        raise HTTPException(status_code=404, detail="No order files found")
    return {"date": date}


@router.get("/orders/pending/{date}")
async def pending_orders(date: str):
    orders = trading_service.service.get_pending_orders(date)
    return {"date": date, "orders": orders}


@router.get("/preflight/{date}", response_model=PretradeResponse)
async def trading_preflight(date: str):
    return PretradeResponse(**trading_service.service.build_pretrade_snapshot(date))


@router.get("/reconcile/{date}")
async def trading_reconcile(date: str):
    return trading_service.service.get_reconciliation(date)


@router.post("/execute", response_model=ExecuteResponse)
async def execute_orders(req: ExecuteRequest):
    result = trading_service.service.execute_orders(req.date)
    if "error" in result:
        status_code = 409 if result.get("pretrade", {}).get("blocked") else 400
        detail = result if len(result) > 1 else result["error"]
        raise HTTPException(status_code=status_code, detail=detail)
    return ExecuteResponse(**result)


@router.post("/liquidate")
async def liquidate_all():
    result = trading_service.service.liquidate_all()
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/history")
async def trade_history(days: int = 7):
    return {"trades": trading_service.service.get_trade_history(days)}
