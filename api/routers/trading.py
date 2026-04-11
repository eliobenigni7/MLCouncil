"""Trading API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from api.auth import require_trading_api_key
from api.rate_limit import limiter
from api.services import trading_service

router = APIRouter(
    prefix="/trading",
    tags=["trading"],
    dependencies=[Depends(require_trading_api_key)],
)


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
@limiter.limit("30/minute")
async def trading_status(request: Request):
    del request
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
@limiter.limit("30/minute")
async def latest_order_date(request: Request):
    del request
    date = trading_service.service.get_latest_order_date()
    if date is None:
        raise HTTPException(status_code=404, detail="No order files found")
    return {"date": date}


@router.get("/orders/pending/{date}")
@limiter.limit("30/minute")
async def pending_orders(request: Request, date: str):
    del request
    orders = trading_service.service.get_pending_orders(date)
    return {"date": date, "orders": orders}


@router.get("/preflight/{date}", response_model=PretradeResponse)
@limiter.limit("20/minute")
async def trading_preflight(request: Request, date: str):
    del request
    return PretradeResponse(**trading_service.service.build_pretrade_snapshot(date))


@router.get("/reconcile/{date}")
@limiter.limit("20/minute")
async def trading_reconcile(request: Request, date: str):
    del request
    return trading_service.service.get_reconciliation(date)


@router.post("/execute", response_model=ExecuteResponse)
@limiter.limit("5/minute")
async def execute_orders(request: Request, req: ExecuteRequest):
    del request
    result = trading_service.service.execute_orders(req.date)
    if "error" in result:
        status_code = 409 if result.get("pretrade", {}).get("blocked") else 400
        detail = result if len(result) > 1 else result["error"]
        raise HTTPException(status_code=status_code, detail=detail)
    return ExecuteResponse(**result)


@router.post("/liquidate")
@limiter.limit("5/minute")
async def liquidate_all(request: Request):
    del request
    result = trading_service.service.liquidate_all()
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/history")
@limiter.limit("20/minute")
async def trade_history(request: Request, days: int = 7):
    del request
    return {"trades": trading_service.service.get_trade_history(days)}
