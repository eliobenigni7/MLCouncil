"""Trading API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.services import trading_service

router = APIRouter(prefix="/trading", tags=["trading"])


class ExecuteRequest(BaseModel):
    date: str


class StatusResponse(BaseModel):
    connected: bool
    paper: bool = True
    account: dict = {}
    positions: list = []
    error: str | None = None


class ExecuteResponse(BaseModel):
    date: str
    orders_submitted: int = 0
    orders_rejected: int = 0
    liquidations: int = 0
    results: list = []
    error: str | None = None


@router.get("/status", response_model=StatusResponse)
async def trading_status():
    status = trading_service.service.get_status()
    return StatusResponse(
        connected=status.get("connected", False),
        paper=status.get("paper", True),
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


@router.post("/execute", response_model=ExecuteResponse)
async def execute_orders(req: ExecuteRequest):
    result = trading_service.service.execute_orders(req.date)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
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
