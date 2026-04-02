from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.services import portfolio_service

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.get("/weights")
async def read_weights():
    return portfolio_service.get_current_weights()


@router.get("/orders/dates")
async def read_order_dates():
    return portfolio_service.get_order_dates()


@router.get("/orders/{date_str}")
async def read_orders_for_date(date_str: str):
    df = portfolio_service.get_orders_for_date(date_str)
    if df is None:
        raise HTTPException(status_code=404, detail=f"No orders for {date_str}")
    return df.to_dict(orient="records")
