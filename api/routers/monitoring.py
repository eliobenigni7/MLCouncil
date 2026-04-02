from __future__ import annotations

from fastapi import APIRouter, Query

from api.services import monitoring_service

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/alerts")
async def read_current_alerts():
    return monitoring_service.get_current_alerts()


@router.get("/alerts/history")
async def read_alert_history(limit: int = Query(30, ge=1, le=365)):
    return monitoring_service.get_alert_history(limit=limit)
