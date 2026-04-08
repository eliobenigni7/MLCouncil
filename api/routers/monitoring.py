from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel

from api.services import monitoring_service

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/alerts")
async def read_current_alerts():
    return monitoring_service.get_current_alerts()


@router.get("/alerts/history")
async def read_alert_history(limit: int = Query(30, ge=1, le=365)):
    return monitoring_service.get_alert_history(limit=limit)


@router.get("/settings")
async def read_runtime_settings():
    return monitoring_service.get_runtime_settings()


class RuntimeSettingsUpdate(BaseModel):
    values: dict[str, str | None]


@router.put("/settings")
async def write_runtime_settings(payload: RuntimeSettingsUpdate):
    return monitoring_service.update_runtime_settings(payload.values)
