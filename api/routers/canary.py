from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from api.services import canary_service

router = APIRouter(prefix="/canary", tags=["canary"])


class PendingApply(BaseModel):
    name: str
    enabled: bool


@router.get("/state")
def state():
    return canary_service.get_state()


@router.get("/flags")
def flags():
    return canary_service.get_flags()


@router.get("/apply/preview")
def apply_preview():
    return canary_service.preview()


@router.post("/apply")
def apply(body: PendingApply):
    return canary_service.apply_pending(body.name, body.enabled)


@router.post("/apply/clear")
def apply_clear(body: PendingApply):
    return canary_service.clear_pending(body.name)
