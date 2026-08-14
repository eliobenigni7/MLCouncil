from __future__ import annotations

from fastapi import APIRouter

from api.services import promotion_service

router = APIRouter(prefix="/promotion", tags=["promotion"])


@router.get("/manifest")
def manifest():
    return promotion_service.get_manifest()


@router.get("/reports")
def reports():
    return promotion_service.get_reports()


@router.get("/shadow-artifacts")
def shadow_artifacts():
    return promotion_service.get_shadow_artifacts()
