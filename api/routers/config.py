from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.services import config_service

router = APIRouter(prefix="/config", tags=["config"])


@router.get("/universe")
async def read_universe():
    return config_service.get_universe()


class UniverseUpdate(BaseModel):
    universe: dict
    settings: dict
    macro: dict


@router.put("/universe")
async def write_universe(data: UniverseUpdate):
    try:
        return config_service.update_universe(data.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/models")
async def read_models():
    return config_service.get_models()


@router.get("/regime-weights")
async def read_regime_weights():
    return config_service.get_regime_weights()


class RegimeWeightsUpdate(BaseModel):
    regime_weights: dict
    weight_clip: dict | None = None
    performance: dict | None = None


@router.put("/regime-weights")
async def write_regime_weights(data: RegimeWeightsUpdate):
    try:
        payload = data.model_dump(exclude_none=True)
        return config_service.update_regime_weights(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
