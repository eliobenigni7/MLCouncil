from __future__ import annotations

import os
import secrets

from fastapi import HTTPException, Request

_PUBLIC_API_PATHS = {
    "/api/health",
    "/api/docs",
    "/api/openapi.json",
}


def is_public_api_path(path: str) -> bool:
    return path in _PUBLIC_API_PATHS


def get_configured_api_key() -> str:
    return os.getenv("MLCOUNCIL_API_KEY", "")


def ensure_request_api_key(request: Request) -> None:
    valid_key = get_configured_api_key()
    if not valid_key:
        return

    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    if not secrets.compare_digest(api_key, valid_key):
        raise HTTPException(status_code=403, detail="Invalid API key")


def require_trading_api_key(request: Request) -> None:
    ensure_request_api_key(request)
