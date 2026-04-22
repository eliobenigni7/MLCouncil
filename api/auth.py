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


def is_api_key_required() -> bool:
    explicit = os.getenv("MLCOUNCIL_REQUIRE_API_KEY")
    if explicit is not None:
        return explicit.strip().lower() in {"1", "true", "yes", "on"}
    return os.getenv("MLCOUNCIL_ENV_PROFILE", "local").strip().lower() == "paper"


def ensure_request_api_key(request: Request) -> None:
    valid_key = get_configured_api_key()
    if not valid_key:
        if is_api_key_required():
            raise HTTPException(
                status_code=503,
                detail="MLCOUNCIL_API_KEY is required but not configured",
            )
        return

    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    if not secrets.compare_digest(api_key, valid_key):
        raise HTTPException(status_code=403, detail="Invalid API key")


def require_trading_api_key(request: Request) -> None:
    ensure_request_api_key(request)
