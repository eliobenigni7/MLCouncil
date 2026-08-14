from __future__ import annotations

import os
import secrets

from fastapi import HTTPException, Request

_PUBLIC_API_PATHS = {
    "/api/health",
    "/api/health/dagster",
    "/api/health/intraday",
    "/api/docs",
    "/api/openapi.json",
    # Login must be reachable without a session or API key (it is the entry point).
    "/api/auth/login",
}


def is_public_api_path(path: str) -> bool:
    return path in _PUBLIC_API_PATHS


def get_configured_api_key() -> str:
    return os.getenv("MLCOUNCIL_API_KEY", "")


def is_api_key_required() -> bool:
    # Paper runtime always requires an API key: the explicit flag cannot
    # downgrade a paper profile (fails closed).
    if os.getenv("MLCOUNCIL_ENV_PROFILE", "local").strip().lower() == "paper":
        return True
    explicit = os.getenv("MLCOUNCIL_REQUIRE_API_KEY")
    if explicit is not None:
        return explicit.strip().lower() in {"1", "true", "yes", "on"}
    return False


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


def request_is_authenticated(request: Request) -> bool:
    """True if a valid session cookie exists OR a valid API key header."""
    from api.session import is_session_valid, request_session_token
    token = request_session_token(request)
    if token and is_session_valid(token):
        return True
    try:
        ensure_request_api_key(request)
        return True
    except HTTPException:
        return False
