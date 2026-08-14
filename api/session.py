from __future__ import annotations

import os
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import Request
from fastapi.responses import Response

SESSION_COOKIE = "mlcouncil_session"
CSRF_COOKIE = "mlcouncil_csrf"
SESSION_TTL_HOURS = 24

_SESSIONS: dict[str, datetime] = {}
_CSRF_TOKENS: dict[str, datetime] = {}


def _is_prod() -> bool:
    return os.getenv("MLCOUNCIL_ENV_PROFILE", "local").strip().lower() in {"prod", "paper"}


def create_session() -> str:
    token = secrets.token_urlsafe(32)
    _SESSIONS[token] = datetime.now(timezone.utc) + timedelta(hours=SESSION_TTL_HOURS)
    return token


def is_session_valid(token: str | None) -> bool:
    if not token:
        return False
    expiry = _SESSIONS.get(token)
    if expiry is None:
        return False
    if datetime.now(timezone.utc) > expiry:
        _SESSIONS.pop(token, None)
        return False
    return True


def get_session(token: str | None) -> str | None:
    return token if is_session_valid(token) else None


def destroy_session(token: str | None) -> None:
    if token:
        _SESSIONS.pop(token, None)


def set_session_cookie(resp: Response, token: str) -> None:
    resp.set_cookie(
        SESSION_COOKIE, token,
        max_age=SESSION_TTL_HOURS * 3600, httponly=True,
        samesite="lax", secure=_is_prod(),
    )


def clear_session_cookie(resp: Response) -> None:
    resp.delete_cookie(SESSION_COOKIE, samesite="lax", secure=_is_prod())


def new_csrf_token() -> str:
    token = secrets.token_urlsafe(32)
    _CSRF_TOKENS[token] = datetime.now(timezone.utc) + timedelta(hours=12)
    return token


def check_csrf(token: str | None, submitted: str | None) -> bool:
    if not token or not submitted:
        return False
    expiry = _CSRF_TOKENS.get(token)
    if expiry is None or datetime.now(timezone.utc) > expiry:
        _CSRF_TOKENS.pop(token, None)
        return False
    return secrets.compare_digest(token, submitted)


def request_session_token(request: Request) -> str | None:
    return request.cookies.get(SESSION_COOKIE)


def request_csrf_token(request: Request) -> str | None:
    return request.cookies.get(CSRF_COOKIE)
