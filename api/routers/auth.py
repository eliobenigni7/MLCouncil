from __future__ import annotations

import os
import secrets

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

from api.errors import ApiError
from api.rate_limit import limiter
from api.session import (CSRF_COOKIE, clear_session_cookie, create_session,
                         destroy_session, is_session_valid, new_csrf_token,
                         request_session_token, set_session_cookie)

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


def _admin_credentials() -> tuple[str, str]:
    return os.getenv("MLCOUNCIL_ADMIN_USERNAME", ""), os.getenv("MLCOUNCIL_ADMIN_PASSWORD", "")


@router.post("/login")
@limiter.limit("5/minute")
def login(request: Request, body: LoginRequest, response: Response):
    # Rate limiting: slowapi is not installed in the test venv, so
    # tests/conftest.py installs a stub whose Limiter.limit() is an identity
    # decorator — under tests this is a no-op; in production (slowapi in
    # requirements_api.txt) the decorator injects request/response itself.
    expected_user, expected_pass = _admin_credentials()
    if not expected_user or not expected_pass:
        raise ApiError(503, "auth_not_configured", "Admin credentials not configured")
    user_ok = secrets.compare_digest(body.username, expected_user)
    pass_ok = secrets.compare_digest(body.password, expected_pass)
    if not (user_ok and pass_ok):
        raise ApiError(401, "invalid_credentials", "Invalid username or password")
    token = create_session()
    set_session_cookie(response, token)
    response.set_cookie(
        CSRF_COOKIE, new_csrf_token(), max_age=12 * 3600,
        httponly=False, samesite="lax", secure=False,
    )
    return {"authenticated": True, "username": body.username}


@router.post("/logout")
def logout(request: Request, response: Response):
    destroy_session(request_session_token(request))
    clear_session_cookie(response)
    response.delete_cookie(CSRF_COOKIE)
    return {"authenticated": False}


@router.get("/me")
def me(request: Request):
    token = request_session_token(request)
    if not token or not is_session_valid(token):
        raise ApiError(401, "not_authenticated", "Not logged in")
    return {"authenticated": True, "username": os.getenv("MLCOUNCIL_ADMIN_USERNAME", "")}
