# tests/test_api_auth.py
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.errors import ApiError, api_error_handler
from api.routers import auth
from api.session import (SESSION_COOKIE, check_csrf, create_session,
                         destroy_session, get_session, is_session_valid,
                         new_csrf_token)


def test_api_error_envelope():
    app = FastAPI()
    app.add_exception_handler(ApiError, api_error_handler)

    @app.get("/boom")
    def boom():
        raise ApiError(404, "artifact_not_found", "Missing artifact", "data/results/equity_curve.parquet")

    client = TestClient(app)
    resp = client.get("/boom")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "artifact_not_found"
    assert body["error"]["message"] == "Missing artifact"

def test_session_lifecycle():
    token = create_session()
    assert is_session_valid(token)
    assert get_session(token) is not None
    destroy_session(token)
    assert not is_session_valid(token)


def test_session_http_cookie():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient

    from api.session import set_session_cookie

    app = FastAPI()

    @app.get("/login")
    def login():
        resp = JSONResponse({"ok": True})
        set_session_cookie(resp, create_session())
        return resp

    client = TestClient(app)
    resp = client.get("/login")
    assert SESSION_COOKIE in resp.headers["set-cookie"]
    assert "HttpOnly" in resp.headers["set-cookie"]


def test_csrf_double_submit():
    token = new_csrf_token()
    assert check_csrf(token, token)
    assert not check_csrf(token, "wrong")


def test_expired_session_rejected():
    token = create_session()
    import api.session as s
    s._SESSIONS[token] = datetime.now(timezone.utc) - timedelta(hours=25)
    assert not is_session_valid(token)

def _auth_app():
    app = FastAPI()
    app.add_exception_handler(ApiError, api_error_handler)
    app.include_router(auth.router, prefix="/api")
    return app


def test_login_success_sets_cookies():
    with patch.dict(os.environ, {"MLCOUNCIL_ADMIN_USERNAME": "admin", "MLCOUNCIL_ADMIN_PASSWORD": "s3cret"}, clear=False):
        client = TestClient(_auth_app())
        resp = client.post("/api/auth/login", json={"username": "admin", "password": "s3cret"})
        assert resp.status_code == 200
        assert "mlcouncil_session" in resp.cookies
        assert "mlcouncil_csrf" in resp.cookies
        assert resp.json()["authenticated"] is True


def test_login_wrong_password_rejected():
    with patch.dict(os.environ, {"MLCOUNCIL_ADMIN_USERNAME": "admin", "MLCOUNCIL_ADMIN_PASSWORD": "s3cret"}, clear=False):
        client = TestClient(_auth_app())
        resp = client.post("/api/auth/login", json={"username": "admin", "password": "nope"})
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "invalid_credentials"


def test_login_unconfigured_returns_503():
    with patch.dict(os.environ, {"MLCOUNCIL_ADMIN_USERNAME": "", "MLCOUNCIL_ADMIN_PASSWORD": ""}, clear=False):
        client = TestClient(_auth_app())
        resp = client.post("/api/auth/login", json={"username": "a", "password": "b"})
        assert resp.status_code == 503


def test_me_requires_session():
    with patch.dict(os.environ, {"MLCOUNCIL_ADMIN_USERNAME": "admin", "MLCOUNCIL_ADMIN_PASSWORD": "s3cret"}, clear=False):
        client = TestClient(_auth_app())
        assert client.get("/api/auth/me").status_code == 401
        login = client.post("/api/auth/login", json={"username": "admin", "password": "s3cret"})
        assert login.status_code == 200
        me = client.get("/api/auth/me")
        assert me.status_code == 200
        assert me.json()["username"] == "admin"


def test_logout_clears_session():
    with patch.dict(os.environ, {"MLCOUNCIL_ADMIN_USERNAME": "admin", "MLCOUNCIL_ADMIN_PASSWORD": "s3cret"}, clear=False):
        client = TestClient(_auth_app())
        client.post("/api/auth/login", json={"username": "admin", "password": "s3cret"})
        assert client.post("/api/auth/logout").status_code == 200
        assert client.get("/api/auth/me").status_code == 401
