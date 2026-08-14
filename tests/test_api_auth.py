# tests/test_api_auth.py
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.errors import ApiError, api_error_handler
from api.auth import is_api_key_required
from api.main import create_app
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

def test_api_key_required_semantics():
    with patch.dict(os.environ, {"MLCOUNCIL_REQUIRE_API_KEY": "true"}, clear=False):
        assert is_api_key_required()
    with patch.dict(os.environ, {"MLCOUNCIL_ENV_PROFILE": "paper", "MLCOUNCIL_REQUIRE_API_KEY": "false"}, clear=False):
        assert is_api_key_required()

def _full_app():
    # api.main is imported at module level: load_runtime_env() inside api.main
    # overwrites MLCOUNCIL_ENV_PROFILE from .env (paper on dev machines), which
    # would make set_session_cookie() emit a Secure cookie that httpx refuses
    # to send over http://testserver. Importing before any env patch keeps the
    # patched profile effective during the test.
    return create_app()


def test_session_auth_flows_through_middleware():
    with patch.dict(os.environ, {
        "MLCOUNCIL_ENV_PROFILE": "local",
        # Deviation from plan: REQUIRE_API_KEY=true (plan said "false"). The
        # conftest autouse fixture stubs get_configured_api_key to "" and
        # request_is_authenticated treats "API key not required" as
        # authenticated, so with "false" an unauthenticated request is never
        # rejected and the 401 assertion cannot hold.
        "MLCOUNCIL_REQUIRE_API_KEY": "true",
        "MLCOUNCIL_ADMIN_USERNAME": "admin",
        "MLCOUNCIL_ADMIN_PASSWORD": "s3cret",
    }, clear=False):
        app = _full_app()
        # raise_server_exceptions=False: /api/pipeline/status calls the real
        # Dagster client (existing tests always mock it), which raises
        # ConnectError when no Dagster server is reachable -> 500, not 401.
        client = TestClient(app, raise_server_exceptions=False)
        # unauth: protected endpoint rejected
        assert client.get("/api/pipeline/status").status_code == 401
        # login then access
        login = client.post("/api/auth/login", json={"username": "admin", "password": "s3cret"})
        assert login.status_code == 200
        resp = client.get("/api/pipeline/status")
        # service may 404/500 on missing data, but must NOT be 401
        assert resp.status_code != 401


def test_legacy_admin_at_admin_prefix():
    with patch.dict(os.environ, {
        "MLCOUNCIL_ENV_PROFILE": "local",
        "MLCOUNCIL_REQUIRE_API_KEY": "false",
        "MLCOUNCIL_ADMIN_USERNAME": "admin",
        "MLCOUNCIL_ADMIN_PASSWORD": "s3cret",
    }, clear=False):
        client = TestClient(_full_app())
        resp = client.get("/admin")
        assert resp.status_code == 200
        assert "admin.html" in resp.text or "MLCouncil" in resp.text
