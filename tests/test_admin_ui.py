import sys
import types
import warnings

from fastapi.testclient import TestClient
from unittest.mock import patch


def _install_slowapi_stub():
    slowapi_mod = types.ModuleType("slowapi")
    util_mod = types.ModuleType("slowapi.util")

    class DummyLimiter:
        def __init__(self, key_func):
            self.key_func = key_func

    slowapi_mod.Limiter = DummyLimiter
    util_mod.get_remote_address = lambda request: "127.0.0.1"
    sys.modules["slowapi"] = slowapi_mod
    sys.modules["slowapi.util"] = util_mod


def test_admin_root_renders_html():
    _install_slowapi_stub()
    from api.main import create_app

    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/")

    assert resp.status_code == 200
    assert "MLCouncil Admin" in resp.text
    assert "API Settings" in resp.text


def test_admin_api_allows_requests_when_api_key_is_not_configured():
    _install_slowapi_stub()
    from api.main import create_app

    app = create_app()
    with patch.dict("os.environ", {}, clear=True):
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/api/config/universe")

    assert resp.status_code == 200


def test_admin_api_requires_key_when_configured():
    _install_slowapi_stub()
    from api.main import create_app

    app = create_app()
    with patch.dict("os.environ", {"MLCOUNCIL_API_KEY": "secret-key"}, clear=True):
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/api/config/universe")

    assert resp.status_code == 401


def test_admin_api_accepts_valid_key_when_configured():
    _install_slowapi_stub()
    from api.main import create_app

    app = create_app()
    with patch.dict("os.environ", {"MLCOUNCIL_API_KEY": "secret-key"}, clear=True):
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get(
                "/api/config/universe",
                headers={"X-API-Key": "secret-key"},
            )

    assert resp.status_code == 200


def test_admin_api_does_not_warn_when_key_is_optional():
    _install_slowapi_stub()
    from api.main import create_app

    with patch.dict("os.environ", {}, clear=True):
        app = create_app()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with TestClient(app, raise_server_exceptions=False):
                pass

    assert not [w for w in caught if "MLCOUNCIL_API_KEY" in str(w.message)]


def test_admin_api_warns_when_required_key_is_missing():
    _install_slowapi_stub()
    from api.main import create_app

    with patch.dict("os.environ", {"MLCOUNCIL_REQUIRE_API_KEY": "true"}, clear=True):
        app = create_app()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with TestClient(app, raise_server_exceptions=False):
                pass

    messages = [str(w.message) for w in caught]
    assert any("MLCOUNCIL_API_KEY is required" in message for message in messages)
