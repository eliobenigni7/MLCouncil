import sys
import types
import warnings
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch


def _install_slowapi_stub():
    slowapi_mod = types.ModuleType("slowapi")
    errors_mod = types.ModuleType("slowapi.errors")
    extension_mod = types.ModuleType("slowapi.extension")
    util_mod = types.ModuleType("slowapi.util")

    class DummyLimiter:
        def __init__(self, key_func):
            self.key_func = key_func

        def limit(self, _value):
            def decorator(func):
                return func

            return decorator

    class RateLimitExceeded(Exception):
        pass

    slowapi_mod.Limiter = DummyLimiter
    errors_mod.RateLimitExceeded = RateLimitExceeded
    extension_mod._rate_limit_exceeded_handler = lambda request, exc: None
    util_mod.get_remote_address = lambda request: "127.0.0.1"
    sys.modules["slowapi"] = slowapi_mod
    sys.modules["slowapi.errors"] = errors_mod
    sys.modules["slowapi.extension"] = extension_mod
    sys.modules["slowapi.util"] = util_mod


def test_admin_root_renders_html():
    _install_slowapi_stub()
    from api.main import create_app

    with patch.dict(
        "os.environ",
        {"MLCOUNCIL_ENV_PROFILE": "local", "MLCOUNCIL_REQUIRE_API_KEY": "false"},
        clear=False,
    ):
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


def test_admin_api_fails_closed_when_required_key_is_missing():
    _install_slowapi_stub()
    from api.main import create_app

    with patch.dict("os.environ", {"MLCOUNCIL_REQUIRE_API_KEY": "true"}, clear=True):
        app = create_app()
        with pytest.raises(RuntimeError, match="MLCOUNCIL_API_KEY is required"):
            with TestClient(app, raise_server_exceptions=False):
                pass


def test_admin_api_fails_closed_in_paper_profile_without_key():
    _install_slowapi_stub()
    from api.main import create_app

    with patch.dict("os.environ", {"MLCOUNCIL_ENV_PROFILE": "paper"}, clear=True):
        app = create_app()
        with pytest.raises(RuntimeError, match="MLCOUNCIL_API_KEY is required"):
            with TestClient(app, raise_server_exceptions=False):
                pass


def test_admin_js_no_legacy_unsafe_interpolation_in_high_risk_sections():
    admin_js = Path(__file__).resolve().parents[1] / "api" / "static" / "js" / "admin.js"
    text = admin_js.read_text(encoding="utf-8")

    assert "tbody.innerHTML = orders.map" not in text
    assert "histBody.innerHTML = history.trades.map" not in text
    assert "tbody.innerHTML = resp.orders.map" not in text
    assert "div.innerHTML = `" not in text


def test_admin_js_disables_execute_when_trading_paused_or_nonpaper_runtime():
    admin_js = Path(__file__).resolve().parents[1] / "api" / "static" / "js" / "admin.js"
    text = admin_js.read_text(encoding="utf-8")

    assert "status.paused || status.kill_switch_active" in text
    assert "Execute Orders is only available in paper runtime profile" in text


def test_admin_ui_includes_api_key_storage_and_sends_header():
    admin_html = Path(__file__).resolve().parents[1] / "api" / "templates" / "admin.html"
    admin_js = Path(__file__).resolve().parents[1] / "api" / "static" / "js" / "admin.js"

    html_text = admin_html.read_text(encoding="utf-8")
    js_text = admin_js.read_text(encoding="utf-8")

    assert "id=\"api-key\"" in html_text
    assert "sessionStorage.getItem('mlcouncil_api_key')" in js_text
    assert "localStorage.getItem('mlcouncil_api_key')" in js_text
    assert "X-API-Key" in js_text
    assert "sessionStorage.setItem('mlcouncil_api_key'" in js_text
    assert "localStorage.removeItem('mlcouncil_api_key')" in js_text


def test_admin_js_migrates_legacy_api_key_from_localstorage():
    admin_js = Path(__file__).resolve().parents[1] / "api" / "static" / "js" / "admin.js"
    text = admin_js.read_text(encoding="utf-8")

    assert "if (!apiKey)" in text
    assert "apiKey = localStorage.getItem('mlcouncil_api_key') || '';" in text
    assert "sessionStorage.setItem('mlcouncil_api_key', apiKey);" in text


def test_admin_js_pending_orders_use_notional_or_quantity_fallback():
    admin_js = Path(__file__).resolve().parents[1] / "api" / "static" / "js" / "admin.js"
    text = admin_js.read_text(encoding="utf-8")

    assert "loadPendingOrders(date)" in text
    assert "o.notional ?? o.quantity ?? o.target_notional ?? o.target_value ?? o.value ?? o.price ?? 0" in text
    assert "$${Number(orderValue || 0).toFixed(0)}" in text
    assert "(o.quantity || 0) * (o.price || 0)" not in text


def test_admin_js_update_single_setting_uses_supplied_value():
    admin_js = Path(__file__).resolve().parents[1] / "api" / "static" / "js" / "admin.js"
    text = admin_js.read_text(encoding="utf-8")

    assert "async function updateSingleSetting(key, value)" in text
    assert "const values = {};" in text
    assert "values[key] = value;" in text
    assert "JSON.stringify({values})" in text
