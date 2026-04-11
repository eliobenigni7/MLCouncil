from __future__ import annotations

from fastapi.testclient import TestClient
from unittest.mock import patch

import pytest


@pytest.fixture(scope="module")
def client():
    from api.main import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def test_trading_preflight_endpoint(client):
    payload = {
        "date": "2026-04-08",
        "paper": True,
        "paused": False,
        "runtime_profile": "paper",
        "lineage": {"pipeline_run_id": "run-preflight"},
        "pretrade": {"blocked": False, "reason": None, "breaches": []},
        "reconciliation": {"symbols_to_open": ["AAPL"], "symbols_to_close": []},
    }

    with patch(
        "api.routers.trading.trading_service.service.build_pretrade_snapshot",
        return_value=payload,
    ):
        resp = client.get("/api/trading/preflight/2026-04-08")

    assert resp.status_code == 200
    body = resp.json()
    assert body["pretrade"]["blocked"] is False
    assert body["lineage"]["pipeline_run_id"] == "run-preflight"


def test_trading_reconcile_endpoint(client):
    payload = {
        "date": "2026-04-08",
        "symbols_to_open": ["AAPL"],
        "symbols_to_close": ["MSFT"],
        "symbols_to_resize": [],
    }

    with patch(
        "api.routers.trading.trading_service.service.get_reconciliation",
        return_value=payload,
    ):
        resp = client.get("/api/trading/reconcile/2026-04-08")

    assert resp.status_code == 200
    assert resp.json()["symbols_to_close"] == ["MSFT"]


def test_execute_orders_returns_conflict_for_blocked_pretrade(client):
    payload = {
        "error": "Trading paused by kill switch",
        "pretrade": {"blocked": True, "reason": "Trading paused by kill switch"},
        "reconciliation": {"symbols_to_open": [], "symbols_to_close": []},
        "lineage": {"pipeline_run_id": "run-blocked"},
    }

    with patch(
        "api.routers.trading.trading_service.service.execute_orders",
        return_value=payload,
    ):
        resp = client.post("/api/trading/execute", json={"date": "2026-04-08"})

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error"] == "Trading paused by kill switch"
    assert detail["pretrade"]["blocked"] is True


def test_trading_endpoints_require_api_key_when_configured(client):
    with patch.dict("os.environ", {"MLCOUNCIL_API_KEY": "secret-key"}, clear=True):
        resp = client.get("/api/trading/status")

    assert resp.status_code == 401
    assert resp.json()["detail"] == "Missing X-API-Key header"


def test_trading_endpoints_accept_valid_api_key(client):
    with patch.dict("os.environ", {"MLCOUNCIL_API_KEY": "secret-key"}, clear=True):
        with patch(
            "api.routers.trading.trading_service.service.get_status",
            return_value={"connected": True, "paper": True},
        ):
            resp = client.get(
                "/api/trading/status",
                headers={"X-API-Key": "secret-key"},
            )

    assert resp.status_code == 200
    assert resp.json()["connected"] is True
