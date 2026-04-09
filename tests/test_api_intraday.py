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


def test_intraday_status_endpoint(client):
    payload = {
        "running": True,
        "paused": False,
        "schedule_minutes": 15,
        "market_session": "regular",
        "last_completed_slot": "2026-04-09T14:30:00+00:00",
        "latest_decision_id": "decision-123",
    }

    with patch(
        "api.routers.intraday.intraday_runtime_service.get_status",
        return_value=payload,
    ):
        resp = client.get("/api/intraday/status")

    assert resp.status_code == 200
    body = resp.json()
    assert body["running"] is True
    assert body["latest_decision_id"] == "decision-123"


def test_intraday_control_start_endpoint(client):
    payload = {
        "running": True,
        "paused": False,
        "schedule_minutes": 15,
        "market_session": "regular",
        "last_completed_slot": None,
        "latest_decision_id": None,
    }

    with patch(
        "api.routers.intraday.intraday_runtime_service.start",
        return_value=payload,
    ):
        resp = client.post("/api/intraday/control/start")

    assert resp.status_code == 200
    assert resp.json()["running"] is True


def test_intraday_latest_decision_endpoint(client):
    payload = {
        "decision_id": "decision-456",
        "as_of": "2026-04-09T14:45:00+00:00",
        "market_session": "regular",
        "agent_trace": {
            "summary": "AAPL remains strongest long candidate.",
            "confidence": 0.74,
        },
        "execution_intents": [{"ticker": "AAPL", "side": "buy"}],
    }

    with patch(
        "api.routers.intraday.intraday_runtime_service.get_latest_decision",
        return_value=payload,
    ):
        resp = client.get("/api/intraday/decisions/latest")

    assert resp.status_code == 200
    body = resp.json()
    assert body["decision_id"] == "decision-456"
    assert body["agent_trace"]["confidence"] == 0.74


def test_intraday_decision_explain_endpoint(client):
    payload = {
        "decision_id": "decision-456",
        "summary": "Momentum + sentiment supported AAPL.",
        "rationale": ["15m momentum positive", "Sentiment stayed positive"],
        "execution_intents": [{"ticker": "AAPL", "side": "buy"}],
    }

    with patch(
        "api.routers.intraday.intraday_runtime_service.explain_decision",
        return_value=payload,
    ):
        resp = client.get("/api/intraday/decisions/decision-456/explain")

    assert resp.status_code == 200
    body = resp.json()
    assert body["decision_id"] == "decision-456"
    assert "rationale" in body


def test_intraday_execute_decision_endpoint(client):
    payload = {
        "orders_submitted": 1,
        "orders_rejected": 0,
        "liquidations": 0,
        "results": [{"symbol": "AAPL", "status": "accepted"}],
        "lineage": {"decision_id": "decision-456", "strategy_version": "intraday-v1"},
        "pretrade": {"blocked": False},
        "reconciliation": {"symbols_to_open": ["AAPL"], "symbols_to_close": []},
    }

    with patch(
        "api.routers.intraday.intraday_runtime_service.execute_decision",
        return_value=payload,
    ):
        resp = client.post("/api/intraday/decisions/decision-456/execute")

    assert resp.status_code == 200
    body = resp.json()
    assert body["orders_submitted"] == 1
    assert body["lineage"]["decision_id"] == "decision-456"
