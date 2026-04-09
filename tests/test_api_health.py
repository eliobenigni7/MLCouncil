import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from api.main import create_app
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("ok", "degraded", "unhealthy")


def test_health_includes_version(client):
    resp = client.get("/api/health")
    body = resp.json()
    assert "version" in body


def test_health_includes_components(client):
    resp = client.get("/api/health")
    body = resp.json()
    assert "components" in body
    assert "data_freshness" in body["components"]
    assert "arctic_store" in body["components"]
    assert "runtime_env" in body["components"]
    assert "trading_operations" in body["components"]
    assert "intraday_supervisor" in body["components"]


def test_health_dagster_endpoint(client):
    mock_ps = type("PS", (), {
        "last_run_id": None,
        "status": "unreachable",
        "start_time": None,
        "end_time": None,
        "partition": None,
    })()
    
    with patch("api.routers.health._dagster_client") as mock_dc:
        mock_dc.get_last_status = AsyncMock(return_value=mock_ps)
        resp = client.get("/api/health/dagster")
    
    assert resp.status_code == 200
    body = resp.json()
    assert body["reachable"] is False


def test_health_treats_empty_arctic_and_missing_alert_file_as_bootstrap_states(client, tmp_path: Path):
    with patch("api.routers.health._data_freshness", return_value={"status": "fresh", "last_order_date": "2026-04-02", "days_ago": 0}):
        with patch("api.routers.health._arctic_store_status", return_value="empty"):
            with patch("api.routers.health._monitoring_status", return_value="idle"):
                with patch(
                    "api.routers.health._runtime_env_status",
                    return_value={"status": "valid", "profile": "local", "valid": True, "missing": [], "errors": []},
                ):
                    with patch(
                        "api.routers.health._latest_operations_status",
                        return_value={"status": "no_runs", "last_date": None},
                    ):
                        resp = client.get("/api/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["components"]["monitoring"] == "idle"
    assert body["components"]["trading_operations"] == "no_runs"


def test_health_degrades_when_runtime_profile_invalid(client):
    with patch("api.routers.health._data_freshness", return_value={"status": "fresh", "last_order_date": "2026-04-02", "days_ago": 0}):
        with patch("api.routers.health._arctic_store_status", return_value="ok"):
            with patch("api.routers.health._monitoring_status", return_value="idle"):
                with patch(
                    "api.routers.health._runtime_env_status",
                    return_value={
                        "status": "invalid",
                        "profile": "paper",
                        "valid": False,
                        "missing": ["ALPACA_PAPER_KEY"],
                        "errors": ["Paper profile must point to Alpaca paper endpoint."],
                    },
                ):
                    with patch(
                        "api.routers.health._latest_operations_status",
                        return_value={"status": "success", "last_date": "2026-04-08"},
                    ):
                        resp = client.get("/api/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["components"]["runtime_env"] == "invalid"
