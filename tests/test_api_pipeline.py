import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from api.main import create_app
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_trigger_pipeline_success(client):
    mock_status = type("PS", (), {
        "last_run_id": "abc-123",
        "status": "STARTED",
        "start_time": "2026-04-02T21:30:00",
        "end_time": None,
        "partition": "2026-04-02",
    })()

    with patch("api.routers.pipeline.dagster_client") as mock_dc:
        mock_dc.trigger_run = AsyncMock(return_value="abc-123")
        mock_dc.get_last_status = AsyncMock(return_value=mock_status)
        resp = client.post("/api/pipeline/run", json={"partition": "2026-04-02"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == "abc-123"


def test_get_pipeline_status(client):
    mock_status = type("PS", (), {
        "last_run_id": "abc-123",
        "status": "SUCCESS",
        "start_time": "2026-04-02T21:30:00",
        "end_time": "2026-04-02T22:00:00",
        "partition": "2026-04-02",
    })()

    with patch("api.routers.pipeline.dagster_client") as mock_dc:
        mock_dc.get_last_status = AsyncMock(return_value=mock_status)
        resp = client.get("/api/pipeline/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "SUCCESS"
    assert body["run_id"] == "abc-123"


def test_pipeline_status_unreachable(client):
    mock_status = type("PS", (), {
        "last_run_id": None,
        "status": "unreachable",
        "start_time": None,
        "end_time": None,
        "partition": None,
    })()

    with patch("api.routers.pipeline.dagster_client") as mock_dc:
        mock_dc.get_last_status = AsyncMock(return_value=mock_status)
        resp = client.get("/api/pipeline/status")
    assert resp.status_code == 200
    assert resp.json()["status"] == "unreachable"


def test_latest_partition_endpoint(client):
    with patch("api.routers.pipeline.dagster_client") as mock_dc:
        mock_dc.get_latest_partition_key = AsyncMock(return_value="2026-04-02")
        resp = client.get("/api/pipeline/latest-partition")
    assert resp.status_code == 200
    assert resp.json() == {"partition": "2026-04-02"}


def test_trigger_pipeline_enqueues_auto_execute_when_enabled(client):
    with patch("api.routers.pipeline.dagster_client") as mock_dc, patch(
        "api.routers.pipeline.schedule_auto_execute"
    ) as mock_schedule, patch.dict(
        "os.environ",
        {"MLCOUNCIL_AUTO_EXECUTE": "true"},
        clear=False,
    ):
        mock_dc.trigger_run = AsyncMock(return_value="run-auto-123")
        resp = client.post("/api/pipeline/run", json={"partition": "2026-04-03"})

    assert resp.status_code == 200
    assert resp.json()["run_id"] == "run-auto-123"
    mock_schedule.assert_called_once_with(run_id="run-auto-123", partition="2026-04-03")


def test_trigger_pipeline_skips_auto_execute_when_disabled(client):
    with patch("api.routers.pipeline.dagster_client") as mock_dc, patch(
        "api.routers.pipeline.schedule_auto_execute"
    ) as mock_schedule, patch.dict(
        "os.environ",
        {"MLCOUNCIL_AUTO_EXECUTE": "false"},
        clear=False,
    ):
        mock_dc.trigger_run = AsyncMock(return_value="run-manual-123")
        resp = client.post("/api/pipeline/run", json={"partition": "2026-04-03"})

    assert resp.status_code == 200
    mock_schedule.assert_not_called()


def test_pipeline_automation_status_endpoint(client):
    payload = {
        "run_id": "run-auto-123",
        "partition": "2026-04-03",
        "status": "executed",
        "created_at": "2026-04-03T21:00:00+00:00",
        "updated_at": "2026-04-03T21:05:00+00:00",
        "execution_result": {"orders_submitted": 3},
        "error": None,
    }

    with patch("api.routers.pipeline.get_task_state", return_value=payload):
        resp = client.get("/api/pipeline/automation/run-auto-123")

    assert resp.status_code == 200
    assert resp.json()["status"] == "executed"
    assert resp.json()["execution_result"]["orders_submitted"] == 3
