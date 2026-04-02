import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from api.main import create_app
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_get_alerts(client):
    resp = client.get("/api/monitoring/alerts")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)


def test_get_alert_history(client):
    resp = client.get("/api/monitoring/alerts/history?limit=10")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
