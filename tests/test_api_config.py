import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from api.main import create_app
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_get_universe(client):
    resp = client.get("/api/config/universe")
    assert resp.status_code == 200
    body = resp.json()
    assert "universe" in body
    assert "tickers" in body["universe"]


def test_get_models(client):
    resp = client.get("/api/config/models")
    assert resp.status_code == 200
    body = resp.json()
    assert "lgbm" in body


def test_get_regime_weights(client):
    resp = client.get("/api/config/regime-weights")
    assert resp.status_code == 200
    body = resp.json()
    assert "regime_weights" in body
