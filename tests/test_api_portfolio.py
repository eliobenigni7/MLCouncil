import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from api.main import create_app
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_get_weights(client):
    resp = client.get("/api/portfolio/weights")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, dict)


def test_get_order_dates(client):
    resp = client.get("/api/portfolio/orders/dates")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)


def test_get_orders_for_date_404(client):
    resp = client.get("/api/portfolio/orders/2099-01-01")
    assert resp.status_code == 404
