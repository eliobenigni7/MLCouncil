# tests/test_api_analytics.py
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.errors import ApiError, api_error_handler
from api.routers import analytics


def _app():
    app = FastAPI()
    app.add_exception_handler(ApiError, api_error_handler)
    app.include_router(analytics.router, prefix="/api")
    return app


def test_equity_endpoint_404_envelope_when_missing(tmp_path, monkeypatch):
    from api.services import analytics_service
    monkeypatch.setattr(analytics_service, "DATA_DIR", tmp_path)
    client = TestClient(_app())
    resp = client.get("/api/analytics/equity")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "artifact_not_found"
