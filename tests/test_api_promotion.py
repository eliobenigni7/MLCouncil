from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.errors import ApiError, api_error_handler
from api.routers import promotion


def _app():
    app = FastAPI()
    app.add_exception_handler(ApiError, api_error_handler)
    app.include_router(promotion.router, prefix="/api")
    return app


def test_manifest_404_when_missing(tmp_path, monkeypatch):
    from api.services import promotion_service
    monkeypatch.setattr(promotion_service, "MANIFEST_PATH", tmp_path / "nope.yaml")
    client = TestClient(_app())
    resp = client.get("/api/promotion/manifest")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "artifact_not_found"
