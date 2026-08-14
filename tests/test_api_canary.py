from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.errors import ApiError, api_error_handler
from api.routers import canary


def _app():
    app = FastAPI()
    app.add_exception_handler(ApiError, api_error_handler)
    app.include_router(canary.router, prefix="/api")
    return app


def test_flags_endpoint_lists_features(tmp_path, monkeypatch):
    from api.services import canary_service
    canary_yaml = tmp_path / "canary.yaml"
    canary_yaml.write_text(
        "features:\n"
        "  - name: online_learning\n    env: MLCOUNCIL_ONLINE_LEARNING\n    value: 'true'\n    enabled: true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(canary_service, "CANARY_CONFIG_PATH", canary_yaml)
    client = TestClient(_app())
    resp = client.get("/api/canary/flags")
    assert resp.status_code == 200
    assert resp.json()["features"][0]["name"] == "online_learning"
