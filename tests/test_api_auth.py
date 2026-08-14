# tests/test_api_auth.py
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.errors import ApiError, api_error_handler


def test_api_error_envelope():
    app = FastAPI()
    app.add_exception_handler(ApiError, api_error_handler)

    @app.get("/boom")
    def boom():
        raise ApiError(404, "artifact_not_found", "Missing artifact", "data/results/equity_curve.parquet")

    client = TestClient(app)
    resp = client.get("/boom")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "artifact_not_found"
    assert body["error"]["message"] == "Missing artifact"
