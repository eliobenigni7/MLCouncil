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


def test_get_runtime_settings(client, tmp_path, monkeypatch):
    from api.services import monitoring_service

    runtime_env = tmp_path / "runtime.env"
    runtime_env.write_text(
        "OPENAI_API_KEY=sk-test\n"
        "ALPACA_API_KEY=legacy-paper-key\n"
        "ALPACA_SECRET_KEY=legacy-paper-secret\n"
        "POLYGON_API_KEY=polygon-test\n"
    )
    monkeypatch.setattr(monitoring_service, "RUNTIME_ENV_PATH", runtime_env)
    monkeypatch.delenv("ALPACA_PAPER_KEY", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_SECRET", raising=False)
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    resp = client.get("/api/monitoring/settings")
    assert resp.status_code == 200

    body = resp.json()
    keys = {item["key"]: item for item in body["settings"]}
    assert keys["OPENAI_API_KEY"]["value"] == "sk-test"
    assert keys["POLYGON_API_KEY"]["value"] == "polygon-test"
    assert keys["ALPACA_PAPER_KEY"]["value"] == "legacy-paper-key"
    assert keys["ALPACA_PAPER_SECRET"]["value"] == "legacy-paper-secret"


def test_get_runtime_settings_prefers_env_over_placeholder_file_values(
    client, tmp_path, monkeypatch
):
    from api.services import monitoring_service

    runtime_env = tmp_path / "runtime.env"
    runtime_env.write_text(
        "ALPACA_PAPER_KEY=replace-me\n"
        "ALPACA_PAPER_SECRET=replace-me\n"
        "MLCOUNCIL_AUTO_EXECUTE=false\n"
    )
    monkeypatch.setattr(monitoring_service, "RUNTIME_ENV_PATH", runtime_env)
    monkeypatch.setenv("ALPACA_API_KEY", "runtime-paper-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "runtime-paper-secret")
    monkeypatch.setenv("MLCOUNCIL_AUTO_EXECUTE", "true")
    monkeypatch.delenv("ALPACA_PAPER_KEY", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_SECRET", raising=False)

    resp = client.get("/api/monitoring/settings")
    assert resp.status_code == 200

    body = resp.json()
    keys = {item["key"]: item for item in body["settings"]}
    assert keys["ALPACA_PAPER_KEY"]["value"] == "runtime-paper-key"
    assert keys["ALPACA_PAPER_SECRET"]["value"] == "runtime-paper-secret"
    assert keys["MLCOUNCIL_AUTO_EXECUTE"]["value"] == "true"


def test_update_runtime_settings_persists_shared_env(client, tmp_path, monkeypatch):
    from api.services import monitoring_service

    runtime_env = tmp_path / "runtime.env"
    monkeypatch.setattr(monitoring_service, "RUNTIME_ENV_PATH", runtime_env)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_BASE_URL", raising=False)

    resp = client.put(
        "/api/monitoring/settings",
        json={
            "values": {
                "OPENAI_API_KEY": "sk-updated",
                "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
            }
        },
    )
    assert resp.status_code == 200

    payload = resp.json()
    keys = {item["key"]: item for item in payload["settings"]}
    assert keys["OPENAI_API_KEY"]["value"] == "sk-updated"
    assert keys["ALPACA_BASE_URL"]["value"] == "https://paper-api.alpaca.markets"
    assert "OPENAI_API_KEY=sk-updated" in runtime_env.read_text()
    assert "ALPACA_BASE_URL=https://paper-api.alpaca.markets" in runtime_env.read_text()
