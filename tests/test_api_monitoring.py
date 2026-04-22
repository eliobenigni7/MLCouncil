import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch


@pytest.fixture(scope="module")
def client():
    from api.main import create_app

    with patch.dict(
        "os.environ",
        {"MLCOUNCIL_ENV_PROFILE": "local", "MLCOUNCIL_REQUIRE_API_KEY": "false"},
        clear=False,
    ):
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
    import runtime_env as runtime_env_module
    from api.services import monitoring_service

    runtime_env = tmp_path / "runtime.env"
    runtime_env.write_text(
        "OPENAI_API_KEY=sk-test\n"
        "ALPACA_API_KEY=legacy-paper-key\n"
        "ALPACA_SECRET_KEY=legacy-paper-secret\n"
        "POLYGON_API_KEY=polygon-test\n"
    )
    monkeypatch.setattr(monitoring_service, "RUNTIME_ENV_PATH", runtime_env)
    monkeypatch.setattr(
        runtime_env_module, "get_project_dotenv_path", lambda: tmp_path / ".env.missing"
    )
    monkeypatch.delenv("ALPACA_PAPER_KEY", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_SECRET", raising=False)
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)

    resp = client.get("/api/monitoring/settings")
    assert resp.status_code == 200

    body = resp.json()
    keys = {item["key"]: item for item in body["settings"]}
    assert keys["OPENAI_API_KEY"]["value"] == monitoring_service.MASKED_SECRET_VALUE
    assert keys["POLYGON_API_KEY"]["value"] == monitoring_service.MASKED_SECRET_VALUE
    assert keys["ALPACA_PAPER_KEY"]["value"] == monitoring_service.MASKED_SECRET_VALUE
    assert keys["ALPACA_PAPER_SECRET"]["value"] == monitoring_service.MASKED_SECRET_VALUE
    assert keys["OPENAI_API_KEY"]["configured"] is True


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
    assert keys["ALPACA_PAPER_KEY"]["value"] == monitoring_service.MASKED_SECRET_VALUE
    assert keys["ALPACA_PAPER_SECRET"]["value"] == monitoring_service.MASKED_SECRET_VALUE
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
    assert keys["OPENAI_API_KEY"]["value"] == monitoring_service.MASKED_SECRET_VALUE
    assert keys["ALPACA_BASE_URL"]["value"] == "https://paper-api.alpaca.markets"
    assert "OPENAI_API_KEY=sk-updated" in runtime_env.read_text()
    assert "ALPACA_BASE_URL=https://paper-api.alpaca.markets" in runtime_env.read_text()


def test_update_runtime_settings_ignores_masked_secret_placeholder(client, tmp_path, monkeypatch):
    from api.services import monitoring_service

    runtime_env = tmp_path / "runtime.env"
    runtime_env.write_text(
        "OPENAI_API_KEY=sk-existing\n"
        "MLCOUNCIL_AUTO_EXECUTE=false\n"
    )
    monkeypatch.setattr(monitoring_service, "RUNTIME_ENV_PATH", runtime_env)

    resp = client.put(
        "/api/monitoring/settings",
        json={
            "values": {
                "OPENAI_API_KEY": monitoring_service.MASKED_SECRET_VALUE,
                "MLCOUNCIL_AUTO_EXECUTE": "true",
            }
        },
    )
    assert resp.status_code == 200

    payload = resp.json()
    keys = {item["key"]: item for item in payload["settings"]}
    assert keys["OPENAI_API_KEY"]["value"] == monitoring_service.MASKED_SECRET_VALUE
    assert keys["MLCOUNCIL_AUTO_EXECUTE"]["value"] == "true"
    contents = runtime_env.read_text()
    assert "OPENAI_API_KEY=sk-existing" in contents
    assert "MLCOUNCIL_AUTO_EXECUTE=true" in contents
