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


def test_get_health_signals_ok_without_artifacts(client, monkeypatch, tmp_path):
    """Senza artefatti su disco /health risponde 200 con tutti i segnali."""
    from api.services import monitoring_service

    monkeypatch.setattr(monitoring_service, "Path", lambda *parts: tmp_path / "results")

    resp = client.get("/api/monitoring/health")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {
        "tda_warning",
        "causal_drift",
        "adwin_drift",
        "ddm_drift",
        "evidently_drift",
    }
    for signal in body.values():
        assert set(signal) == {"level", "value", "threshold", "note"}
        assert signal["level"] == "ok"
        assert signal["note"]


def test_get_health_signals_reads_artifacts(client, monkeypatch, tmp_path):
    """Con artefatti presenti i livelli riflettono le soglie dei check."""
    import json as _json

    from api.services import monitoring_service

    results = tmp_path / "results"
    results.mkdir(parents=True)
    (results / "causal_drift_latest.json").write_text(
        _json.dumps({"change_fraction": 0.4, "status": "alert", "is_alert": True})
    )
    (results / "tda_warning_latest.json").write_text(
        _json.dumps({"is_alert": True, "beta1_proxy": 0.41, "threshold": 0.35})
    )
    monkeypatch.setattr(monitoring_service, "Path", lambda *parts: results)

    resp = client.get("/api/monitoring/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["causal_drift"]["level"] == "alert"
    assert body["causal_drift"]["value"] == 0.4
    assert body["causal_drift"]["threshold"] == 0.25
    assert body["tda_warning"]["level"] == "alert"
    assert body["evidently_drift"]["level"] == "ok"
    assert body["adwin_drift"]["level"] == "ok"


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
