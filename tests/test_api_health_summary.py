from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient


def test_health_includes_validation_and_runtime_summary(monkeypatch, tmp_path):
    validation_dir = tmp_path / "validation"
    validation_dir.mkdir()
    (validation_dir / "2026-04-20.json").write_text(
        json.dumps(
            {
                "date": "2026-04-20",
                "walk_forward_window_count": 3,
                "oos_sharpe": 0.12,
                "pbo": 0.2,
            }
        )
    )

    runtime_path = tmp_path / "runtime.paper.env"
    runtime_path.write_text(
        "\n".join(
            [
                "ALPACA_BASE_URL=https://api.alpaca.markets",
                "ALPACA_PAPER_KEY=paper-key",
                "ALPACA_PAPER_SECRET=paper-secret",
                "MLCOUNCIL_MAX_DAILY_ORDERS=20",
                "MLCOUNCIL_MAX_TURNOVER=0.30",
                "MLCOUNCIL_MAX_POSITION_SIZE=0.10",
                "MLCOUNCIL_AUTOMATION_PAUSED=false",
            ]
        )
        + "\n"
    )

    monkeypatch.setenv("MLCOUNCIL_ENV_PROFILE", "paper")
    monkeypatch.setenv("MLCOUNCIL_RUNTIME_ENV_PATH", str(runtime_path))

    import api.routers.health as health_module
    import runtime_env

    monkeypatch.setattr(health_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(health_module, "VALIDATION_DIRS", [validation_dir])

    runtime_env.load_runtime_env(override=True)

    from api.main import create_app

    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/api/health")

    body = resp.json()
    assert resp.status_code == 200
    assert "validation_backtest_summary" in body
    assert body["validation_backtest_summary"]["latest"]["walk_forward_window_count"] == 3
    assert body["validation_backtest_summary"]["latest"]["oos_sharpe"] == 0.12
    assert body["validation_backtest_summary"]["latest"]["pbo"] == 0.2
    assert "oos_return" not in body["validation_backtest_summary"]["latest"]
    assert "pbo_proxy" not in body["validation_backtest_summary"]["latest"]
    assert "config_runtime_summary" in body
    assert body["config_runtime_summary"]["status"] == "inconsistent"
    assert body["config_runtime_summary"]["issues"]


def test_health_ignores_operations_json_when_building_validation_summary(monkeypatch, tmp_path):
    operations_dir = tmp_path / "operations"
    operations_dir.mkdir()
    (operations_dir / "2026-04-21.json").write_text(
        json.dumps(
            {
                "date": "2026-04-21",
                "trade_status": "success",
                "walk_forward_window_count": 999,
                "oos_sharpe": 9.9,
                "pbo": 0.0,
            }
        )
    )

    monkeypatch.setenv("MLCOUNCIL_ENV_PROFILE", "paper")
    monkeypatch.setenv("MLCOUNCIL_RUNTIME_ENV_PATH", str(tmp_path / "runtime.paper.env"))

    import api.routers.health as health_module

    monkeypatch.setattr(health_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(health_module, "VALIDATION_DIRS", [tmp_path / "validation", tmp_path / "backtests"])

    summary = health_module._latest_validation_backtest_summary()

    assert summary["status"] == "no_data"


def test_admin_status_badge_supports_consistent_inconsistent_and_error():
    from pathlib import Path

    js = Path("api/static/js/admin.js").read_text(encoding="utf-8")
    assert "'consistent': 'badge-ok'" in js
    assert "'inconsistent': 'badge-warning'" in js
    assert "'error': 'badge-error'" in js
