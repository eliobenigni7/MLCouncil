from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_build_supervisor_prefers_polygon_and_openai_when_keys_are_present(monkeypatch):
    from api.services import intraday_runtime_service as runtime_service

    sentinel_adapter = object()
    sentinel_agent = object()

    monkeypatch.setenv("POLYGON_API_KEY", "polygon-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MLCOUNCIL_INTRADAY_AGENT_PROVIDER", "openai")
    monkeypatch.setenv("MLCOUNCIL_INTRADAY_INTERVAL_MINUTES", "15")

    monkeypatch.setattr(runtime_service, "PolygonMarketDataAdapter", lambda **kwargs: sentinel_adapter)
    monkeypatch.setattr(runtime_service, "OpenAIIntradayAgent", lambda **kwargs: sentinel_agent)

    supervisor = runtime_service._build_supervisor()

    assert supervisor.market_data_adapter is sentinel_adapter
    assert supervisor.agent_orchestrator is sentinel_agent


def test_build_supervisor_uses_rule_based_agent_by_default(monkeypatch):
    from api.services import intraday_runtime_service as runtime_service

    sentinel_adapter = object()

    monkeypatch.setenv("POLYGON_API_KEY", "polygon-test")
    monkeypatch.delenv("MLCOUNCIL_AUTO_EXECUTE", raising=False)
    monkeypatch.delenv("MLCOUNCIL_INTRADAY_AGENT_PROVIDER", raising=False)
    monkeypatch.setattr(runtime_service, "PolygonMarketDataAdapter", lambda **kwargs: sentinel_adapter)

    supervisor = runtime_service._build_supervisor()

    assert supervisor.market_data_adapter is sentinel_adapter
    assert supervisor.agent_orchestrator.__class__.__name__ == "FallbackIntradayAgent"
    assert supervisor.executor is None


def test_build_supervisor_enables_executor_only_when_auto_execute_is_true(monkeypatch):
    from api.services import intraday_runtime_service as runtime_service

    sentinel_adapter = object()

    monkeypatch.setenv("POLYGON_API_KEY", "polygon-test")
    monkeypatch.setenv("MLCOUNCIL_AUTO_EXECUTE", "true")
    monkeypatch.setattr(
        runtime_service,
        "PolygonMarketDataAdapter",
        lambda **kwargs: sentinel_adapter,
    )

    supervisor = runtime_service._build_supervisor()

    assert supervisor.market_data_adapter is sentinel_adapter
    assert callable(supervisor.executor)


def test_intraday_supervisor_script_does_not_exit_immediately_with_import_error():
    state_path = ROOT / "data" / "intraday" / "supervisor_state.json"
    original_state = state_path.read_text() if state_path.exists() else None
    proc = subprocess.Popen(
        [sys.executable, "scripts/run_intraday_supervisor.py"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        time.sleep(1.0)
        return_code = proc.poll()
        if return_code is not None:
            _, stderr = proc.communicate(timeout=5)
            assert "ModuleNotFoundError: No module named 'api'" not in stderr
            raise AssertionError(f"supervisor script exited early with code {return_code}: {stderr}")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        if original_state is None:
            state_path.unlink(missing_ok=True)
        else:
            state_path.write_text(original_state)
