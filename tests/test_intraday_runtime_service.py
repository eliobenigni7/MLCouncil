from __future__ import annotations


def test_build_supervisor_prefers_polygon_and_openai_when_keys_are_present(monkeypatch):
    from api.services import intraday_runtime_service as runtime_service

    sentinel_adapter = object()
    sentinel_agent = object()

    monkeypatch.setenv("POLYGON_API_KEY", "polygon-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MLCOUNCIL_INTRADAY_INTERVAL_MINUTES", "15")

    monkeypatch.setattr(runtime_service, "PolygonMarketDataAdapter", lambda **kwargs: sentinel_adapter)
    monkeypatch.setattr(runtime_service, "OpenAIIntradayAgent", lambda **kwargs: sentinel_agent)

    supervisor = runtime_service._build_supervisor()

    assert supervisor.market_data_adapter is sentinel_adapter
    assert supervisor.agent_orchestrator is sentinel_agent
