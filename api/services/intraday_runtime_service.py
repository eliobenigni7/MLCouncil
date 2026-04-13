from __future__ import annotations

import os
from pathlib import Path

from api.services import trading_service
from council.mlflow_utils import log_intraday_decision
from intraday.agent import FallbackIntradayAgent, OpenAIIntradayAgent
from intraday.market_data import AlpacaMarketDataAdapter, PolygonMarketDataAdapter
from intraday.supervisor import IntradaySupervisor
from runtime_env import load_runtime_env

load_runtime_env()

_ROOT = Path(__file__).parents[2]


def _load_intraday_universe() -> list[str]:
    env_value = os.getenv("MLCOUNCIL_INTRADAY_UNIVERSE", "")
    if env_value.strip():
        return [
            ticker.strip().upper() for ticker in env_value.split(",") if ticker.strip()
        ]
    return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA"]


def _build_supervisor() -> IntradaySupervisor:
    schedule_minutes = int(os.getenv("MLCOUNCIL_INTRADAY_INTERVAL_MINUTES", "15"))
    storage_dir = _ROOT / "data" / "intraday"

    fallback_adapter = None
    try:
        fallback_adapter = AlpacaMarketDataAdapter()
    except Exception:
        fallback_adapter = None

    try:
        adapter = PolygonMarketDataAdapter(fallback_adapter=fallback_adapter)
    except Exception:
        from intraday.market_data import MarketSnapshot
        from datetime import datetime
        from zoneinfo import ZoneInfo

        class _FallbackAdapter:
            def get_market_snapshot(self, *, as_of: datetime, universe: list[str]):
                bars = {
                    ticker: {"close": 0.0, "return_15m": 0.0} for ticker in universe
                }
                return MarketSnapshot(
                    as_of=as_of.astimezone(ZoneInfo("America/New_York")),
                    universe=universe,
                    bars=bars,
                    source="bootstrap-fallback",
                    session="regular",
                )

            def get_news_snapshot(self, *, as_of: datetime, universe: list[str]):
                del as_of, universe
                return []

        adapter = _FallbackAdapter()

    agent_provider = (
        os.getenv("MLCOUNCIL_INTRADAY_AGENT_PROVIDER", "rule-based").strip().lower()
    )
    if agent_provider == "openai":
        try:
            agent = OpenAIIntradayAgent(
                fallback_agent=FallbackIntradayAgent(),
            )
        except Exception:
            agent = FallbackIntradayAgent()
    else:
        agent = FallbackIntradayAgent()

    return IntradaySupervisor(
        market_data_adapter=adapter,
        agent_orchestrator=agent,
        storage_dir=storage_dir,
        universe=_load_intraday_universe(),
        schedule_minutes=schedule_minutes,
        executor=lambda payload: trading_service.service.execute_intraday_decision(
            payload
        ),
    )


_supervisor = _build_supervisor()


def get_status():
    return _supervisor.get_status()


def get_health():
    return _supervisor.get_health()


def start():
    return _supervisor.start()


def pause():
    return _supervisor.pause()


def resume():
    return _supervisor.resume()


def stop():
    return _supervisor.stop()


def get_latest_decision():
    return _supervisor.get_latest_decision()


def explain_decision(decision_id: str):
    return _supervisor.explain_decision(decision_id)


def list_decisions(limit: int = 20):
    return _supervisor.list_decisions(limit=limit)


def run_cycle():
    payload = _supervisor.run_cycle().to_dict()
    _log_decision(payload)
    try:
        trading_service.service.execute_intraday_decision(payload)
    except Exception:
        pass
    return payload


def execute_decision(decision_id: str):
    full_payload = _supervisor.get_latest_decision()
    if full_payload is None or full_payload.get("decision_id") != decision_id:
        full_payload = _load_decision_payload(decision_id)
    if full_payload is None:
        raise KeyError(decision_id)
    return trading_service.service.execute_intraday_decision(full_payload)


def _load_decision_payload(decision_id: str):
    for payload in _supervisor.list_decisions(limit=500):
        if payload.get("decision_id") == decision_id:
            return payload
    return None


def _log_decision(payload: dict):
    if os.getenv("MLCOUNCIL_INTRADAY_LOG_TO_MLFLOW", "false").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    try:
        log_intraday_decision(
            decision=payload,
            agent_trace=payload.get("agent_trace", {}),
            pipeline_run_id=f"intraday-{payload.get('decision_id', 'unknown')}",
            data_version=str(payload.get("data_snapshot_version", "unknown")),
            feature_version=str(
                payload.get("feature_snapshot", {}).get("version", "unknown")
            ),
            environment=os.getenv("MLCOUNCIL_ENV_PROFILE", "local"),
            model_name="intraday-council",
            model_version=str(payload.get("strategy_version", "intraday-v1")),
        )
    except Exception:
        return
