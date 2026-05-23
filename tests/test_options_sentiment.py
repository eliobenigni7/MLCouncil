"""Tests for options-implied sentiment scaffold (Phase 2.6)."""

from __future__ import annotations

import json

import httpx
import pandas as pd
import pytest


def test_options_sentiment_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MLCOUNCIL_OPTIONS_SENTIMENT", raising=False)
    from models.options_sentiment import options_sentiment_enabled

    assert options_sentiment_enabled() is False


def test_metrics_from_chain_put_call_and_skew():
    from models.options_sentiment import metrics_from_chain

    contracts = [
        {
            "details": {"contract_type": "put"},
            "day": {"volume": 100},
            "greeks": {"implied_volatility": 0.35},
        },
        {
            "details": {"contract_type": "call"},
            "day": {"volume": 50},
            "greeks": {"implied_volatility": 0.25},
        },
    ]
    m = metrics_from_chain("AAPL", contracts)
    assert m.put_call_ratio == pytest.approx(2.0)
    assert m.skew_proxy > 0


def test_polygon_client_parses_snapshot(monkeypatch):
    from models.options_sentiment import PolygonOptionsClient

    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        assert "/v3/snapshot/options/AAPL" in str(request.url)
        body = {
            "results": [
                {"details": {"contract_type": "call"}, "day": {"volume": 10}, "greeks": {"implied_volatility": 0.2}},
                {"details": {"contract_type": "put"}, "day": {"volume": 20}, "greeks": {"implied_volatility": 0.3}},
            ]
        }
        return httpx.Response(200, json=body)

    transport = httpx.MockTransport(handler)
    client = PolygonOptionsClient(client=httpx.Client(transport=transport, base_url="https://api.polygon.io"))
    chain = client.fetch_chain_snapshot("AAPL")
    assert len(chain) == 2
    client.close()


def test_log_shadow_signals_roundtrip(tmp_path):
    from models.options_sentiment import log_shadow_signals

    signals = pd.Series([0.5, -0.2], index=["AAPL", "MSFT"], name="options_sentiment_shadow")
    path = log_shadow_signals("2024-06-01", signals, output_path=tmp_path / "shadow.parquet")
    df = pd.read_parquet(path)
    assert len(df) == 2
    assert "put_call_ratio" in df.columns
