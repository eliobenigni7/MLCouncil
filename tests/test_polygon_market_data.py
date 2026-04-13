from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
import pytest


def test_polygon_market_data_adapter_builds_rich_snapshot_and_news(monkeypatch):
    from intraday.market_data import PolygonMarketDataAdapter

    monkeypatch.setenv("POLYGON_API_KEY", "polygon-test")
    requests_seen: list[str] = []

    class FallbackAdapter:
        def get_market_snapshot(self, *, as_of, universe):
            from intraday.market_data import MarketSnapshot

            return MarketSnapshot(
                as_of=as_of,
                universe=universe,
                source="alpaca-fallback",
                session="regular",
                bars={
                    "AAPL": {
                        "close": 201.0,
                        "return_15m": 0.015,
                        "day_change_pct": 0.0203,
                        "spread_bps": 9.95,
                        "intraday_range_pct": 0.0227,
                        "volume_ratio": 0.0,
                        "distance_from_vwap": 0.0045,
                        "minute_volume": 12500.0,
                    },
                    "MSFT": {
                        "close": 412.0,
                        "return_15m": -0.019,
                        "day_change_pct": -0.0144,
                        "spread_bps": 9.7,
                        "intraday_range_pct": 0.0108,
                        "volume_ratio": 0.0,
                        "distance_from_vwap": -0.0036,
                        "minute_volume": 9800.0,
                    },
                },
            )

        def get_news_snapshot(self, *, as_of, universe):
            del as_of, universe
            return []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(str(request.url))
        path = request.url.path

        if path == "/v2/aggs/ticker/AAPL/prev":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"T": "AAPL", "v": 5000000.0, "vw": 200.1, "o": 198.0, "c": 197.0, "h": 202.0, "l": 197.5},
                    ]
                },
            )

        if path == "/v2/aggs/ticker/MSFT/prev":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"T": "MSFT", "v": 4300000.0, "vw": 413.5, "o": 415.0, "c": 418.0, "h": 416.0, "l": 411.5},
                    ]
                },
            )

        if path == "/v2/reference/news":
            ticker = request.url.params.get("ticker")
            payload = {
                "results": [
                    {
                        "id": f"{ticker}-1",
                        "title": f"{ticker} headline",
                        "description": f"{ticker} detailed news",
                        "published_utc": "2026-04-09T14:35:00Z",
                        "publisher": {"name": "Polygon News"},
                        "tickers": [ticker],
                    }
                ]
            }
            return httpx.Response(200, json=payload)

        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.polygon.io",
    )
    adapter = PolygonMarketDataAdapter(
        client=client,
        base_url="https://api.polygon.io",
        fallback_adapter=FallbackAdapter(),
    )
    as_of = datetime(2026, 4, 9, 10, 45, tzinfo=ZoneInfo("America/New_York"))

    snapshot = adapter.get_market_snapshot(as_of=as_of, universe=["AAPL", "MSFT"])
    news_items = adapter.get_news_snapshot(as_of=as_of, universe=["AAPL", "MSFT"])

    assert snapshot.source == "polygon-prev+alpaca-fallback"
    assert snapshot.bars["AAPL"]["return_15m"] == pytest.approx(0.015)
    assert snapshot.bars["AAPL"]["spread_bps"] == pytest.approx(9.95)
    assert snapshot.bars["MSFT"]["intraday_range_pct"] == pytest.approx(0.0108)
    assert snapshot.bars["AAPL"]["previous_day_volume"] == pytest.approx(5000000.0)
    assert len(news_items) == 2
    assert news_items[0]["source"] == "Polygon News"
    assert any("/v2/aggs/ticker/AAPL/prev" in url for url in requests_seen)
    assert any("/v2/reference/news" in url for url in requests_seen)


def test_polygon_market_data_adapter_requires_api_key(monkeypatch):
    from intraday.market_data import PolygonMarketDataAdapter

    monkeypatch.delenv("POLYGON_API_KEY", raising=False)

    with pytest.raises(EnvironmentError, match="POLYGON_API_KEY"):
        PolygonMarketDataAdapter(base_url="https://api.polygon.io")


def test_polygon_market_data_adapter_uses_crypto_prev_endpoint_and_enriches_zero_fallback(monkeypatch):
    from intraday.market_data import PolygonMarketDataAdapter

    monkeypatch.setenv("POLYGON_API_KEY", "polygon-test")
    requests_seen: list[str] = []

    class FallbackAdapter:
        def get_market_snapshot(self, *, as_of, universe):
            from intraday.market_data import MarketSnapshot

            return MarketSnapshot(
                as_of=as_of,
                universe=universe,
                source="alpaca-fallback",
                session="regular",
                bars={
                    "BTCUSD": {
                        "close": 0.0,
                        "return_15m": 0.0,
                        "day_change_pct": 0.0,
                        "spread_bps": 0.0,
                        "intraday_range_pct": 0.0,
                        "volume_ratio": 0.0,
                        "distance_from_vwap": 0.0,
                        "minute_volume": 0.0,
                    }
                },
            )

        def get_news_snapshot(self, *, as_of, universe):
            del as_of, universe
            return []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(str(request.url))
        if request.url.path == "/v2/aggs/ticker/X:BTCUSD/prev":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "T": "X:BTCUSD",
                            "v": 12345.0,
                            "vw": 65000.0,
                            "o": 64000.0,
                            "c": 66000.0,
                            "h": 66200.0,
                            "l": 63800.0,
                        }
                    ]
                },
            )
        if request.url.path == "/v2/reference/news":
            return httpx.Response(200, json={"results": []})
        return httpx.Response(404, json={"results": []})

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.polygon.io",
    )
    adapter = PolygonMarketDataAdapter(
        client=client,
        base_url="https://api.polygon.io",
        fallback_adapter=FallbackAdapter(),
    )
    as_of = datetime(2026, 4, 12, 3, 15, tzinfo=ZoneInfo("America/New_York"))

    snapshot = adapter.get_market_snapshot(as_of=as_of, universe=["BTCUSD"])

    btc = snapshot.bars["BTCUSD"]
    assert btc["close"] == pytest.approx(66000.0)
    assert btc["day_change_pct"] == pytest.approx((66000.0 / 64000.0) - 1.0)
    assert btc["intraday_range_pct"] == pytest.approx((66200.0 - 63800.0) / 64000.0)
    assert btc["distance_from_vwap"] == pytest.approx((66000.0 / 65000.0) - 1.0)
    assert btc["previous_day_volume"] == pytest.approx(12345.0)
    assert any("/v2/aggs/ticker/X:BTCUSD/prev" in url for url in requests_seen)
