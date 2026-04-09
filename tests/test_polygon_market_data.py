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

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(str(request.url))
        path = request.url.path

        if path == "/v2/snapshot/locale/us/markets/stocks/tickers":
            payload = {
                "results": [
                    {
                        "ticker": "AAPL",
                        "lastTrade": {"p": 201.0},
                        "lastQuote": {"P": 201.1, "p": 200.9},
                        "min": {"v": 12500.0},
                        "day": {"o": 198.0, "c": 201.0, "h": 202.0, "l": 197.5, "v": 250000.0, "vw": 200.1},
                        "prevDay": {"c": 197.0, "v": 5000000.0},
                    },
                    {
                        "ticker": "MSFT",
                        "lastTrade": {"p": 412.0},
                        "lastQuote": {"P": 412.2, "p": 411.8},
                        "min": {"v": 9800.0},
                        "day": {"o": 415.0, "c": 412.0, "h": 416.0, "l": 411.5, "v": 180000.0, "vw": 413.5},
                        "prevDay": {"c": 418.0, "v": 4300000.0},
                    },
                ]
            }
            return httpx.Response(200, json=payload)

        if path == "/v2/aggs/ticker/AAPL/range/5/minute/2026-04-09/2026-04-09":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"t": 1, "c": 198.0, "v": 1500.0},
                        {"t": 2, "c": 199.0, "v": 1600.0},
                        {"t": 3, "c": 200.0, "v": 1700.0},
                        {"t": 4, "c": 201.0, "v": 1800.0},
                    ]
                },
            )

        if path == "/v2/aggs/ticker/MSFT/range/5/minute/2026-04-09/2026-04-09":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"t": 1, "c": 420.0, "v": 1200.0},
                        {"t": 2, "c": 418.0, "v": 1100.0},
                        {"t": 3, "c": 415.0, "v": 1000.0},
                        {"t": 4, "c": 412.0, "v": 900.0},
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
    adapter = PolygonMarketDataAdapter(client=client, base_url="https://api.polygon.io")
    as_of = datetime(2026, 4, 9, 10, 45, tzinfo=ZoneInfo("America/New_York"))

    snapshot = adapter.get_market_snapshot(as_of=as_of, universe=["AAPL", "MSFT"])
    news_items = adapter.get_news_snapshot(as_of=as_of, universe=["AAPL", "MSFT"])

    assert snapshot.source == "polygon"
    assert snapshot.bars["AAPL"]["return_15m"] == pytest.approx((201.0 / 198.0) - 1.0)
    assert snapshot.bars["AAPL"]["spread_bps"] == pytest.approx(((201.1 - 200.9) / 201.0) * 10000, rel=1e-3)
    assert snapshot.bars["MSFT"]["intraday_range_pct"] == pytest.approx((416.0 - 411.5) / 415.0)
    assert snapshot.bars["AAPL"]["volume_ratio"] > 0
    assert len(news_items) == 2
    assert news_items[0]["source"] == "Polygon News"
    assert any("/v2/reference/news" in url for url in requests_seen)


def test_polygon_market_data_adapter_requires_api_key(monkeypatch):
    from intraday.market_data import PolygonMarketDataAdapter

    monkeypatch.delenv("POLYGON_API_KEY", raising=False)

    with pytest.raises(EnvironmentError, match="POLYGON_API_KEY"):
        PolygonMarketDataAdapter(base_url="https://api.polygon.io")
