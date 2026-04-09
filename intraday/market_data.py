from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Protocol

import httpx


@dataclass(slots=True)
class MarketSnapshot:
    as_of: datetime
    universe: list[str]
    bars: dict[str, dict[str, float]]
    source: str
    session: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["as_of"] = self.as_of.isoformat()
        return payload


class MarketDataAdapter(Protocol):
    def get_market_snapshot(self, *, as_of: datetime, universe: list[str]) -> MarketSnapshot:
        ...

    def get_news_snapshot(self, *, as_of: datetime, universe: list[str]) -> list[dict[str, Any]]:
        ...


class AlpacaMarketDataAdapter:
    """Fallback intraday feed using the broker adapter when richer data is unavailable."""

    def __init__(self):
        from execution.alpaca_adapter import AlpacaConfig, AlpacaLiveNode

        self._node = AlpacaLiveNode(AlpacaConfig.from_env())

    def get_market_snapshot(self, *, as_of: datetime, universe: list[str]) -> MarketSnapshot:
        bars: dict[str, dict[str, float]] = {}
        for ticker in universe:
            price = self._node.get_latest_price(ticker) or 0.0
            bars[ticker] = {
                "close": float(price),
                "return_15m": 0.0,
            }
        return MarketSnapshot(
            as_of=as_of,
            universe=universe,
            bars=bars,
            source="alpaca-fallback",
            session="regular",
        )

    def get_news_snapshot(self, *, as_of: datetime, universe: list[str]) -> list[dict[str, Any]]:
        del as_of, universe
        return []


class PolygonMarketDataAdapter:
    """Primary intraday market and news feed backed by Polygon REST endpoints."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        base_url: str = "https://api.polygon.io",
        timeout: float = 20.0,
        fallback_adapter: MarketDataAdapter | None = None,
    ) -> None:
        resolved_key = api_key or os.getenv("POLYGON_API_KEY")
        if not resolved_key:
            raise EnvironmentError("POLYGON_API_KEY is required for PolygonMarketDataAdapter")

        self.api_key = resolved_key
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(base_url=self.base_url, timeout=timeout)
        self._fallback_adapter = fallback_adapter

    def get_market_snapshot(self, *, as_of: datetime, universe: list[str]) -> MarketSnapshot:
        try:
            snapshot = self._client.get(
                "/v2/snapshot/locale/us/markets/stocks/tickers",
                params={
                    "tickers": ",".join(universe),
                    "apiKey": self.api_key,
                },
            )
            snapshot.raise_for_status()
            body = snapshot.json()
            lookup = {
                item.get("ticker"): item
                for item in body.get("results", [])
                if isinstance(item, dict)
            }

            bars: dict[str, dict[str, float]] = {}
            session_date = as_of.date().isoformat()
            for ticker in universe:
                item = lookup.get(ticker, {})
                aggregate = self._client.get(
                    f"/v2/aggs/ticker/{ticker}/range/5/minute/{session_date}/{session_date}",
                    params={
                        "adjusted": "true",
                        "sort": "asc",
                        "limit": 12,
                        "apiKey": self.api_key,
                    },
                )
                aggregate.raise_for_status()
                agg_body = aggregate.json()
                bars[ticker] = self._build_rich_bar(
                    ticker=ticker,
                    snapshot=item,
                    aggregate_payload=agg_body,
                )

            return MarketSnapshot(
                as_of=as_of,
                universe=universe,
                bars=bars,
                source="polygon",
                session="regular",
            )
        except Exception:
            if self._fallback_adapter is not None:
                return self._fallback_adapter.get_market_snapshot(as_of=as_of, universe=universe)
            raise

    def get_news_snapshot(self, *, as_of: datetime, universe: list[str]) -> list[dict[str, Any]]:
        try:
            payloads: list[dict[str, Any]] = []
            for ticker in universe:
                response = self._client.get(
                    "/v2/reference/news",
                    params={
                        "ticker": ticker,
                        "limit": 3,
                        "order": "desc",
                        "sort": "published_utc",
                        "apiKey": self.api_key,
                    },
                )
                response.raise_for_status()
                body = response.json()
                for item in body.get("results", []):
                    if not isinstance(item, dict):
                        continue
                    payloads.append(
                        {
                            "ticker": ticker,
                            "headline": str(item.get("title", "")),
                            "summary": str(item.get("description", "")),
                            "published_at": str(item.get("published_utc", as_of.isoformat())),
                            "source": str(item.get("publisher", {}).get("name", "polygon")),
                            "sentiment_score": 0.0,
                        }
                    )
            return payloads
        except Exception:
            if self._fallback_adapter is not None:
                return self._fallback_adapter.get_news_snapshot(as_of=as_of, universe=universe)
            raise

    @staticmethod
    def _build_rich_bar(
        *,
        ticker: str,
        snapshot: dict[str, Any],
        aggregate_payload: dict[str, Any],
    ) -> dict[str, float]:
        del ticker
        last_trade = float(snapshot.get("lastTrade", {}).get("p", 0.0) or 0.0)
        ask = float(snapshot.get("lastQuote", {}).get("P", 0.0) or 0.0)
        bid = float(snapshot.get("lastQuote", {}).get("p", 0.0) or 0.0)
        day = snapshot.get("day", {})
        prev_day = snapshot.get("prevDay", {})
        minute = snapshot.get("min", {})

        aggs = aggregate_payload.get("results", []) or []
        start_close = float(aggs[0].get("c", last_trade) or last_trade) if aggs else last_trade
        end_close = float(aggs[-1].get("c", last_trade) or last_trade) if aggs else last_trade

        return_15m = ((end_close / start_close) - 1.0) if start_close else 0.0
        intraday_range_pct = (
            (float(day.get("h", 0.0) or 0.0) - float(day.get("l", 0.0) or 0.0))
            / float(day.get("o", 1.0) or 1.0)
        )
        day_change_pct = (
            (float(day.get("c", last_trade) or last_trade) / float(prev_day.get("c", 1.0) or 1.0)) - 1.0
            if float(prev_day.get("c", 0.0) or 0.0)
            else 0.0
        )
        spread_bps = ((ask - bid) / last_trade) * 10000 if last_trade and ask and bid else 0.0
        volume_ratio = (
            float(day.get("v", 0.0) or 0.0) / float(prev_day.get("v", 1.0) or 1.0)
            if float(prev_day.get("v", 0.0) or 0.0)
            else 0.0
        )
        distance_from_vwap = (
            (last_trade / float(day.get("vw", last_trade) or last_trade)) - 1.0
            if float(day.get("vw", 0.0) or 0.0)
            else 0.0
        )
        return {
            "close": last_trade,
            "return_15m": return_15m,
            "day_change_pct": day_change_pct,
            "spread_bps": spread_bps,
            "intraday_range_pct": intraday_range_pct,
            "volume_ratio": volume_ratio,
            "distance_from_vwap": distance_from_vwap,
            "minute_volume": float(minute.get("v", 0.0) or 0.0),
        }
