from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Protocol

import httpx
from runtime_env import get_secret


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
        try:
            from alpaca.data.requests import StockSnapshotRequest

            request = StockSnapshotRequest(symbol_or_symbols=universe)
            snapshots = self._node._data_client.get_stock_snapshot(request)
        except Exception:
            snapshots = {}

        def _field(payload: Any, name: str, default: Any = None) -> Any:
            if payload is None:
                return default
            if isinstance(payload, dict):
                return payload.get(name, default)
            return getattr(payload, name, default)

        for ticker in universe:
            snapshot = snapshots.get(ticker, {}) if isinstance(snapshots, dict) else {}
            latest_trade = _field(snapshot, "latest_trade", {}) or {}
            latest_quote = _field(snapshot, "latest_quote", {}) or {}
            daily_bar = _field(snapshot, "daily_bar", {}) or {}
            previous_daily_bar = _field(snapshot, "previous_daily_bar", {}) or {}
            minute_bar = _field(snapshot, "minute_bar", {}) or {}

            close = float(
                _field(latest_trade, "price")
                or _field(minute_bar, "close")
                or _field(daily_bar, "close")
                or self._node.get_latest_price(ticker)
                or 0.0
            )
            minute_open = float(_field(minute_bar, "open") or close or 0.0)
            day_open = float(_field(daily_bar, "open") or close or 0.0)
            prev_close = float(_field(previous_daily_bar, "close") or 0.0)
            ask = float(_field(latest_quote, "ask_price") or 0.0)
            bid = float(_field(latest_quote, "bid_price") or 0.0)
            day_vwap = float(_field(daily_bar, "vwap") or 0.0)

            bars[ticker] = {
                "close": close,
                "return_15m": ((close / minute_open) - 1.0) if minute_open else 0.0,
                "day_change_pct": ((close / prev_close) - 1.0) if prev_close else 0.0,
                "spread_bps": ((ask - bid) / close) * 10000 if close and ask and bid else 0.0,
                "intraday_range_pct": (
                    (float(_field(daily_bar, "high") or 0.0) - float(_field(daily_bar, "low") or 0.0))
                    / day_open
                ) if day_open else 0.0,
                "volume_ratio": 0.0,
                "distance_from_vwap": ((close / day_vwap) - 1.0) if day_vwap else 0.0,
                "minute_volume": float(_field(minute_bar, "volume") or 0.0),
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
    """Polygon adapter constrained to endpoints available on lower-tier plans.

    Market data is built by enriching a fallback intraday snapshot (typically Alpaca)
    with Polygon previous-day aggregates, while news comes directly from Polygon.
    This avoids real-time Polygon endpoints that may require higher entitlements.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        base_url: str = "https://api.polygon.io",
        timeout: float = 20.0,
        fallback_adapter: MarketDataAdapter | None = None,
    ) -> None:
        resolved_key = api_key or get_secret("POLYGON_API_KEY") or os.getenv("POLYGON_API_KEY")
        if not resolved_key:
            raise EnvironmentError("POLYGON_API_KEY is required for PolygonMarketDataAdapter")

        self.api_key = resolved_key
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(base_url=self.base_url, timeout=timeout)
        self._fallback_adapter = fallback_adapter
        self._prev_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._news_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}

    def get_market_snapshot(self, *, as_of: datetime, universe: list[str]) -> MarketSnapshot:
        try:
            fallback_snapshot = None
            if self._fallback_adapter is not None:
                fallback_snapshot = self._fallback_adapter.get_market_snapshot(
                    as_of=as_of,
                    universe=universe,
                )

            previous_day = self._fetch_previous_day_aggregates(universe)
            bars: dict[str, dict[str, float]] = {}
            for ticker in universe:
                base_bar = (
                    dict(fallback_snapshot.bars.get(ticker, {}))
                    if fallback_snapshot is not None
                    else {}
                )
                prev = previous_day.get(ticker, {})
                close = float(base_bar.get("close") or prev.get("c") or 0.0)
                prev_close = float(prev.get("c") or 0.0)
                prev_open = float(prev.get("o") or close or 0.0)
                prev_high = float(prev.get("h") or close or 0.0)
                prev_low = float(prev.get("l") or close or 0.0)
                prev_vwap = float(prev.get("vw") or 0.0)

                bars[ticker] = {
                    "close": close,
                    "return_15m": float(base_bar.get("return_15m") or 0.0),
                    "day_change_pct": (
                        float(base_bar.get("day_change_pct"))
                        if "day_change_pct" in base_bar
                        else ((close / prev_close) - 1.0) if prev_close else 0.0
                    ),
                    "spread_bps": float(base_bar.get("spread_bps") or 0.0),
                    "intraday_range_pct": (
                        float(base_bar.get("intraday_range_pct"))
                        if "intraday_range_pct" in base_bar
                        else ((prev_high - prev_low) / prev_open) if prev_open else 0.0
                    ),
                    "volume_ratio": float(base_bar.get("volume_ratio") or 0.0),
                    "distance_from_vwap": (
                        float(base_bar.get("distance_from_vwap"))
                        if "distance_from_vwap" in base_bar
                        else ((close / prev_vwap) - 1.0) if prev_vwap else 0.0
                    ),
                    "minute_volume": float(base_bar.get("minute_volume") or 0.0),
                    "previous_day_volume": float(prev.get("v") or 0.0),
                }

            return MarketSnapshot(
                as_of=as_of,
                universe=universe,
                bars=bars,
                source=(
                    "polygon-prev+alpaca-fallback"
                    if fallback_snapshot is not None
                    else "polygon-prev"
                ),
                session=(fallback_snapshot.session if fallback_snapshot is not None else "regular"),
            )
        except Exception:
            if self._fallback_adapter is not None:
                return self._fallback_adapter.get_market_snapshot(as_of=as_of, universe=universe)
            raise

    def get_news_snapshot(self, *, as_of: datetime, universe: list[str]) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        errors = 0
        for ticker in universe:
            cached_news = self._cached_news(ticker)
            if cached_news is not None:
                payloads.extend(cached_news)
                continue
            try:
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
                ticker_payloads: list[dict[str, Any]] = []
                for item in body.get("results", []):
                    if not isinstance(item, dict):
                        continue
                    ticker_payloads.append(
                        {
                            "ticker": ticker,
                            "headline": str(item.get("title", "")),
                            "summary": str(item.get("description", "")),
                            "published_at": str(item.get("published_utc", as_of.isoformat())),
                            "source": str(item.get("publisher", {}).get("name", "polygon")),
                            "sentiment_score": 0.0,
                        }
                    )
                self._news_cache[ticker] = (time.time(), ticker_payloads)
                payloads.extend(ticker_payloads)
            except Exception:
                errors += 1
                continue

        if payloads:
            return payloads
        if errors and self._fallback_adapter is not None:
            return self._fallback_adapter.get_news_snapshot(as_of=as_of, universe=universe)
        return payloads

    def _fetch_previous_day_aggregates(self, universe: list[str]) -> dict[str, dict[str, Any]]:
        payloads: dict[str, dict[str, Any]] = {}
        for ticker in universe:
            cached_payload = self._cached_prev_aggregate(ticker)
            if cached_payload is not None:
                payloads[ticker] = cached_payload
                continue
            try:
                response = self._client.get(
                    f"/v2/aggs/ticker/{ticker}/prev",
                    params={
                        "adjusted": "true",
                        "apiKey": self.api_key,
                    },
                )
                response.raise_for_status()
                body = response.json()
                results = body.get("results", []) or []
                if results and isinstance(results[0], dict):
                    payloads[ticker] = results[0]
                    self._prev_cache[ticker] = (time.time(), results[0])
            except Exception:
                continue
        return payloads

    def _cached_prev_aggregate(self, ticker: str) -> dict[str, Any] | None:
        cached = self._prev_cache.get(ticker)
        if cached is None:
            return None
        stored_at, payload = cached
        if time.time() - stored_at > 6 * 60 * 60:
            self._prev_cache.pop(ticker, None)
            return None
        return payload

    def _cached_news(self, ticker: str) -> list[dict[str, Any]] | None:
        cached = self._news_cache.get(ticker)
        if cached is None:
            return None
        stored_at, payload = cached
        if time.time() - stored_at > 15 * 60:
            self._news_cache.pop(ticker, None)
            return None
        return payload
