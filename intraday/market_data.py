from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
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
        self._crypto_client = None
        self._init_crypto_client()

    def _init_crypto_client(self) -> None:
        try:
            from alpaca.data.historical.crypto import CryptoHistoricalDataClient

            api_key = self._node.config.paper_key or self._node.config.live_key
            api_secret = self._node.config.paper_secret or self._node.config.live_secret
            self._crypto_client = CryptoHistoricalDataClient(api_key, api_secret)
        except Exception:
            self._crypto_client = None

    def get_market_snapshot(self, *, as_of: datetime, universe: list[str]) -> MarketSnapshot:
        bars: dict[str, dict[str, float]] = {}
        from execution.alpaca_adapter import AlpacaLiveNode

        equity_universe = [ticker for ticker in universe if not AlpacaLiveNode._is_crypto(ticker)]
        crypto_universe = [ticker for ticker in universe if AlpacaLiveNode._is_crypto(ticker)]

        snapshots: dict[str, Any] = {}
        if equity_universe:
            try:
                from alpaca.data.requests import StockSnapshotRequest

                request = StockSnapshotRequest(symbol_or_symbols=equity_universe)
                snapshots = self._node._data_client.get_stock_snapshot(request)
            except Exception:
                snapshots = {}

        def _field(payload: Any, name: str, default: Any = None) -> Any:
            if payload is None:
                return default
            if isinstance(payload, dict):
                return payload.get(name, default)
            return getattr(payload, name, default)

        for ticker in equity_universe:
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

        if crypto_universe:
            crypto_bars = self._build_crypto_bars(as_of=as_of, universe=crypto_universe)
            bars.update(crypto_bars)
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

    def _build_crypto_bars(self, *, as_of: datetime, universe: list[str]) -> dict[str, dict[str, float]]:
        if self._crypto_client is None:
            return {
                ticker: {
                    "close": 0.0,
                    "return_15m": 0.0,
                    "day_change_pct": 0.0,
                    "spread_bps": 0.0,
                    "intraday_range_pct": 0.0,
                    "volume_ratio": 0.0,
                    "distance_from_vwap": 0.0,
                    "minute_volume": 0.0,
                }
                for ticker in universe
            }

        from alpaca.data.requests import (
            CryptoBarsRequest,
            CryptoLatestQuoteRequest,
            CryptoLatestTradeRequest,
        )
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        feed_symbols = [self._normalize_crypto_symbol(ticker) for ticker in universe]
        trade_map: dict[str, Any] = {}
        quote_map: dict[str, Any] = {}
        bars_1m_map: dict[str, list[Any]] = {}
        bars_15m_map: dict[str, list[Any]] = {}
        bars_1d_map: dict[str, list[Any]] = {}
        try:
            trade_map = self._crypto_client.get_crypto_latest_trade(
                CryptoLatestTradeRequest(symbol_or_symbols=feed_symbols)
            ) or {}
        except Exception:
            trade_map = {}
        try:
            quote_map = self._crypto_client.get_crypto_latest_quote(
                CryptoLatestQuoteRequest(symbol_or_symbols=feed_symbols)
            ) or {}
        except Exception:
            quote_map = {}
        try:
            bars_1m = self._crypto_client.get_crypto_bars(
                CryptoBarsRequest(
                    symbol_or_symbols=feed_symbols,
                    timeframe=TimeFrame.Minute,
                    start=as_of - timedelta(minutes=30),
                    end=as_of,
                    limit=20,
                )
            )
            bars_1m_map = dict(getattr(bars_1m, "data", {}) or {})
        except Exception:
            bars_1m_map = {}
        try:
            bars_15m = self._crypto_client.get_crypto_bars(
                CryptoBarsRequest(
                    symbol_or_symbols=feed_symbols,
                    timeframe=TimeFrame(15, TimeFrameUnit.Minute),
                    start=as_of - timedelta(hours=8),
                    end=as_of,
                    limit=4,
                )
            )
            bars_15m_map = dict(getattr(bars_15m, "data", {}) or {})
        except Exception:
            bars_15m_map = {}
        try:
            bars_1d = self._crypto_client.get_crypto_bars(
                CryptoBarsRequest(
                    symbol_or_symbols=feed_symbols,
                    timeframe=TimeFrame.Day,
                    start=as_of - timedelta(days=3),
                    end=as_of,
                    limit=2,
                )
            )
            bars_1d_map = dict(getattr(bars_1d, "data", {}) or {})
        except Exception:
            bars_1d_map = {}

        bars: dict[str, dict[str, float]] = {}
        for ticker in universe:
            symbol = self._normalize_crypto_symbol(ticker)
            trade = self._lookup_symbol_payload(trade_map, symbol)
            quote = self._lookup_symbol_payload(quote_map, symbol)
            one_minute = self._lookup_symbol_payload(bars_1m_map, symbol)
            fifteen_minute = self._lookup_symbol_payload(bars_15m_map, symbol)
            one_day = self._lookup_symbol_payload(bars_1d_map, symbol)

            last_1m = one_minute[-1] if isinstance(one_minute, list) and one_minute else None
            first_1m = one_minute[0] if isinstance(one_minute, list) and one_minute else None
            last_15m = fifteen_minute[-1] if isinstance(fifteen_minute, list) and fifteen_minute else None
            day_bar = one_day[-1] if isinstance(one_day, list) and one_day else None

            close = float(
                self._field(trade, "price")
                or self._field(last_15m, "close")
                or self._field(last_15m, "c")
                or self._field(last_1m, "close")
                or self._field(last_1m, "c")
                or self._field(day_bar, "close")
                or self._field(day_bar, "c")
                or 0.0
            )
            open_15m = float(
                self._field(last_15m, "open")
                or self._field(last_15m, "o")
                or self._field(first_1m, "open")
                or self._field(first_1m, "o")
                or close
                or 0.0
            )
            day_open = float(
                self._field(day_bar, "open")
                or self._field(day_bar, "o")
                or close
                or 0.0
            )
            day_high = float(
                self._field(day_bar, "high")
                or self._field(day_bar, "h")
                or close
                or 0.0
            )
            day_low = float(
                self._field(day_bar, "low")
                or self._field(day_bar, "l")
                or close
                or 0.0
            )
            day_vwap = float(
                self._field(day_bar, "vwap")
                or self._field(day_bar, "vw")
                or 0.0
            )
            day_volume = float(
                self._field(day_bar, "volume")
                or self._field(day_bar, "v")
                or 0.0
            )
            ask = float(self._field(quote, "ask_price") or self._field(quote, "ap") or 0.0)
            bid = float(self._field(quote, "bid_price") or self._field(quote, "bp") or 0.0)
            minute_volume = float(
                self._field(last_1m, "volume")
                or self._field(last_1m, "v")
                or 0.0
            )

            bars[ticker] = {
                "close": close,
                "return_15m": ((close / open_15m) - 1.0) if open_15m else 0.0,
                "day_change_pct": ((close / day_open) - 1.0) if day_open else 0.0,
                "spread_bps": ((ask - bid) / close) * 10000 if close and ask and bid else 0.0,
                "intraday_range_pct": ((day_high - day_low) / day_open) if day_open else 0.0,
                "volume_ratio": 0.0,
                "distance_from_vwap": ((close / day_vwap) - 1.0) if day_vwap else 0.0,
                "minute_volume": minute_volume,
                "previous_day_volume": day_volume,
            }

        return bars

    @staticmethod
    def _field(payload: Any, name: str, default: Any = None) -> Any:
        if payload is None:
            return default
        if isinstance(payload, dict):
            return payload.get(name, default)
        return getattr(payload, name, default)

    @classmethod
    def _lookup_symbol_payload(cls, payload_map: dict[str, Any], symbol: str) -> Any:
        variants = {
            symbol,
            symbol.replace("/", ""),
            symbol.replace("/", "-"),
            symbol.replace("-", "/"),
        }
        for key in variants:
            if key in payload_map:
                return payload_map[key]
        return None

    @staticmethod
    def _normalize_crypto_symbol(symbol: str) -> str:
        upper = str(symbol).strip().upper()
        compact = upper.replace("-", "").replace("/", "")
        if compact.endswith("USD") and len(compact) > 3:
            return f"{compact[:-3]}/USD"
        return upper.replace("-", "/")


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
                base_close = float(base_bar.get("close") or 0.0)
                close = base_close if base_close > 0 else float(prev.get("c") or 0.0)
                prev_close = float(prev.get("c") or 0.0)
                prev_open = float(prev.get("o") or close or 0.0)
                prev_high = float(prev.get("h") or close or 0.0)
                prev_low = float(prev.get("l") or close or 0.0)
                prev_vwap = float(prev.get("vw") or 0.0)
                base_day_change = self._safe_float(base_bar.get("day_change_pct"))
                base_intraday_range = self._safe_float(base_bar.get("intraday_range_pct"))
                base_distance_vwap = self._safe_float(base_bar.get("distance_from_vwap"))

                bars[ticker] = {
                    "close": close,
                    "return_15m": float(base_bar.get("return_15m") or 0.0),
                    "day_change_pct": self._select_metric(
                        base_value=base_day_change,
                        fallback_value=((close / prev_open) - 1.0) if prev_open else 0.0,
                        base_price=base_close,
                    ),
                    "spread_bps": float(base_bar.get("spread_bps") or 0.0),
                    "intraday_range_pct": self._select_metric(
                        base_value=base_intraday_range,
                        fallback_value=((prev_high - prev_low) / prev_open) if prev_open else 0.0,
                        base_price=base_close,
                    ),
                    "volume_ratio": float(base_bar.get("volume_ratio") or 0.0),
                    "distance_from_vwap": self._select_metric(
                        base_value=base_distance_vwap,
                        fallback_value=((close / prev_vwap) - 1.0) if prev_vwap else 0.0,
                        base_price=base_close,
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
            for candidate in self._polygon_symbol_candidates(ticker):
                try:
                    response = self._client.get(
                        f"/v2/aggs/ticker/{candidate}/prev",
                        params={
                            "adjusted": "true",
                            "apiKey": self.api_key,
                        },
                    )
                except Exception:
                    continue
                if not response.is_success:
                    continue
                body = response.json()
                results = body.get("results", []) or []
                if results and isinstance(results[0], dict):
                    payloads[ticker] = results[0]
                    self._prev_cache[ticker] = (time.time(), results[0])
                    break
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

    @staticmethod
    def _is_crypto_symbol(ticker: str) -> bool:
        symbol = str(ticker).strip().upper().replace("/", "").replace("-", "")
        if len(symbol) < 6:
            return False
        quote = symbol[-3:]
        base = symbol[:-3]
        return quote == "USD" and base.isalpha()

    @classmethod
    def _polygon_symbol_candidates(cls, ticker: str) -> list[str]:
        raw = str(ticker).strip().upper()
        if cls._is_crypto_symbol(raw):
            normalized = raw.replace("/", "").replace("-", "")
            return [f"X:{normalized}", normalized]
        return [raw]

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _select_metric(*, base_value: float | None, fallback_value: float, base_price: float) -> float:
        # Quando il fallback adapter non ha dati reali (tipico crypto fuori RTH),
        # evita di propagare zeri sintetici e usa il dato Polygon precedente.
        if base_value is None:
            return float(fallback_value)
        if base_price <= 0 and base_value == 0.0:
            return float(fallback_value)
        return float(base_value)
