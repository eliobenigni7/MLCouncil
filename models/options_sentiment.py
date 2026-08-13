"""Options-implied sentiment scaffold (Phase 2.6 — shadow only).

Fetches Polygon options snapshot metrics per ticker and derives:
- **put/call ratio** (volume or open-interest weighted when available)
- **skew proxy** — normalized put-minus-call IV spread at near-ATM strikes

Production daily path is unchanged unless ``MLCOUNCIL_OPTIONS_SENTIMENT=true``
(explicit opt-in). Shadow outputs are written to
``data/results/options_sentiment_shadow.parquet`` for walk-forward comparison.

Requires ``POLYGON_API_KEY`` (same pattern as ``intraday.market_data.PolygonMarketDataAdapter``).
Canary status: shadow — target: P-3 — expiry: 2028-06-01 (promote via canary o retire)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
from loguru import logger

from runtime_env import get_secret

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_SHADOW_PARQUET = Path("data/results/options_sentiment_shadow.parquet")
_DEFAULT_BASE_URL = "https://api.polygon.io"


def options_sentiment_enabled() -> bool:
    """True when shadow options sentiment should be computed/logged."""
    return os.getenv("MLCOUNCIL_OPTIONS_SENTIMENT", "false").strip().lower() in _TRUTHY


def shadow_output_path() -> Path:
    return _SHADOW_PARQUET


@dataclass
class OptionsSentimentMetrics:
    """Per-ticker options-implied sentiment features."""

    ticker: str
    put_call_ratio: float
    skew_proxy: float
    put_volume: float = 0.0
    call_volume: float = 0.0
    as_of: str = ""
    source: str = "polygon-options-snapshot"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_signal(self) -> float:
        """Map metrics to a bounded cross-sectional signal in [-1, 1]."""
        # Elevated put/call → bearish; positive skew_proxy (put IV > call IV) → bearish
        pcr = float(np.clip(self.put_call_ratio, 0.25, 4.0))
        pcr_score = -float(np.tanh(pcr - 1.0))  # PCR>1 → negative
        skew_score = -float(np.clip(self.skew_proxy, -1.0, 1.0))
        return float(np.clip(0.6 * pcr_score + 0.4 * skew_score, -1.0, 1.0))


class PolygonOptionsClient:
    """Minimal Polygon REST client for options chain snapshots."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 25.0,
    ) -> None:
        resolved = api_key or get_secret("POLYGON_API_KEY") or os.getenv("POLYGON_API_KEY")
        if not resolved:
            raise EnvironmentError("POLYGON_API_KEY is required for PolygonOptionsClient")
        self.api_key = resolved
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(base_url=self.base_url, timeout=timeout)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def fetch_chain_snapshot(self, ticker: str) -> list[dict[str, Any]]:
        """Return raw option contract dicts from Polygon snapshot endpoint."""
        symbol = ticker.upper().replace(".", "")
        response = self._client.get(
            f"/v3/snapshot/options/{symbol}",
            params={"apiKey": self.api_key},
        )
        response.raise_for_status()
        body = response.json()
        results = body.get("results", body)
        if isinstance(results, dict):
            return list(results.get("options", results.get("contracts", [])) or [])
        if isinstance(results, list):
            return [r for r in results if isinstance(r, dict)]
        return []


def _contract_side(contract: dict[str, Any]) -> str:
    details = contract.get("details") or contract
    cp = str(details.get("contract_type") or details.get("type") or "").lower()
    if cp in {"put", "p"}:
        return "put"
    if cp in {"call", "c"}:
        return "call"
    sym = str(details.get("ticker") or contract.get("ticker") or "")
    if sym.endswith("P") or "P0" in sym:
        return "put"
    if sym.endswith("C") or "C0" in sym:
        return "call"
    return ""


def _contract_volume(contract: dict[str, Any]) -> float:
    day = contract.get("day") or {}
    for key in ("volume", "v"):
        if key in day:
            return float(day[key] or 0.0)
    greeks = contract.get("greeks") or {}
    return float(contract.get("volume") or greeks.get("volume") or 0.0)


def _contract_iv(contract: dict[str, Any]) -> float:
    greeks = contract.get("greeks") or {}
    for key in ("implied_volatility", "iv"):
        if key in greeks:
            return float(greeks[key] or 0.0)
        if key in contract:
            return float(contract[key] or 0.0)
    return float(contract.get("implied_volatility") or 0.0)


def metrics_from_chain(ticker: str, contracts: list[dict[str, Any]], *, as_of: str = "") -> OptionsSentimentMetrics:
    """Aggregate put/call ratio and IV skew proxy from a chain snapshot."""
    put_vol = call_vol = 0.0
    put_ivs: list[float] = []
    call_ivs: list[float] = []

    for contract in contracts:
        side = _contract_side(contract)
        vol = _contract_volume(contract)
        iv = _contract_iv(contract)
        if side == "put":
            put_vol += vol
            if iv > 0:
                put_ivs.append(iv)
        elif side == "call":
            call_vol += vol
            if iv > 0:
                call_ivs.append(iv)

    if call_vol > 0:
        pcr = put_vol / call_vol
    elif put_vol > 0:
        pcr = 4.0
    else:
        pcr = 1.0

    put_iv_mean = float(np.mean(put_ivs)) if put_ivs else 0.0
    call_iv_mean = float(np.mean(call_ivs)) if call_ivs else 0.0
    if put_iv_mean > 0 and call_iv_mean > 0:
        skew_proxy = float(np.clip((put_iv_mean - call_iv_mean) / max(call_iv_mean, 1e-6), -1.0, 1.0))
    else:
        skew_proxy = float(np.clip(pcr - 1.0, -1.0, 1.0))

    return OptionsSentimentMetrics(
        ticker=ticker.upper(),
        put_call_ratio=float(pcr),
        skew_proxy=skew_proxy,
        put_volume=put_vol,
        call_volume=call_vol,
        as_of=as_of,
    )


class OptionsSentimentModel:
    """Shadow options-implied sentiment scorer (not wired to council weights)."""

    def __init__(self, client: PolygonOptionsClient | None = None) -> None:
        self._client = client

    def _client_or_create(self) -> PolygonOptionsClient:
        if self._client is None:
            self._client = PolygonOptionsClient()
        return self._client

    def score_ticker(self, ticker: str, *, as_of: date | None = None) -> OptionsSentimentMetrics:
        as_of_str = (as_of or date.today()).isoformat()
        try:
            client = self._client_or_create()
            contracts = client.fetch_chain_snapshot(ticker)
            return metrics_from_chain(ticker, contracts, as_of=as_of_str)
        except Exception as exc:
            logger.debug(f"options sentiment fetch failed for {ticker}: {exc}")
            return OptionsSentimentMetrics(
                ticker=ticker.upper(),
                put_call_ratio=1.0,
                skew_proxy=0.0,
                as_of=as_of_str,
                source="fallback-neutral",
                metadata={"error": str(exc)},
            )

    def predict_shadow(
        self,
        tickers: list[str],
        *,
        as_of: date | None = None,
    ) -> pd.Series:
        """Cross-sectional z-scored shadow signals (index=ticker)."""
        if not tickers:
            return pd.Series(dtype=float, name="options_sentiment_shadow")

        rows = [self.score_ticker(t, as_of=as_of).to_signal() for t in tickers]
        raw = pd.Series(rows, index=[t.upper() for t in tickers], name="options_sentiment_shadow")
        std = float(raw.std())
        if std > 1e-9:
            raw = (raw - raw.mean()) / std
        return raw

    def close(self) -> None:
        if self._client is not None:
            self._client.close()


def log_shadow_signals(
    partition_date: str,
    signals: pd.Series,
    *,
    metrics: list[OptionsSentimentMetrics] | None = None,
    output_path: Path | None = None,
) -> Path:
    """Append partition row(s) to shadow parquet (wide + diagnostic columns)."""
    dest = output_path or shadow_output_path()
    dest.parent.mkdir(parents=True, exist_ok=True)

    metric_map = {m.ticker: m for m in (metrics or [])}
    payload = pd.DataFrame(
        {
            "partition_date": partition_date,
            "ticker": signals.index.astype(str),
            "options_sentiment_shadow": signals.reindex(signals.index).fillna(0.0).values,
            "put_call_ratio": [
                metric_map.get(t, OptionsSentimentMetrics(t, 1.0, 0.0)).put_call_ratio
                for t in signals.index.astype(str)
            ],
            "skew_proxy": [
                metric_map.get(t, OptionsSentimentMetrics(t, 1.0, 0.0)).skew_proxy
                for t in signals.index.astype(str)
            ],
        }
    )

    if dest.exists():
        existing = pd.read_parquet(dest)
        existing = existing[existing["partition_date"].astype(str) != str(partition_date)]
        payload = pd.concat([existing, payload], ignore_index=True)

    payload.to_parquet(dest, index=False)
    logger.info("options sentiment shadow logged → {} ({} rows)", dest, len(payload))
    return dest


def run_shadow_batch(
    tickers: list[str],
    *,
    partition_date: str | None = None,
) -> dict[str, Any]:
    """Score universe and persist shadow parquet when enabled."""
    if not options_sentiment_enabled():
        return {"status": "disabled", "shadow_path": str(shadow_output_path())}

    as_of = date.fromisoformat(partition_date) if partition_date else date.today()
    model = OptionsSentimentModel()
    try:
        metrics = [model.score_ticker(t, as_of=as_of) for t in tickers]
        signals = model.predict_shadow(tickers, as_of=as_of)
        path = log_shadow_signals(partition_date or as_of.isoformat(), signals, metrics=metrics)
        return {
            "status": "ok",
            "shadow_path": str(path),
            "n_tickers": len(tickers),
            "mean_signal": float(signals.mean()) if len(signals) else 0.0,
        }
    finally:
        model.close()
