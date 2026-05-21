"""Order Flow Imbalance (OFI) microstructure alpha — shadow challenger (T2.4).

Computes intraday OFI from L2 book snapshots and cross-sectionally z-scores
signals. **Not wired into the daily council aggregator** until walk-forward
promotion (``MLCOUNCIL_MICROSTRUCTURE_PROMOTED``); shadow runs log signals only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from data.ingest.orderbook import (
    OrderBookFeed,
    ingest_orderbook_snapshots,
    synthetic_book_sequence,
)
from intraday.market_data import BookSnapshot, compute_ofi

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_DEFAULT_LEVELS = 5


def microstructure_shadow_enabled() -> bool:
    """Shadow logging/compute enabled (default true until promoted)."""
    if microstructure_promoted():
        return False
    raw = os.getenv("MLCOUNCIL_MICROSTRUCTURE_SHADOW", "true").strip().lower()
    return raw in _TRUTHY


def microstructure_promoted() -> bool:
    """When true, council may include microstructure weights (post T1.1 gate)."""
    return os.getenv("MLCOUNCIL_MICROSTRUCTURE_PROMOTED", "").strip().lower() in _TRUTHY


@dataclass
class OFIResult:
    """Per-symbol OFI computation metadata."""

    symbol: str
    ofi: float
    cumulative_bid: float
    cumulative_ask: float
    snapshot_count: int
    feed: str = "synthetic"
    shadow_mode: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class MicrostructureModel:
    """OFI-based intraday alpha challenger (shadow mode by default)."""

    name = "microstructure"

    def __init__(self, *, levels: int = _DEFAULT_LEVELS) -> None:
        self.levels = levels
        self._last_snapshot: dict[str, BookSnapshot] = {}

    @staticmethod
    def ofi_from_snapshots(
        snapshots: list[BookSnapshot],
        *,
        levels: int | None = None,
    ) -> float:
        """Aggregate OFI over a snapshot sequence (sum of consecutive deltas)."""
        if len(snapshots) < 2:
            if snapshots:
                return 0.0
            return 0.0
        total = 0.0
        previous = snapshots[0]
        for current in snapshots[1:]:
            total += compute_ofi(current, previous, levels=levels)
            previous = current
        return float(total)

    def update_snapshot(self, snapshot: BookSnapshot) -> OFIResult:
        """Ingest one book snapshot; return incremental OFI vs previous."""
        prev = self._last_snapshot.get(snapshot.symbol)
        ofi = compute_ofi(snapshot, prev, levels=self.levels)
        self._last_snapshot[snapshot.symbol] = snapshot
        return OFIResult(
            symbol=snapshot.symbol,
            ofi=ofi,
            cumulative_bid=snapshot.cumulative_bid_volume(self.levels),
            cumulative_ask=snapshot.cumulative_ask_volume(self.levels),
            snapshot_count=1 if prev is None else 2,
            shadow_mode=not microstructure_promoted(),
        )

    def compute_ofi_for_symbol(
        self,
        symbol: str,
        *,
        as_of: datetime | None = None,
        feed: str | None = None,
    ) -> OFIResult:
        """Fetch book snapshots via ingest layer and return latest OFI."""
        ingest = ingest_orderbook_snapshots(symbol=symbol, as_of=as_of, feed=feed)
        if not ingest.snapshots:
            return OFIResult(
                symbol=symbol,
                ofi=0.0,
                cumulative_bid=0.0,
                cumulative_ask=0.0,
                snapshot_count=0,
                feed=ingest.feed.value,
                shadow_mode=not microstructure_promoted(),
                metadata=ingest.to_dict(),
            )
        ofi = self.ofi_from_snapshots(ingest.snapshots, levels=self.levels)
        last = ingest.snapshots[-1]
        return OFIResult(
            symbol=symbol,
            ofi=ofi,
            cumulative_bid=last.cumulative_bid_volume(self.levels),
            cumulative_ask=last.cumulative_ask_volume(self.levels),
            snapshot_count=len(ingest.snapshots),
            feed=ingest.feed.value,
            shadow_mode=not microstructure_promoted(),
            metadata=ingest.to_dict(),
        )

    def predict_from_ofi(
        self,
        ofi_by_ticker: dict[str, float],
    ) -> pd.Series:
        """Cross-sectional z-score of raw OFI values → council signal scale."""
        if not ofi_by_ticker:
            return pd.Series(dtype=float)
        tickers = list(ofi_by_ticker.keys())
        values = np.array([ofi_by_ticker[t] for t in tickers], dtype=float)
        if len(values) == 1 or np.std(values) < 1e-12:
            z = np.zeros_like(values)
        else:
            z = (values - values.mean()) / values.std()
        return pd.Series(z, index=tickers, name="microstructure")

    def predict(
        self,
        features: pl.DataFrame | None = None,
        *,
        tickers: list[str] | None = None,
        as_of: datetime | None = None,
        feed: str | None = None,
    ) -> pd.Series:
        """Generate shadow OFI signals for ``tickers`` (or features['ticker'])."""
        del features
        universe = tickers or ["AAPL"]
        ofi_map: dict[str, float] = {}
        resolved_feed = feed or os.getenv("MLCOUNCIL_ORDERBOOK_FEED", OrderBookFeed.SYNTHETIC.value)
        for ticker in universe:
            if resolved_feed == OrderBookFeed.SYNTHETIC.value and len(universe) == 1:
                snaps = synthetic_book_sequence(symbol=ticker, as_of=as_of)
                ofi_map[ticker] = self.ofi_from_snapshots(snaps, levels=self.levels)
            else:
                result = self.compute_ofi_for_symbol(ticker, as_of=as_of, feed=resolved_feed)
                ofi_map[ticker] = result.ofi
        return self.predict_from_ofi(ofi_map)

    def fit(self, features: pl.DataFrame, targets: pd.Series) -> None:
        """No training in v1; OFI is rule-based from book deltas."""
        del features, targets

    def shadow_log_payload(self, signals: pd.Series, *, as_of: datetime | None = None) -> dict[str, Any]:
        """Structured payload for MLflow / shadow signal logging."""
        return {
            "model": self.name,
            "shadow_mode": microstructure_shadow_enabled(),
            "promoted": microstructure_promoted(),
            "as_of": (as_of or datetime.utcnow()).isoformat(),
            "signals": {str(k): float(v) for k, v in signals.items()},
            "levels": self.levels,
        }


def build_synthetic_fixture_ofi() -> tuple[float, list[BookSnapshot]]:
    """Return expected OFI (=5) and the canonical two-tick fixture."""
    snaps = synthetic_book_sequence(symbol="AAPL")
    ofi = MicrostructureModel.ofi_from_snapshots(snaps)
    return ofi, snaps
