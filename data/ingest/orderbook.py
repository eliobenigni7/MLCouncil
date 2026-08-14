"""L2 order book ingest skeleton (T2.4 microstructure / OFI).

Live L2 feeds (Alpaca elite, Databento) are stubbed until subscription is approved.
Tests and shadow-mode inference use :func:`synthetic_book_sequence` fixtures.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Protocol

from loguru import logger

from intraday.market_data import BookLevel, BookSnapshot, compute_ofi

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LEVELS = 5


class OrderBookFeed(str, Enum):
    """Supported L2 feed identifiers (live paths deferred)."""

    SYNTHETIC = "synthetic"
    ALPACA = "alpaca"
    DATABENTO = "databento"


@dataclass(slots=True)
class OrderBookIngestResult:
    """Outcome of a book snapshot fetch."""

    snapshots: list[BookSnapshot]
    feed: OrderBookFeed
    deferred_live: bool
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "feed": self.feed.value,
            "deferred_live": self.deferred_live,
            "snapshot_count": len(self.snapshots),
            "message": self.message,
        }


class OrderBookAdapter(Protocol):
    def fetch_snapshots(
        self,
        *,
        symbol: str,
        as_of: datetime,
        levels: int = _DEFAULT_LEVELS,
    ) -> list[BookSnapshot]:
        ...


class SyntheticOrderBookAdapter:
    """Deterministic L2 sequence for tests and local shadow runs."""

    def fetch_snapshots(
        self,
        *,
        symbol: str,
        as_of: datetime,
        levels: int = _DEFAULT_LEVELS,
    ) -> list[BookSnapshot]:
        del levels
        return synthetic_book_sequence(symbol=symbol, as_of=as_of)


class AlpacaOrderBookAdapter:
    """Placeholder for Alpaca L2 / market-data v2 book endpoint."""

    def fetch_snapshots(
        self,
        *,
        symbol: str,
        as_of: datetime,
        levels: int = _DEFAULT_LEVELS,
    ) -> list[BookSnapshot]:
        del symbol, as_of, levels
        raise NotImplementedError(
            "Alpaca L2 order book ingest is deferred until an L2 subscription is active. "
            "Use OrderBookFeed.SYNTHETIC or synthetic_book_sequence() for tests."
        )


class DatabentoOrderBookAdapter:
    """Placeholder for Databento MBP/L2 historical or live feed."""

    def fetch_snapshots(
        self,
        *,
        symbol: str,
        as_of: datetime,
        levels: int = _DEFAULT_LEVELS,
    ) -> list[BookSnapshot]:
        del symbol, as_of, levels
        raise NotImplementedError(
            "Databento L2 ingest is deferred until subscription budget is approved. "
            "See docs/adr/2026-05-21-microstructure-ofi.md."
        )


def resolve_feed(name: str | None = None) -> OrderBookFeed:
    raw = (name or os.getenv("MLCOUNCIL_ORDERBOOK_FEED", "synthetic")).strip().lower()
    try:
        return OrderBookFeed(raw)
    except ValueError as exc:
        raise ValueError(
            f"Unknown order book feed {raw!r}; choose from "
            f"{[f.value for f in OrderBookFeed]}"
        ) from exc


def build_adapter(feed: OrderBookFeed | str | None = None) -> OrderBookAdapter:
    resolved = feed if isinstance(feed, OrderBookFeed) else resolve_feed(feed)
    if resolved == OrderBookFeed.SYNTHETIC:
        return SyntheticOrderBookAdapter()
    if resolved == OrderBookFeed.ALPACA:
        return AlpacaOrderBookAdapter()
    if resolved == OrderBookFeed.DATABENTO:
        return DatabentoOrderBookAdapter()
    raise ValueError(f"Unsupported feed: {resolved}")


def ingest_orderbook_snapshots(
    *,
    symbol: str,
    as_of: datetime | None = None,
    feed: str | None = None,
    levels: int = _DEFAULT_LEVELS,
) -> OrderBookIngestResult:
    """Fetch L2 snapshots for a symbol; live feeds return deferred stub errors."""
    as_of = as_of or datetime.now(timezone.utc)
    resolved = resolve_feed(feed)
    adapter = build_adapter(resolved)
    deferred_live = resolved != OrderBookFeed.SYNTHETIC

    if deferred_live:
        logger.warning(
            "Order book feed {} requested but live L2 is deferred; no snapshots ingested",
            resolved.value,
        )
        return OrderBookIngestResult(
            snapshots=[],
            feed=resolved,
            deferred_live=True,
            message="Live L2 subscription deferred; enable SYNTHETIC feed for shadow runs.",
        )

    snapshots = adapter.fetch_snapshots(symbol=symbol, as_of=as_of, levels=levels)
    return OrderBookIngestResult(
        snapshots=snapshots,
        feed=resolved,
        deferred_live=False,
        message="ok",
    )


def synthetic_book_sequence(
    *,
    symbol: str = "AAPL",
    as_of: datetime | None = None,
) -> list[BookSnapshot]:
    """Two-tick fixture for OFI validation (Lo & MacKinlay style cumulative depth delta).

    t0 cumulative bid=300, ask=250
    t1 cumulative bid=310, ask=255  → OFI = (310-300) - (255-250) = 5
    """
    as_of = as_of or datetime(2026, 5, 21, 15, 30, tzinfo=timezone.utc)
    t0 = as_of
    t1 = as_of + timedelta(seconds=1)

    def _book(
        ts: datetime,
        bid_sizes: list[float],
        ask_sizes: list[float],
    ) -> BookSnapshot:
        bids = [
            BookLevel(price=100.0 - i * 0.01, size=size)
            for i, size in enumerate(bid_sizes)
        ]
        asks = [
            BookLevel(price=100.05 + i * 0.01, size=size)
            for i, size in enumerate(ask_sizes)
        ]
        return BookSnapshot(
            symbol=symbol,
            as_of=ts,
            bids=bids,
            asks=asks,
            levels=len(bid_sizes),
        )

    return [
        _book(t0, [100, 80, 60, 40, 20], [90, 70, 50, 30, 10]),
        _book(t1, [110, 80, 60, 40, 20], [95, 70, 50, 30, 10]),
    ]


def iter_ofi_from_snapshots(
    snapshots: list[BookSnapshot],
    *,
    levels: int | None = None,
) -> Iterator[tuple[datetime, float]]:
    """Yield (as_of, OFI) for each snapshot after the first."""
    if len(snapshots) < 2:
        return
    previous = snapshots[0]
    for current in snapshots[1:]:
        yield current.as_of, compute_ofi(current, previous, levels=levels)
        previous = current
