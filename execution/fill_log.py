"""Structured fill log for cost-calibration feedback (Phase 2 of the
self-calibrating cost-model track, ADR-0003).

Each :class:`FillRecord` captures the minimal information required to compute
implementation shortfall::

    IS_bps = 10_000 * (fill_price - decision_price) / decision_price * sign(side)

Records are appended to month-partitioned parquet files under
``data/operations/fills/YYYY-MM.parquet`` so that the calibration job can
read a rolling window efficiently. The writer is append-safe: it loads any
existing parquet for the month, concatenates, and rewrites atomically via a
``.tmp`` sibling + ``os.replace``.

Design notes
------------
* Append-only schema; never mutate historical records.
* ``decision_price`` must be captured at the moment the upstream decision
  was made (last close for daily market orders, signal-time mid for
  intraday, limit price for limit orders). Best-effort fallbacks are noted
  in the docstring of :meth:`FillRecord.implementation_shortfall_bps`.
* ``pipeline_run_id`` and ``config_hash`` provide audit lineage that joins
  cleanly to ``daily_orders`` and MLflow runs.
* ``commission_bps`` and ``slippage_bps_assumed`` capture what the cost
  model *thought* the trade would cost; calibration learns the gap between
  this assumption and the realised IS.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import polars as pl

_ROOT = Path(__file__).resolve().parents[1]
FILL_LOG_DIR = _ROOT / "data" / "operations" / "fills"


_SCHEMA: dict[str, pl.DataType] = {
    "fill_id": pl.Utf8,
    "order_id": pl.Utf8,
    "ticker": pl.Utf8,
    "side": pl.Utf8,
    "qty": pl.Float64,
    "fill_price": pl.Float64,
    "decision_price": pl.Float64,
    "decision_ts": pl.Datetime("us", "UTC"),
    "fill_ts": pl.Datetime("us", "UTC"),
    "broker": pl.Utf8,
    "venue": pl.Utf8,
    "pipeline_run_id": pl.Utf8,
    "config_hash": pl.Utf8,
    "commission_bps": pl.Float64,
    "slippage_bps_assumed": pl.Float64,
    "cost_calibration_version": pl.Utf8,
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class FillRecord:
    """A single normalised fill, suitable for calibration analytics.

    All prices are in the same currency as the OHLCV feed (USD). ``qty`` is
    signed by the ``side`` column (positive value, semantic sign carried
    via ``side``) to remain consistent with broker conventions.
    """

    fill_id: str
    order_id: str
    ticker: str
    side: str  # "buy" or "sell"
    qty: float
    fill_price: float
    decision_price: float
    decision_ts: datetime
    fill_ts: datetime = field(default_factory=_utcnow)
    broker: str = "alpaca"
    venue: str = "ALPACA"
    pipeline_run_id: str = ""
    config_hash: str = ""
    commission_bps: float = 0.0
    slippage_bps_assumed: float = 0.0
    cost_calibration_version: str = ""

    # ---- validation ------------------------------------------------------
    def __post_init__(self) -> None:
        if self.side not in {"buy", "sell"}:
            raise ValueError(f"side must be 'buy' or 'sell', got {self.side!r}")
        if self.qty <= 0:
            raise ValueError(f"qty must be positive (sign carried by side), got {self.qty}")
        if self.fill_price <= 0:
            raise ValueError(f"fill_price must be positive, got {self.fill_price}")
        if self.decision_price <= 0:
            raise ValueError(f"decision_price must be positive, got {self.decision_price}")

    # ---- analytics -------------------------------------------------------
    @property
    def sign(self) -> int:
        return 1 if self.side == "buy" else -1

    def implementation_shortfall_bps(self) -> float:
        """Realised slippage vs decision price, in basis points.

        Positive values mean the broker filled us at a worse price than the
        decision reference (cost to the strategy). Negative values mean
        price-improvement.
        """
        slip = (self.fill_price - self.decision_price) / self.decision_price
        return 10_000.0 * slip * self.sign

    def to_dict(self) -> dict:
        d = asdict(self)
        # ensure datetimes are ISO strings when serialised to JSON elsewhere
        d["decision_ts"] = self.decision_ts
        d["fill_ts"] = self.fill_ts
        return d


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _month_partition(ts: datetime) -> str:
    return ts.strftime("%Y-%m")


def _partition_path(ts: datetime, base: Path = FILL_LOG_DIR) -> Path:
    return base / f"{_month_partition(ts)}.parquet"


def _records_to_df(records: Iterable[FillRecord]) -> pl.DataFrame:
    rows = [r.to_dict() for r in records]
    if not rows:
        return pl.DataFrame(schema=_SCHEMA)
    df = pl.DataFrame(rows)
    # enforce schema
    return df.cast({k: v for k, v in _SCHEMA.items() if k in df.columns})


def append_fills(records: Iterable[FillRecord], base: Path = FILL_LOG_DIR) -> Path:
    """Append a batch of :class:`FillRecord` to the monthly parquet.

    All records in *records* must share the same month, which is determined
    from their ``fill_ts``. Mixing months raises ``ValueError`` to keep the
    partition contract obvious. Use multiple calls for cross-month batches.

    Returns the parquet path that was rewritten.
    """
    records = list(records)
    if not records:
        return base  # nothing to do

    months = {_month_partition(r.fill_ts) for r in records}
    if len(months) > 1:
        raise ValueError(
            f"append_fills(): batch crosses month boundaries: {sorted(months)}; "
            "split the batch per month"
        )

    base.mkdir(parents=True, exist_ok=True)
    new_df = _records_to_df(records)
    path = _partition_path(records[0].fill_ts, base=base)

    if path.exists():
        existing = pl.read_parquet(path)
        combined = pl.concat([existing, new_df], how="vertical_relaxed")
        # de-duplicate on fill_id (idempotent re-runs)
        combined = combined.unique(subset=["fill_id"], keep="last")
    else:
        combined = new_df

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    combined.write_parquet(tmp_path)
    os.replace(tmp_path, path)
    return path


def append_fill(record: FillRecord, base: Path = FILL_LOG_DIR) -> Path:
    """Convenience wrapper around :func:`append_fills` for a single record."""
    return append_fills([record], base=base)


def read_fills(
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    base: Path = FILL_LOG_DIR,
) -> pl.DataFrame:
    """Read the union of monthly parquet partitions, optionally filtered."""
    if not base.exists():
        return pl.DataFrame(schema=_SCHEMA)

    parts = sorted(base.glob("*.parquet"))
    if not parts:
        return pl.DataFrame(schema=_SCHEMA)

    df = pl.concat([pl.read_parquet(p) for p in parts], how="vertical_relaxed")
    if start is not None:
        df = df.filter(pl.col("fill_ts") >= start)
    if end is not None:
        df = df.filter(pl.col("fill_ts") <= end)
    return df.sort("fill_ts")
