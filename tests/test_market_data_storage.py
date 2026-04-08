from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from data.ingest.market_data import _save_by_year


def _ohlcv_row(valid_time: date, close: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["AAPL"],
            "valid_time": [valid_time],
            "transaction_time": [datetime(2026, 4, 3, tzinfo=timezone.utc)],
            "open": [close - 1.0],
            "high": [close + 1.0],
            "low": [close - 2.0],
            "close": [close],
            "adj_close": [close],
            "volume": [1_000_000],
        }
    ).with_columns(pl.col("transaction_time").cast(pl.Datetime("us", "UTC")))


def test_save_by_year_appends_history_instead_of_overwriting(tmp_path: Path):
    """Il salvataggio giornaliero non deve cancellare lo storico annuale esistente."""
    first_day = _ohlcv_row(date(2026, 4, 1), 100.0)
    second_day = _ohlcv_row(date(2026, 4, 2), 101.0)

    _save_by_year(first_day, tmp_path)
    _save_by_year(second_day, tmp_path)

    stored = pl.read_parquet(tmp_path / "ohlcv" / "AAPL" / "2026.parquet").sort("valid_time")

    assert stored.height == 2
    assert stored["valid_time"].to_list() == [date(2026, 4, 1), date(2026, 4, 2)]
