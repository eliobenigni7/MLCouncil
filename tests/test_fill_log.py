"""Tests for execution.fill_log."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from execution.fill_log import (
    FillRecord,
    append_fill,
    append_fills,
    read_fills,
)


@pytest.fixture
def base_dir(tmp_path: Path) -> Path:
    return tmp_path / "fills"


def _make_record(
    fill_id: str = "F1",
    *,
    ticker: str = "AAPL",
    side: str = "buy",
    qty: float = 100.0,
    fill_price: float = 200.5,
    decision_price: float = 200.0,
    fill_ts: datetime | None = None,
    **kwargs,
) -> FillRecord:
    fill_ts = fill_ts or datetime(2026, 5, 21, 14, 30, tzinfo=timezone.utc)
    decision_ts = fill_ts - timedelta(minutes=5)
    return FillRecord(
        fill_id=fill_id,
        order_id="O1",
        ticker=ticker,
        side=side,
        qty=qty,
        fill_price=fill_price,
        decision_price=decision_price,
        decision_ts=decision_ts,
        fill_ts=fill_ts,
        broker="alpaca",
        venue="ALPACA",
        pipeline_run_id="run-2026-05-21",
        config_hash="abc123",
        commission_bps=1.0,
        slippage_bps_assumed=3.0,
        cost_calibration_version="static-v0",
        **kwargs,
    )


class TestFillRecord:
    def test_implementation_shortfall_buy_pays_up(self):
        r = _make_record(side="buy", fill_price=200.5, decision_price=200.0)
        # paid 0.25% more than decision → +25 bps cost
        assert r.implementation_shortfall_bps() == pytest.approx(25.0)

    def test_implementation_shortfall_buy_price_improvement(self):
        r = _make_record(side="buy", fill_price=199.0, decision_price=200.0)
        # filled below decision price → negative bps (price improvement)
        assert r.implementation_shortfall_bps() == pytest.approx(-50.0)

    def test_implementation_shortfall_sell_gets_lower(self):
        r = _make_record(side="sell", fill_price=199.0, decision_price=200.0)
        # sold 0.5% below decision → +50 bps cost (sign flips)
        assert r.implementation_shortfall_bps() == pytest.approx(50.0)

    def test_implementation_shortfall_sell_price_improvement(self):
        r = _make_record(side="sell", fill_price=201.0, decision_price=200.0)
        # sold above decision price → negative bps (price improvement)
        assert r.implementation_shortfall_bps() == pytest.approx(-50.0)

    def test_validation_rejects_bad_side(self):
        with pytest.raises(ValueError, match="side"):
            _make_record(side="long")

    def test_validation_rejects_non_positive_qty(self):
        with pytest.raises(ValueError, match="qty"):
            _make_record(qty=0)

    def test_validation_rejects_non_positive_price(self):
        with pytest.raises(ValueError, match="fill_price"):
            _make_record(fill_price=0)
        with pytest.raises(ValueError, match="decision_price"):
            _make_record(decision_price=-1.0)


class TestPersistence:
    def test_append_single_creates_partition(self, base_dir: Path):
        r = _make_record()
        path = append_fill(r, base=base_dir)
        assert path.exists()
        assert path.name == "2026-05.parquet"
        df = pl.read_parquet(path)
        assert df.height == 1
        assert df["fill_id"][0] == "F1"

    def test_append_multiple_dedup_on_fill_id(self, base_dir: Path):
        r1 = _make_record("F1")
        r2 = _make_record("F2")
        append_fills([r1, r2], base=base_dir)
        # re-append F1 with different price → should overwrite (keep last)
        r1_updated = _make_record("F1", fill_price=210.0)
        append_fill(r1_updated, base=base_dir)
        df = read_fills(base=base_dir).sort("fill_id")
        assert df.height == 2
        f1 = df.filter(pl.col("fill_id") == "F1").row(0, named=True)
        assert f1["fill_price"] == pytest.approx(210.0)

    def test_append_rejects_cross_month_batch(self, base_dir: Path):
        r1 = _make_record("F1", fill_ts=datetime(2026, 5, 31, 23, 30, tzinfo=timezone.utc))
        r2 = _make_record("F2", fill_ts=datetime(2026, 6, 1, 1, 0, tzinfo=timezone.utc))
        with pytest.raises(ValueError, match="month boundaries"):
            append_fills([r1, r2], base=base_dir)

    def test_read_fills_filters_by_window(self, base_dir: Path):
        r1 = _make_record("F1", fill_ts=datetime(2026, 5, 20, 10, 0, tzinfo=timezone.utc))
        r2 = _make_record("F2", fill_ts=datetime(2026, 5, 21, 10, 0, tzinfo=timezone.utc))
        append_fills([r1, r2], base=base_dir)
        df = read_fills(
            start=datetime(2026, 5, 21, tzinfo=timezone.utc),
            base=base_dir,
        )
        assert df.height == 1
        assert df["fill_id"][0] == "F2"

    def test_read_fills_empty_when_no_partition(self, base_dir: Path):
        df = read_fills(base=base_dir)
        assert df.height == 0
        assert "fill_id" in df.columns

    def test_atomic_write_no_partial_file(self, base_dir: Path, monkeypatch):
        """If parquet write fails mid-flight, the previous file is preserved."""
        r1 = _make_record("F1")
        append_fill(r1, base=base_dir)

        # patch os.replace to raise, ensuring the existing parquet stays intact
        import execution.fill_log as fl

        def boom(src, dst):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(fl.os, "replace", boom)
        with pytest.raises(RuntimeError):
            append_fill(_make_record("F2"), base=base_dir)

        df = read_fills(base=base_dir)
        # F1 still there, F2 not committed
        assert df.height == 1
        assert df["fill_id"][0] == "F1"
