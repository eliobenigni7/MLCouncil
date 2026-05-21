"""Tests for the OMS → fill_log integration (Task 2.2).

Verifies that add_fill() mirrors fills into the structured monthly parquet
log without breaking existing OMS behaviour.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from execution import fill_log as fl_module
from execution.fill_log import read_fills
from execution.oms import Fill, OrderManager


@pytest.fixture
def oms_with_fill_log(tmp_path: Path, monkeypatch):
    """OrderManager scoped to a tmp dir, with fill_log redirected to tmp."""
    # redirect fill_log to a tmp directory so we don't touch the real repo
    tmp_fills = tmp_path / "fills"
    monkeypatch.setattr(fl_module, "FILL_LOG_DIR", tmp_fills)

    orders_dir = tmp_path / "orders"
    orders_dir.mkdir()
    # redirect OMS pending-orders & fills sidecar dir
    import execution.oms as oms_module

    monkeypatch.setattr(oms_module, "OMS_DIR", tmp_path / "oms")
    (tmp_path / "oms").mkdir()

    oms = OrderManager(orders_dir=orders_dir)
    return oms, tmp_fills


def test_add_fill_writes_structured_record(oms_with_fill_log):
    oms, tmp_fills = oms_with_fill_log
    order = oms.create_order(
        symbol="AAPL",
        quantity=100,
        side="buy",
        order_type="market",
        decision_price=200.0,
        tags={"pipeline_run_id": "run-1", "config_hash": "cfg-1"},
    )
    fill = Fill(
        fill_id="f1",
        order_id=order.order_id,
        symbol="AAPL",
        quantity=100,
        price=200.5,
        commission=0.40,  # 0.40 USD / (100 * 200.5) = ~0.2 bps
        timestamp=datetime(2026, 5, 21, 14, 30, tzinfo=timezone.utc),
        venue="ALPACA",
    )
    oms.add_fill(order, fill)

    df = read_fills(base=tmp_fills)
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["fill_id"] == "f1"
    assert row["ticker"] == "AAPL"
    assert row["side"] == "buy"
    assert row["decision_price"] == pytest.approx(200.0)
    assert row["fill_price"] == pytest.approx(200.5)
    assert row["pipeline_run_id"] == "run-1"
    assert row["config_hash"] == "cfg-1"
    assert row["commission_bps"] == pytest.approx(0.40 / 200.5 * 100, rel=0.05)  # ~0.2 bps


def test_add_fill_does_not_log_when_decision_price_unavailable(oms_with_fill_log):
    """If decision_price falls back to fill.price (no decision context),
    the implementation shortfall is 0 by construction and the record is
    still useful for cost-model audits."""
    oms, tmp_fills = oms_with_fill_log
    order = oms.create_order(symbol="MSFT", quantity=10, side="sell", order_type="market")
    fill = Fill(
        fill_id="f2",
        order_id=order.order_id,
        symbol="MSFT",
        quantity=10,
        price=420.0,
        commission=0.10,
        timestamp=datetime(2026, 5, 21, 15, 0, tzinfo=timezone.utc),
    )
    oms.add_fill(order, fill)
    df = read_fills(base=tmp_fills)
    # decision_price falls back to fill_price → IS == 0
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["decision_price"] == pytest.approx(420.0)
    assert row["fill_price"] == pytest.approx(420.0)


def test_fill_log_failure_does_not_break_add_fill(oms_with_fill_log, monkeypatch):
    """If the fill_log raises, the order lifecycle still completes."""
    oms, _ = oms_with_fill_log

    def boom(*args, **kwargs):
        raise RuntimeError("fill log down")

    monkeypatch.setattr("execution.fill_log.append_fill", boom)

    order = oms.create_order(symbol="GOOGL", quantity=5, side="buy", decision_price=150.0)
    fill = Fill(
        fill_id="f3",
        order_id=order.order_id,
        symbol="GOOGL",
        quantity=5,
        price=151.0,
        commission=0.05,
        timestamp=datetime(2026, 5, 21, 15, 30, tzinfo=timezone.utc),
    )
    oms.add_fill(order, fill)
    # Order still marked FILLED despite telemetry failure
    assert order.filled_quantity == 5
    assert order.avg_fill_price == pytest.approx(151.0)


def test_create_order_decision_price_defaults_to_limit_price(oms_with_fill_log):
    oms, _ = oms_with_fill_log
    order = oms.create_order(
        symbol="NVDA",
        quantity=20,
        side="buy",
        order_type="limit",
        limit_price=900.0,
    )
    assert order.decision_price == pytest.approx(900.0)
