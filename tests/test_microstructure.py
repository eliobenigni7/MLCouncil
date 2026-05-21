"""Tests for T2.4 microstructure / OFI challenger (synthetic L2 only)."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from data.ingest.orderbook import (
    OrderBookFeed,
    build_adapter,
    ingest_orderbook_snapshots,
    iter_ofi_from_snapshots,
    synthetic_book_sequence,
)
from intraday.market_data import BookLevel, BookSnapshot, compute_ofi
from models.microstructure import (
    MicrostructureModel,
    build_synthetic_fixture_ofi,
    microstructure_promoted,
    microstructure_shadow_enabled,
)


class TestComputeOfi:
    def test_fixture_ofi_equals_five(self):
        snaps = synthetic_book_sequence(symbol="AAPL")
        assert len(snaps) == 2

        t0_bid = snaps[0].cumulative_bid_volume()
        t0_ask = snaps[0].cumulative_ask_volume()
        assert t0_bid == 300.0
        assert t0_ask == 250.0

        t1_bid = snaps[1].cumulative_bid_volume()
        t1_ask = snaps[1].cumulative_ask_volume()
        assert t1_bid == 310.0
        assert t1_ask == 255.0

        ofi = compute_ofi(snaps[1], snaps[0])
        assert ofi == pytest.approx(5.0)
        assert ofi == pytest.approx((t1_bid - t0_bid) - (t1_ask - t0_ask))

    def test_build_synthetic_fixture_helper(self):
        ofi, snaps = build_synthetic_fixture_ofi()
        assert ofi == pytest.approx(5.0)
        assert len(snaps) == 2

    def test_iter_ofi_yields_single_step(self):
        snaps = synthetic_book_sequence()
        steps = list(iter_ofi_from_snapshots(snaps))
        assert len(steps) == 1
        assert steps[0][1] == pytest.approx(5.0)

    def test_first_snapshot_without_previous_returns_zero(self):
        snap = synthetic_book_sequence()[0]
        assert compute_ofi(snap, None) == 0.0


class TestOrderBookIngest:
    def test_synthetic_ingest_returns_two_snapshots(self):
        as_of = datetime(2026, 5, 21, 15, 30, tzinfo=timezone.utc)
        result = ingest_orderbook_snapshots(
            symbol="MSFT",
            as_of=as_of,
            feed=OrderBookFeed.SYNTHETIC.value,
        )
        assert result.deferred_live is False
        assert result.feed == OrderBookFeed.SYNTHETIC
        assert len(result.snapshots) == 2
        assert result.snapshots[0].symbol == "MSFT"

    def test_alpaca_feed_deferred_without_snapshots(self):
        result = ingest_orderbook_snapshots(
            symbol="AAPL",
            feed=OrderBookFeed.ALPACA.value,
        )
        assert result.deferred_live is True
        assert result.snapshots == []

    def test_alpaca_adapter_raises_not_implemented(self):
        adapter = build_adapter(OrderBookFeed.ALPACA)
        with pytest.raises(NotImplementedError, match="deferred"):
            adapter.fetch_snapshots(
                symbol="AAPL",
                as_of=datetime(2026, 5, 21, tzinfo=timezone.utc),
            )


class TestMicrostructureModel:
    def test_shadow_mode_default(self, monkeypatch):
        monkeypatch.delenv("MLCOUNCIL_MICROSTRUCTURE_PROMOTED", raising=False)
        monkeypatch.setenv("MLCOUNCIL_MICROSTRUCTURE_SHADOW", "true")
        assert microstructure_shadow_enabled() is True
        assert microstructure_promoted() is False

    def test_ofi_from_snapshots_matches_formula(self):
        model = MicrostructureModel()
        snaps = synthetic_book_sequence()
        assert model.ofi_from_snapshots(snaps) == pytest.approx(5.0)

    def test_predict_zscores_cross_section(self):
        model = MicrostructureModel()
        signals = model.predict(tickers=["AAPL", "MSFT"])
        assert len(signals) == 2
        assert abs(float(signals.std())) == pytest.approx(1.0, abs=1e-6) or np.std(signals) < 1e-12

    def test_update_snapshot_incremental(self):
        model = MicrostructureModel()
        snaps = synthetic_book_sequence()
        r0 = model.update_snapshot(snaps[0])
        assert r0.ofi == 0.0
        r1 = model.update_snapshot(snaps[1])
        assert r1.ofi == pytest.approx(5.0)
        assert r1.shadow_mode is True

    def test_custom_book_levels_respected(self):
        snap0 = BookSnapshot(
            symbol="TEST",
            as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
            bids=[BookLevel(10.0, 50.0), BookLevel(9.99, 25.0)],
            asks=[BookLevel(10.01, 40.0), BookLevel(10.02, 20.0)],
            levels=2,
        )
        snap1 = BookSnapshot(
            symbol="TEST",
            as_of=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            bids=[BookLevel(10.0, 60.0), BookLevel(9.99, 25.0)],
            asks=[BookLevel(10.01, 45.0), BookLevel(10.02, 20.0)],
            levels=2,
        )
        # delta bid = 10, delta ask = 5 → OFI = 5
        assert compute_ofi(snap1, snap0, levels=2) == pytest.approx(5.0)
