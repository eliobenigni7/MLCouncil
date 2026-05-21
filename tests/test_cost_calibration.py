"""Tests for council.cost_calibration."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from council.cost_calibration import (
    CalibrationArtifact,
    CalibrationHashError,
    CostCalibrator,
    compute_is_bps,
    load_calibration,
    run_calibration_job,
    ticker_tier,
    write_calibration,
)
from execution.fill_log import FillRecord, append_fills


def _make_fills_df(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# math
# ---------------------------------------------------------------------------


class TestComputeIsBps:
    def test_buy_pays_up_is_positive(self):
        df = _make_fills_df([
            {"ticker": "AAPL", "side": "buy", "fill_price": 200.5, "decision_price": 200.0,
             "fill_ts": datetime(2026, 5, 1, tzinfo=timezone.utc)}
        ])
        out = compute_is_bps(df)
        assert out["is_bps"][0] == pytest.approx(25.0)

    def test_sell_below_decision_is_positive(self):
        df = _make_fills_df([
            {"ticker": "AAPL", "side": "sell", "fill_price": 199.0, "decision_price": 200.0,
             "fill_ts": datetime(2026, 5, 1, tzinfo=timezone.utc)}
        ])
        out = compute_is_bps(df)
        assert out["is_bps"][0] == pytest.approx(50.0)

    def test_empty_frame_returns_empty(self):
        df = pl.DataFrame({"ticker": [], "side": [], "fill_price": [], "decision_price": []})
        out = compute_is_bps(df)
        assert "is_bps" in out.columns


# ---------------------------------------------------------------------------
# tier mapping
# ---------------------------------------------------------------------------


def test_ticker_tier_known_and_default():
    assert ticker_tier("AAPL") == "mega"
    assert ticker_tier("ETSY") == "mid"
    assert ticker_tier("BTCUSD") == "crypto"
    assert ticker_tier("UNKNOWN") == "default"


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


class TestCostCalibrator:
    def _make_fills(self, ticker: str, n: int, is_bps: float) -> pl.DataFrame:
        # synthesize n fills with constant implementation shortfall.
        # fill_price = decision_price * (1 + is_bps/10000 * sign)
        rows = []
        for i in range(n):
            decision_price = 100.0
            slip_pct = is_bps / 10_000.0
            fill_price = decision_price * (1.0 + slip_pct)
            rows.append({
                "ticker": ticker,
                "side": "buy",
                "qty": 1.0,
                "fill_price": fill_price,
                "decision_price": decision_price,
                "fill_ts": datetime(2026, 5, 1, tzinfo=timezone.utc) + timedelta(minutes=i),
            })
        return pl.DataFrame(rows)

    def test_kappa_matches_constant_is_bps(self):
        fills = self._make_fills("AAPL", n=40, is_bps=4.0)
        art = CostCalibrator(min_fills=30).calibrate(fills)
        assert art.kappa_by_ticker["AAPL"] == pytest.approx(4.0)
        assert art.fill_count_by_ticker["AAPL"] == 40
        # AAPL → mega tier
        assert art.kappa_by_tier["mega"] == pytest.approx(4.0)

    def test_tickers_below_min_fills_excluded(self):
        fills = pl.concat([
            self._make_fills("AAPL", n=40, is_bps=3.0),
            self._make_fills("ETSY", n=10, is_bps=20.0),
        ])
        art = CostCalibrator(min_fills=30).calibrate(fills)
        assert "AAPL" in art.kappa_by_ticker
        assert "ETSY" not in art.kappa_by_ticker
        # Tier roll-up: mega has 40 fills (≥30) → included.
        # mid has only 10 → excluded.
        assert "mega" in art.kappa_by_tier
        assert "mid" not in art.kappa_by_tier

    def test_empty_fills_returns_empty_artifact(self):
        empty = pl.DataFrame({
            "ticker": [],
            "side": [],
            "fill_price": [],
            "decision_price": [],
            "fill_ts": [],
        }, schema={
            "ticker": pl.Utf8,
            "side": pl.Utf8,
            "fill_price": pl.Float64,
            "decision_price": pl.Float64,
            "fill_ts": pl.Datetime("us", "UTC"),
        })
        art = CostCalibrator().calibrate(empty)
        assert art.fill_sample_count == 0
        assert art.kappa_by_ticker == {}


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def _make_artifact(self) -> CalibrationArtifact:
        return CalibrationArtifact(
            generated_at=datetime(2026, 5, 21, 23, 0, tzinfo=timezone.utc),
            calibration_window_end=datetime(2026, 5, 21, 21, 30, tzinfo=timezone.utc),
            fill_sample_count=120,
            min_fills=30,
            kappa_by_ticker={"AAPL": 2.5, "MSFT": 3.1},
            fill_count_by_ticker={"AAPL": 60, "MSFT": 60},
            kappa_by_tier={"mega": 2.8},
            fill_count_by_tier={"mega": 120},
            pipeline_run_id="run-1",
            config_hash="cfg-1",
        )

    def test_round_trip(self, tmp_path: Path):
        art = self._make_artifact()
        out = tmp_path / "cost_calibration.json"
        version = write_calibration(art, path=out)
        assert (tmp_path / "cost_calibration.json.manifest").exists()
        loaded = load_calibration(out)
        assert loaded.kappa_by_ticker == art.kappa_by_ticker
        assert loaded.version == version

    def test_tampered_artifact_raises(self, tmp_path: Path):
        art = self._make_artifact()
        out = tmp_path / "cost_calibration.json"
        write_calibration(art, path=out)
        # tamper with the JSON
        data = out.read_text()
        out.write_text(data.replace("2.5", "9.9"))
        with pytest.raises(CalibrationHashError, match="hash mismatch"):
            load_calibration(out)

    def test_missing_manifest_raises(self, tmp_path: Path):
        art = self._make_artifact()
        out = tmp_path / "cost_calibration.json"
        write_calibration(art, path=out)
        (tmp_path / "cost_calibration.json.manifest").unlink()
        with pytest.raises(CalibrationHashError, match="Missing manifest"):
            load_calibration(out)

    def test_bypass_manifest_only_with_explicit_flag(self, tmp_path: Path):
        art = self._make_artifact()
        out = tmp_path / "cost_calibration.json"
        write_calibration(art, path=out)
        (tmp_path / "cost_calibration.json.manifest").unlink()
        loaded = load_calibration(out, require_manifest=False)
        assert loaded.kappa_by_ticker["AAPL"] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# end-to-end job
# ---------------------------------------------------------------------------


class TestRunCalibrationJob:
    def test_empty_fills_dir_returns_none(self, tmp_path: Path):
        out = tmp_path / "calibration.json"
        result = run_calibration_job(
            fills_dir=tmp_path / "fills",
            out_path=out,
        )
        assert result is None
        assert not out.exists()

    def test_end_to_end_writes_artifact(self, tmp_path: Path):
        fills_dir = tmp_path / "fills"
        # populate the fill log with 35 deterministic buys
        records = []
        base_ts = datetime(2026, 5, 21, 14, 0, tzinfo=timezone.utc)
        for i in range(35):
            records.append(FillRecord(
                fill_id=f"f{i}",
                order_id=f"o{i}",
                ticker="AAPL",
                side="buy",
                qty=10.0,
                fill_price=200.06,  # +3 bps
                decision_price=200.0,
                decision_ts=base_ts,
                fill_ts=base_ts + timedelta(minutes=i),
                broker="alpaca",
                venue="ALPACA",
            ))
        append_fills(records, base=fills_dir)

        out = tmp_path / "calibration.json"
        result = run_calibration_job(
            fills_dir=fills_dir,
            out_path=out,
            min_fills=30,
            pipeline_run_id="run-test",
            config_hash="cfg-test",
        )
        assert result is not None
        assert "AAPL" in result.kappa_by_ticker
        assert result.kappa_by_ticker["AAPL"] == pytest.approx(3.0, abs=0.05)
        assert out.exists()
        assert out.with_suffix(out.suffix + ".manifest").exists()

        # round-trip with hash verification succeeds
        loaded = load_calibration(out)
        assert loaded.kappa_by_ticker["AAPL"] == pytest.approx(3.0, abs=0.05)
        assert loaded.pipeline_run_id == "run-test"
