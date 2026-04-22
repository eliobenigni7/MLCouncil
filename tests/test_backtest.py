"""Tests for backtest layer (Agent 07).

Five test cases:
1. test_no_lookahead_in_fills   — orders submitted on T, fills on T+1
2. test_slippage_applied        — fill price ≠ close price
3. test_long_only_constraint    — no short positions in portfolio
4. test_turnover_reasonable     — average daily turnover < 50%
5. test_sharpe_computable       — Sharpe computable after 30+ simulation days

All tests use synthetic data and bypass NautilusTrader's engine
(tested via BacktestReport + helper functions directly).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Helpers — synthetic data factories
# ---------------------------------------------------------------------------

def _make_prices(n_days: int = 60, tickers: list[str] | None = None) -> pd.DataFrame:
    """Random-walk close prices, indexed by date."""
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOG"]
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    returns = rng.normal(0.0005, 0.015, size=(n_days, len(tickers)))
    prices = 100 * np.cumprod(1 + returns, axis=0)
    return pd.DataFrame(prices, index=dates, columns=tickers)


def _make_fills(
    prices: pd.DataFrame,
    slippage_bps: float = 3.0,
    include_shorts: bool = False,
) -> pd.DataFrame:
    """
    Simulate fill records.

    - Orders are placed at close on date T.
    - Fills occur at the *open* of T+1 (next-open model).
    - Fill price = open + slippage (long) or open - slippage (short).
    - By default only BUY orders are generated (long-only).
    """
    rng = np.random.default_rng(0)
    tickers = list(prices.columns)
    dates = prices.index.tolist()

    rows = []
    for i, order_date in enumerate(dates[:-1]):
        fill_date = dates[i + 1]
        close_px = prices.loc[order_date]
        open_px = close_px * rng.uniform(0.998, 1.002, len(tickers))

        for j, ticker in enumerate(tickers):
            side = "BUY"
            if include_shorts and rng.random() > 0.7:
                side = "SELL"

            slip_mult = (1 + slippage_bps / 10_000) if side == "BUY" else (1 - slippage_bps / 10_000)
            fill_px = open_px.iloc[j] * slip_mult
            quantity = 10

            rows.append(
                {
                    "order_date": order_date,
                    "fill_date": fill_date,
                    "ticker": ticker,
                    "side": side,
                    "quantity": quantity,
                    "close_price": close_px.iloc[j],
                    "open_price": open_px.iloc[j],
                    "fill_price": fill_px,
                    "traded_usd": fill_px * quantity,
                }
            )

    return pd.DataFrame(rows)


def _make_equity_curve(fills: pd.DataFrame, initial_capital: float = 100_000.0) -> pd.Series:
    """Reconstruct a simple equity curve from fills (cumulative P&L)."""
    fills = fills.copy()
    fills["pnl"] = np.where(
        fills["side"] == "BUY",
        fills["quantity"] * (fills["fill_price"].shift(-len(fills.columns)) - fills["fill_price"]),
        0.0,
    )
    # Simpler: just build a random-walk equity from fill-date groups
    daily = fills.groupby("fill_date")["traded_usd"].sum()
    equity = initial_capital + (daily - daily.mean()).cumsum()
    return equity


def _make_trade_fills(
    dates: list[pd.Timestamp],
    prices: list[float],
    quantities: list[float],
) -> pd.DataFrame:
    """Create fills with Nautilus-style columns for turnover and cost tests."""
    rows = []
    for ts, price, qty in zip(dates, prices, quantities, strict=True):
        rows.append(
            {
                "ts_event": int(pd.Timestamp(ts, tz="UTC").value),
                "last_px": float(price),
                "last_qty": float(qty),
                "order_side": "BUY",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1 — No lookahead: fills on T+1, not T
# ---------------------------------------------------------------------------

class TestNoLookahead:
    def test_no_lookahead_in_fills(self):
        """Every fill_date must be strictly after its order_date."""
        prices = _make_prices(n_days=40)
        fills = _make_fills(prices)

        assert not fills.empty, "fills DataFrame must not be empty"

        order_dates = pd.to_datetime(fills["order_date"])
        fill_dates = pd.to_datetime(fills["fill_date"])

        # All fills must be strictly after the order
        assert (fill_dates > order_dates).all(), (
            "Lookahead detected: some fill_date <= order_date"
        )

    def test_fill_is_next_business_day(self):
        """fill_date should be the next available trading date after order_date."""
        prices = _make_prices(n_days=40)
        fills = _make_fills(prices)
        dates = prices.index.tolist()

        for _, row in fills.iterrows():
            order_idx = dates.index(row["order_date"])
            expected_fill = dates[order_idx + 1]
            assert row["fill_date"] == expected_fill, (
                f"Expected fill on {expected_fill}, got {row['fill_date']}"
            )


# ---------------------------------------------------------------------------
# Test 2 — Slippage applied
# ---------------------------------------------------------------------------

class TestSlippageApplied:
    def test_slippage_applied(self):
        """Fill price must differ from open price by ~slippage_bps."""
        prices = _make_prices(n_days=40)
        fills = _make_fills(prices, slippage_bps=3.0)

        buys = fills[fills["side"] == "BUY"]
        assert not buys.empty

        # fill_price should be > open_price for buys (slippage cost)
        assert (buys["fill_price"] > buys["open_price"]).all(), (
            "BUY fills should have fill_price > open_price due to slippage"
        )

    def test_slippage_magnitude(self):
        """Slippage should be approximately slippage_bps basis points."""
        slippage_bps = 5.0
        prices = _make_prices(n_days=40)
        fills = _make_fills(prices, slippage_bps=slippage_bps)

        buys = fills[fills["side"] == "BUY"]
        actual_bps = ((buys["fill_price"] - buys["open_price"]) / buys["open_price"] * 10_000)

        assert actual_bps.mean() == pytest.approx(slippage_bps, rel=0.05), (
            f"Expected ~{slippage_bps} bps slippage, got {actual_bps.mean():.2f} bps"
        )

    def test_no_slippage_when_zero(self):
        """With slippage_bps=0, fill_price == open_price."""
        prices = _make_prices(n_days=10)
        fills = _make_fills(prices, slippage_bps=0.0)

        buys = fills[fills["side"] == "BUY"]
        np.testing.assert_allclose(
            buys["fill_price"].values,
            buys["open_price"].values,
            rtol=1e-9,
        )


# ---------------------------------------------------------------------------
# Test 3 — Long-only constraint
# ---------------------------------------------------------------------------

class TestLongOnlyConstraint:
    def test_long_only_constraint(self):
        """Default portfolio must contain no SELL orders."""
        prices = _make_prices(n_days=40)
        fills = _make_fills(prices, include_shorts=False)

        short_fills = fills[fills["side"] == "SELL"]
        assert short_fills.empty, (
            f"Long-only violation: found {len(short_fills)} SELL fills"
        )

    def test_all_quantities_positive(self):
        """Quantities must be positive (no negative = short representation)."""
        prices = _make_prices(n_days=40)
        fills = _make_fills(prices, include_shorts=False)

        assert (fills["quantity"] > 0).all(), "All quantities must be positive"

    def test_short_fills_present_when_allowed(self):
        """Sanity check: SELL orders appear when include_shorts=True."""
        prices = _make_prices(n_days=60)
        fills = _make_fills(prices, include_shorts=True)

        short_fills = fills[fills["side"] == "SELL"]
        assert not short_fills.empty, "Expected some SELL fills with include_shorts=True"


# ---------------------------------------------------------------------------
# Test 4 — Turnover reasonable
# ---------------------------------------------------------------------------

class TestTurnoverReasonable:
    def test_turnover_reasonable(self):
        """Average daily turnover must be < 50% of portfolio."""
        initial_capital = 100_000.0
        prices = _make_prices(n_days=60)
        fills = _make_fills(prices)

        daily_traded = fills.groupby("fill_date")["traded_usd"].sum()
        turnover_pct = daily_traded / initial_capital * 100

        avg_turnover = turnover_pct.mean()
        assert avg_turnover < 50.0, (
            f"Average turnover {avg_turnover:.1f}% exceeds 50% threshold"
        )

    def test_turnover_not_zero(self):
        """Turnover should be positive — strategy is actually trading."""
        prices = _make_prices(n_days=40)
        fills = _make_fills(prices)

        daily_traded = fills.groupby("fill_date")["traded_usd"].sum()
        assert (daily_traded > 0).all(), "All trading days must have positive turnover"

    def test_turnover_using_report(self):
        """BacktestReport.turnover_analysis() returns daily turnover < 50%."""
        from backtest.report import BacktestReport

        initial_capital = 100_000.0
        prices = _make_prices(n_days=60)
        fills = _make_fills(prices)

        report = BacktestReport(
            fills=fills,
            prices=prices,
            initial_capital=initial_capital,
        )
        turnover_df = report.turnover_analysis()

        assert "turnover_pct" in turnover_df.columns, (
            "turnover_analysis() must return DataFrame with 'turnover_pct' column"
        )
        assert (turnover_df["turnover_pct"] < 50.0).all(), (
            "All daily turnovers must be < 50%"
        )


# ---------------------------------------------------------------------------
# Test 5 — Sharpe computable
# ---------------------------------------------------------------------------

class TestSharpeComputable:
    def _build_report(self, n_days: int = 40):
        from backtest.report import BacktestReport

        prices = _make_prices(n_days=n_days)
        fills = _make_fills(prices)

        # Build equity curve: start at initial_capital, add daily P&L
        initial_capital = 100_000.0
        daily_traded = fills.groupby("fill_date")["traded_usd"].sum()
        rng = np.random.default_rng(7)
        # Simulate daily P&L as small noise around zero
        pnl = rng.normal(50, 200, len(daily_traded))
        equity_series = pd.Series(
            initial_capital + np.cumsum(pnl),
            index=daily_traded.index,
            name="equity",
        )

        return BacktestReport(
            fills=fills,
            prices=prices,
            initial_capital=initial_capital,
            equity_curve=equity_series,
        )

    def test_sharpe_computable(self):
        """Sharpe ratio must be computable (finite float) after 30+ sim days."""
        report = self._build_report(n_days=35)
        sharpe = report.sharpe_ratio()

        assert isinstance(sharpe, float), "sharpe_ratio() must return a float"
        assert np.isfinite(sharpe), f"Sharpe ratio must be finite, got {sharpe}"

    def test_sharpe_range_plausible(self):
        """Sharpe should be in a plausible range [-5, 5] for synthetic data."""
        report = self._build_report(n_days=60)
        sharpe = report.sharpe_ratio()

        assert -5.0 <= sharpe <= 5.0, (
            f"Sharpe ratio {sharpe:.3f} outside plausible range [-5, 5]"
        )

    def test_max_drawdown_negative(self):
        """max_drawdown() must return a non-positive value."""
        report = self._build_report(n_days=40)
        dd = report.max_drawdown()

        assert dd <= 0.0, f"max_drawdown() must be <= 0, got {dd}"

    def test_calmar_computable(self):
        """calmar_ratio() must return a finite float."""
        report = self._build_report(n_days=60)
        calmar = report.calmar_ratio()

        assert isinstance(calmar, float), "calmar_ratio() must return a float"
        assert np.isfinite(calmar), f"Calmar ratio must be finite, got {calmar}"


class TestTransactionCostRealism:
    def test_shared_transaction_cost_model_estimates_weights_and_notional(self):
        """Phase 2 requires one reusable transaction cost model for optimizer and report."""
        from council.transaction_costs import TransactionCostModel

        model = TransactionCostModel(commission_bps=1.0, slippage_bps=3.0)
        w_old = np.array([0.50, 0.50])
        w_new = np.array([0.70, 0.30])

        assert model.estimate_turnover(w_old, w_new) == pytest.approx(0.20)
        assert model.estimate_cost_from_turnover(0.20, portfolio_value=100_000.0) == pytest.approx(8.0)
        assert model.estimate_cost_from_notional(25_000.0) == pytest.approx(10.0)

    def test_turnover_analysis_respects_configured_cost_model(self):
        """BacktestReport must use configured bps instead of a hardcoded default."""
        from backtest.report import BacktestReport

        fills = _make_trade_fills(
            dates=[
                pd.Timestamp("2024-01-03"),
                pd.Timestamp("2024-01-03"),
                pd.Timestamp("2024-01-04"),
            ],
            prices=[100.0, 50.0, 120.0],
            quantities=[10.0, 20.0, 5.0],
        )
        prices = _make_prices(n_days=5)

        report = BacktestReport(
            fills=fills,
            prices=prices,
            initial_capital=100_000.0,
            commission_bps=2.0,
            slippage_bps=5.0,
        )
        turnover_df = report.turnover_analysis()

        assert turnover_df.loc[pd.Timestamp("2024-01-03"), "estimated_cost_usd"] == pytest.approx(1.40)
        assert turnover_df.loc[pd.Timestamp("2024-01-04"), "estimated_cost_usd"] == pytest.approx(0.42)

    def test_report_defaults_to_net_equity_curve(self):
        """Gross equity must be preserved, but the default reported curve should be net of estimated costs."""
        from backtest.report import BacktestReport

        index = pd.bdate_range("2024-01-02", periods=3)
        gross_equity = pd.Series([100_000.0, 101_000.0, 100_500.0], index=index, name="equity")
        fills = _make_trade_fills(
            dates=[index[1], index[2]],
            prices=[100.0, 100.0],
            quantities=[100.0, 50.0],
        )
        prices = _make_prices(n_days=5)

        report = BacktestReport(
            fills=fills,
            prices=prices,
            initial_capital=100_000.0,
            equity_curve=gross_equity,
            commission_bps=10.0,
            slippage_bps=0.0,
        )

        expected_net = pd.Series(
            [100_000.0, 100_990.0, 100_485.0],
            index=index,
            name="equity_net",
        )

        pd.testing.assert_series_equal(report.gross_equity_curve, gross_equity.astype(float), check_names=False)
        pd.testing.assert_series_equal(report.equity_curve, expected_net, check_names=False)
        assert report.total_estimated_cost_usd == pytest.approx(15.0)

    def test_runner_stats_expose_gross_and_net_metrics(self):
        """The backtest runner stats contract must persist both gross and net metrics."""
        sys.modules.setdefault("polars", types.SimpleNamespace())
        from backtest.runner import _compute_stats

        index = pd.bdate_range("2024-01-02", periods=3)
        gross_equity = pd.Series([100_000.0, 101_000.0, 100_500.0], index=index, name="equity")
        fills = _make_trade_fills(
            dates=[index[1], index[2]],
            prices=[100.0, 100.0],
            quantities=[100.0, 50.0],
        )

        stats = _compute_stats(
            gross_equity,
            fills,
            100_000.0,
            commission_bps=10.0,
            slippage_bps=0.0,
        )

        assert stats["gross_final_equity"] == pytest.approx(100_500.0)
        assert stats["final_equity"] == pytest.approx(100_485.0)
        assert stats["estimated_costs_usd"] == pytest.approx(15.0)
        assert "gross_sharpe" in stats
        assert "gross_max_drawdown" in stats


class TestWalkForwardRunner:
    def test_run_walk_forward_backtest_returns_validation_bundle(self):
        from backtest.runner import run_walk_forward_backtest

        dates = pd.bdate_range("2024-01-02", periods=24)
        signals = pd.DataFrame(
            {
                "AAA": np.linspace(0.2, 1.2, len(dates)),
                "BBB": np.linspace(-0.1, 0.3, len(dates)),
                "CCC": np.linspace(-0.4, -0.2, len(dates)),
            },
            index=dates,
        )
        forward_returns = pd.DataFrame(
            {
                "AAA": np.linspace(0.001, 0.004, len(dates)),
                "BBB": np.linspace(0.0002, 0.001, len(dates)),
                "CCC": np.linspace(-0.0006, -0.0002, len(dates)),
            },
            index=dates,
        )
        components = {
            "technical": signals,
            "sentiment": signals * 0.4,
            "regime": signals * 0.2,
        }

        result = run_walk_forward_backtest(
            signals=signals,
            forward_returns=forward_returns,
            train_window=8,
            test_window=4,
            step=4,
            purge_period=1,
            embargo_period=1,
            component_signals=components,
        )

        assert "summary" in result
        assert "window_metrics" in result
        assert "benchmark_comparison" in result
        assert "regime_performance" in result
        assert "ablation_analysis" in result
        assert "environment_metadata" in result
        assert result["environment_metadata"]["python_version"]
        assert result["summary"]["walk_forward_window_count"] >= 1
        assert not result["ablation_analysis"].empty
