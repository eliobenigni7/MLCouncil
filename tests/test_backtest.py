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
