"""Deterministic strategy backtest utilities for MLCouncil.

This module provides a non-Nautilus fallback that simulates a daily rebalanced
portfolio from target weights and forward returns, applying shared transaction
cost estimates. It is intended to produce dashboard- and analysis-friendly
artifacts when the full backtest engine is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from council.transaction_costs import TransactionCostModel


@dataclass(frozen=True)
class StrategyBacktestInputs:
    weights: pd.DataFrame
    forward_returns: pd.DataFrame
    initial_capital: float = 100_000.0
    cost_model: Optional[TransactionCostModel] = None


def _normalize_long_only_weights(row: pd.Series) -> pd.Series:
    weights = row.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    total = float(weights.sum())
    if total <= 1e-12:
        return weights * 0.0
    return weights / total


def _annualized_sharpe(returns: pd.Series) -> float:
    clean = returns.dropna().astype(float)
    if len(clean) < 2:
        return 0.0
    std = float(clean.std())
    if std < 1e-12:
        return 0.0
    return float(clean.mean() / std * np.sqrt(252))


def _max_drawdown(equity: pd.Series) -> float:
    clean = equity.dropna().astype(float)
    if clean.empty:
        return 0.0
    rolling_max = clean.cummax()
    drawdown = (clean - rolling_max) / rolling_max
    return float(drawdown.min())


def simulate_weight_backtest(
    *,
    weights: pd.DataFrame,
    forward_returns: pd.DataFrame,
    initial_capital: float = 100_000.0,
    cost_model: Optional[TransactionCostModel] = None,
) -> "BacktestResult":
    """Simulate a daily rebalanced long-only strategy.

    Parameters
    ----------
    weights:
        Target weights per rebalance date. Index = date, columns = tickers.
    forward_returns:
        Forward 1-day returns aligned to the same rebalance dates.
    initial_capital:
        Starting portfolio value.
    cost_model:
        Shared commission + slippage estimator.

    Returns
    -------
    BacktestResult
        Lightweight result object with gross/net equity curves and stats.
    """
    from backtest.runner import BacktestResult

    cost_model = cost_model or TransactionCostModel.from_env()

    if weights.empty or forward_returns.empty:
        empty = pd.Series(dtype=float, name="equity")
        return BacktestResult(
            fills=pd.DataFrame(),
            positions=pd.DataFrame(),
            equity_curve=empty,
            gross_equity_curve=empty,
            stats={"sharpe": 0.0, "gross_sharpe": 0.0, "max_drawdown": 0.0, "gross_max_drawdown": 0.0,
                   "cagr": 0.0, "gross_cagr": 0.0, "calmar": 0.0, "gross_calmar": 0.0,
                   "n_trades": 0, "n_years": 0.0, "final_equity": initial_capital,
                   "gross_final_equity": initial_capital, "estimated_costs_usd": 0.0, "turnover": 0.0},
            strategy_fills=pd.DataFrame(),
        )

    common_index = weights.index.intersection(forward_returns.index)
    common_cols = weights.columns.intersection(forward_returns.columns)
    if len(common_index) == 0 or len(common_cols) == 0:
        empty = pd.Series(dtype=float, name="equity")
        return BacktestResult(
            fills=pd.DataFrame(),
            positions=pd.DataFrame(),
            equity_curve=empty,
            gross_equity_curve=empty,
            stats={"sharpe": 0.0, "gross_sharpe": 0.0, "max_drawdown": 0.0, "gross_max_drawdown": 0.0,
                   "cagr": 0.0, "gross_cagr": 0.0, "calmar": 0.0, "gross_calmar": 0.0,
                   "n_trades": 0, "n_years": 0.0, "final_equity": initial_capital,
                   "gross_final_equity": initial_capital, "estimated_costs_usd": 0.0, "turnover": 0.0},
            strategy_fills=pd.DataFrame(),
        )

    w = weights.loc[common_index, common_cols].copy().astype(float)
    fwd = forward_returns.loc[common_index, common_cols].copy().astype(float).fillna(0.0)
    w = w.apply(_normalize_long_only_weights, axis=1)

    prev_weights = pd.Series(0.0, index=common_cols, dtype=float)
    gross_capital = float(initial_capital)
    net_capital = float(initial_capital)

    gross_values: list[float] = []
    net_values: list[float] = []
    gross_returns: list[float] = []
    net_returns: list[float] = []
    turnovers: list[float] = []
    costs: list[float] = []
    rows: list[dict[str, float | str]] = []

    for d in common_index:
        w_t = w.loc[d].fillna(0.0).astype(float)
        r_t = fwd.loc[d].fillna(0.0).astype(float)

        gross_ret = float((w_t * r_t).sum())
        turnover = cost_model.estimate_turnover(prev_weights.values, w_t.values)
        cost_usd = cost_model.estimate_cost_from_turnover(turnover, portfolio_value=net_capital)
        net_ret = gross_ret - (cost_usd / net_capital if net_capital > 0 else 0.0)

        gross_capital *= 1.0 + gross_ret
        net_capital *= 1.0 + net_ret

        gross_values.append(gross_capital)
        net_values.append(net_capital)
        gross_returns.append(gross_ret)
        net_returns.append(net_ret)
        turnovers.append(turnover)
        costs.append(cost_usd)
        rows.append(
            {
                "date": d,
                "gross_return": gross_ret,
                "net_return": net_ret,
                "turnover": turnover,
                "estimated_cost_usd": cost_usd,
                "portfolio_value_start": net_capital / (1.0 + net_ret) if (1.0 + net_ret) != 0 else net_capital,
                "portfolio_value_end": net_capital,
            }
        )
        prev_weights = w_t

    index = pd.DatetimeIndex(common_index)
    gross_equity = pd.Series(gross_values, index=index, name="equity")
    net_equity = pd.Series(net_values, index=index, name="equity")
    gross_ret_series = pd.Series(gross_returns, index=index, name="gross_return")
    net_ret_series = pd.Series(net_returns, index=index, name="net_return")

    n_years = len(net_equity) / 252.0
    gross_final = float(gross_equity.iloc[-1])
    net_final = float(net_equity.iloc[-1])
    gross_cagr = (gross_final / initial_capital) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0
    net_cagr = (net_final / initial_capital) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0
    gross_mdd = _max_drawdown(gross_equity)
    net_mdd = _max_drawdown(net_equity)
    gross_sharpe = _annualized_sharpe(gross_ret_series)
    net_sharpe = _annualized_sharpe(net_ret_series)
    turnover_mean = float(np.mean(turnovers)) if turnovers else 0.0
    total_costs = float(np.sum(costs)) if costs else 0.0

    stats = {
        "sharpe": net_sharpe,
        "gross_sharpe": gross_sharpe,
        "max_drawdown": net_mdd,
        "gross_max_drawdown": gross_mdd,
        "cagr": float(net_cagr),
        "gross_cagr": float(gross_cagr),
        "calmar": float(net_cagr / abs(net_mdd)) if abs(net_mdd) > 1e-12 else float("inf"),
        "gross_calmar": float(gross_cagr / abs(gross_mdd)) if abs(gross_mdd) > 1e-12 else float("inf"),
        "n_trades": int(len(common_index)),
        "n_years": round(n_years, 2),
        "final_equity": net_final,
        "gross_final_equity": gross_final,
        "estimated_costs_usd": total_costs,
        "turnover": turnover_mean,
    }

    strategy_fills = pd.DataFrame(rows)
    strategy_fills["date"] = pd.to_datetime(strategy_fills["date"])
    strategy_fills = strategy_fills.set_index("date")

    return BacktestResult(
        fills=pd.DataFrame(),
        positions=pd.DataFrame(),
        equity_curve=net_equity,
        gross_equity_curve=gross_equity,
        stats=stats,
        strategy_fills=strategy_fills,
    )
