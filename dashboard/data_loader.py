"""Data loader for the MLCouncil Streamlit dashboard.

All functions use st.cache_data to avoid re-loading on every rerender.
Public deployment: only normalized metrics are exposed (no raw capital/orders).

Real artifacts are preferred. When a dataset is unavailable, loaders return an
empty state instead of synthetic demo values.
"""

from __future__ import annotations

import json
import pickle
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_ORDERS_DIR = _ROOT / "data" / "orders"
_OPERATIONS_DIR = _ROOT / "data" / "operations"
_PAPER_TRADES_DIR = _ROOT / "data" / "paper_trades"
_RAW_DIR = _ROOT / "data" / "raw"
_RISK_DIR = _ROOT / "data" / "risk"
_RESULTS_DIR = _ROOT / "data" / "results"
_UNKNOWN_REGIME = {"regime": "unknown", "bull": 0.0, "bear": 0.0, "transition": 0.0}
_ATTRIBUTION_COLUMNS = [
    "date",
    "model_name",
    "weight",
    "ic_rolling_30d",
    "sharpe_rolling_60d",
    "pnl_contribution",
]
_REGIME_HISTORY_COLUMNS = ["date", "regime", "prob_bull", "prob_bear", "prob_transition"]
_PORTFOLIO_SNAPSHOT_COLUMNS = ["ticker", "weight_target", "weight_current", "signal"]


def _flatten_universe_tickers(universe: dict) -> list[str]:
    tickers = universe.get("tickers")
    if isinstance(tickers, list):
        return tickers

    flattened: list[str] = []
    for key, value in universe.items():
        if key == "settings" or not isinstance(value, list):
            continue
        flattened.extend(str(ticker) for ticker in value)
    return list(dict.fromkeys(flattened))


def _empty_series(name: str) -> pd.Series:
    return pd.Series(dtype=float, name=name)


def _empty_attribution() -> pd.DataFrame:
    return pd.DataFrame(columns=_ATTRIBUTION_COLUMNS)


def _empty_regime_history() -> pd.DataFrame:
    return pd.DataFrame(columns=_REGIME_HISTORY_COLUMNS)


def _empty_portfolio_snapshot() -> pd.DataFrame:
    return pd.DataFrame(columns=_PORTFOLIO_SNAPSHOT_COLUMNS)


def _load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normalize_trade_snapshot(payload) -> Optional[dict]:
    if isinstance(payload, dict):
        if any(key in payload for key in ("account", "orders", "pretrade", "reconciliation")):
            return payload
        return None

    if isinstance(payload, list):
        for item in reversed(payload):
            if isinstance(item, dict) and any(
                key in item for key in ("account", "orders", "pretrade", "reconciliation")
            ):
                return item

    return None


def _load_equity_from_risk_reports() -> Optional[pd.Series]:
    if not _RISK_DIR.exists():
        return None

    rows = []
    for path in sorted(_RISK_DIR.glob("risk_report_*.json"))[-252:]:
        try:
            payload = _load_json(path)
            portfolio_value = float(payload.get("portfolio_value", 0.0) or 0.0)
            if portfolio_value <= 0:
                continue
            rows.append(
                {
                    "date": pd.Timestamp(path.stem.replace("risk_report_", "")),
                    "equity": portfolio_value,
                }
            )
        except Exception:
            pass

    if not rows:
        return None

    out = pd.DataFrame(rows).drop_duplicates(subset="date").sort_values("date")
    return out.set_index("date")["equity"]


def _load_equity_from_trade_logs() -> Optional[pd.Series]:
    if not _PAPER_TRADES_DIR.exists():
        return None

    rows = []
    for path in sorted(_PAPER_TRADES_DIR.glob("*.json"))[-252:]:
        try:
            payload = _normalize_trade_snapshot(_load_json(path))
            if payload is None:
                continue
            account = payload.get("account", {})
            portfolio_value = float(account.get("portfolio_value", 0.0) or 0.0)
            if portfolio_value <= 0:
                continue
            rows.append({"date": pd.Timestamp(path.stem), "equity": portfolio_value})
        except Exception:
            pass

    if not rows:
        return None

    out = pd.DataFrame(rows).drop_duplicates(subset="date").sort_values("date")
    return out.set_index("date")["equity"]


# ============================================================================
# Equity curve
# ============================================================================

@st.cache_data(ttl=3600)
def load_equity_curve(mode: str = "Paper Trading") -> pd.Series:
    """Load equity curve from backtest result or paper trading logs.

    Returns a Series normalized to 100 at inception (public-safe).
    Columns: DatetimeIndex, values: normalized portfolio value.
    """
    equity = _try_load_equity_from_disk(mode)
    if equity is None or equity.empty:
        return _empty_series("equity_normalized")

    # Normalize to 100 — hides actual capital from public view
    if equity.iloc[0] > 0:
        equity = equity / equity.iloc[0] * 100.0
    equity.name = "equity_normalized"
    return equity


def _try_load_equity_from_disk(mode: str) -> Optional[pd.Series]:
    """Try to load a real equity curve from disk artifacts."""
    # 1. Pickled BacktestResult
    result_pkl = _RESULTS_DIR / "backtest_result.pkl"
    if result_pkl.exists():
        try:
            with open(result_pkl, "rb") as f:
                result = pickle.load(f)
            curve = getattr(result, "equity_curve", None)
            if curve is not None and not curve.empty:
                return curve
        except Exception:
            pass

    # 2. Parquet equity log in data/results/
    equity_pq = _RESULTS_DIR / "equity_curve.parquet"
    if equity_pq.exists():
        try:
            df = pd.read_parquet(equity_pq)
            col = next((c for c in ["equity", "value", "portfolio_value"] if c in df.columns), None)
            if col:
                return df[col].dropna()
        except Exception:
            pass

    # 3. Paper trading daily snapshots: data/orders/YYYY-MM-DD.parquet
    if mode == "Paper Trading" and _ORDERS_DIR.exists():
        try:
            snapshots = sorted(_ORDERS_DIR.glob("*.parquet"))
            if snapshots:
                rows = []
                for pq in snapshots[-252:]:  # last year max
                    try:
                        df = pd.read_parquet(pq)
                        if "portfolio_value" in df.columns:
                            d = pd.Timestamp(pq.stem)
                            rows.append({"date": d, "equity": float(df["portfolio_value"].iloc[-1])})
                    except Exception:
                        pass
                if rows:
                    out = pd.DataFrame(rows).set_index("date")["equity"]
                    out.index = pd.to_datetime(out.index)
                    return out
        except Exception:
            pass

    if mode == "Paper Trading":
        for loader in (_load_equity_from_risk_reports, _load_equity_from_trade_logs):
            equity = loader()
            if equity is not None and not equity.empty:
                equity.index = pd.to_datetime(equity.index)
                return equity.sort_index()

    return None


def _synthetic_equity_curve(mode: str) -> pd.Series:
    """Generate a plausible synthetic equity curve for demo mode."""
    rng = np.random.default_rng(42)
    n = 504 if mode == "Backtest" else 126  # 2 years backtest, 6 months paper
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    # Slight positive drift, realistic vol
    daily_ret = rng.normal(0.0006, 0.012, size=n)
    equity = 100_000 * np.cumprod(1 + daily_ret)
    return pd.Series(equity, index=dates, name="equity")


# ============================================================================
# Benchmark (SPY)
# ============================================================================

@st.cache_data(ttl=3600)
def load_benchmark(mode: str = "Paper Trading") -> pd.Series:
    """Load SPY benchmark matching equity curve dates."""
    equity = load_equity_curve(mode)
    if equity.empty:
        return _empty_series("SPY")

    start = equity.index[0]
    end = equity.index[-1]

    # Try local parquet
    spy_pq = _RAW_DIR / "ohlcv" / "SPY"
    if spy_pq.exists():
        try:
            import polars as pl
            frames = [pl.read_parquet(p) for p in sorted(spy_pq.glob("*.parquet"))]
            if frames:
                df = pl.concat(frames).to_pandas()
                col = next((c for c in ["adj_close", "close"] if c in df.columns), None)
                date_col = next((c for c in ["valid_time", "date"] if c in df.columns), None)
                if col and date_col:
                    spy = df.set_index(date_col)[col].dropna()
                    spy.index = pd.to_datetime(spy.index)
                    spy = spy[(spy.index >= start) & (spy.index <= end)]
                    if not spy.empty:
                        return spy / spy.iloc[0] * 100.0
        except Exception:
            pass

    sp500_pq = _RAW_DIR / "macro" / "sp500.parquet"
    if sp500_pq.exists():
        try:
            df = pd.read_parquet(sp500_pq)
            if {"valid_time", "sp500_price"}.issubset(df.columns):
                spy = df.set_index("valid_time")["sp500_price"].dropna()
                spy.index = pd.to_datetime(spy.index)
                spy = spy[(spy.index >= start) & (spy.index <= end)]
                spy = spy.reindex(equity.index, method="ffill").dropna()
                if not spy.empty:
                    spy = spy / spy.iloc[0] * 100.0
                    spy.name = "SPY"
                    return spy
        except Exception:
            pass

    return _empty_series("SPY")


# ============================================================================
# Returns
# ============================================================================

@st.cache_data(ttl=3600)
def load_daily_returns(mode: str = "Paper Trading") -> pd.Series:
    """Daily returns derived from equity curve."""
    equity = load_equity_curve(mode)
    returns = equity.pct_change().dropna()
    returns.name = "returns"
    return returns


# ============================================================================
# Model attribution
# ============================================================================

@st.cache_data(ttl=3600)
def load_model_attribution(
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> pd.DataFrame:
    """Load per-model P&L attribution DataFrame.

    Returns
    -------
    pd.DataFrame with columns:
        date, model_name, weight, ic_rolling_30d, sharpe_rolling_60d,
        pnl_contribution
    """
    result = _try_load_attribution_from_disk(start, end)
    if result is not None and not result.empty:
        return result
    return _empty_attribution()


def _try_load_attribution_from_disk(
    start: Optional[date],
    end: Optional[date],
) -> Optional[pd.DataFrame]:
    """Try to load attribution data from persisted CouncilAggregator state."""
    # Check for pickled aggregator
    agg_pkl = _RESULTS_DIR / "aggregator.pkl"
    if agg_pkl.exists():
        try:
            with open(agg_pkl, "rb") as f:
                agg = pickle.load(f)
            # Build multi-date attribution
            dates = sorted(agg._weights_log.keys())
            if start:
                dates = [d for d in dates if d >= start]
            if end:
                dates = [d for d in dates if d <= end]
            if not dates:
                return None
            frames = []
            for d in dates:
                df = agg.get_attribution(d)
                df["date"] = pd.Timestamp(d)
                frames.append(df)
            return pd.concat(frames, ignore_index=True)
        except Exception:
            pass

    # Check for parquet attribution log
    attr_pq = _RESULTS_DIR / "attribution.parquet"
    if attr_pq.exists():
        try:
            df = pd.read_parquet(attr_pq)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                if start:
                    df = df[df["date"] >= pd.Timestamp(start)]
                if end:
                    df = df[df["date"] <= pd.Timestamp(end)]
            return df if not df.empty else None
        except Exception:
            pass

    return None


def _synthetic_attribution(
    start: Optional[date],
    end: Optional[date],
) -> pd.DataFrame:
    """Synthetic attribution data for demo mode."""
    rng = np.random.default_rng(7)
    if end is None:
        end = date.today()
    if start is None:
        start = end - timedelta(days=180)

    dates = pd.bdate_range(start=start, end=end)
    models = ["lgbm", "sentiment", "hmm"]
    # Regime-like weight evolution
    regimes = ["bull"] * 60 + ["transition"] * 20 + ["bear"] * 30 + ["bull"] * 90
    base_weights = {
        "bull":       {"lgbm": 0.50, "sentiment": 0.30, "hmm": 0.20},
        "bear":       {"lgbm": 0.40, "sentiment": 0.20, "hmm": 0.40},
        "transition": {"lgbm": 0.45, "sentiment": 0.25, "hmm": 0.30},
    }

    rows = []
    for i, d in enumerate(dates):
        regime = regimes[i % len(regimes)]
        bw = base_weights[regime]
        for m in models:
            ic = rng.normal(0.04, 0.025)
            w = bw[m] * (1 + rng.normal(0, 0.05))
            rows.append({
                "date": d,
                "model_name": m,
                "weight": float(np.clip(w, 0.05, 0.70)),
                "ic_rolling_30d": float(ic),
                "sharpe_rolling_60d": float(rng.normal(1.2, 0.4)),
                "pnl_contribution": float(ic * w * rng.normal(1, 0.3)),
            })

    return pd.DataFrame(rows)


# ============================================================================
# IC history per model (for ic_rolling_chart)
# ============================================================================

@st.cache_data(ttl=3600)
def load_ic_history() -> pd.DataFrame:
    """Load rolling IC (30-day) per model over time.

    Returns
    -------
    pd.DataFrame with columns: date, lgbm, sentiment, hmm
    """
    attr = load_model_attribution()
    if attr.empty:
        return pd.DataFrame()

    pivot = (
        attr[["date", "model_name", "ic_rolling_30d"]]
        .pivot_table(index="date", columns="model_name", values="ic_rolling_30d")
        .sort_index()
    )
    pivot.columns.name = None
    return pivot.reset_index()


# ============================================================================
# Weights history (for weight_evolution_chart)
# ============================================================================

@st.cache_data(ttl=3600)
def load_weights_history() -> pd.DataFrame:
    """Load council weights evolution over time.

    Returns
    -------
    pd.DataFrame with columns: date, lgbm, sentiment, hmm  (weights sum ~1)
    """
    attr = load_model_attribution()
    if attr.empty:
        return pd.DataFrame()

    pivot = (
        attr[["date", "model_name", "weight"]]
        .pivot_table(index="date", columns="model_name", values="weight")
        .sort_index()
    )
    pivot.columns.name = None
    return pivot.reset_index()


# ============================================================================
# Current regime
# ============================================================================

@st.cache_data(ttl=300)
def load_current_regime() -> dict:
    """Load current regime and probabilities.

    Returns
    -------
    dict with keys: regime (str), bull (float), bear (float), transition (float)
    """
    result = _try_load_regime_from_disk()
    if result:
        return result
    return dict(_UNKNOWN_REGIME)


def _try_load_regime_from_disk() -> Optional[dict]:
    """Try to load regime from latest pipeline output."""
    # Check for JSON/parquet regime snapshot
    regime_json = _RESULTS_DIR / "current_regime.json"
    if regime_json.exists():
        try:
            with open(regime_json) as f:
                return json.load(f)
        except Exception:
            pass

    # Check latest orders parquet for regime column
    if _ORDERS_DIR.exists():
        try:
            snapshots = sorted(_ORDERS_DIR.glob("*.parquet"))
            if snapshots:
                df = pd.read_parquet(snapshots[-1])
                if "regime" in df.columns:
                    regime = str(df["regime"].iloc[-1])
                    probs = {
                        "regime": regime,
                        "bull": float(df.get("prob_bull", pd.Series([0.5])).iloc[-1]),
                        "bear": float(df.get("prob_bear", pd.Series([0.2])).iloc[-1]),
                        "transition": float(df.get("prob_transition", pd.Series([0.3])).iloc[-1]),
                    }
                    return probs
        except Exception:
            pass

    return None


# ============================================================================
# Regime history (for timeline chart)
# ============================================================================

@st.cache_data(ttl=3600)
def load_regime_history() -> pd.DataFrame:
    """Load historical regime classifications.

    Returns
    -------
    pd.DataFrame with columns: date, regime, prob_bull, prob_bear, prob_transition
    """
    hist_pq = _RESULTS_DIR / "regime_history.parquet"
    if hist_pq.exists():
        try:
            return pd.read_parquet(hist_pq)
        except Exception:
            pass

    return _empty_regime_history()


def _synthetic_regime_history() -> pd.DataFrame:
    """Synthetic regime history for demo."""
    equity = load_equity_curve()
    dates = equity.index
    rng = np.random.default_rng(13)

    # Markov-like regime sequence
    states = ["bull", "bear", "transition"]
    trans = {
        "bull":       [0.93, 0.03, 0.04],
        "bear":       [0.05, 0.90, 0.05],
        "transition": [0.30, 0.20, 0.50],
    }
    regimes = ["bull"]
    for _ in range(len(dates) - 1):
        probs = trans[regimes[-1]]
        regimes.append(rng.choice(states, p=probs))

    rows = []
    for d, r in zip(dates, regimes):
        if r == "bull":
            pb, pbe, pt = rng.dirichlet([7, 1, 2])
        elif r == "bear":
            pb, pbe, pt = rng.dirichlet([1, 7, 2])
        else:
            pb, pbe, pt = rng.dirichlet([2, 2, 6])
        rows.append({
            "date": d,
            "regime": r,
            "prob_bull": float(pb),
            "prob_bear": float(pbe),
            "prob_transition": float(pt),
        })

    return pd.DataFrame(rows)


# ============================================================================
# Portfolio snapshot
# ============================================================================

@st.cache_data(ttl=300)
def load_portfolio_snapshot() -> pd.DataFrame:
    """Load current positions and target weights.

    Returns
    -------
    pd.DataFrame with columns: ticker, weight_target, weight_current, signal
    Public-safe: no USD values, only normalized weights.
    """
    if _ORDERS_DIR.exists():
        try:
            snapshots = sorted(_ORDERS_DIR.glob("*.parquet"))
            if snapshots:
                df = pd.read_parquet(snapshots[-1])
                snapshot = pd.DataFrame(
                    {
                        "ticker": df.get("ticker", df.get("symbol", pd.Series(dtype=str))),
                        "weight_target": df.get("target_weight", pd.Series(dtype=float)),
                        "weight_current": df.get("weight_current", pd.Series(dtype=float)),
                        "signal": df.get("signal", pd.Series(dtype=float)),
                    }
                )
                return snapshot.head(20)
        except Exception:
            pass

    if _PAPER_TRADES_DIR.exists():
        try:
            snapshots = sorted(_PAPER_TRADES_DIR.glob("*.json"))
            if snapshots:
                payload = _normalize_trade_snapshot(_load_json(snapshots[-1]))
                if payload is not None:
                    account = payload.get("account", {})
                    portfolio_value = float(account.get("portfolio_value", 0.0) or 0.0)
                    rows = []
                    for order in payload.get("orders", []):
                        requested_notional = float(order.get("requested_notional", 0.0) or 0.0)
                        weight_target = (
                            requested_notional / portfolio_value
                            if portfolio_value > 0 and requested_notional > 0
                            else None
                        )
                        rows.append(
                            {
                                "ticker": order.get("symbol", order.get("ticker")),
                                "weight_target": weight_target,
                                "weight_current": None,
                                "signal": None,
                            }
                        )
                    if rows:
                        return pd.DataFrame(rows).head(20)
        except Exception:
            pass

    return _empty_portfolio_snapshot()


def _synthetic_portfolio_snapshot() -> pd.DataFrame:
    """Synthetic portfolio snapshot for demo."""
    import yaml
    try:
        with open(_ROOT / "config" / "universe.yaml") as f:
            cfg = yaml.safe_load(f)
        tickers = _flatten_universe_tickers(cfg["universe"])
    except Exception:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

    rng = np.random.default_rng(55)
    n = len(tickers)
    raw = rng.dirichlet(np.ones(n) * 2)
    signals = rng.normal(0, 1, n)
    return pd.DataFrame({
        "ticker": tickers,
        "weight_target": np.round(raw, 4),
        "weight_current": np.round(raw + rng.normal(0, 0.01, n), 4).clip(0),
        "signal": np.round(signals, 3),
    })


# ============================================================================
# Aggregate sidebar metrics
# ============================================================================

@st.cache_data(ttl=300)
def load_sidebar_metrics() -> dict:
    """Aggregate dashboard metrics for the sidebar.

    Returns
    -------
    dict with keys: sharpe_ytd, max_dd, ic_30d, regime, regime_prob,
                    sharpe_delta, dd_delta, ic_delta
    """
    returns = load_daily_returns()
    regime_info = load_current_regime()

    if returns.empty:
        return {
            "sharpe_ytd": 0.0, "max_dd": 0.0, "ic_30d": 0.0,
            "regime": "N/A", "regime_prob": 0.0,
            "sharpe_delta": 0.0, "dd_delta": 0.0, "ic_delta": 0.0,
        }

    # YTD returns
    ytd_start = pd.Timestamp(date.today().year, 1, 1)
    ytd_returns = returns[returns.index >= ytd_start]
    if ytd_returns.empty:
        ytd_returns = returns

    # Sharpe YTD
    rfr_daily = 0.05 / 252
    sharpe_ytd = (
        float((ytd_returns - rfr_daily).mean() / ytd_returns.std() * np.sqrt(252))
        if ytd_returns.std() > 0 else 0.0
    )

    # Yesterday Sharpe (for delta)
    prev_returns = returns[returns.index < returns.index[-1]]
    ytd_prev = prev_returns[prev_returns.index >= ytd_start]
    if not ytd_prev.empty and ytd_prev.std() > 0:
        sharpe_prev = float((ytd_prev - rfr_daily).mean() / ytd_prev.std() * np.sqrt(252))
    else:
        sharpe_prev = sharpe_ytd

    # Max drawdown (YTD)
    equity = load_equity_curve()
    equity_ytd = equity[equity.index >= ytd_start]
    if equity_ytd.empty:
        equity_ytd = equity
    rolling_max = equity_ytd.cummax()
    dd_series = (equity_ytd - rolling_max) / rolling_max
    max_dd = float(dd_series.min()) if not dd_series.empty else 0.0
    
    # Drawdown delta (yesterday's dd vs current dd)
    dd_delta = 0.0
    if len(equity_ytd) >= 2:
        dd_today = float(dd_series.iloc[-1])
        dd_prev = float(dd_series.iloc[-2])
        dd_delta = round(dd_today - dd_prev, 4)

    # IC 30d (latest from attribution)
    attr = load_model_attribution()
    ic_30d = 0.0
    ic_prev = 0.0
    if not attr.empty and "ic_rolling_30d" in attr.columns:
        latest_date = attr["date"].max()
        yesterday = latest_date - pd.Timedelta(days=1)
        latest_ic = attr[attr["date"] == latest_date]["ic_rolling_30d"].mean()
        prev_ic = attr[attr["date"] >= yesterday]["ic_rolling_30d"].mean()
        ic_30d = float(latest_ic) if not np.isnan(latest_ic) else 0.0
        ic_prev = float(prev_ic) if not np.isnan(prev_ic) else ic_30d

    regime = regime_info.get("regime", "N/A").capitalize()
    regime_prob = float(regime_info.get(regime_info.get("regime", "bull"), 0.0))

    return {
        "sharpe_ytd": round(sharpe_ytd, 3),
        "max_dd": round(max_dd * 100, 2),  # in %
        "ic_30d": round(ic_30d, 4),
        "regime": regime,
        "regime_prob": round(regime_prob * 100, 1),
        "sharpe_delta": round(sharpe_ytd - sharpe_prev, 3),
        "dd_delta": dd_delta,
        "ic_delta": round(ic_30d - ic_prev, 4),
    }
