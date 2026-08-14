"""Analytics service: port of ``dashboard/data_loader.py`` (13 loaders).

Stessa priorità degli artifact e stesse trasformazioni del loader Streamlit,
così i numeri combaciano byte-per-byte. Gli artifact mancanti sollevano
``ApiError(404, "artifact_not_found", ...)`` e la SPA renderizza stati vuoti.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from api.errors import ApiError

DATA_DIR = Path(os.getenv("MLCOUNCIL_DATA_DIR", "data"))

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


def _results_dir_for_tag(results_tag: str | None) -> Path:
    if not results_tag:
        return DATA_DIR / "results"
    return DATA_DIR / "results_snapshots" / results_tag


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


def _densify_business_days(series: pd.Series) -> pd.Series:
    """Reindex a time series to business days and forward-fill gaps.

    Dashboard artifacts are expected to be daily equity curves. Some sources
    persist only sparse rebalance snapshots; filling the missing business days
    keeps the chart continuous and makes rolling-window metrics usable.
    """
    if series is None or series.empty:
        return series

    out = series.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    if len(out) < 2:
        return out

    full_index = pd.bdate_range(out.index.min(), out.index.max())
    out = out.reindex(full_index).ffill()
    out.name = series.name
    return out


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


def _series_json(s: pd.Series) -> dict:
    s = s.dropna()
    return {"dates": [d.isoformat() for d in s.index], "values": [float(v) for v in s.values]}


def _records(df: pd.DataFrame) -> dict:
    out = []
    for _, row in df.iterrows():
        rec = {}
        for col, val in row.items():
            if pd.isna(val):
                rec[col] = None
            elif hasattr(val, "isoformat"):
                rec[col] = val.isoformat()
            else:
                try:
                    rec[col] = float(val)
                except (TypeError, ValueError):
                    rec[col] = val
        out.append(rec)
    return {"records": out}


def _artifact(path: Path, what: str) -> Path:
    if not path.exists():
        raise ApiError(404, "artifact_not_found", f"{what} not available yet", str(path))
    return path


# ============================================================================
# Equity curve
# ============================================================================


def _equity_series(mode: str = "Paper Trading", results_tag: str | None = None) -> pd.Series:
    """Load equity curve from backtest result or paper trading logs.

    Returns a Series normalized to 100 at inception (public-safe).
    """
    equity = _try_load_equity_from_disk(mode, results_tag=results_tag)
    if equity is None or equity.empty:
        raise ApiError(
            404,
            "artifact_not_found",
            "Equity curve not available yet",
            str(_results_dir_for_tag(results_tag)),
        )

    # Normalize to 100 — hides actual capital from public view
    if equity.iloc[0] > 0:
        equity = equity / equity.iloc[0] * 100.0
    equity = _densify_business_days(equity)
    equity.name = "equity_normalized"
    return equity


def load_equity_curve(mode: str = "Paper Trading", results_tag: str | None = None) -> dict:
    """Equity curve normalized to 100 at inception (public-safe)."""
    return _series_json(_equity_series(mode, results_tag=results_tag))


def _try_load_equity_from_disk(mode: str, results_tag: str | None = None) -> Optional[pd.Series]:
    """Try to load a real equity curve from disk artifacts (same priority as data_loader)."""
    results_dir = _results_dir_for_tag(results_tag)
    # 1. Pickled BacktestResult
    result_pkl = results_dir / "backtest_result.pkl"
    if result_pkl.exists():
        try:
            from council.pickle_security import trusted_pickle_load

            result = trusted_pickle_load(result_pkl, require_hash=True)
            curve = getattr(result, "equity_curve", None)
            if curve is not None and not curve.empty:
                return curve
        except Exception:
            pass

    # 2. Parquet equity log in data/results/
    equity_pq = results_dir / "equity_curve.parquet"
    if equity_pq.exists():
        try:
            df = pd.read_parquet(equity_pq)
            col = next((c for c in ["equity", "value", "portfolio_value"] if c in df.columns), None)
            if col:
                return df[col].dropna()
        except Exception:
            pass

    if results_tag is not None:
        return None

    # 3. Paper trading daily snapshots: data/orders/YYYY-MM-DD.parquet
    orders_dir = DATA_DIR / "orders"
    if mode == "Paper Trading" and orders_dir.exists():
        try:
            snapshots = sorted(orders_dir.glob("*.parquet"))
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


def _load_equity_from_risk_reports() -> Optional[pd.Series]:
    risk_dir = DATA_DIR / "risk"
    if not risk_dir.exists():
        return None

    rows = []
    for path in sorted(risk_dir.glob("risk_report_*.json"))[-252:]:
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
    paper_dir = DATA_DIR / "paper_trades"
    if not paper_dir.exists():
        return None

    rows = []
    for path in sorted(paper_dir.glob("*.json"))[-252:]:
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
# Benchmark (SPY)
# ============================================================================


def load_benchmark(mode: str = "Paper Trading", results_tag: str | None = None) -> dict:
    """SPY benchmark matching equity curve dates, normalized to 100."""
    equity = _equity_series(mode, results_tag=results_tag)

    start = equity.index[0]
    end = equity.index[-1]

    # Try local parquet
    spy_pq = DATA_DIR / "raw" / "ohlcv" / "SPY"
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
                        spy = spy / spy.iloc[0] * 100.0
                        spy = spy.reindex(equity.index, method="ffill").dropna()
                        if not spy.empty:
                            spy = _densify_business_days(spy)
                            spy.name = "SPY"
                            return _series_json(spy)
        except Exception:
            pass

    sp500_pq = DATA_DIR / "raw" / "macro" / "sp500.parquet"
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
                    return _series_json(spy)
        except Exception:
            pass

    raise ApiError(404, "artifact_not_found", "Benchmark not available yet", str(spy_pq))


# ============================================================================
# Returns
# ============================================================================


def _returns_series(mode: str = "Paper Trading", results_tag: str | None = None) -> pd.Series:
    """Daily returns derived from equity curve (raises 404 when no data)."""
    equity = _equity_series(mode, results_tag=results_tag)
    returns = equity.pct_change().dropna()
    if returns.empty:
        raise ApiError(
            404,
            "artifact_not_found",
            "Returns not available yet",
            str(_results_dir_for_tag(results_tag)),
        )
    returns.name = "returns"
    return returns


def load_daily_returns(mode: str = "Paper Trading", results_tag: str | None = None) -> dict:
    """Daily returns derived from equity curve."""
    return _series_json(_returns_series(mode, results_tag=results_tag))


# ============================================================================
# Model attribution
# ============================================================================


def _attribution_df(start: Optional[date] = None, end: Optional[date] = None) -> pd.DataFrame:
    """Per-model P&L attribution DataFrame (raises 404 when no artifact)."""
    result = _try_load_attribution_from_disk(start, end)
    if result is not None and not result.empty:
        return result
    raise ApiError(
        404,
        "artifact_not_found",
        "Model attribution not available yet",
        str(DATA_DIR / "results" / "attribution.parquet"),
    )


def load_model_attribution(start: Optional[date] = None, end: Optional[date] = None) -> dict:
    """Per-model P&L attribution: date, model_name, weight, ic_rolling_30d,
    sharpe_rolling_60d, pnl_contribution."""
    return _records(_attribution_df(start, end))


def _try_load_attribution_from_disk(
    start: Optional[date],
    end: Optional[date],
) -> Optional[pd.DataFrame]:
    """Try to load attribution data from persisted CouncilAggregator state."""
    results_dir = DATA_DIR / "results"
    # Check for pickled aggregator
    agg_pkl = results_dir / "aggregator.pkl"
    if agg_pkl.exists():
        try:
            from council.pickle_security import trusted_pickle_load

            agg = trusted_pickle_load(agg_pkl, require_hash=True)
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
    attr_pq = results_dir / "attribution.parquet"
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


# ============================================================================
# IC history per model (for ic_rolling_chart)
# ============================================================================


def load_ic_history() -> dict:
    """Rolling IC (30-day) per model over time: date, lgbm, sentiment, hmm."""
    attr = _attribution_df()
    pivot = (
        attr[["date", "model_name", "ic_rolling_30d"]]
        .pivot_table(index="date", columns="model_name", values="ic_rolling_30d")
        .sort_index()
    )
    pivot.columns.name = None
    return _records(pivot.reset_index())


# ============================================================================
# Weights history (for weight_evolution_chart)
# ============================================================================


def load_weights_history() -> dict:
    """Council weights evolution over time: date, lgbm, sentiment, hmm (sum ~1)."""
    attr = _attribution_df()
    pivot = (
        attr[["date", "model_name", "weight"]]
        .pivot_table(index="date", columns="model_name", values="weight")
        .sort_index()
    )
    pivot.columns.name = None
    return _records(pivot.reset_index())


# ============================================================================
# Current regime
# ============================================================================


def _regime_info() -> dict:
    result = _try_load_regime_from_disk()
    return result if result else dict(_UNKNOWN_REGIME)


def load_current_regime() -> dict:
    """Current regime and probabilities: regime, bull, bear, transition."""
    result = _try_load_regime_from_disk()
    if result:
        return result
    raise ApiError(
        404,
        "artifact_not_found",
        "Current regime not available yet",
        str(DATA_DIR / "results" / "current_regime.json"),
    )


def _try_load_regime_from_disk() -> Optional[dict]:
    """Try to load regime from latest pipeline output."""
    results_dir = DATA_DIR / "results"
    # Check for JSON/parquet regime snapshot
    regime_json = results_dir / "current_regime.json"
    if regime_json.exists():
        try:
            with open(regime_json) as f:
                return json.load(f)
        except Exception:
            pass

    # Check latest orders parquet for regime column
    orders_dir = DATA_DIR / "orders"
    if orders_dir.exists():
        try:
            snapshots = sorted(orders_dir.glob("*.parquet"))
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


def load_regime_history() -> dict:
    """Historical regime classifications: date, regime, prob_bull, prob_bear, prob_transition."""
    hist_pq = DATA_DIR / "results" / "regime_history.parquet"
    _artifact(hist_pq, "Regime history")
    try:
        df = pd.read_parquet(hist_pq)
        # Normalize column name: pipeline writes "valid_time", dashboard expects "date"
        if "valid_time" in df.columns:
            df = df.rename(columns={"valid_time": "date"})
        return _records(df)
    except Exception:
        raise ApiError(404, "artifact_not_found", "Regime history not available yet", str(hist_pq))


# ============================================================================
# Portfolio snapshot
# ============================================================================


def load_portfolio_snapshot() -> dict:
    """Current positions and target weights (normalized weights, no USD values)."""
    orders_dir = DATA_DIR / "orders"
    if orders_dir.exists():
        try:
            snapshots = sorted(orders_dir.glob("*.parquet"))
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
                return _records(snapshot.head(20))
        except Exception:
            pass

    paper_dir = DATA_DIR / "paper_trades"
    if paper_dir.exists():
        try:
            snapshots = sorted(paper_dir.glob("*.json"))
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
                        return _records(pd.DataFrame(rows).head(20))
        except Exception:
            pass

    raise ApiError(404, "artifact_not_found", "Portfolio snapshot not available yet", str(orders_dir))


# ============================================================================
# Aggregate sidebar metrics
# ============================================================================


def load_sidebar_metrics() -> dict:
    """Aggregate dashboard metrics for the sidebar.

    Keys: sharpe_ytd, max_dd, ic_30d, regime, regime_prob,
          sharpe_delta, dd_delta, ic_delta
    """
    returns = _returns_series()
    regime_info = _regime_info()

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
    equity = _equity_series()
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
    attr = _attribution_df()
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


# ============================================================================
# Optimization diagnostics + council weights log (math-trace)
# ============================================================================


def load_optimization_diagnostics(as_of: date) -> dict:
    """Persisted portfolio optimizer diagnostics for a date (raw JSON)."""
    path = DATA_DIR / "results" / "optimization_diagnostics" / f"{as_of.isoformat()}.json"
    _artifact(path, "Optimization diagnostics")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        raise ApiError(404, "artifact_not_found", "Optimization diagnostics not available yet", str(path))


def load_council_weights_log_entry(as_of: date) -> dict:
    """Council aggregator weights_log entry for math-trace panel."""
    agg_pkl = DATA_DIR / "results" / "aggregator.pkl"
    _artifact(agg_pkl, "Council aggregator weights log")
    try:
        from council.pickle_security import trusted_pickle_load

        agg = trusted_pickle_load(agg_pkl, require_hash=True)
        key = as_of
        if key not in agg._weights_log:
            for k in sorted(agg._weights_log.keys(), reverse=True):
                if k <= as_of:
                    key = k
                    break
            else:
                return {}
        return dict(agg._weights_log.get(key, {}))
    except Exception:
        return {}


# ============================================================================
# Fill quality
# ============================================================================


def load_fill_quality_summary() -> dict:
    """Per-ticker fill quality: median IS, lookup slippage, calibrated kappa."""
    from council.cost_calibration import DEFAULT_CALIBRATION_PATH, load_calibration
    from council.transaction_costs import estimate_slippage_bps, get_calibration_path
    from execution.fill_log import read_fills

    fills_dir = DATA_DIR / "operations" / "fills"
    _artifact(fills_dir, "Fill quality")
    try:
        fills = read_fills(base=fills_dir)
    except Exception:
        raise ApiError(404, "artifact_not_found", "Fill quality not available yet", str(fills_dir))

    import polars as pl
    from council.cost_calibration import compute_is_bps

    if fills.height == 0:
        return {"records": []}

    if "is_bps" not in fills.columns:
        fills = compute_is_bps(fills)

    summary = fills.group_by("ticker").agg(
        pl.col("is_bps").median().alias("median_is_bps"),
        pl.len().alias("fill_count"),
    )
    pdf = summary.to_pandas()
    pdf["lookup_slippage_bps"] = pdf["ticker"].map(estimate_slippage_bps)

    calib_path = get_calibration_path() or DEFAULT_CALIBRATION_PATH
    kappa_map: dict[str, float] = {}
    if calib_path.exists():
        try:
            artifact = load_calibration(calib_path)
            kappa_map = {**artifact.kappa_by_ticker, **artifact.kappa_by_tier}
        except Exception:
            pass
    pdf["kappa_calibrated_bps"] = pdf["ticker"].map(
        lambda t: kappa_map.get(t, float("nan"))
    )
    return _records(pdf)
