"""Backtest Playground orchestrator.

A lightweight, Alpaca-free backtest entry point designed for the Streamlit
``Backtest Playground`` page. Lets the user experiment with:

  - Backtest window (start/end dates), initial capital, costs
  - Universe (subset of ``config/universe.yaml``)
  - Council regime weights (bull / bear / transition)
  - Portfolio constraints (max position, turnover, vol, sector cap, etc.)

Pipeline
--------
1. Load OHLCV for the universe via ``backtest.runner._load_ohlcv_polars``
   (local parquet + yfinance fallback — no Alpaca).
2. Generate per-model proxy signals (lgbm/sentiment/hmm) directly from
   prices. Proxies are fast (no LGBM/HMM training) but respond to the
   council weights exactly the same way real signals would.
3. Build a temp ``regime_weights.yaml`` from the user's choices and a
   patched ``os.environ`` for portfolio constraints.
4. For each rebalance date: ``CouncilAggregator.aggregate`` → council
   signal → ``PortfolioConstructor.optimize`` → target weights.
5. Replay the weight matrix through ``backtest.simulator.simulate_weight_backtest``.
6. Persist a snapshot to ``data/results_playground/<timestamp>/`` for
   later comparison.

The proxies are clearly labelled in the UI; users wanting the full
LGBM/FinBERT/HMM stack should run ``scripts/run_strategy_backtest.py``.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

RESULTS_DIR = _ROOT / "data" / "results_playground"
REBALANCE_EVERY = 5
COV_LOOKBACK = 60
MOMENTUM_WINDOW = 20
REGIME_FAST_MA = 50
REGIME_SLOW_MA = 200
BENCHMARK_TICKER = "SPY"

ProgressCb = Callable[[float, str], None]


# ===========================================================================
# Params / Result
# ===========================================================================

@dataclass
class PlaygroundParams:
    start_date: str
    end_date: str
    universe: list[str]
    initial_capital: float = 100_000.0
    slippage_bps: float = 3.0
    commission_bps: float = 0.5
    regime_weights: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "bull":       {"lgbm": 0.55, "sentiment": 0.25, "hmm": 0.20},
            "bear":       {"lgbm": 0.35, "sentiment": 0.15, "hmm": 0.50},
            "transition": {"lgbm": 0.45, "sentiment": 0.20, "hmm": 0.35},
        }
    )
    weight_clip_min: float = 0.05
    weight_clip_max: float = 0.60
    ic_rolling_window: int = 60
    sharpe_rolling_window: int = 120
    use_orthogonality: bool = True
    max_correlation: float = 0.65
    max_position: float = 0.08
    max_turnover: float = 0.20
    max_vol_ann: float = 0.30
    sector_cap: float = 0.45
    min_signal_strength: float = 0.20
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PlaygroundParams":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PlaygroundResult:
    equity_curve: pd.Series
    gross_equity_curve: pd.Series
    weights: pd.DataFrame
    stats: dict
    benchmark_curve: pd.Series
    council_contributions: pd.DataFrame
    params: PlaygroundParams
    snapshot_path: Optional[Path] = None
    elapsed_seconds: float = 0.0


# ===========================================================================
# Env-var patching for PortfolioConstructor
# ===========================================================================

_PORTFOLIO_ENV_KEYS = {
    "max_position": "MLCOUNCIL_MAX_POSITION_SIZE",
    "max_turnover": "MLCOUNCIL_MAX_TURNOVER",
    "max_vol_ann":  "MLCOUNCIL_MAX_VOL_ANN",
    "sector_cap":   "MLCOUNCIL_MAX_SECTOR_EXPOSURE",
    "min_signal_strength": "MLCOUNCIL_MIN_SIGNAL_STRENGTH",
    "slippage_bps": "MLCOUNCIL_SLIPPAGE_BPS",
    "commission_bps": "MLCOUNCIL_COMMISSION_BPS",
}


@contextmanager
def _patched_portfolio_env(params: PlaygroundParams):
    """Patch portfolio-related env vars for the duration of a run."""
    overrides = {
        "MLCOUNCIL_MAX_POSITION_SIZE": str(params.max_position),
        "MLCOUNCIL_MAX_TURNOVER":      str(params.max_turnover),
        "MLCOUNCIL_MAX_VOL_ANN":       str(params.max_vol_ann),
        "MLCOUNCIL_MAX_SECTOR_EXPOSURE": str(params.sector_cap),
        "MLCOUNCIL_MIN_SIGNAL_STRENGTH": str(params.min_signal_strength),
        "MLCOUNCIL_SLIPPAGE_BPS":  str(params.slippage_bps),
        "MLCOUNCIL_COMMISSION_BPS": str(params.commission_bps),
        # Force daily vol cap to be derived from annual, for predictability.
        "MLCOUNCIL_MAX_VOL_DAILY": "0",
    }
    original: dict[str, Optional[str]] = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            os.environ[k] = v
        yield
    finally:
        for k, prev in original.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


def _write_regime_yaml(params: PlaygroundParams) -> Path:
    cfg = {
        "regime_weights": params.regime_weights,
        "weight_clip": {"min": params.weight_clip_min, "max": params.weight_clip_max},
        "performance": {
            "min_history_days": 60,
            "ic_rolling_window": params.ic_rolling_window,
            "sharpe_rolling_window": params.sharpe_rolling_window,
        },
        "orthogonality": {
            "max_correlation": params.max_correlation,
            "correlation_window": 90,
            "auto_downweight": params.use_orthogonality,
            "downweight_factor": 0.5,
        },
    }
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="playground_regime_", delete=False, encoding="utf-8"
    )
    yaml.safe_dump(cfg, tmp)
    tmp.close()
    return Path(tmp.name)


# ===========================================================================
# Data loading
# ===========================================================================

def _load_universe_panel(
    tickers: Iterable[str],
    start: str,
    end: str,
    progress_cb: Optional[ProgressCb] = None,
) -> pd.DataFrame:
    """Load adjusted close prices for the universe; index=date, columns=ticker.

    Pads the start date by ``COV_LOOKBACK + MOMENTUM_WINDOW + REGIME_SLOW_MA``
    business days so signals/regime/covariance are warm by ``start``.
    """
    from backtest.runner import _load_ohlcv_polars

    pad_start = (
        pd.to_datetime(start) - pd.tseries.offsets.BusinessDay(REGIME_SLOW_MA + 10)
    ).strftime("%Y-%m-%d")
    tickers = list(dict.fromkeys(tickers))
    frames: dict[str, pd.Series] = {}
    n = len(tickers)
    for i, ticker in enumerate(tickers):
        if progress_cb:
            progress_cb(0.05 + 0.30 * (i / max(n, 1)), f"Loading OHLCV: {ticker}")
        df = _load_ohlcv_polars(ticker, pad_start, end)
        if df.is_empty():
            continue
        try:
            pdf = df.to_pandas()
        except Exception:
            continue
        if "valid_time" not in pdf.columns:
            continue
        pdf["valid_time"] = pd.to_datetime(pdf["valid_time"])
        price_col = "adj_close" if "adj_close" in pdf.columns else "close"
        s = (
            pdf.set_index("valid_time")[price_col]
            .astype(float)
            .sort_index()
            .rename(ticker)
        )
        frames[ticker] = s

    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames.values(), axis=1).sort_index()
    return panel


def _load_benchmark(start: str, end: str) -> pd.Series:
    from backtest.runner import _load_ohlcv_polars

    pad_start = (
        pd.to_datetime(start) - pd.tseries.offsets.BusinessDay(REGIME_SLOW_MA + 10)
    ).strftime("%Y-%m-%d")
    df = _load_ohlcv_polars(BENCHMARK_TICKER, pad_start, end)
    if df.is_empty():
        return pd.Series(dtype=float, name=BENCHMARK_TICKER)
    pdf = df.to_pandas()
    pdf["valid_time"] = pd.to_datetime(pdf["valid_time"])
    price_col = "adj_close" if "adj_close" in pdf.columns else "close"
    return pdf.set_index("valid_time")[price_col].astype(float).sort_index().rename(BENCHMARK_TICKER)


# ===========================================================================
# Proxy signal generation
# ===========================================================================

def _zscore_row(row: pd.Series) -> pd.Series:
    clean = row.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return pd.Series(0.0, index=row.index)
    mean = clean.mean()
    std = clean.std()
    if std < 1e-12:
        return pd.Series(0.0, index=row.index)
    return ((row - mean) / std).fillna(0.0)


def _build_proxy_signals(prices: pd.DataFrame, benchmark: pd.Series) -> dict[str, pd.DataFrame]:
    """Generate transparent proxy signals: lgbm, sentiment, hmm.

    These are not the production models; they are deterministic functions
    of price action chosen so the council mixing math is exercised end-to-end.
    """
    returns_1d = prices.pct_change(fill_method=None)

    # --- lgbm proxy: cross-sectional 20-day momentum z-score (long bias)
    mom_20 = prices.pct_change(MOMENTUM_WINDOW)
    lgbm = mom_20.apply(_zscore_row, axis=1)

    # --- hmm proxy: cross-sectional mean-reversion (negative short-term return)
    mr_5 = -returns_1d.rolling(5).sum()
    hmm = mr_5.apply(_zscore_row, axis=1)

    # --- sentiment proxy: ewm of 1-day return cross-sectional z-score
    # (persistent low-frequency signal that mimics a sentiment trend)
    ewm_r = returns_1d.ewm(span=10, adjust=False).mean()
    sentiment = ewm_r.apply(_zscore_row, axis=1)

    # Align all on the same index/columns and forward fill small gaps
    for df in (lgbm, hmm, sentiment):
        df.replace([np.inf, -np.inf], 0.0, inplace=True)

    return {"lgbm": lgbm.fillna(0.0), "sentiment": sentiment.fillna(0.0), "hmm": hmm.fillna(0.0)}


def _classify_regime(benchmark: pd.Series, d: pd.Timestamp) -> str:
    """Bull/bear/transition based on SPY fast/slow MA crossover."""
    history = benchmark.loc[:d]
    if len(history) < REGIME_SLOW_MA:
        return "transition"
    fast = history.iloc[-REGIME_FAST_MA:].mean()
    slow = history.iloc[-REGIME_SLOW_MA:].mean()
    spot = history.iloc[-1]
    if spot > fast > slow:
        return "bull"
    if spot < fast < slow:
        return "bear"
    return "transition"


# ===========================================================================
# Main orchestrator
# ===========================================================================

def run_playground_backtest(
    params: PlaygroundParams,
    progress_cb: Optional[ProgressCb] = None,
) -> PlaygroundResult:
    """Run a council + portfolio backtest with user-supplied parameters."""
    from council.aggregator import CouncilAggregator
    from council.portfolio import PortfolioConstructor
    from council.transaction_costs import TransactionCostModel
    from backtest.simulator import simulate_weight_backtest

    t0 = datetime.utcnow()

    def _pp(p: float, msg: str) -> None:
        if progress_cb:
            progress_cb(p, msg)

    _pp(0.02, "Loading universe data…")

    if not params.universe:
        raise ValueError("Universe must contain at least one ticker.")
    if pd.to_datetime(params.start_date) >= pd.to_datetime(params.end_date):
        raise ValueError("start_date must be strictly before end_date.")

    prices = _load_universe_panel(
        params.universe, params.start_date, params.end_date, progress_cb=progress_cb
    )
    if prices.empty:
        raise RuntimeError(
            "No OHLCV data could be loaded for the requested window. "
            "Check connectivity to yfinance or pre-populate data/raw/ohlcv/."
        )

    benchmark = _load_benchmark(params.start_date, params.end_date)
    if benchmark.empty:
        benchmark = prices.mean(axis=1)  # crude fallback

    _pp(0.40, "Building proxy model signals…")
    signals_panel = _build_proxy_signals(prices, benchmark)

    # Restrict to the user's [start, end] window for the rebalance loop
    window = (prices.index >= pd.to_datetime(params.start_date)) & (
        prices.index <= pd.to_datetime(params.end_date)
    )
    backtest_dates = prices.index[window]
    if len(backtest_dates) < REBALANCE_EVERY + 1:
        raise RuntimeError("Backtest window is too short — pick at least ~2 months.")

    rebalance_dates = backtest_dates[::REBALANCE_EVERY]

    _pp(0.45, "Configuring council and portfolio…")
    regime_yaml = _write_regime_yaml(params)

    weights_rows: dict[pd.Timestamp, pd.Series] = {}
    contributions_rows: list[dict] = []
    current_weights = pd.Series(0.0, index=prices.columns, dtype=float)

    with _patched_portfolio_env(params):
        aggregator = CouncilAggregator(
            config_path=str(regime_yaml),
            use_orthogonality=params.use_orthogonality,
        )
        portfolio = PortfolioConstructor()

        total = len(rebalance_dates)
        for i, d in enumerate(rebalance_dates):
            frac = 0.50 + 0.45 * (i / max(total, 1))
            _pp(frac, f"Rebalance {i + 1}/{total} — {d.strftime('%Y-%m-%d')}")

            # 1. Per-model signals at date d
            sig_today: dict[str, pd.Series] = {}
            for model_name, df in signals_panel.items():
                if d not in df.index:
                    continue
                row = df.loc[d].dropna()
                if not row.empty:
                    sig_today[model_name] = row

            if not sig_today:
                continue

            regime = _classify_regime(benchmark, d)

            try:
                council_signal = aggregator.aggregate(sig_today, regime=regime, date=d.date())
            except Exception as exc:  # noqa: BLE001
                # Skip pathological dates rather than abort the whole run.
                continue

            # 2. Covariance from prior COV_LOOKBACK trading days
            prior = prices.loc[:d].iloc[-(COV_LOOKBACK + 1):].pct_change().dropna(how="all")
            if prior.empty or prior.shape[0] < 5:
                continue
            cov = prior.cov()

            # 3. Optimise weights
            multipliers = pd.Series(1.0, index=council_signal.index)
            try:
                target_w = portfolio.optimize(
                    alpha_signals=council_signal,
                    position_multipliers=multipliers,
                    current_weights=current_weights.reindex(council_signal.index).fillna(0.0),
                    returns_covariance=cov,
                    portfolio_value=params.initial_capital,
                    days_since_last_rebalance=REBALANCE_EVERY,
                )
            except Exception:
                continue

            target_w = target_w.reindex(prices.columns).fillna(0.0)
            weights_rows[d] = target_w
            current_weights = target_w

            log_entry = aggregator._weights_log.get(d.date(), {})  # noqa: SLF001
            contributions = log_entry.get("contributions", {})
            row = {"date": d, "regime": regime}
            row.update({f"contrib_{k}": float(v) for k, v in contributions.items()})
            cfg_weights = log_entry.get("weights", {})
            row.update({f"weight_{k}": float(v) for k, v in cfg_weights.items()})
            contributions_rows.append(row)

    if not weights_rows:
        raise RuntimeError(
            "No rebalances were produced — the portfolio optimiser rejected every date. "
            "Try relaxing constraints (e.g. lower min_signal_strength, larger universe)."
        )

    weights_df = pd.DataFrame.from_dict(weights_rows, orient="index").sort_index()
    weights_df = weights_df.reindex(backtest_dates).ffill().fillna(0.0)

    forward_returns = prices.pct_change(fill_method=None).shift(-1).loc[backtest_dates]
    forward_returns = forward_returns.reindex(columns=weights_df.columns).fillna(0.0)

    _pp(0.95, "Simulating equity curve…")
    cost_model = TransactionCostModel(
        commission_bps=params.commission_bps,
        slippage_bps=params.slippage_bps,
    )
    sim = simulate_weight_backtest(
        weights=weights_df,
        forward_returns=forward_returns,
        initial_capital=params.initial_capital,
        cost_model=cost_model,
    )

    contributions_df = (
        pd.DataFrame(contributions_rows).set_index("date").sort_index()
        if contributions_rows else pd.DataFrame()
    )

    bench_window = benchmark.reindex(backtest_dates).dropna()

    snapshot_path = _persist_snapshot(
        params=params,
        equity=sim.equity_curve,
        gross_equity=sim.gross_equity_curve,
        weights=weights_df,
        stats=sim.stats,
        contributions=contributions_df,
        benchmark=bench_window,
    )

    try:
        regime_yaml.unlink(missing_ok=True)
    except Exception:
        pass

    elapsed = (datetime.utcnow() - t0).total_seconds()
    _pp(1.0, f"Done in {elapsed:.1f}s")

    return PlaygroundResult(
        equity_curve=sim.equity_curve,
        gross_equity_curve=sim.gross_equity_curve,
        weights=weights_df,
        stats=sim.stats,
        benchmark_curve=bench_window,
        council_contributions=contributions_df,
        params=params,
        snapshot_path=snapshot_path,
        elapsed_seconds=elapsed,
    )


# ===========================================================================
# Snapshot persistence
# ===========================================================================

def _persist_snapshot(
    *,
    params: PlaygroundParams,
    equity: pd.Series,
    gross_equity: pd.Series,
    weights: pd.DataFrame,
    stats: dict,
    contributions: pd.DataFrame,
    benchmark: pd.Series,
    base_dir: Optional[Path] = None,
) -> Path:
    base = base_dir or RESULTS_DIR
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out = base / stamp
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "params.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(params.to_dict(), f, sort_keys=False)

    equity.rename("equity").to_frame().to_parquet(out / "equity_curve.parquet")
    if not gross_equity.empty:
        gross_equity.rename("gross_equity").to_frame().to_parquet(out / "gross_equity.parquet")
    weights.to_parquet(out / "weights.parquet")
    if not contributions.empty:
        contributions.to_parquet(out / "contributions.parquet")
    if not benchmark.empty:
        benchmark.rename("benchmark").to_frame().to_parquet(out / "benchmark.parquet")

    with open(out / "stats.json", "w", encoding="utf-8") as f:
        json.dump({k: _jsonify(v) for k, v in stats.items()}, f, indent=2)

    return out


def _jsonify(v):
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    if isinstance(v, float) and (np.isinf(v) or np.isnan(v)):
        return None
    return v


# ===========================================================================
# Snapshot browsing helpers
# ===========================================================================

def list_snapshots(base_dir: Optional[Path] = None) -> pd.DataFrame:
    base = base_dir or RESULTS_DIR
    if not base.exists():
        return pd.DataFrame(
            columns=["timestamp", "start_date", "end_date", "n_tickers",
                     "sharpe", "max_drawdown", "cagr", "final_equity", "path"]
        )

    rows = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        stats_path = child / "stats.json"
        params_path = child / "params.yaml"
        try:
            stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
            params = yaml.safe_load(params_path.read_text()) if params_path.exists() else {}
        except Exception:
            continue
        rows.append({
            "timestamp": child.name,
            "start_date": params.get("start_date", ""),
            "end_date": params.get("end_date", ""),
            "n_tickers": len(params.get("universe", []) or []),
            "sharpe": stats.get("sharpe"),
            "max_drawdown": stats.get("max_drawdown"),
            "cagr": stats.get("cagr"),
            "final_equity": stats.get("final_equity"),
            "note": params.get("note", ""),
            "path": str(child),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    return df


def load_snapshot_equity(snapshot_dir: Path) -> pd.Series:
    p = Path(snapshot_dir) / "equity_curve.parquet"
    if not p.exists():
        return pd.Series(dtype=float)
    return pd.read_parquet(p)["equity"]


def load_snapshot_params(snapshot_dir: Path) -> dict:
    p = Path(snapshot_dir) / "params.yaml"
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


# ===========================================================================
# Universe helpers (for the UI)
# ===========================================================================

def load_available_universe() -> list[str]:
    """Return the flat list of equity tickers from config/universe.yaml."""
    cfg_path = _ROOT / "config" / "universe.yaml"
    try:
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    except FileNotFoundError:
        return []
    universe = cfg.get("universe", {}) or {}
    tickers: list[str] = []
    for bucket in ("large_cap", "mid_cap", "small_cap"):
        for t in universe.get(bucket, []) or []:
            tickers.append(str(t).upper())
    # Dedup preserving order
    seen: set[str] = set()
    out: list[str] = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out
