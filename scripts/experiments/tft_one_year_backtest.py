#!/usr/bin/env python3
"""TFT challenger: one-year backtest 2021-2025 usando TFT invece di LightGBM."""
from __future__ import annotations

import json, sys, time, os
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest.validation import build_purged_walk_forward_splits, run_walk_forward_analysis
from council.aggregation.aggregator import CouncilAggregator
from council.transaction_costs import TransactionCostModel
from backtest.simulator import simulate_weight_backtest
from data.features.alpha158 import build_macro_context, compute_alpha158
from data.features.target import compute_targets, training_rank_column
from models.regime import RegimeModel
from models.tft import TemporalFusionAlpha
from scripts.one_year_backtest import _compute_proxy_sentiment, load_universe, load_ohlcv, macro_path

# Force linear mode (same as one_year_backtest.py)
os.environ["MLCOUNCIL_AGGREGATOR_MODE"] = "linear"
os.environ["MLCOUNCIL_MAX_VOL_DAILY"] = "0.025"
os.environ["MLCOUNCIL_MAX_TURNOVER"] = "0.15"
os.environ["MLCOUNCIL_MAX_POSITION_SIZE"] = "0.15"
os.environ["MLCOUNCIL_MAX_SECTOR_EXPOSURE"] = "0.45"
os.environ["MLCOUNCIL_POSITION_SIZING"] = "conformal"

RESULTS_DIR = ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("TFT CHALLENGER: 2021-01-01 → 2026-01-01")
print("=" * 60)

# Load data
universe = load_universe()
ohlcv = load_ohlcv(universe)
macro = build_macro_context(
    vix_path=macro_path("vix"),
    treasuries_path=macro_path("treasuries"),
    sp500_path=macro_path("sp500"),
)

ohlcv = ohlcv.filter(
    (pl.col("valid_time") >= pl.lit(pd.Timestamp("2021-01-01").date()))
    & (pl.col("valid_time") <= pl.lit(pd.Timestamp("2026-01-01").date()))
)
features = compute_alpha158(ohlcv, macro_df=macro if not macro.is_empty() else None)
fcols = [c for c in features.columns if c not in {"ticker", "valid_time"}]
if fcols:
    features = features.with_columns(
        [pl.when(pl.col(c).is_infinite() | pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c) for c in fcols]
    ).with_columns([pl.col(c).fill_null(0.0).alias(c) for c in fcols])

targets = compute_targets(ohlcv, horizons=[1, 5, 10], risk_adjusted=True)

ret_pd = targets.select(["ticker", "valid_time", "ret_fwd_1d"]).to_pandas()
ret_pd["valid_time"] = pd.to_datetime(ret_pd["valid_time"]).dt.date
forward_returns = ret_pd.pivot(index="valid_time", columns="ticker", values="ret_fwd_1d").sort_index()
forward_returns.index = pd.to_datetime(forward_returns.index)

all_dates = sorted(features["valid_time"].unique().to_list())
print(f"  Universe: {len(universe)} tickers, {len(all_dates)} trading days")

# Walk-forward setup
train_days = 6 * 21
test_window = max(10, min(21, len(all_dates) // 12))
splits = build_purged_walk_forward_splits(
    all_dates, train_window=min(train_days, len(all_dates) - test_window - 1),
    test_window=test_window, step=test_window, purge_period=1, embargo_period=1,
)
print(f"  Windows: {len(splits)}")

# --- Use TFT instead of LightGBM ---
tft = TemporalFusionAlpha()
# Train TFT on initial window
train_start, train_end = splits[0].train_start.date(), splits[0].train_end.date()
ft = features.filter(
    (pl.col("valid_time") >= pl.lit(train_start)) & (pl.col("valid_time") <= pl.lit(train_end))
)
rank_col = training_rank_column(5)
tg = (
    targets
    .filter((pl.col("valid_time") >= pl.lit(train_start)) & (pl.col("valid_time") <= pl.lit(train_end)))
    .select(["ticker", "valid_time", rank_col])
    .to_pandas()
)
tg["valid_time"] = pd.to_datetime(tg["valid_time"]).dt.date
tg = tg.set_index(["ticker", "valid_time"])[rank_col].dropna()
tft.fit(ft, tg)
print(f"  TFT trained on {train_start} → {train_end} ({len(tg)} targets)")

agg = CouncilAggregator()
signal_rows = []
weight_rows = []
last_target_w = None
rebalance_count = 0

t0 = time.time()

for wid, split in enumerate(splits, start=1):
    train_start_d, train_end_d = split.train_start.date(), split.train_end.date()
    test_start_d, test_end_d = split.test_start.date(), split.test_end.date()

    ft_win = features.filter(
        (pl.col("valid_time") >= pl.lit(train_start_d)) & (pl.col("valid_time") <= pl.lit(train_end_d))
    )
    fe = features.filter(
        (pl.col("valid_time") >= pl.lit(test_start_d)) & (pl.col("valid_time") <= pl.lit(test_end_d))
    )
    if ft_win.is_empty() or fe.is_empty():
        continue

    # Retrain TFT each window
    tg_win = (
        targets
        .filter((pl.col("valid_time") >= pl.lit(train_start_d)) & (pl.col("valid_time") <= pl.lit(train_end_d)))
        .select(["ticker", "valid_time", rank_col])
        .to_pandas()
    )
    tg_win["valid_time"] = pd.to_datetime(tg_win["valid_time"]).dt.date
    tg_win = tg_win.set_index(["ticker", "valid_time"])[rank_col].dropna()
    try:
        tft.fit(ft_win, tg_win)
    except Exception as e:
        print(f"  [window {wid}] TFT fit failed: {e}, reusing previous model")

    mt = macro.filter(pl.col("valid_time") <= pl.lit(train_end_d)) if not macro.is_empty() else macro
    hmm = RegimeModel()
    if not mt.is_empty():
        hmm.fit(mt)

    from council.sizing.conformal import ConformalPositionSizer
    sizer = ConformalPositionSizer()
    tg_pivot = targets.filter(pl.col("valid_time") <= pl.lit(train_end_d)).to_pandas()
    tg_pivot["valid_time"] = pd.to_datetime(tg_pivot["valid_time"]).dt.date
    calib = tg_pivot.set_index(["ticker", "valid_time"])[rank_col].dropna()
    ft_pd = ft_win.to_pandas().set_index(["ticker", "valid_time"])
    fcols_list = [c for c in ft_pd.columns if c not in {"ticker", "valid_time"} and c in fcols]
    X_calib = ft_pd[fcols_list].values
    y_calib = calib.reindex(ft_pd.index).dropna().values
    if len(X_calib) > 10 and len(y_calib) > 10:
        sizer.fit(X_calib[:min(2000, len(X_calib))], y_calib[:min(2000, len(y_calib))])

    test_dates = sorted(fe["valid_time"].unique().to_list())
    for d in test_dates:
        if pd.Timestamp(d).dayofweek >= 5:
            continue
        day_feat = fe.filter(pl.col("valid_time") == d)
        if day_feat.is_empty():
            continue
        ts = pd.Timestamp(d)

        # TFT predict
        try:
            sig = tft.predict(day_feat)
        except Exception as e:
            print(f"  TFT predict failed {d}: {e}")
            continue

        if sig.empty or len(sig) == 0:
            # TFT returned no signals — skip this day
            if last_target_w is not None:
                weight_rows.append(last_target_w.rename(ts))
            continue

        if macro.is_empty():
            regime = "transition"
            regime_embedding = None
        else:
            mtoday = macro.filter(pl.col("valid_time") <= pl.lit(ts))
            regime = hmm.predict_regime(mtoday) if not mtoday.is_empty() else "transition"
            try:
                probs = hmm.predict_probabilities(mtoday)
                regime_embedding = np.array([probs.get("bull", 0.0), probs.get("bear", 0.0), probs.get("transition", 0.0)], dtype=float)
            except Exception:
                regime_embedding = None

        zeros = pd.Series(0.0, index=sig.index)
        sentiment_signal = _compute_proxy_sentiment(ohlcv, pl.lit(d), sig.index.tolist())
        council = agg.aggregate(
            {"lgbm": sig, "sentiment": sentiment_signal, "hmm": zeros},
            regime=regime, regime_embedding=regime_embedding, date=ts.date(),
        )
        signal_rows.append(council.rename(ts))

        if last_target_w is None or rebalance_count % 3 == 0:
            from scripts import run_pipeline as rp
            try:
                tw = rp.step_portfolio(
                    council, sizer, fcols_list, fe, ohlcv, ts.date(),
                    current_weights=last_target_w, save_orders=False, emit_report=False,
                )
            except Exception as e:
                print(f"  [portfolio fallback] {d}: {e}")
                tw = last_target_w
            last_target_w = tw.copy() if tw is not None else last_target_w
        else:
            tw = last_target_w.reindex(council.index).fillna(0.0)
        if tw is not None:
            weight_rows.append(tw.rename(ts))
        rebalance_count += 1

    elapsed = time.time() - t0
    if wid % 5 == 0:
        print(f"  [window {wid}/{len(splits)}] {elapsed:.0f}s elapsed")

# Build results
weights_df = pd.DataFrame(weight_rows).sort_index().fillna(0.0)
weights_df.index = pd.to_datetime(weights_df.index)
if weights_df.index.has_duplicates:
    weights_df = weights_df.groupby(level=0).mean()

signals_df = pd.DataFrame(signal_rows).sort_index().fillna(0.0)
signals_df.index = pd.to_datetime(signals_df.index)
if signals_df.index.has_duplicates:
    signals_df = signals_df.groupby(level=0).mean()
signals_df = signals_df.reindex(weights_df.index).fillna(0.0)

aligned_returns = forward_returns.loc[weights_df.index.intersection(forward_returns.index)]
wf = run_walk_forward_analysis(
    signals=signals_df.loc[aligned_returns.index],
    forward_returns=aligned_returns,
    train_window=min(train_days, len(all_dates) // 2),
    test_window=test_window, step=test_window,
    purge_period=1, embargo_period=1,
)
sim = simulate_weight_backtest(
    weights=weights_df, forward_returns=aligned_returns,
    initial_capital=100_000.0, cost_model=TransactionCostModel.from_env(),
)

result = {
    "model": "TFT",
    "window": "2021-2025",
    **sim.stats,
    "oos_sharpe": wf["summary"].get("oos_sharpe", 0),
    "oos_max_drawdown": wf["summary"].get("oos_max_drawdown", 0),
    "pbo": wf["summary"].get("pbo", 0),
    "windows": len(splits),
    "elapsed_seconds": round(time.time() - t0, 1),
}

out_path = RESULTS_DIR / "tft_challenger_2021_2025.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nDone. Results → {out_path}")
print(f"  Sharpe: {result.get('sharpe', 0):.4f}")
print(f"  Gross Sharpe: {result.get('gross_sharpe', 0):.4f}")
print(f"  CAGR: {result.get('cagr', 0)*100:.2f}%")
print(f"  Max DD: {result.get('max_drawdown', 0)*100:.2f}%")
print(f"  OOS Sharpe: {result.get('oos_sharpe', 0):.4f}")
print(f"  PBO: {result.get('pbo', 0)*100:.1f}%")
print(f"  Windows: {result.get('windows', 0)}")
print(f"  {result['elapsed_seconds']}s total")
