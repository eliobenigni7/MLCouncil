"""One-year rolling window backtest — iterates until Sharpe > 1."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest.validation import (
    build_purged_walk_forward_splits,
    run_walk_forward_analysis,
)
from council.aggregator import CouncilAggregator
from council.transaction_costs import TransactionCostModel
from backtest.simulator import simulate_weight_backtest
from data.features.alpha158 import build_macro_context, compute_alpha158
from data.features.target import compute_targets, training_rank_column
from models.regime import RegimeModel
from models.technical import TechnicalModel

RESULTS_DIR = ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_universe() -> set[str]:
    with open(ROOT / "config" / "universe.yaml") as f:
        cfg = yaml.safe_load(f) or {}
    tickers: set[str] = set()
    for bucket in ("large_cap", "mid_cap"):
        tickers.update(str(t).upper() for t in cfg.get("universe", {}).get(bucket, []) or [])
    return tickers


def load_ohlcv(allowed: set[str]) -> pl.DataFrame:
    frames = []
    raw_dir = ROOT / "data" / "raw" / "ohlcv"
    for tdir in sorted(raw_dir.iterdir()):
        if not tdir.is_dir():
            continue
        t = tdir.name.upper()
        if t not in allowed:
            continue
        tf = []
        for pq in sorted(tdir.glob("*.parquet")):
            try:
                df = pl.read_parquet(pq)
            except Exception:
                continue
            if "symbol" in df.columns:
                df = df.drop("symbol")
            if "ticker" not in df.columns:
                df = df.with_columns(pl.lit(t).alias("ticker"))
            if "transaction_time" in df.columns:
                df = df.drop("transaction_time")
            if "valid_time" in df.columns:
                vt = df["valid_time"]
                if vt.dtype == pl.Datetime:
                    df = df.with_columns(vt.dt.replace_time_zone("UTC").cast(pl.Date))
                elif vt.dtype != pl.Date:
                    df = df.with_columns(vt.cast(pl.Date))
            keep = [c for c in ["ticker", "valid_time", "open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
            if keep:
                tf.append(df.select(keep))
        if tf:
            frames.append(
                pl.concat(tf, how="vertical_relaxed")
                .unique(subset=["ticker", "valid_time"], keep="last")
                .sort(["ticker", "valid_time"])
            )
    return pl.concat(frames, how="vertical_relaxed").unique(["ticker", "valid_time"]).sort(["ticker", "valid_time"])


def macro_path(name: str) -> str | None:
    p = ROOT / "data" / "raw" / "macro" / f"{name}.parquet"
    return str(p) if p.exists() else None


def run_one_year_backtest(
    year_start: str,
    year_end: str,
    train_window_months: int = 6,
    force_linear: bool = True,
    rebalance_every: int = 5,
    vol_daily: float = 0.0095,
    max_pos: float = 0.08,
    max_turnover_env: float = 0.20,
) -> dict:
    """Run a focused backtest on a ~1-year window. Returns stats dict."""

    # Load data
    universe = load_universe()
    ohlcv = load_ohlcv(universe)
    macro = build_macro_context(
        vix_path=macro_path("vix"),
        treasuries_path=macro_path("treasuries"),
        sp500_path=macro_path("sp500"),
    )

    # Filter to window
    ohlcv = ohlcv.filter(
        (pl.col("valid_time") >= pl.lit(pd.Timestamp(year_start).date()))
        & (pl.col("valid_time") <= pl.lit(pd.Timestamp(year_end).date()))
    )
    if ohlcv.is_empty():
        return {"error": f"No data for {year_start}–{year_end}"}

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
    print(f"  Universe: {len(universe)} tickers, {len(all_dates)} trading days in window")

    if len(all_dates) < 60:
        return {"error": f"Only {len(all_dates)} days, need ≥60"}

    # Train/test split within the window
    train_days = train_window_months * 21  # ~21 trading days/month
    test_window = max(rebalance_every, min(63, max(10, len(all_dates) // 4)))

    splits = build_purged_walk_forward_splits(
        all_dates,
        train_window=min(train_days, len(all_dates) - test_window - 1),
        test_window=test_window,
        step=test_window,
        purge_period=1,
        embargo_period=1,
    )
    if not splits:
        return {"error": "Could not build walk-forward splits"}

    print(f"  Windows: {len(splits)}, train ~{train_days}d, test ~{test_window}d")

    # Force linear mode if requested (bypass MoE — test our new regime weights)
    if force_linear:
        import os
        os.environ["MLCOUNCIL_AGGREGATOR_MODE"] = "linear"
        os.environ["MLCOUNCIL_MAX_VOL_DAILY"] = str(vol_daily)
        os.environ["MLCOUNCIL_MAX_TURNOVER"] = str(max_turnover_env)
        os.environ["MLCOUNCIL_MAX_POSITION_SIZE"] = str(max_pos)
        os.environ["MLCOUNCIL_MAX_SECTOR_EXPOSURE"] = "0.45"
        os.environ["MLCOUNCIL_POSITION_SIZING"] = "conformal"

    agg = CouncilAggregator()
    lgbm_history_rows: list[pd.Series] = []
    signal_rows: list[pd.Series] = []
    weight_rows: list[pd.Series] = []
    last_target_w: pd.Series | None = None
    rebalance_count = 0

    t0 = time.time()

    for wid, split in enumerate(splits, start=1):
        train_start, train_end = split.train_start.date(), split.train_end.date()
        test_start, test_end = split.test_start.date(), split.test_end.date()

        ft = features.filter(
            (pl.col("valid_time") >= pl.lit(train_start)) & (pl.col("valid_time") <= pl.lit(train_end))
        )
        fe = features.filter(
            (pl.col("valid_time") >= pl.lit(test_start)) & (pl.col("valid_time") <= pl.lit(test_end))
        )
        if ft.is_empty() or fe.is_empty():
            continue

        rank_col = training_rank_column(5)
        tg = (
            targets
            .filter((pl.col("valid_time") >= pl.lit(train_start)) & (pl.col("valid_time") <= pl.lit(train_end)))
            .select(["ticker", "valid_time", rank_col])
            .to_pandas()
        )
        tg["valid_time"] = pd.to_datetime(tg["valid_time"]).dt.date
        tg = tg.set_index(["ticker", "valid_time"])[rank_col].dropna()

        lgbm = TechnicalModel(config_path=str(ROOT / "config" / "models.yaml"))
        lgbm.fit(ft, tg)

        mt = macro.filter(pl.col("valid_time") <= pl.lit(train_end)) if not macro.is_empty() else macro
        hmm = RegimeModel()
        if not mt.is_empty():
            hmm.fit(mt)

        # Conformal sizer
        from council.conformal import ConformalPositionSizer
        sizer = ConformalPositionSizer()
        tg_pivot = targets.filter(pl.col("valid_time") <= pl.lit(train_end)).to_pandas()
        tg_pivot["valid_time"] = pd.to_datetime(tg_pivot["valid_time"]).dt.date
        calib = tg_pivot.set_index(["ticker", "valid_time"])[rank_col].dropna()
        ft_pd = ft.to_pandas().set_index(["ticker", "valid_time"])
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
            sig = lgbm.predict(day_feat)
            lgbm_history_rows.append(pd.Series(sig.to_dict(), name=ts))
            hist_df = pd.DataFrame(lgbm_history_rows).sort_index()
            if not hist_df.empty:
                hist_df.index = pd.to_datetime(hist_df.index)

            rets = forward_returns.loc[:ts]
            if not hist_df.empty and not rets.empty:
                agg.update_performance({"lgbm": hist_df}, rets, date=ts.date())

            if macro.is_empty():
                regime = "transition"
            else:
                mtoday = macro.filter(pl.col("valid_time") <= pl.lit(ts))
                regime = hmm.predict_regime(mtoday) if not mtoday.is_empty() else "transition"

            zeros = pd.Series(0.0, index=sig.index)
            council = agg.aggregate({"lgbm": sig, "hmm": zeros}, regime=regime, date=ts.date())
            signal_rows.append(council.rename(ts))

            if last_target_w is None or rebalance_count % rebalance_every == 0:
                # Build weights via optimizer
                from scripts import run_pipeline as rp
                tw = rp.step_portfolio(
                    council, sizer, fcols_list, fe, ohlcv, ts.date(),
                    current_weights=last_target_w,
                    save_orders=False, emit_report=False,
                )
                last_target_w = tw.copy() if tw is not None else last_target_w
            else:
                tw = last_target_w.reindex(council.index).fillna(0.0)

            if tw is not None:
                weight_rows.append(tw.rename(ts))
            rebalance_count += 1

        elapsed = time.time() - t0
        if wid % 5 == 0:
            print(f"    [window {wid}/{len(splits)}] {elapsed:.0f}s elapsed")

    if not weight_rows:
        return {"error": "No weights produced"}

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
        test_window=test_window,
        step=test_window,
        purge_period=1,
        embargo_period=1,
    )

    sim = simulate_weight_backtest(
        weights=weights_df,
        forward_returns=aligned_returns,
        initial_capital=100_000.0,
        cost_model=TransactionCostModel.from_env(),
    )

    total_elapsed = time.time() - t0
    stats = {
        **sim.stats,
        "oos_sharpe": wf["summary"].get("oos_sharpe", 0),
        "oos_max_drawdown": wf["summary"].get("oos_max_drawdown", 0),
        "pbo": wf["summary"].get("pbo", 0),
        "windows": len(splits),
        "elapsed_seconds": round(total_elapsed, 1),
        "tickers": len(universe),
        "train_months": train_window_months,
        "rebalance_every": rebalance_every,
    }
    return stats


if __name__ == "__main__":
    # Cycle through yearly windows with production-aligned risk limits (~15% turnover).
    window_configs = [
        # (year_start, year_end, train_months, rebalance, vol_daily, max_pos, max_turnover)
        ("2025-05-01", "2026-05-20", 6, 3, 0.025, 0.15, 0.15),
        ("2024-01-01", "2025-01-01", 6, 3, 0.025, 0.15, 0.15),
        ("2023-01-01", "2024-01-01", 6, 3, 0.025, 0.15, 0.15),
        ("2022-01-01", "2023-01-01", 6, 3, 0.025, 0.15, 0.15),
        ("2021-01-01", "2022-01-01", 6, 3, 0.025, 0.15, 0.15),
        ("2020-01-01", "2021-01-01", 6, 3, 0.025, 0.15, 0.15),
    ]

    for start, end, train_mo, reb, vol_d, pos, tover in window_configs:
        print(f"\n{'='*60}")
        print(f"Backtest: {start} → {end} (train={train_mo}mo, reb={reb}d, vol_daily={vol_d}, pos={pos}, tover={tover})")
        print('=' * 60)
        t0 = time.time()
        result = run_one_year_backtest(
            start, end,
            train_window_months=train_mo,
            rebalance_every=reb,
            vol_daily=vol_d,
            max_pos=pos,
            max_turnover_env=tover,
        )
        elapsed = time.time() - t0
        print(f"Time: {elapsed:.0f}s")
        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Sharpe:          {result.get('sharpe', 'N/A'):.4f}")
            print(f"  Gross Sharpe:    {result.get('gross_sharpe', 'N/A'):.4f}")
            print(f"  CAGR:            {result.get('cagr', 0)*100:.2f}%")
            print(f"  Gross CAGR:      {result.get('gross_cagr', 0)*100:.2f}%")
            print(f"  Max DD:          {result.get('max_drawdown', 0)*100:.2f}%")
            print(f"  OOS Sharpe:      {result.get('oos_sharpe', 0):.4f}")
            print(f"  Turnover:        {result.get('turnover', 0)*100:.2f}%")
            print(f"  PBO:             {result.get('pbo', 0)*100:.1f}%")
            print(f"  Windows:         {result.get('windows', 0)}")
            sharpe = result.get("sharpe", 0)
            if sharpe > 1.0:
                print(f"\n  >>> SHARPE {sharpe:.4f} > 1.0 ✅ TARGET RAGGIUNTO <<<")
            else:
                print(f"\n  >>> Sharpe {sharpe:.4f} — serve > 1.0 ❌ <<<")
