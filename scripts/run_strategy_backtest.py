"""Run a coherent end-to-end strategy backtest for MLCouncil.

This script uses the actual daily strategy components:
- LightGBM technical model trained on the pre-cutoff window
- HMM regime detector trained on the pre-cutoff macro window
- CouncilAggregator regime weights
- Daily portfolio construction from the live-style step_portfolio logic
- Deterministic portfolio simulation with shared transaction cost model

The goal is to generate dashboard artifacts that reflect the real strategy
logic, rather than a synthetic walk-forward proxy.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest.simulator import simulate_weight_backtest
from backtest.validation import build_purged_walk_forward_splits, run_walk_forward_analysis
from council.aggregator import CouncilAggregator
from council.transaction_costs import TransactionCostModel
from data.features.alpha158 import build_macro_context, compute_alpha158
from data.features.target import compute_targets
from models.regime import RegimeModel
from models.technical import TechnicalModel
from scripts import run_pipeline as rp

RESULTS_DIR = ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REBALANCE_EVERY = 3
EQUITY_UNIVERSE = {"AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"}


def _load_configured_universe() -> set[str]:
    config_path = ROOT / "config" / "universe.yaml"
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return set(EQUITY_UNIVERSE)

    tickers: set[str] = set()
    universe = cfg.get("universe", {})
    if isinstance(universe, dict):
        for bucket in ("large_cap", "mid_cap"):
            tickers.update(str(t).upper() for t in universe.get(bucket, []) or [])
    crypto = cfg.get("crypto_universe", {})
    if isinstance(crypto, dict):
        tickers.update(str(t).upper() for t in crypto.get("large_cap", []) or [])
    return tickers or set(EQUITY_UNIVERSE)


def _load_ohlcv() -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    raw_dir = ROOT / "data" / "raw" / "ohlcv"
    if not raw_dir.exists():
        raise FileNotFoundError(f"OHLCV directory not found: {raw_dir}")

    allowed = _load_configured_universe()
    for ticker_dir in sorted(raw_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name.upper()
        if ticker not in allowed:
            continue
        for pq in sorted(ticker_dir.glob("*.parquet")):
            try:
                df = pl.read_parquet(pq)
            except Exception as exc:
                print(f"[warn] skipping {pq.name}: {exc}")
                continue

            if "symbol" in df.columns:
                df = df.drop("symbol")
            if "ticker" not in df.columns:
                df = df.with_columns(pl.lit(ticker).alias("ticker"))
            if "transaction_time" in df.columns:
                df = df.drop("transaction_time")
            if "valid_time" in df.columns:
                if df["valid_time"].dtype == pl.Datetime:
                    df = df.with_columns(pl.col("valid_time").dt.replace_time_zone("UTC").cast(pl.Date))
                elif df["valid_time"].dtype != pl.Date:
                    df = df.with_columns(pl.col("valid_time").cast(pl.Date))

            frames.append(df)

    if not frames:
        raise SystemExit("No OHLCV data found")

    return pl.concat(frames).unique(["ticker", "valid_time"]).sort(["ticker", "valid_time"])


def _macro_path(name: str) -> str | None:
    p = ROOT / "data" / "raw" / "macro" / f"{name}.parquet"
    return str(p) if p.exists() else None


def _build_macro_today(macro: pl.DataFrame, d: pd.Timestamp) -> pl.DataFrame:
    if macro.is_empty():
        return macro
    return macro.filter(pl.col("valid_time") <= pl.lit(pd.Timestamp(d).date()))



def main() -> None:
    print("=" * 72)
    print("MLCouncil — coherent strategy backtest")
    print("=" * 72)

    print("[1/7] Load data")
    ohlcv = _load_ohlcv()
    macro = build_macro_context(
        vix_path=_macro_path("vix"),
        treasuries_path=_macro_path("treasuries"),
        sp500_path=_macro_path("sp500"),
    )
    features = compute_alpha158(ohlcv, macro_df=macro if not macro.is_empty() else None)
    feature_cols = [c for c in features.columns if c not in {"ticker", "valid_time"}]
    if feature_cols:
        features = features.with_columns(
            [
                pl.when(pl.col(c).is_infinite() | pl.col(c).is_nan())
                .then(None)
                .otherwise(pl.col(c))
                .alias(c)
                for c in feature_cols
            ]
        ).with_columns([pl.col(c).fill_null(0.0).alias(c) for c in feature_cols])
    targets = compute_targets(ohlcv, horizons=[1], risk_adjusted=False)

    ret_pd = targets.select(["ticker", "valid_time", "ret_fwd_1d"]).to_pandas()
    ret_pd["valid_time"] = pd.to_datetime(ret_pd["valid_time"]).dt.date
    forward_returns = ret_pd.pivot(index="valid_time", columns="ticker", values="ret_fwd_1d").sort_index()
    forward_returns.index = pd.to_datetime(forward_returns.index)

    all_dates = sorted(features["valid_time"].unique().to_list())
    if len(all_dates) < 6:
        raise SystemExit("Not enough historical dates to run a walk-forward backtest")

    train_window = min(max(63, len(all_dates) // 10), max(len(all_dates) - 2, 1))
    test_window = max(REBALANCE_EVERY, min(63, max(10, len(all_dates) // 8)))
    train_window = min(train_window, max(len(all_dates) - test_window - 1, 1))
    splits = build_purged_walk_forward_splits(
        all_dates,
        train_window=train_window,
        test_window=test_window,
        step=test_window,
        purge_period=1,
        embargo_period=1,
    )
    if not splits:
        raise SystemExit("Unable to build walk-forward splits from the available history")

    print(
        f"    rows={features.shape[0]}, dates={len(all_dates)}, "
        f"windows={len(splits)}, train_window={train_window}, test_window={test_window}"
    )

    print("[2/7] Train rolling walk-forward windows")
    agg = CouncilAggregator()
    lgbm_history_rows: list[pd.Series] = []
    signal_rows: list[pd.Series] = []
    weight_rows: list[pd.Series] = []
    last_target_w: pd.Series | None = None
    global_rebalance_count = 0
    last_regime_payload: dict[str, float | str] = {"regime": "transition", "bull": 0.0, "bear": 0.0, "transition": 0.0}

    for window_id, split in enumerate(splits, start=1):
        train_start = split.train_start.date()
        train_end = split.train_end.date()
        test_start = split.test_start.date()
        test_end = split.test_end.date()

        feat_train = features.filter(
            (pl.col("valid_time") >= pl.lit(train_start))
            & (pl.col("valid_time") <= pl.lit(train_end))
        )
        feat_test = features.filter(
            (pl.col("valid_time") >= pl.lit(test_start))
            & (pl.col("valid_time") <= pl.lit(test_end))
        )
        if feat_train.is_empty() or feat_test.is_empty():
            print(f"    [window {window_id}/{len(splits)}] skipped (empty train/test slice)")
            continue

        print(
            f"    [window {window_id}/{len(splits)}] "
            f"train {train_start} → {train_end} | test {test_start} → {test_end}"
        )

        targets_train = (
            targets
            .filter(
                (pl.col("valid_time") >= pl.lit(train_start))
                & (pl.col("valid_time") <= pl.lit(train_end))
            )
            .select(["ticker", "valid_time", "rank_fwd_1d"])
            .to_pandas()
        )
        targets_train["valid_time"] = pd.to_datetime(targets_train["valid_time"]).dt.date
        targets_train = targets_train.set_index(["ticker", "valid_time"])["rank_fwd_1d"].dropna()

        lgbm = TechnicalModel(config_path=str(ROOT / "config" / "models.yaml"))
        lgbm.fit(feat_train, targets_train)

        macro_train = macro.filter(pl.col("valid_time") <= pl.lit(train_end)) if not macro.is_empty() else macro
        hmm = RegimeModel()
        if not macro_train.is_empty():
            hmm.fit(macro_train)

        sizer, feat_cols = rp.step_conformal(lgbm, feat_train, targets)

        test_dates = sorted(feat_test["valid_time"].unique().to_list())
        for d in test_dates:
            day_feat = feat_test.filter(pl.col("valid_time") == d)
            if day_feat.is_empty():
                continue

            ts = pd.Timestamp(d)
            lgbm_signal = lgbm.predict(day_feat)
            lgbm_history_rows.append(pd.Series(lgbm_signal.to_dict(), name=ts))
            lgbm_hist_df = pd.DataFrame(lgbm_history_rows).sort_index()
            if not lgbm_hist_df.empty:
                lgbm_hist_df.index = pd.to_datetime(lgbm_hist_df.index)

            returns_hist = forward_returns.loc[:ts]
            if not lgbm_hist_df.empty and not returns_hist.empty:
                agg.update_performance({"lgbm": lgbm_hist_df}, returns_hist, date=ts.date())

            if macro.is_empty():
                regime = "transition"
            else:
                macro_today = _build_macro_today(macro, ts)
                regime = hmm.predict_regime(macro_today) if not macro_today.is_empty() else "transition"

            zero_signal = pd.Series(0.0, index=lgbm_signal.index)
            council_signal = agg.aggregate({"lgbm": lgbm_signal, "hmm": zero_signal}, regime=regime, date=ts.date())
            signal_rows.append(council_signal.rename(ts))

            last_regime_payload = {"regime": regime}
            if not macro.is_empty():
                macro_last = _build_macro_today(macro, ts)
                if not macro_last.is_empty():
                    last_regime_payload = {
                        "regime": hmm.predict_regime(macro_last),
                        **{k: float(v) for k, v in hmm.predict_probabilities(macro_last).items()},
                    }

            if last_target_w is None or global_rebalance_count % REBALANCE_EVERY == 0:
                target_w = rp.step_portfolio(
                    council_signal,
                    sizer,
                    feat_cols,
                    feat_test,
                    ohlcv,
                    ts.date(),
                    save_orders=False,
                    emit_report=False,
                )
                last_target_w = target_w.copy()
            else:
                target_w = last_target_w.reindex(council_signal.index).fillna(0.0)

            weight_rows.append(target_w.rename(ts))
            global_rebalance_count += 1

    if not weight_rows:
        raise SystemExit("No walk-forward weights were produced")

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
    walk_forward_result = run_walk_forward_analysis(
        signals=signals_df.loc[aligned_returns.index],
        forward_returns=aligned_returns,
        train_window=train_window,
        test_window=test_window,
        step=test_window,
        purge_period=1,
        embargo_period=1,
    )

    print("[3/7] Simulate portfolio")
    sim = simulate_weight_backtest(
        weights=weights_df,
        forward_returns=aligned_returns.reindex(weights_df.index),
        initial_capital=100_000.0,
        cost_model=TransactionCostModel.from_env(),
    )

    print("[4/7] Persist dashboard artifacts")
    with open(RESULTS_DIR / "backtest_result.pkl", "wb") as f:
        pickle.dump(sim, f, protocol=pickle.HIGHEST_PROTOCOL)
    sim.equity_curve.to_frame().to_parquet(RESULTS_DIR / "equity_curve.parquet", index=True)
    weights_df.to_parquet(RESULTS_DIR / "strategy_weights.parquet", index=True)
    signals_df.to_parquet(RESULTS_DIR / "walk_forward_signals.parquet", index=True)
    walk_forward_result["window_metrics"].to_parquet(RESULTS_DIR / "walk_forward_windows.parquet", index=False)
    walk_forward_result["oos_returns"].to_frame().to_parquet(RESULTS_DIR / "walk_forward_oos_returns.parquet", index=True)
    walk_forward_result["benchmark_comparison"].to_parquet(RESULTS_DIR / "walk_forward_benchmark.parquet", index=False)
    walk_forward_result["regime_performance"].to_parquet(RESULTS_DIR / "walk_forward_regime.parquet", index=False)
    walk_forward_result["ablation_analysis"].to_parquet(RESULTS_DIR / "walk_forward_ablation.parquet", index=False)

    with open(RESULTS_DIR / "current_regime.json", "w", encoding="utf-8") as f:
        json.dump(last_regime_payload, f, indent=2)

    with open(RESULTS_DIR / "walk_forward_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "simulation": sim.stats,
                "walk_forward": walk_forward_result["summary"],
            },
            f,
            indent=2,
            default=float,
        )
    weights_df.to_csv(RESULTS_DIR / "strategy_weights.csv")

    # Save regime history if macro is available.
    if not macro.is_empty():
        history_df = hmm.get_regime_history(macro)
        if "valid_time" in history_df.columns:
            history_df = history_df.rename(columns={"valid_time": "date"})
        history_df.to_parquet(RESULTS_DIR / "regime_history.parquet", index=False)

    agg.save(RESULTS_DIR / "aggregator.pkl")

    # Attribution parquet from the accumulated weights log.
    attr_rows = []
    for log_date in sorted(agg._weights_log):
        try:
            df = agg.get_attribution(log_date)
            if not df.empty:
                df = df.copy()
                df["date"] = pd.Timestamp(log_date)
                attr_rows.append(df)
        except Exception as exc:
            print(f"[warn] attribution for {log_date} unavailable: {exc}")

    if attr_rows:
        attr_df = pd.concat(attr_rows, ignore_index=True)
        attr_df["date"] = pd.to_datetime(attr_df["date"])
        attr_df = attr_df.sort_values(["date", "model_name"])
        attr_df.to_parquet(RESULTS_DIR / "attribution.parquet", index=False)

    print("[5/7] Done")
    print("Simulation stats:\n" + json.dumps(sim.stats, indent=2, default=float))
    print("Walk-forward summary:\n" + json.dumps(walk_forward_result["summary"], indent=2, default=float))
    print(f"Backtest equity points: {len(sim.equity_curve)}")
    print(f"Walk-forward windows: {len(walk_forward_result['window_metrics'])}")
    print(f"Weights rows: {len(weights_df)}")
    print(f"Artifacts written to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
