#!/usr/bin/env python
"""Standalone TFT alpha challenger training (shadow mode — T2.1).

Trains ``TemporalFusionAlpha`` on Alpha158 features + forward-return targets,
saves checkpoint to ``models/checkpoints/tft_challenger.pkl``, and writes shadow
signals to ``data/results/tft_shadow_signals.parquet`` for walk-forward CI.

Does **not** modify the daily Dagster pipeline or council aggregator.

Usage:
    python scripts/train_tft.py --start 2021-01-01 --end 2024-12-31
    python scripts/train_tft.py --help
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.features.alpha158 import build_macro_context, compute_alpha158  # noqa: E402
from data.features.target import compute_targets, training_rank_column  # noqa: E402
from models.tft import (  # noqa: E402
    TemporalFusionAlpha,
    build_shadow_signal_matrix,
    write_shadow_signals,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TFT shadow challenger")
    p.add_argument("--start", type=str, default="2021-01-01", help="Training start (YYYY-MM-DD)")
    p.add_argument("--end", type=str, default=None, help="Training end (YYYY-MM-DD)")
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "models.yaml",
        help="models.yaml path",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models" / "checkpoints" / "tft_challenger.pkl",
        help="Output checkpoint path",
    )
    p.add_argument(
        "--shadow-parquet",
        type=Path,
        default=ROOT / "data" / "results" / "tft_shadow_signals.parquet",
        help="Shadow signal output for walk-forward CI",
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "ohlcv",
        help="OHLCV parquet root",
    )
    p.add_argument("--max-epochs", type=int, default=None, help="Override tft.max_epochs")
    return p.parse_args()


def _normalize_ohlcv(df: pl.DataFrame) -> pl.DataFrame:
    for c in df.columns:
        if df[c].dtype == pl.Datetime:
            df = df.with_columns(pl.col(c).dt.replace_time_zone("UTC").cast(pl.Date))
    return df


def load_ohlcv(raw_dir: Path) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    if not raw_dir.exists():
        raise FileNotFoundError(f"OHLCV directory not found: {raw_dir}")

    for ticker_dir in sorted(raw_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        if ticker in ("BTCUSD", "ETHUSD"):
            continue
        historical = ticker_dir / f"{ticker}.parquet"
        if not historical.exists():
            continue
        try:
            df = _normalize_ohlcv(pl.read_parquet(historical))
            if "ticker" not in df.columns:
                df = df.with_columns(pl.lit(ticker).alias("ticker"))
            frames.append(df)
        except Exception as exc:
            print(f"  Skipping {historical}: {exc}")

    if not frames:
        raise ValueError("No OHLCV data found")
    return pl.concat(frames).unique(["ticker", "valid_time"]).sort(["ticker", "valid_time"])


def _filter_dates(df: pl.DataFrame, start: date, end: date | None) -> pl.DataFrame:
    out = df.filter(pl.col("valid_time") >= start)
    if end is not None:
        out = out.filter(pl.col("valid_time") <= end)
    return out


def main() -> int:
    args = _parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else None

    print("=" * 60)
    print("MLCouncil TFT Shadow Challenger Training")
    print("=" * 60)

    print("\n[1/5] Loading OHLCV...")
    ohlcv = load_ohlcv(args.raw_dir)
    ohlcv = _filter_dates(ohlcv, start, end)
    print(f"  Rows: {ohlcv.shape[0]}, tickers: {ohlcv['ticker'].n_unique()}")

    print("\n[2/5] Macro context...")
    macro_dir = ROOT / "data" / "raw" / "macro"
    macro = None
    if macro_dir.exists():

        def _path(name: str) -> str | None:
            p = macro_dir / f"{name}.parquet"
            return str(p) if p.exists() else None

        try:
            macro = build_macro_context(
                vix_path=_path("vix"),
                treasuries_path=_path("treasuries"),
                sp500_path=_path("sp500"),
            )
        except Exception as exc:
            print(f"  Macro skipped: {exc}")

    print("\n[3/5] Alpha158 features...")
    features = compute_alpha158(ohlcv, macro_df=macro)
    features = _filter_dates(features, start, end)

    print("\n[4/5] Targets...")
    train_horizon = 5
    targets_pl = compute_targets(ohlcv, horizons=[train_horizon], risk_adjusted=False)
    rank_col = training_rank_column(train_horizon)
    targets_df = targets_pl.select(["ticker", "valid_time", rank_col]).to_pandas()
    targets_df["valid_time"] = pd.to_datetime(targets_df["valid_time"]).dt.date
    targets = pd.Series(
        targets_df[rank_col].values,
        index=pd.MultiIndex.from_frame(
            targets_df[["ticker", "valid_time"]], names=["ticker", "valid_time"]
        ),
        name="target",
    )
    targets = targets.dropna()
    print(f"  Valid targets: {len(targets)}")

    print("\n[5/5] Training TFT...")
    model = TemporalFusionAlpha(config_path=str(args.config))
    if args.max_epochs is not None:
        model._params["max_epochs"] = args.max_epochs

    model.fit(features, targets)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.checkpoint))
    print(f"  Checkpoint: {args.checkpoint}")

    latency_ms = model.measure_inference_latency_ms(features.tail(500))
    print(f"  Inference latency (sample): {latency_ms:.1f} ms CPU")

    shadow = build_shadow_signal_matrix(features, model)
    write_shadow_signals(shadow, args.shadow_parquet)
    # Walk-forward CI expects walkforward_signals_tft.parquet
    wf_path = ROOT / "data" / "results" / "walkforward_signals_tft.parquet"
    write_shadow_signals(shadow, wf_path)
    print(f"  Shadow signals: {args.shadow_parquet}")
    print(f"  Walk-forward cache: {wf_path}")

    if model._train_loss_history:
        print(f"  Final train loss: {model._train_loss_history[-1]:.4f}")

    top = model.get_selection_weights().head(10)
    print("\n  Top 10 VSN weights:")
    for feat, w in top.items():
        print(f"    {feat:40s} {w:.4f}")

    print("\n" + "=" * 60)
    print("TFT shadow training complete (not wired to daily pipeline).")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
