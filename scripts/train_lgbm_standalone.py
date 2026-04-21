#!/usr/bin/env python
"""Standalone LightGBM training script for MLCouncil.

Loads historical OHLCV from data/raw/ohlcv/, computes Alpha158 features,
builds forward-return targets, trains TechnicalModel with CPCV, and saves
checkpoint to models/checkpoints/lgbm_latest.pkl.

Usage:
    source .venv/bin/activate
    python scripts/train_lgbm_standalone.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.features.alpha158 import compute_alpha158, build_macro_context
from data.features.target import compute_targets
from models.technical import TechnicalModel


def _normalize_ohlcv(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize datetime columns for consistent concatenation."""
    for c in df.columns:
        dtype = df[c].dtype
        if dtype == pl.Datetime:
            # Cast to Date (UTC) for consistency
            df = df.with_columns(
                pl.col(c).dt.replace_time_zone("UTC").cast(pl.Date)
            )
    return df


def load_ohlcv(raw_dir: Path | None = None) -> pl.DataFrame:
    """Load all historical OHLCV from raw parquet files."""
    if raw_dir is None:
        raw_dir = ROOT / "data" / "raw" / "ohlcv"

    frames: list[pl.DataFrame] = []
    if not raw_dir.exists():
        raise FileNotFoundError(f"OHLCV directory not found: {raw_dir}")

    for ticker_dir in sorted(raw_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        # Skip crypto tickers and non-equity
        if ticker in ("BTCUSD", "ETHUSD"):
            continue
        # Load the full historical file (e.g., AAPL.parquet)
        historical = ticker_dir / f"{ticker}.parquet"
        if not historical.exists():
            print(f"  No historical file for {ticker}")
            continue
        try:
            df = _normalize_ohlcv(pl.read_parquet(historical))
            if "ticker" not in df.columns:
                df = df.with_columns(pl.lit(ticker).alias("ticker"))
            frames.append(df)
        except Exception as e:
            print(f"  Skipping {historical}: {e}")

    if not frames:
        raise ValueError("No OHLCV data found")

    result = pl.concat(frames).unique(["ticker", "valid_time"]).sort(
        ["ticker", "valid_time"]
    )
    print(f"Loaded {result.shape[0]} rows, {result['ticker'].n_unique()} tickers")
    return result


def main():
    print("=" * 60)
    print("MLCouncil LightGBM Training (Standalone)")
    print("=" * 60)

    # 1. Load OHLCV
    print("\n[1/5] Loading OHLCV data...")
    ohlcv = load_ohlcv()
    print(f"  Date range: {ohlcv['valid_time'].min()} to {ohlcv['valid_time'].max()}")
    print(f"  Tickers: {sorted(ohlcv['ticker'].unique().to_list())}")

    # 2. Load macro data
    print("\n[2/5] Loading macro data...")
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
            print(f"  Macro data loaded: {macro.shape[0]} rows")
        except Exception as e:
            print(f"  Macro data not available: {e}")
    else:
        print("  No macro directory found")

    # 3. Compute Alpha158 features
    print("\n[3/5] Computing Alpha158 features...")
    features = compute_alpha158(ohlcv, macro_df=macro)
    print(f"  Features shape: {features.shape}")
    print(f"  Feature columns: {len(features.columns)}")

    # 4. Build targets
    print("\n[4/5] Building forward-return targets...")
    targets_pl = compute_targets(ohlcv, horizons=[5], risk_adjusted=False)
    print(f"  Targets shape: {targets_pl.shape}")

    # Convert targets to pandas Series with MultiIndex (ticker, valid_time)
    targets_df = targets_pl.select(["ticker", "valid_time", "rank_fwd_5d"]).to_pandas()
    targets_df["valid_time"] = pd.to_datetime(targets_df["valid_time"]).dt.date
    targets = pd.Series(
        targets_df["rank_fwd_5d"].values,
        index=pd.MultiIndex.from_frame(
            targets_df[["ticker", "valid_time"]], names=["ticker", "valid_time"]
        ),
        name="target",
    )
    targets = targets.dropna()
    print(f"  Valid targets: {len(targets)}")

    # 5. Train LightGBM
    print("\n[5/5] Training LightGBM with CPCV...")
    model = TechnicalModel(config_path=str(ROOT / "config" / "models.yaml"))
    model.fit(features, targets)

    # Save checkpoint
    checkpoint_path = ROOT / "models" / "checkpoints" / "lgbm_latest.pkl"
    model.save(str(checkpoint_path))
    print(f"\n  Checkpoint saved: {checkpoint_path}")

    # Print results
    if model._fold_metrics:
        ics = [m["ic"] for m in model._fold_metrics]
        print(f"\n  CPCV Results:")
        print(f"    Folds: {len(ics)}")
        print(f"    Mean IC: {np.mean(ics):.4f}")
        print(f"    Std IC:  {np.std(ics):.4f}")
        print(f"    ICIR:    {np.mean(ics) / np.std(ics) if np.std(ics) > 1e-9 else 0:.4f}")
        print(f"    Best IC: {max(ics):.4f}")

    if model._shap_importance is not None:
        print(f"\n  Top 10 SHAP features:")
        for _, row in model._shap_importance.head(10).iterrows():
            print(f"    {row['feature']:40s} {row['shap_importance']:.4f}")

    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
