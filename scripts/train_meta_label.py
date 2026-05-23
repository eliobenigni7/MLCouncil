#!/usr/bin/env python
"""Train meta-label classifier from historical LGBM signals and realized returns.

Requires ``models/checkpoints/lgbm_latest.pkl`` and OHLCV under ``data/raw/ohlcv/``.

Usage:
    python scripts/train_meta_label.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.features.alpha158 import build_macro_context, compute_alpha158
from data.features.target import compute_targets
from models.meta_label import MetaLabelClassifier
from models.regime_features import append_regime_features, load_regime_history, regime_features_enabled
from models.technical import TechnicalModel
from scripts.train_lgbm_standalone import load_ohlcv


def main() -> None:
    ckpt = ROOT / "models" / "checkpoints" / "lgbm_latest.pkl"
    if not ckpt.exists():
        raise FileNotFoundError(f"LightGBM checkpoint missing: {ckpt}")

    ohlcv = load_ohlcv()
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
        except Exception:
            macro = None

    features = compute_alpha158(ohlcv, macro_df=macro)
    if regime_features_enabled():
        hist = load_regime_history(ROOT / "data" / "results" / "regime_history.parquet")
        features = append_regime_features(features, hist, interactions=True)

    targets_pl = compute_targets(ohlcv, horizons=[1], risk_adjusted=False)
    targets_df = targets_pl.select(["ticker", "valid_time", "ret_fwd_1d"]).to_pandas()
    targets_df["valid_time"] = pd.to_datetime(targets_df["valid_time"]).dt.date
    forward = pd.Series(
        targets_df["ret_fwd_1d"].values,
        index=pd.MultiIndex.from_frame(
            targets_df[["ticker", "valid_time"]],
            names=["ticker", "valid_time"],
        ),
    ).dropna()

    lgbm = TechnicalModel(config_path=str(ROOT / "config" / "models.yaml"))
    lgbm.load(ckpt)
    regime_hist = (
        load_regime_history(ROOT / "data" / "results" / "regime_history.parquet")
        if regime_features_enabled()
        else None
    )

    primary_parts: list[pd.Series] = []
    for dt in sorted(features["valid_time"].unique().to_list()):
        day_feat = features.filter(pl.col("valid_time") == dt)
        if day_feat.is_empty():
            continue
        sig = lgbm.predict(day_feat, regime_history=regime_hist)
        primary_parts.append(
            pd.Series(
                sig.values,
                index=pd.MultiIndex.from_arrays(
                    [sig.index.tolist(), [dt] * len(sig)],
                    names=["ticker", "valid_time"],
                ),
            )
        )

    primary = pd.concat(primary_parts)

    meta = MetaLabelClassifier(config_path=str(ROOT / "config" / "models.yaml"))
    meta.fit(features, primary, forward)

    out = ROOT / "models" / "checkpoints" / "meta_label_latest.pkl"
    meta.save(out)
    print(f"Meta-label checkpoint saved: {out}")


if __name__ == "__main__":
    main()
