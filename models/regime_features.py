"""Regime-conditioned feature augmentation for the technical LightGBM path.

Joins HMM regime labels and posterior probabilities on ``valid_time`` and
optionally adds interactions with top Alpha158 features when
``MLCOUNCIL_REGIME_FEATURES=true``.
Canary status: shadow — target: P-1.1 — expiry: 2027-02-01 (promote via canary o retire)
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import polars as pl

REGIME_STATES: tuple[str, ...] = ("bull", "bear", "transition")
_TRUTHY = frozenset({"1", "true", "yes", "on"})

_DEFAULT_HISTORY = Path("data/results/regime_history.parquet")


def regime_features_enabled() -> bool:
    """Return True when regime columns should be merged into LGBM features."""
    return os.getenv("MLCOUNCIL_REGIME_FEATURES", "").strip().lower() in _TRUTHY


def load_regime_history(path: str | Path | None = None) -> pd.DataFrame:
    """Load per-date regime labels and probabilities from parquet."""
    p = Path(path) if path is not None else _DEFAULT_HISTORY
    if not p.exists():
        return pd.DataFrame(columns=["valid_time", "regime"])

    hist = pd.read_parquet(p)
    if "valid_time" not in hist.columns:
        return pd.DataFrame(columns=["valid_time", "regime"])

    hist = hist.copy()
    if pd.api.types.is_datetime64_any_dtype(hist["valid_time"]):
        hist["valid_time"] = hist["valid_time"].dt.date
    else:
        hist["valid_time"] = pd.to_datetime(hist["valid_time"], errors="coerce").dt.date

    return hist.sort_values("valid_time").drop_duplicates("valid_time", keep="last")


def _regime_frame_for_merge(regime_history: pd.DataFrame) -> pd.DataFrame:
    """Build one-hot and probability columns keyed by valid_time."""
    if regime_history.empty:
        return pd.DataFrame(columns=["valid_time"])

    df = regime_history.copy()
    for state in REGIME_STATES:
        df[f"regime_state_{state}"] = (df["regime"] == state).astype(float)

    for state in REGIME_STATES:
        prob_col = f"prob_{state}"
        out_col = f"regime_prob_{state}"
        if prob_col in df.columns:
            df[out_col] = df[prob_col].astype(float)
        else:
            df[out_col] = df[f"regime_state_{state}"]

    keep = ["valid_time"] + [f"regime_state_{s}" for s in REGIME_STATES]
    keep += [f"regime_prob_{s}" for s in REGIME_STATES]
    return df[keep]


def append_regime_features(
    features: pl.DataFrame,
    regime_history: pd.DataFrame,
    *,
    interactions: bool = False,
    top_feature_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Merge regime one-hot / posterior features (and optional interactions) on date."""
    if features.is_empty() or regime_history.empty:
        return features

    regime_df = _regime_frame_for_merge(regime_history)
    regime_pl = pl.from_pandas(regime_df)

    feat = features.clone()
    if feat["valid_time"].dtype == pl.Datetime:
        feat = feat.with_columns(pl.col("valid_time").dt.date())
    elif feat["valid_time"].dtype != pl.Date:
        feat = feat.with_columns(pl.col("valid_time").cast(pl.Date))

    merged = feat.join(regime_pl, on="valid_time", how="left")

    regime_cols = [f"regime_state_{s}" for s in REGIME_STATES] + [
        f"regime_prob_{s}" for s in REGIME_STATES
    ]
    for col in regime_cols:
        if col in merged.columns:
            merged = merged.with_columns(pl.col(col).fill_null(0.0))

    if interactions and top_feature_cols:
        base_cols = [c for c in top_feature_cols if c in merged.columns]
        for feat_col in base_cols[:10]:
            for state in REGIME_STATES:
                onehot = f"regime_state_{state}"
                if onehot in merged.columns:
                    merged = merged.with_columns(
                        (pl.col(feat_col) * pl.col(onehot)).alias(
                            f"{feat_col}_x_{onehot}"
                        )
                    )

    return merged


def regime_feature_column_names(
    *,
    interactions: bool = False,
    top_feature_cols: list[str] | None = None,
) -> list[str]:
    """Column names produced by ``append_regime_features`` (for tests)."""
    cols = [f"regime_state_{s}" for s in REGIME_STATES]
    cols += [f"regime_prob_{s}" for s in REGIME_STATES]
    if interactions and top_feature_cols:
        for feat_col in top_feature_cols[:10]:
            for state in REGIME_STATES:
                cols.append(f"{feat_col}_x_regime_state_{state}")
    return cols
