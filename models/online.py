"""Incremental daily LightGBM updates for the production inference path.

Uses ``lightgbm.Booster.refit()`` on the champion checkpoint loaded via
``TechnicalModel``. Does **not** replace walk-forward champion promotion;
that remains ``council/walkforward_promotion_gate.py``.

Enable with ``MLCOUNCIL_ONLINE_LEARNING=true``.
Canary status: shadow — target: P-1.1 — expiry: 2027-02-01 (promote via canary o retire)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from models.technical import TechnicalModel

_DEFAULT_IC_THRESHOLD = 0.05
_DEFAULT_REFIT_DAYS = 60
_DEFAULT_EVAL_DAYS = 10
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def online_learning_enabled() -> bool:
    """Return True when daily incremental refit is enabled."""
    return os.getenv("MLCOUNCIL_ONLINE_LEARNING", "").strip().lower() in _TRUTHY


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return int(raw)


@dataclass
class OnlineUpdateResult:
    """Outcome of a daily incremental refit attempt."""

    accepted: bool
    ic_baseline: float
    ic_today: float
    ic_threshold: float
    refit_rows: int
    eval_rows: int
    drift_detected: bool = False
    drift_detector: str | None = None
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class IncrementalLightGBM:
    """Wrapper around a fitted ``TechnicalModel`` for booster-level refit."""

    def __init__(self, model: TechnicalModel) -> None:
        if model._model is None:
            raise RuntimeError("IncrementalLightGBM requires a fitted TechnicalModel")
        self._model = model

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> IncrementalLightGBM:
        model = TechnicalModel()
        model.load(path)
        return cls(model)

    @property
    def technical_model(self) -> TechnicalModel:
        return self._model

    @staticmethod
    def spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(TechnicalModel._ic(y_true, y_pred))

    def _aligned_frame(
        self,
        features: pl.DataFrame,
        targets: pd.Series,
    ) -> tuple[pd.DataFrame, list[str]]:
        df, feat_cols = self._model._to_pandas_aligned(features, targets)
        df = df.dropna(subset=["__target__"] + feat_cols)
        return df, feat_cols

    def evaluate_ic(
        self,
        features: pl.DataFrame,
        targets: pd.Series,
    ) -> float:
        """Spearman IC on an evaluation slice."""
        df, feat_cols = self._aligned_frame(features, targets)
        if len(df) < 10 or not feat_cols:
            return 0.0
        preds = self._model._model.predict(df[feat_cols])
        return self.spearman_ic(df["__target__"].values, preds)

    def refit(
        self,
        features: pl.DataFrame,
        targets: pd.Series,
        *,
        additional_rounds: int | None = None,
    ) -> int:
        """Incrementally update the underlying booster with new labeled rows.

        Returns the number of rows used for refit.
        """
        df, feat_cols = self._aligned_frame(features, targets)
        if len(df) < 5 or not feat_cols:
            return 0

        booster = self._model._model.booster_
        rounds = additional_rounds
        if rounds is None:
            rounds = max(1, int(self._model._params.get("n_estimators", 100) // 10))

        booster.refit(
            df[feat_cols].values,
            df["__target__"].values,
            max_iter=rounds,
        )
        return len(df)

    def should_accept_update(
        self,
        ic_today: float,
        ic_baseline: float,
        *,
        threshold: float = _DEFAULT_IC_THRESHOLD,
    ) -> bool:
        """Accept incremental update unless IC regressed beyond threshold."""
        return ic_today >= ic_baseline - threshold

    def predict(self, features: pl.DataFrame) -> pd.Series:
        return self._model.predict(features)

    def save_checkpoint(self, path: str | Path, *, compute_hash: bool = True) -> str:
        return self._model.save(path, compute_hash=compute_hash)


def split_refit_eval_slices(
    features: pl.DataFrame,
    targets: pd.Series,
    *,
    refit_days: int,
    eval_days: int,
) -> tuple[pl.DataFrame, pd.Series, pl.DataFrame, pd.Series]:
    """Split labeled history into refit (older) and IC eval (recent) windows."""
    dates = sorted(features["valid_time"].unique().to_list())
    if not dates:
        empty = features.head(0)
        return empty, targets.iloc[0:0], empty, targets.iloc[0:0]

    n_eval = max(1, min(eval_days, max(1, len(dates) // 5)))
    eval_dates = set(dates[-n_eval:])
    eval_start = min(eval_dates)
    refit_dates = {d for d in dates if d < eval_start}
    if refit_days and len(refit_dates) > refit_days:
        refit_dates = set(sorted(refit_dates)[-refit_days:])

    refit_feat = features.filter(pl.col("valid_time").is_in(list(refit_dates)))
    eval_feat = features.filter(pl.col("valid_time").is_in(list(eval_dates)))

    def _slice_targets(feat: pl.DataFrame) -> pd.Series:
        if feat.is_empty():
            return targets.iloc[0:0]
        keys = set(
            zip(
                feat["ticker"].to_list(),
                feat["valid_time"].to_list(),
                strict=False,
            )
        )
        if isinstance(targets.index, pd.MultiIndex):
            mask = [
                (t, d) in keys
                for t, d in zip(
                    targets.index.get_level_values(0),
                    targets.index.get_level_values(1),
                    strict=False,
                )
            ]
            return targets[mask]
        return targets

    return (
        refit_feat,
        _slice_targets(refit_feat),
        eval_feat,
        _slice_targets(eval_feat),
    )


def build_targets_series(
    targets_pl: pl.DataFrame,
    horizon_col: str | None = None,
    *,
    horizon: int = 1,
) -> pd.Series:
    """Convert ``compute_targets`` output to MultiIndex Series."""
    from data.features.target import training_rank_column

    horizon_col = horizon_col or training_rank_column(horizon)
    pdf = targets_pl.select(["ticker", "valid_time", horizon_col]).to_pandas()
    if pd.api.types.is_datetime64_any_dtype(pdf["valid_time"]):
        pdf["valid_time"] = pdf["valid_time"].dt.date
    else:
        pdf["valid_time"] = pd.to_datetime(pdf["valid_time"]).dt.date
    pdf = pdf.dropna(subset=[horizon_col])
    return pd.Series(
        pdf[horizon_col].values,
        index=pd.MultiIndex.from_frame(
            pdf[["ticker", "valid_time"]],
            names=["ticker", "valid_time"],
        ),
        name=horizon_col,
    )


def filter_features_from_date(
    features: pl.DataFrame,
    *,
    as_of: date,
    lookback_days: int,
) -> pl.DataFrame:
    """Keep rows with valid_time in (as_of - lookback, as_of]."""
    start = as_of - timedelta(days=int(lookback_days * 1.5))
    return features.filter(
        (pl.col("valid_time") > start) & (pl.col("valid_time") <= as_of)
    )


def equal_weight_daily_returns(ohlcv: pl.DataFrame, n_days: int = 60) -> pd.Series:
    """Equal-weight cross-sectional daily returns for ADWIN monitoring."""
    if ohlcv.is_empty() or "adj_close" not in ohlcv.columns:
        return pd.Series(dtype=float)

    df = (
        ohlcv.sort(["ticker", "valid_time"])
        .with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1).over("ticker") - 1.0).alias(
                "ret"
            )
        )
        .drop_nulls("ret")
    )
    daily = (
        df.group_by("valid_time")
        .agg(pl.col("ret").mean().alias("ew_ret"))
        .sort("valid_time")
    )
    pdf = daily.to_pandas()
    if pdf.empty:
        return pd.Series(dtype=float)
    pdf = pdf.set_index("valid_time")["ew_ret"].tail(n_days)
    return pdf


def run_daily_incremental_update(
    model: TechnicalModel,
    checkpoint_path: Path,
    *,
    features_history: pl.DataFrame,
    targets: pd.Series,
    ohlcv: pl.DataFrame | None = None,
    ic_threshold: float | None = None,
    refit_days: int | None = None,
    eval_days: int | None = None,
) -> tuple[TechnicalModel, OnlineUpdateResult]:
    """Attempt daily refit; rollback to pre-update model if IC gate fails.

    Parameters
    ----------
    model:
        Champion ``TechnicalModel`` loaded from checkpoint.
    checkpoint_path:
        Path to ``lgbm_latest.pkl`` (updated only when gate passes).
    features_history:
        Labeled feature rows for refit + IC eval windows.
    targets:
        MultiIndex targets aligned with ``features_history``.
    ohlcv:
        Optional OHLCV for ADWIN on equal-weight daily returns.
    """
    threshold = ic_threshold if ic_threshold is not None else _env_float(
        "MLCOUNCIL_ONLINE_IC_THRESHOLD", _DEFAULT_IC_THRESHOLD
    )
    refit_window = refit_days if refit_days is not None else _env_int(
        "MLCOUNCIL_ONLINE_REFIT_DAYS", _DEFAULT_REFIT_DAYS
    )
    eval_window = eval_days if eval_days is not None else _env_int(
        "MLCOUNCIL_ONLINE_EVAL_DAYS", _DEFAULT_EVAL_DAYS
    )

    incremental = IncrementalLightGBM(model)

    refit_feat, refit_tgt, eval_feat, eval_tgt = split_refit_eval_slices(
        features_history,
        targets,
        refit_days=refit_window,
        eval_days=eval_window,
    )

    ic_baseline = incremental.evaluate_ic(eval_feat, eval_tgt) if not eval_feat.is_empty() else 0.0

    refit_rows = 0
    if not refit_feat.is_empty():
        refit_rows = incremental.refit(refit_feat, refit_tgt)

    ic_today = incremental.evaluate_ic(eval_feat, eval_tgt) if not eval_feat.is_empty() else ic_baseline
    accepted = incremental.should_accept_update(ic_today, ic_baseline, threshold=threshold)

    drift_detected = False
    drift_detector: str | None = None
    if ohlcv is not None and not ohlcv.is_empty():
        try:
            from council.risk.drift import ADWINDetector

            returns = equal_weight_daily_returns(ohlcv, n_days=60)
            detector = ADWINDetector(window_days=60)
            drift_detected = detector.update_series(returns)
            if drift_detected:
                drift_detector = "adwin"
        except Exception:
            pass

    if accepted and refit_rows > 0:
        incremental.save_checkpoint(checkpoint_path)
        message = f"incremental refit accepted (IC {ic_today:.4f} >= {ic_baseline - threshold:.4f})"
        final_model = incremental.technical_model
    else:
        model.load(checkpoint_path)
        final_model = model
        if refit_rows == 0:
            message = "incremental refit skipped (insufficient labeled rows)"
        else:
            message = (
                f"incremental refit rejected (IC {ic_today:.4f} < "
                f"{ic_baseline - threshold:.4f}); champion unchanged"
            )

    result = OnlineUpdateResult(
        accepted=accepted and refit_rows > 0,
        ic_baseline=ic_baseline,
        ic_today=ic_today,
        ic_threshold=threshold,
        refit_rows=refit_rows,
        eval_rows=int(eval_feat.shape[0]) if not eval_feat.is_empty() else 0,
        drift_detected=drift_detected,
        drift_detector=drift_detector,
        message=message,
        metadata={
            "refit_days": refit_window,
            "eval_days": eval_window,
        },
    )
    return final_model, result
