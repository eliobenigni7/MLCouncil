"""Conditional quantile regression sizing + stacking meta-learner (T3.2 shadow).

``MLCOUNCIL_POSITION_SIZING=cqr`` selects CQR; default ``conformal`` keeps
``council/sizing/conformal.py`` (MAPIE Jackknife+) as production path.
Canary status: shadow — target: P-1.1 — expiry: 2027-02-01 (promote via canary o retire)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

DEFAULT_CQR_CHECKPOINT = (
    Path(__file__).resolve().parents[2] / "models" / "checkpoints" / "cqr_sizer.pkl"
)
DEFAULT_STACKING_CHECKPOINT = (
    Path(__file__).resolve().parents[2] / "models" / "checkpoints" / "stacking_meta.pkl"
)
SHADOW_STACKING_DIR = Path(__file__).resolve().parents[2] / "data" / "results" / "shadow_stacking"


def position_sizing_mode() -> str:
    """``conformal`` (default), ``cqr``, or ``kelly``."""
    raw = os.getenv("MLCOUNCIL_POSITION_SIZING", "conformal").strip().lower()
    if raw in ("kelly", "fractional_kelly", "fractional-kelly"):
        return "kelly"
    return raw if raw in ("conformal", "cqr") else "conformal"


def position_sizer_checkpoint_name() -> str:
    mode = position_sizing_mode()
    if mode == "kelly":
        return "kelly_sizer.pkl"
    return "cqr_sizer.pkl" if mode == "cqr" else "conformal_sizer.pkl"


def get_position_sizer(coverage: float = 0.85):
    """Factory: conformal (MAPIE), CQR shadow sizer, or FractionalKellySizer."""
    mode = position_sizing_mode()
    if mode == "kelly":
        from council.sizing.fractional_kelly import FractionalKellySizer

        return FractionalKellySizer()
    if mode == "cqr":
        return CQRPositionSizer(coverage=coverage)
    from council.sizing.conformal import ConformalPositionSizer

    return ConformalPositionSizer(coverage=coverage)


def stacking_shadow_enabled() -> bool:
    raw = os.getenv("MLCOUNCIL_STACKING_SHADOW", "").strip().lower()
    return raw in ("true", "1", "yes", "on")


class CQRPositionSizer:
    """CQR-style interval sizing scaffold (quantile + split conformal residual).

    Mirrors ``ConformalPositionSizer`` API for drop-in shadow comparison.
    """

    _MIN_MULT: float = 0.2
    _MAX_MULT: float = 2.0

    def __init__(self, coverage: float = 0.85) -> None:
        if not 0.5 < coverage < 1.0:
            raise ValueError(f"coverage must be in (0.5, 1.0), got {coverage}")
        self.coverage = coverage
        self._alpha = 1.0 - coverage
        self._lower_q = self._alpha / 2.0
        self._upper_q = 1.0 - self._alpha / 2.0
        self._residual_lower: float = 0.0
        self._residual_upper: float = 0.0
        self._coef: np.ndarray | None = None
        self._n_features: int | None = None

    def fit(self, X_calib: np.ndarray, y_calib: np.ndarray) -> None:
        from sklearn.linear_model import QuantileRegressor

        X_calib = np.asarray(X_calib, dtype=float)
        y_calib = np.asarray(y_calib, dtype=float)
        if X_calib.ndim != 2:
            raise ValueError(f"X_calib must be 2-D, got shape {X_calib.shape}")
        if len(X_calib) != len(y_calib):
            raise ValueError("X_calib and y_calib length mismatch")

        self._n_features = X_calib.shape[1]
        mid = QuantileRegressor(quantile=0.5, alpha=1.0, solver="highs")
        lo = QuantileRegressor(quantile=self._lower_q, alpha=1.0, solver="highs")
        hi = QuantileRegressor(quantile=self._upper_q, alpha=1.0, solver="highs")
        mid.fit(X_calib, y_calib)
        lo.fit(X_calib, y_calib)
        hi.fit(X_calib, y_calib)
        self._coef = np.asarray(mid.coef_, dtype=float).ravel()
        self._coef_lower = np.asarray(lo.coef_, dtype=float).ravel()
        self._coef_upper = np.asarray(hi.coef_, dtype=float).ravel()
        self._intercept_lower = float(lo.intercept_)
        self._intercept_upper = float(hi.intercept_)

        preds = mid.predict(X_calib)
        residuals = y_calib - preds
        n = len(residuals)
        split = max(1, int(n * 0.5))
        cal_residuals = residuals[split:]
        self._residual_lower = float(np.quantile(cal_residuals, self._lower_q))
        self._residual_upper = float(np.quantile(cal_residuals, self._upper_q))

        logger.debug(
            f"CQRPositionSizer fit: n={n} coverage={self.coverage} "
            f"residual_band=({self._residual_lower:.4f}, {self._residual_upper:.4f})"
        )

    def _predict_quantiles(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._coef is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = np.asarray(X, dtype=float)
        preds = X @ self._coef
        if hasattr(self, "_coef_lower"):
            lower = X @ self._coef_lower + self._intercept_lower + self._residual_lower
            upper = X @ self._coef_upper + self._intercept_upper + self._residual_upper
        else:
            lower = preds + self._residual_lower
            upper = preds + self._residual_upper
        return preds, lower, upper

    def get_intervals(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._predict_quantiles(X)

    def compute_position_multipliers(
        self,
        council_signal: pd.Series,
        X: np.ndarray,
    ) -> pd.Series:
        _, lower, upper = self.get_intervals(np.asarray(X, dtype=float))
        width = upper - lower
        median_w = float(np.median(width))
        if median_w < 1e-9:
            multipliers = np.ones(len(width))
        else:
            width_norm = width / median_w
            multipliers = np.clip(
                np.exp(1.0 - width_norm), self._MIN_MULT, self._MAX_MULT
            )
        return pd.Series(
            multipliers, index=council_signal.index, name="position_multiplier"
        )

    def filter_low_confidence(
        self,
        council_signal: pd.Series,
        X: np.ndarray,
        threshold_percentile: float = 90,
    ) -> pd.Series:
        _, lower, upper = self.get_intervals(np.asarray(X, dtype=float))
        width = upper - lower
        threshold = float(np.percentile(width, threshold_percentile))
        filtered = council_signal.copy().astype(float)
        wide_mask = pd.Series(width >= threshold, index=council_signal.index)
        filtered[wide_mask] = 0.0
        return filtered


class StackingMetaLearner:
    """Meta-learner on base model outputs (XGB scaffold, Ridge fallback)."""

    def __init__(self, *, use_xgb: bool | None = None) -> None:
        if use_xgb is None:
            use_xgb = os.getenv("MLCOUNCIL_STACKING_BACKEND", "ridge").strip().lower() == "xgb"
        self._use_xgb = use_xgb
        self._model: Any = None
        self._feature_names: list[str] = []

    def fit(
        self,
        model_outputs: pd.DataFrame,
        y: pd.Series | np.ndarray,
        extra_features: pd.DataFrame | None = None,
    ) -> "StackingMetaLearner":
        X = model_outputs.copy()
        if extra_features is not None:
            X = pd.concat([X, extra_features], axis=1)
        X = X.fillna(0.0)
        self._feature_names = list(X.columns)
        y_arr = np.asarray(y, dtype=float)

        if self._use_xgb:
            try:
                import xgboost as xgb

                self._model = xgb.XGBRegressor(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.05,
                    random_state=42,
                )
                self._model.fit(X.values, y_arr)
                return self
            except ImportError:
                logger.warning("xgboost not installed; StackingMetaLearner uses Ridge")

        from sklearn.linear_model import Ridge

        self._model = Ridge(alpha=1.0)
        self._model.fit(X.values, y_arr)
        self._use_xgb = False
        return self

    def predict(
        self,
        model_outputs: pd.DataFrame,
        extra_features: pd.DataFrame | None = None,
    ) -> pd.Series:
        if self._model is None:
            raise RuntimeError("StackingMetaLearner not fitted")
        X = model_outputs.reindex(columns=self._feature_names, fill_value=0.0)
        if extra_features is not None:
            X = pd.concat([X, extra_features.reindex(X.index, fill_value=0.0)], axis=1)
            X = X.reindex(columns=self._feature_names, fill_value=0.0)
        preds = self._model.predict(X.values)
        return pd.Series(preds, index=model_outputs.index, name="stacked_signal")

    def save(self, path: str | Path) -> None:
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "use_xgb": self._use_xgb,
                    "feature_names": self._feature_names,
                    "model": self._model,
                },
                fh,
            )

    @classmethod
    def load(cls, path: str | Path) -> "StackingMetaLearner":
        import pickle

        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        meta = cls(use_xgb=bool(payload.get("use_xgb", False)))
        meta._feature_names = list(payload["feature_names"])
        meta._model = payload["model"]
        return meta


def evaluate_cqr_coverage_by_vol_quintile(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    vol_proxy: np.ndarray,
) -> pd.DataFrame:
    """Empirical coverage per volatility quintile (gating metric for T3.2)."""
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    vol_proxy = np.asarray(vol_proxy, dtype=float)
    covered = (y_true >= lower) & (y_true <= upper)
    df = pd.DataFrame({"covered": covered, "vol": vol_proxy})
    df["quintile"] = pd.qcut(df["vol"], q=5, labels=False, duplicates="drop")
    return (
        df.groupby("quintile", observed=True)["covered"]
        .mean()
        .reset_index()
        .rename(columns={"covered": "empirical_coverage"})
    )


def log_stacking_shadow(
    partition_date: str,
    council_signal: pd.Series,
    stacked_signal: pd.Series,
    *,
    out_dir: Path | None = None,
) -> Path:
    """Write shadow stacking comparison parquet (no production effect)."""
    out_dir = out_dir or SHADOW_STACKING_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = council_signal.index.union(stacked_signal.index)
    payload = pd.DataFrame(
        {
            "ticker": idx,
            "council_signal": council_signal.reindex(idx).fillna(0.0).values,
            "stacked_signal": stacked_signal.reindex(idx).fillna(0.0).values,
        }
    )
    path = out_dir / f"{partition_date}.parquet"
    payload.to_parquet(path, index=False)
    logger.info(f"stacking shadow logged → {path} ({len(payload)} rows)")
    return path
