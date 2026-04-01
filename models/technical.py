"""LightGBM technical alpha model with Combinatorial Purged Cross-Validation.

Training protocol
-----------------
1. Split unique dates into n_splits folds.
2. Generate C(n_splits, n_test_folds) (train, test) pairs via CPCV.
3. Apply embargo: remove embargo_days before each test fold from training.
4. Train one LightGBM per fold; select the model with highest mean OOF IC.
5. Log params, fold ICs, ICIR, and SHAP importance to MLflow.

Prediction
----------
predict() returns cross-sectional z-scores per date, suitable for direct
use as position sizing signals by the council ensemble.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from itertools import combinations
from typing import Generator

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import shap
import yaml

from .base import BaseModel

# ---------------------------------------------------------------------------
# CPCV split generator
# ---------------------------------------------------------------------------

_EXCLUDE_COLS = {"ticker", "valid_time", "transaction_time"}


def cpcv_split(
    dates: list,
    n_splits: int = 6,
    embargo_days: int = 5,
    n_test_folds: int = 2,
) -> Generator[tuple[list, list], None, None]:
    """Combinatorial Purged Cross-Validation split generator.

    Splits `dates` into `n_splits` non-overlapping folds.
    Each combination of `n_test_folds` folds becomes a test set;
    the remaining folds minus the embargo zone form the training set.

    Embargo rule: the `embargo_days` calendar-index positions immediately
    before each test fold's start are removed from the training set.
    This prevents information leakage when labels overlap across the
    train/test boundary (e.g., 5-day forward returns).

    Parameters
    ----------
    dates:
        Ordered list of unique dates (any comparable type).
    n_splits:
        Number of folds to split into.
    embargo_days:
        Number of date positions to strip from training before each test fold.
    n_test_folds:
        How many folds to use as test in each combination (default 2).

    Yields
    ------
    (train_dates, test_dates): sorted lists of dates.
    """
    dates = sorted(set(dates))
    n = len(dates)

    if n < n_splits:
        raise ValueError(f"Need at least {n_splits} dates, got {n}.")

    # Build fold index ranges using integer offsets
    fold_idx: list[list[int]] = []
    for i in range(n_splits):
        start = (i * n) // n_splits
        end = ((i + 1) * n) // n_splits
        fold_idx.append(list(range(start, end)))

    for test_combo in combinations(range(n_splits), n_test_folds):
        # Collect test indices
        test_idx: set[int] = set()
        for fi in test_combo:
            test_idx.update(fold_idx[fi])

        # Start with all non-test indices as training candidates
        train_idx: set[int] = set(range(n)) - test_idx

        # Apply embargo: remove embargo_days before each test fold start
        for fi in test_combo:
            fold_start = fold_idx[fi][0]
            embargo_start = max(0, fold_start - embargo_days)
            for j in range(embargo_start, fold_start):
                train_idx.discard(j)

        train_dates = [dates[i] for i in sorted(train_idx)]
        test_dates = [dates[i] for i in sorted(test_idx)]

        if train_dates and test_dates:
            yield train_dates, test_dates


# ---------------------------------------------------------------------------
# LightGBM model
# ---------------------------------------------------------------------------

class TechnicalModel(BaseModel):
    """LightGBM regression model trained with CPCV on Alpha158 features.

    The best fold model (highest OOF Spearman IC) is retained for inference.
    SHAP values are computed on a random sample of training data after fit().
    """

    name = "lgbm"

    def __init__(self, config_path: str = "config/models.yaml") -> None:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self._params: dict = cfg.get("lgbm", {})
        cpcv_cfg = cfg.get("cpcv", {})
        self._n_splits: int = cpcv_cfg.get("n_splits", 6)
        self._embargo_days: int = cpcv_cfg.get("embargo_days", 5)
        self._n_test_folds: int = cpcv_cfg.get("n_test_folds", 2)

        self._model: lgb.LGBMRegressor | None = None
        self._feature_cols: list[str] | None = None
        self._last_trained: str | None = None
        self._n_features: int | None = None
        self._shap_importance: pd.DataFrame | None = None
        self._fold_metrics: list[dict] | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _feature_cols_from(self, features: pl.DataFrame) -> list[str]:
        return [c for c in features.columns if c not in _EXCLUDE_COLS]

    def _to_pandas_aligned(
        self, features: pl.DataFrame, targets: pd.Series
    ) -> pd.DataFrame:
        """Merge features and targets into a single pandas DataFrame."""
        feat_cols = self._feature_cols_from(features)
        df = features.select(["ticker", "valid_time"] + feat_cols).to_pandas()

        # Normalize valid_time to Python date so merges work regardless of
        # how polars Date → pandas conversion lands (date / datetime64 / object).
        if pd.api.types.is_datetime64_any_dtype(df["valid_time"]):
            df["valid_time"] = df["valid_time"].dt.date
        else:
            df["valid_time"] = pd.to_datetime(df["valid_time"]).dt.date

        if isinstance(targets.index, pd.MultiIndex):
            # MultiIndex (ticker, valid_time) — reset to flat for merge
            t = targets.reset_index()
            t.columns = ["ticker", "valid_time", "__target__"]
            # Normalize target index dates too
            if pd.api.types.is_datetime64_any_dtype(t["valid_time"]):
                t["valid_time"] = t["valid_time"].dt.date
            else:
                t["valid_time"] = pd.to_datetime(t["valid_time"], errors="coerce").dt.date
            df = df.merge(t, on=["ticker", "valid_time"], how="inner")
        else:
            # Flat index aligned row-by-row
            df = df.copy()
            df["__target__"] = targets.values

        return df, feat_cols

    @staticmethod
    def _ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Spearman rank correlation (Information Coefficient)."""
        from scipy.stats import spearmanr
        if len(y_true) < 3:
            return 0.0
        corr, _ = spearmanr(y_pred, y_true)
        return float(corr) if not np.isnan(corr) else 0.0

    @staticmethod
    def _zscore_series(s: pd.Series) -> pd.Series:
        std = s.std()
        if std == 0:
            return s - s.mean()
        return (s - s.mean()) / std

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, features: pl.DataFrame, targets: pd.Series) -> None:
        df, feat_cols = self._to_pandas_aligned(features, targets)
        df = df.dropna(subset=["__target__"] + feat_cols)

        self._feature_cols = feat_cols
        self._n_features = len(feat_cols)

        dates = sorted(df["valid_time"].unique().tolist())

        best_ic = -np.inf
        best_model: lgb.LGBMRegressor | None = None
        fold_metrics: list[dict] = []

        # --- MLflow (optional — no crash if server is down) ---
        mlflow_run = None
        try:
            import mlflow
            mlflow_run = mlflow.start_run(run_name=f"lgbm_{date.today()}")
            mlflow.log_params(self._params)
        except Exception:
            mlflow_run = None

        try:
            for fold_idx, (train_dates, test_dates) in enumerate(
                cpcv_split(
                    dates,
                    n_splits=self._n_splits,
                    embargo_days=self._embargo_days,
                    n_test_folds=self._n_test_folds,
                )
            ):
                train_mask = df["valid_time"].isin(set(train_dates))
                test_mask = df["valid_time"].isin(set(test_dates))

                X_train = df.loc[train_mask, feat_cols]
                y_train = df.loc[train_mask, "__target__"]
                X_test = df.loc[test_mask, feat_cols]
                y_test = df.loc[test_mask, "__target__"]

                if len(X_train) < 30 or len(X_test) < 5:
                    continue

                model = lgb.LGBMRegressor(**self._params)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_test, y_test)],
                    callbacks=[
                        lgb.early_stopping(50, verbose=False),
                        lgb.log_evaluation(-1),
                    ],
                )

                fold_ic = self._ic(y_test.values, model.predict(X_test))
                fold_metrics.append({"fold": fold_idx, "ic": fold_ic})

                if fold_ic > best_ic:
                    best_ic = fold_ic
                    best_model = model

            # Fallback: train on full data if no fold produced a valid model
            if best_model is None:
                best_model = lgb.LGBMRegressor(**self._params)
                best_model.fit(df[feat_cols], df["__target__"])
                best_ic = 0.0

            # Aggregate fold metrics
            ic_values = [m["ic"] for m in fold_metrics]
            ic_mean = float(np.mean(ic_values)) if ic_values else 0.0
            ic_std = float(np.std(ic_values)) if ic_values else 0.0
            icir = ic_mean / ic_std if ic_std > 1e-9 else 0.0

            if mlflow_run is not None:
                try:
                    import mlflow
                    mlflow.log_metrics(
                        {"ic_mean": ic_mean, "ic_std": ic_std, "icir": icir}
                    )
                    mlflow.lightgbm.log_model(best_model, "lgbm_model")
                except Exception:
                    pass

            # SHAP importance
            try:
                explainer = shap.TreeExplainer(best_model)
                n_sample = min(500, len(df))
                sample = df[feat_cols].sample(n_sample, random_state=42)
                shap_vals = explainer.shap_values(sample)
                self._shap_importance = (
                    pd.DataFrame(
                        {
                            "feature": feat_cols,
                            "shap_importance": np.abs(shap_vals).mean(axis=0),
                        }
                    )
                    .sort_values("shap_importance", ascending=False)
                    .reset_index(drop=True)
                )
                if mlflow_run is not None:
                    try:
                        import mlflow
                        mlflow.log_dict(
                            self._shap_importance.head(20)
                            .set_index("feature")["shap_importance"]
                            .to_dict(),
                            "shap_importance.json",
                        )
                    except Exception:
                        pass
            except Exception:
                self._shap_importance = None

            self._model = best_model
            self._fold_metrics = fold_metrics
            self._last_trained = datetime.now().isoformat()

        finally:
            if mlflow_run is not None:
                try:
                    import mlflow
                    mlflow.end_run()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, features: pl.DataFrame) -> pd.Series:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        feat_cols = self._feature_cols
        df = features.select(
            ["ticker", "valid_time"] + feat_cols
        ).to_pandas()

        # Normalize valid_time dtype for groupby
        if pd.api.types.is_datetime64_any_dtype(df["valid_time"]):
            df["valid_time"] = df["valid_time"].dt.date
        else:
            df["valid_time"] = pd.to_datetime(df["valid_time"], errors="coerce").dt.date

        X = df[feat_cols].fillna(0.0)
        raw_scores = self._model.predict(X)

        df = df[["ticker", "valid_time"]].copy()
        df["score"] = raw_scores

        # Cross-sectional z-score per date
        df["score_z"] = df.groupby("valid_time")["score"].transform(
            self._zscore_series
        )

        result = pd.Series(
            df["score_z"].values,
            index=pd.Index(df["ticker"].values, name="ticker"),
        )
        return result

    # ------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------

    def get_shap_importance(self) -> pd.DataFrame:
        """Return top-20 features by mean absolute SHAP value.

        Used for drift detection and feature monitoring.
        """
        if self._shap_importance is None:
            raise RuntimeError("No SHAP data. Call fit() first.")
        return self._shap_importance.head(20).copy()
