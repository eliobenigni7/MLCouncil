"""Automated model retraining pipeline with kill switch evaluation.

This module provides:
- IC degradation monitoring
- Automatic retraining triggers
- Model validation before deployment (kill switch)
- Model registry integration with MLflow

Usage:
    from data.retraining import RetrainingPipeline, ModelValidator

    pipeline = RetrainingPipeline()
    result = pipeline.evaluate_and_retrain_if_needed()
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from runtime_env import load_runtime_env

from backtest.validation import (
    build_purged_walk_forward_splits,
    compute_strategy_returns,
    derive_regime_labels,
    estimate_turnover_from_signals,
    summarize_benchmark_comparison,
    summarize_regime_performance,
    summarize_walk_forward_metrics,
)
from council.mlflow_utils import MLflowTracker, build_run_tags, validate_promotion_gate

_ROOT = Path(__file__).parents[1]
MODELS_DIR = _ROOT / "models"

load_runtime_env()
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


@dataclass
class ICStatistics:
    ic_mean: float
    ic_std: float
    ic_rolling_30d: float
    sharpe_rolling_60d: float
    decay_rate: float
    n_observations: int
    is_degraded: bool
    last_updated: datetime


@dataclass
class ValidationResult:
    is_valid: bool
    candidate_sharpe: float
    production_sharpe: float
    sharpe_degradation: float
    candidate_max_dd: float
    production_max_dd: float
    dd_worsened: bool
    ic_positive: bool
    candidate_oos_sharpe: float = 0.0
    candidate_pbo: float = 1.0
    walk_forward_window_count: int = 0
    messages: list[str] = field(default_factory=list)


@dataclass
class RetrainingResult:
    should_retrain: bool
    reason: str
    ic_stats: Optional[ICStatistics]
    validation_result: Optional[ValidationResult]
    candidate_deployed: bool
    timestamp: datetime


class ICTracker:
    def __init__(
        self,
        ic_history_path: Optional[Path] = None,
        min_history: int = 30,
    ):
        self.ic_history_path = ic_history_path or (MODELS_DIR / "ic_history.json")
        self.min_history = min_history
        self._ic_by_model: dict[str, dict[str, float]] = {}

    def load(self) -> None:
        if self.ic_history_path.exists():
            try:
                data = json.loads(self.ic_history_path.read_text())
                self._ic_by_model = {
                    model: {d: float(v) for d, v in history.items()}
                    for model, history in data.items()
                }
            except Exception:
                self._ic_by_model = {}

    def save(self) -> None:
        self.ic_history_path.parent.mkdir(parents=True, exist_ok=True)
        self.ic_history_path.write_text(json.dumps(self._ic_by_model, indent=2))

    def add_ic(
        self,
        model_name: str,
        date: str,
        ic_value: float,
    ) -> None:
        if model_name not in self._ic_by_model:
            self._ic_by_model[model_name] = {}
        self._ic_by_model[model_name][date] = ic_value

    def get_ic_series(self, model_name: str) -> pd.Series:
        if model_name not in self._ic_by_model:
            return pd.Series(dtype=float)
        history = self._ic_by_model[model_name]
        return pd.Series(history).sort_index()

    def get_statistics(self, model_name: str) -> Optional[ICStatistics]:
        ic_series = self.get_ic_series(model_name)
        if len(ic_series) < self.min_history:
            return None

        recent_30 = ic_series.tail(30)
        recent_60 = ic_series.tail(60)

        ic_mean = float(ic_series.mean())
        ic_std = float(ic_series.std())
        ic_rolling_30d = float(recent_30.mean())

        if len(recent_60) >= 2:
            sharpe = (recent_60.mean() / (recent_60.std() + 1e-9)) * np.sqrt(252)
        else:
            sharpe = 0.0

        if len(ic_series) >= 20:
            old_ic = ic_series[:-10].mean()
            new_ic = ic_series[-10:].mean()
            decay_rate = (old_ic - new_ic) / (abs(old_ic) + 1e-8)
        else:
            decay_rate = 0.0

        is_degraded = (
            ic_rolling_30d < 0.05
            or (decay_rate > 0.3 and ic_mean > 0)
            or ic_series.tail(10).mean() < 0
        )

        return ICStatistics(
            ic_mean=ic_mean,
            ic_std=ic_std,
            ic_rolling_30d=ic_rolling_30d,
            sharpe_rolling_60d=float(sharpe),
            decay_rate=float(decay_rate),
            n_observations=len(ic_series),
            is_degraded=is_degraded,
            last_updated=datetime.now(),
        )

    def get_all_statistics(self) -> dict[str, ICStatistics]:
        stats = {}
        for model_name in self._ic_by_model:
            stat = self.get_statistics(model_name)
            if stat:
                stats[model_name] = stat
        return stats


class ModelValidator:
    def __init__(
        self,
        max_sharpe_degradation: float = 0.2,
        max_dd_worsening: float = 0.05,
        min_ic: float = 0.0,
        min_oos_sharpe: float = 0.0,
        max_pbo: float = 0.5,
        min_walk_forward_windows: int = 1,
    ):
        self.max_sharpe_degradation = max_sharpe_degradation
        self.max_dd_worsening = max_dd_worsening
        self.min_ic = min_ic
        self.min_oos_sharpe = min_oos_sharpe
        self.max_pbo = max_pbo
        self.min_walk_forward_windows = min_walk_forward_windows

    def validate(
        self,
        candidate_metrics: dict,
        production_metrics: dict,
    ) -> ValidationResult:
        messages = []

        candidate_sharpe = candidate_metrics.get("sharpe", 0)
        production_sharpe = production_metrics.get("sharpe", 0)
        sharpe_degradation = production_sharpe - candidate_sharpe

        candidate_max_dd = abs(candidate_metrics.get("max_drawdown", 0))
        production_max_dd = abs(production_metrics.get("max_drawdown", 0))
        dd_worsened = candidate_max_dd > (production_max_dd + self.max_dd_worsening)

        ic_positive = candidate_metrics.get("ic_mean", 0) > self.min_ic
        candidate_oos_sharpe = float(candidate_metrics.get("oos_sharpe", 0.0))
        candidate_pbo = float(candidate_metrics.get("pbo", 1.0))
        walk_forward_window_count = int(candidate_metrics.get("walk_forward_window_count", 0))

        is_valid = True

        if sharpe_degradation > self.max_sharpe_degradation:
            is_valid = False
            messages.append(
                f"CANDIDATE_REJECTED: Sharpe degraded by {sharpe_degradation:.2f} "
                f"(limit: {self.max_sharpe_degradation})"
            )

        if dd_worsened:
            is_valid = False
            messages.append(
                f"CANDIDATE_REJECTED: Max drawdown worsened from "
                f"{production_max_dd:.2%} to {candidate_max_dd:.2%}"
            )

        if not ic_positive:
            is_valid = False
            messages.append(
                f"CANDIDATE_REJECTED: IC mean {candidate_metrics.get('ic_mean', 0):.4f} "
                f"below minimum {self.min_ic}"
            )

        if candidate_oos_sharpe <= self.min_oos_sharpe:
            is_valid = False
            messages.append(
                f"CANDIDATE_REJECTED: OOS Sharpe {candidate_oos_sharpe:.2f} "
                f"below minimum {self.min_oos_sharpe:.2f}"
            )

        if walk_forward_window_count < self.min_walk_forward_windows:
            is_valid = False
            messages.append(
                f"CANDIDATE_REJECTED: walk-forward windows {walk_forward_window_count} "
                f"below minimum {self.min_walk_forward_windows}"
            )

        if candidate_pbo > self.max_pbo:
            is_valid = False
            messages.append(
                f"CANDIDATE_REJECTED: pbo {candidate_pbo:.2f} exceeds limit {self.max_pbo:.2f}"
            )

        if is_valid:
            messages.append("CANDIDATE_ACCEPTED: All validation checks passed")

        return ValidationResult(
            is_valid=is_valid,
            candidate_sharpe=candidate_sharpe,
            production_sharpe=production_sharpe,
            sharpe_degradation=sharpe_degradation,
            candidate_max_dd=candidate_max_dd,
            production_max_dd=production_max_dd,
            dd_worsened=dd_worsened,
            ic_positive=ic_positive,
            candidate_oos_sharpe=candidate_oos_sharpe,
            candidate_pbo=candidate_pbo,
            walk_forward_window_count=walk_forward_window_count,
            messages=messages,
        )


class ModelRegistry:
    def __init__(self, checkpoints_dir: Path = CHECKPOINTS_DIR):
        self.checkpoints_dir = checkpoints_dir
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self._tracker = None

    @property
    def tracker(self):
        if self._tracker is None:
            try:
                self._tracker = MLflowTracker(tracking_uri=MLFLOW_TRACKING_URI)
            except Exception:
                return None
        return self._tracker

    def save_model(
        self,
        model: any,
        name: str,
        version: str,
        metrics: Optional[dict] = None,
        tags: Optional[dict] = None,
        pipeline_run_id: str = "retraining-manual",
        data_version: str = "unknown",
        feature_version: str = "unknown",
        environment: str = "paper",
    ) -> Path:
        path = self.checkpoints_dir / f"{name}_{version}.pkl"

        try:
            import joblib
            joblib.dump(model, path)
        except Exception:
            with open(path, "wb") as f:
                pickle.dump(model, f)

        if self.tracker and metrics:
            try:
                run_tags = build_run_tags(
                    model_name=name,
                    pipeline_run_id=pipeline_run_id,
                    data_version=data_version,
                    feature_version=feature_version,
                    environment=environment,
                    model_version=version,
                    extra_tags=tags or {},
                )
                validate_promotion_gate(metrics=metrics, tags=run_tags)
                with self.tracker.managed_run(
                    run_kind="retraining",
                    model_name=name,
                    pipeline_run_id=pipeline_run_id,
                    data_version=data_version,
                    feature_version=feature_version,
                    environment=environment,
                    model_version=version,
                    extra_tags=tags or {},
                    run_name=f"{name}_{version}",
                ):
                    self.tracker.log_metrics(metrics)
                    self.tracker.log_model(model, name)
            except Exception:
                pass

        return path

    def deploy_latest(self, name: str) -> bool:
        candidates = sorted(
            self.checkpoints_dir.glob(f"{name}_*.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not candidates:
            return False

        latest = candidates[0]
        latest_stem = latest.stem
        latest_path = self.checkpoints_dir / f"{latest_stem.replace(f'{name}_', 'latest_')}.pkl"

        import shutil
        shutil.copy(latest, latest_path.replace(f"_{name}_", "_latest."))

        return True

    def load_latest(self, name: str) -> Optional[any]:
        latest = self.checkpoints_dir / f"{name}_latest.pkl"
        if not latest.exists():
            return None

        try:
            import joblib
            return joblib.load(latest)
        except Exception:
            with open(latest, "rb") as f:
                return pickle.load(f)


class RetrainingPipeline:
    def __init__(
        self,
        ic_tracker: Optional[ICTracker] = None,
        validator: Optional[ModelValidator] = None,
        registry: Optional[ModelRegistry] = None,
        retrain_ic_threshold: float = 0.05,
        max_days_between_retrain: int = 30,
    ):
        self.ic_tracker = ic_tracker or ICTracker()
        self.validator = validator or ModelValidator()
        self.registry = registry or ModelRegistry()
        self.retrain_ic_threshold = retrain_ic_threshold
        self.max_days_between_retrain = max_days_between_retrain

        self._last_train_date: Optional[datetime] = None

    def _build_model(self, model_name: str):
        if model_name == "lgbm":
            from models.technical import TechnicalModel
            return TechnicalModel()
        if model_name == "hmm":
            from models.regime import RegimeModel
            return RegimeModel()
        raise ValueError(f"Unknown model type: {model_name}")

    def _score_predictions(
        self,
        predictions,
        actuals,
    ) -> dict[str, float]:
        from scipy.stats import spearmanr

        ic_values = []
        common_dates = predictions.index.intersection(actuals.index)
        for date in common_dates:
            pred = predictions.loc[date]
            actual = actuals.loc[date]
            if hasattr(pred, "dropna"):
                pred = pred.dropna()
            if hasattr(actual, "dropna"):
                actual = actual.dropna()
            if not hasattr(pred, "index") or not hasattr(actual, "index"):
                continue
            common = pred.index.intersection(actual.index)
            if len(common) < 3:
                continue
            ic, _ = spearmanr(pred[common], actual[common])
            if not np.isnan(ic):
                ic_values.append(float(ic))

        ic_mean = float(np.mean(ic_values)) if ic_values else 0.0
        sharpe = (
            float(ic_mean / (np.std(ic_values) + 1e-9) * np.sqrt(252))
            if len(ic_values) >= 2
            else 0.0
        )

        aligned_returns = actuals.loc[common_dates]
        if isinstance(aligned_returns, pd.DataFrame):
            daily_returns = aligned_returns.mean(axis=1).fillna(0.0)
        else:
            daily_returns = pd.Series(aligned_returns, index=common_dates, dtype=float).fillna(0.0)
        equity = (1 + daily_returns).cumprod()
        peak = equity.cummax()
        max_dd = float(((equity - peak) / peak).min()) if not equity.empty else 0.0

        return {
            "ic_mean": ic_mean,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "turnover": 0.0,
            "n_validation_days": len(ic_values),
        }

    @staticmethod
    def _to_cross_sectional_frame(payload) -> pd.DataFrame:
        """Best-effort conversion to date-indexed cross-sectional DataFrame."""
        if isinstance(payload, pd.DataFrame):
            frame = payload.copy()
        elif isinstance(payload, pd.Series):
            frame = payload.to_frame(name="value")
        else:
            try:
                frame = pd.DataFrame(payload)
            except Exception:
                return pd.DataFrame()

        if frame.empty:
            return frame

        try:
            frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
        except Exception:
            return pd.DataFrame()
        return frame.sort_index()

    @staticmethod
    def _returns_sharpe(returns: pd.Series) -> float:
        clean = returns.dropna()
        if len(clean) < 2:
            return 0.0
        std = float(clean.std())
        if std < 1e-12:
            return 0.0
        return float(clean.mean() / std * np.sqrt(252))

    @staticmethod
    def _returns_max_drawdown(returns: pd.Series) -> float:
        clean = returns.fillna(0.0)
        if clean.empty:
            return 0.0
        equity = (1.0 + clean).cumprod()
        peak = equity.cummax()
        drawdown = equity / peak - 1.0
        return float(drawdown.min())

    def _compute_walk_forward_diagnostics(
        self,
        model_name: str,
        features: pd.DataFrame,
        targets,
        validation_window: int,
    ) -> dict[str, float]:
        unique_dates = pd.DatetimeIndex(pd.to_datetime(pd.Index(features.index).unique())).sort_values()
        if len(unique_dates) < 6:
            return summarize_walk_forward_metrics(pd.DataFrame())

        test_window = min(validation_window, max(2, len(unique_dates) // 5))
        train_window = max(4, len(unique_dates) - (test_window * 2))
        if train_window + test_window > len(unique_dates):
            train_window = max(2, len(unique_dates) - test_window)

        splits = build_purged_walk_forward_splits(
            unique_dates,
            train_window=train_window,
            test_window=test_window,
            step=test_window,
            purge_period=1,
            embargo_period=1,
        )

        rows = []
        oos_signal_frames: list[pd.DataFrame] = []
        oos_target_frames: list[pd.DataFrame] = []
        for window_id, split in enumerate(splits):
            train_mask = (
                (pd.to_datetime(features.index) >= split.train_start)
                & (pd.to_datetime(features.index) <= split.train_end)
            )
            test_mask = (
                (pd.to_datetime(features.index) >= split.test_start)
                & (pd.to_datetime(features.index) <= split.test_end)
            )
            if int(train_mask.sum()) < 3 or int(test_mask.sum()) < 3:
                continue

            wf_model = self._build_model(model_name)
            train_features = features.loc[train_mask]
            train_targets = targets.loc[train_mask]
            test_features = features.loc[test_mask]
            test_targets = targets.loc[test_mask]

            wf_model.fit(train_features, train_targets)
            train_predictions = wf_model.predict(train_features)
            test_predictions = wf_model.predict(test_features)

            train_pred_frame = self._to_cross_sectional_frame(train_predictions)
            test_pred_frame = self._to_cross_sectional_frame(test_predictions)
            train_target_frame = self._to_cross_sectional_frame(train_targets)
            test_target_frame = self._to_cross_sectional_frame(test_targets)

            # Fallback for model outputs that cannot be interpreted as
            # date-indexed cross-sectional matrices.
            if (
                train_pred_frame.empty
                or test_pred_frame.empty
                or train_target_frame.empty
                or test_target_frame.empty
            ):
                train_metrics = self._score_predictions(train_predictions, train_targets)
                test_metrics = self._score_predictions(test_predictions, test_targets)
                rows.append(
                    {
                        "window_id": window_id,
                        "train_sharpe": train_metrics["sharpe"],
                        "test_sharpe": test_metrics["sharpe"],
                        "test_max_drawdown": test_metrics["max_drawdown"],
                        "test_turnover": test_metrics["turnover"],
                    }
                )
                continue

            train_strategy_returns = compute_strategy_returns(
                train_pred_frame,
                train_target_frame,
            )
            test_strategy_returns = compute_strategy_returns(
                test_pred_frame,
                test_target_frame,
            )
            rows.append(
                {
                    "window_id": window_id,
                    "train_sharpe": self._returns_sharpe(train_strategy_returns),
                    "test_sharpe": self._returns_sharpe(test_strategy_returns),
                    "test_max_drawdown": self._returns_max_drawdown(test_strategy_returns),
                    "test_turnover": estimate_turnover_from_signals(test_pred_frame),
                }
            )

            if not test_pred_frame.empty and not test_target_frame.empty:
                oos_signal_frames.append(test_pred_frame)
                oos_target_frames.append(test_target_frame)

        summary = summarize_walk_forward_metrics(pd.DataFrame(rows))

        if oos_signal_frames and oos_target_frames:
            oos_signals = pd.concat(oos_signal_frames).sort_index()
            oos_targets = pd.concat(oos_target_frames).sort_index()
            if oos_signals.index.has_duplicates:
                oos_signals = oos_signals.groupby(level=0).mean()
            if oos_targets.index.has_duplicates:
                oos_targets = oos_targets.groupby(level=0).mean()

            oos_returns = compute_strategy_returns(oos_signals, oos_targets)
            benchmark_input = {"equal_weight": oos_targets.mean(axis=1).astype(float)}
            benchmark_df = summarize_benchmark_comparison(oos_returns, benchmark_input)
            regime_labels = derive_regime_labels(benchmark_input["equal_weight"])
            regime_df = summarize_regime_performance(oos_returns, regime_labels)

            if not benchmark_df.empty:
                eq_row = benchmark_df.loc[
                    benchmark_df["benchmark"] == "equal_weight"
                ]
                if not eq_row.empty:
                    summary["equal_weight_sharpe_delta"] = float(eq_row.iloc[0]["sharpe_delta"])
                    summary["equal_weight_cagr_delta"] = float(eq_row.iloc[0]["cagr_delta"])
            summary["regime_count"] = int(regime_df["regime"].nunique()) if not regime_df.empty else 0

        return summary

    def load_state(self) -> None:
        state_file = MODELS_DIR / "retraining_state.json"
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text())
                self._last_train_date = datetime.fromisoformat(state["last_train_date"])
            except Exception:
                pass
        self.ic_tracker.load()

    def save_state(self) -> None:
        state_file = MODELS_DIR / "retraining_state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "last_train_date": self._last_train_date.isoformat() if self._last_train_date else None,
        }
        state_file.write_text(json.dumps(state, indent=2))
        self.ic_tracker.save()

    def should_retrain(self) -> tuple[bool, str]:
        if self._last_train_date is None:
            return True, "first_training"

        days_since = (datetime.now() - self._last_train_date).days
        if days_since > self.max_days_between_retrain:
            return True, f"max_days_exceeded ({days_since} days)"

        stats = self.ic_tracker.get_all_statistics()
        degraded_models = [m for m, s in stats.items() if s.is_degraded]

        if degraded_models:
            return True, f"ic_degradation: {', '.join(degraded_models)}"

        return False, "no_retrain_needed"

    def train_candidate_model(
        self,
        model_name: str,
        features: pd.DataFrame,
        targets: pd.Series,
        validation_window: int = 60,
    ) -> tuple[any, dict]:
        model = self._build_model(model_name)

        split_idx = len(features) - validation_window
        if split_idx < 100:
            split_idx = int(len(features) * 0.8)

        train_features = features.iloc[:split_idx]
        train_targets = targets.iloc[:split_idx]
        val_features = features.iloc[split_idx:]
        val_targets = targets.iloc[split_idx:]

        model.fit(train_features, train_targets)

        val_predictions = model.predict(val_features)
        metrics = self._score_predictions(val_predictions, val_targets)
        metrics.update(
            self._compute_walk_forward_diagnostics(
                model_name=model_name,
                features=features,
                targets=targets,
                validation_window=validation_window,
            )
        )

        return model, metrics

    def evaluate_and_retrain_if_needed(
        self,
        model_name: str = "lgbm",
        features: Optional[pd.DataFrame] = None,
        targets: Optional[pd.Series] = None,
    ) -> RetrainingResult:
        self.load_state()

        should_train, reason = self.should_retrain()

        all_stats = self.ic_tracker.get_all_statistics()
        model_stats = all_stats.get(model_name)

        validation_result = None
        candidate_deployed = False

        if not should_train:
            return RetrainingResult(
                should_retrain=False,
                reason=reason,
                ic_stats=model_stats,
                validation_result=None,
                candidate_deployed=False,
                timestamp=datetime.now(),
            )

        if features is None or targets is None:
            return RetrainingResult(
                should_retrain=True,
                reason=f"{reason} (but no training data provided)",
                ic_stats=model_stats,
                validation_result=None,
                candidate_deployed=False,
                timestamp=datetime.now(),
            )

        production_model = self.registry.load_latest(model_name)
        production_metrics = {}
        if production_model:
            try:
                prod_features = features.tail(60)
                prod_targets = targets.loc[prod_features.index]
                prod_predictions = production_model.predict(prod_features)
                production_metrics = self._score_predictions(prod_predictions, prod_targets)
            except Exception:
                pass

        candidate_model, candidate_metrics = self.train_candidate_model(
            model_name, features, targets
        )

        validation_result = self.validator.validate(candidate_metrics, production_metrics)

        if validation_result.is_valid:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.registry.save_model(
                candidate_model,
                model_name,
                version,
                metrics=candidate_metrics,
                tags={"retrain_reason": reason},
                pipeline_run_id=f"retraining-{version}",
                data_version=f"{model_name}-targets-{len(targets)}",
                feature_version=f"{model_name}-features-{len(features.columns)}",
            )
            candidate_deployed = self.registry.deploy_latest(model_name)
            self._last_train_date = datetime.now()

        self.save_state()

        return RetrainingResult(
            should_retrain=True,
            reason=reason,
            ic_stats=model_stats,
            validation_result=validation_result,
            candidate_deployed=candidate_deployed,
            timestamp=datetime.now(),
        )


def get_retraining_status() -> dict:
    tracker = ICTracker()
    tracker.load()
    stats = tracker.get_all_statistics()

    return {
        "models": {
            name: {
                "ic_mean": s.ic_mean,
                "ic_rolling_30d": s.ic_rolling_30d,
                "sharpe_60d": s.sharpe_rolling_60d,
                "is_degraded": s.is_degraded,
                "decay_rate": s.decay_rate,
                "n_observations": s.n_observations,
            }
            for name, s in stats.items()
        },
        "last_evaluated": datetime.now().isoformat(),
    }
