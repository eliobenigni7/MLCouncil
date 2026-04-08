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
    ):
        self.max_sharpe_degradation = max_sharpe_degradation
        self.max_dd_worsening = max_dd_worsening
        self.min_ic = min_ic

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
        if model_name == "lgbm":
            from models.technical import TechnicalModel
            model = TechnicalModel()
        elif model_name == "hmm":
            from models.regime import RegimeModel
            model = RegimeModel()
        else:
            raise ValueError(f"Unknown model type: {model_name}")

        split_idx = len(features) - validation_window
        if split_idx < 100:
            split_idx = int(len(features) * 0.8)

        train_features = features.iloc[:split_idx]
        train_targets = targets.iloc[:split_idx]
        val_features = features.iloc[split_idx:]
        val_targets = targets.iloc[split_idx:]

        model.fit(train_features, train_targets)

        val_predictions = model.predict(val_features)

        from scipy.stats import spearmanr
        ic_values = []
        for date in val_predictions.index:
            if date in val_targets.index:
                pred = val_predictions.loc[date].dropna()
                actual = val_targets.loc[date].dropna()
                common = pred.index.intersection(actual.index)
                if len(common) >= 3:
                    ic, _ = spearmanr(pred[common], actual[common])
                    if not np.isnan(ic):
                        ic_values.append(ic)

        ic_mean = np.mean(ic_values) if ic_values else 0
        sharpe = ic_mean / (np.std(ic_values) + 1e-9) * np.sqrt(252) if len(ic_values) >= 2 else 0

        returns = val_targets.loc[val_predictions.index]
        equity = (1 + returns).cumprod()
        peak = equity.cummax()
        max_dd = float(((equity - peak) / peak).min())

        metrics = {
            "ic_mean": ic_mean,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "turnover": 0.0,
            "n_validation_days": len(ic_values),
        }

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
                prod_predictions = production_model.predict(features.tail(60))
                prod_ic = []
                prod_targets = targets.tail(60)
                for date in prod_predictions.index:
                    if date in prod_targets.index:
                        pred = prod_predictions.loc[date].dropna()
                        actual = prod_targets.loc[date].dropna()
                        common = pred.index.intersection(actual.index)
                        if len(common) >= 3:
                            ic, _ = spearmanr(pred[common], actual[common])
                            if not np.isnan(ic):
                                prod_ic.append(ic)
                if prod_ic:
                    production_metrics["ic_mean"] = np.mean(prod_ic)
                    production_metrics["sharpe"] = np.mean(prod_ic) / (np.std(prod_ic) + 1e-9) * np.sqrt(252)
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
