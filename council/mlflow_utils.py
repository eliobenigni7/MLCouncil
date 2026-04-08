"""MLflow integration utilities for MLCouncil.

Provides helpers for:
- Model registry
- IC tracking and logging
- Experiment versioning
- Model lineage

Usage:
    from council.mlflow_utils import mlflow_start_run, log_ic_metrics

    with mlflow_start_run("lgbm_model", tags={"env": "production"}):
        log_ic_metrics(ic_series, model_name="lgbm")
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional

import pandas as pd
from runtime_env import load_runtime_env

load_runtime_env()

_MLFLOW_AVAILABLE = True
try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:
    _MLFLOW_AVAILABLE = False


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = "mlcouncil"
REQUIRED_RUN_TAGS = (
    "model_name",
    "pipeline_run_id",
    "data_version",
    "feature_version",
    "environment",
)
REQUIRED_PROMOTION_METRICS = (
    "sharpe",
    "max_drawdown",
    "turnover",
    "oos_sharpe",
    "oos_max_drawdown",
    "oos_turnover",
    "walk_forward_window_count",
    "pbo",
)
MAX_PROMOTION_PBO = 0.50
MIN_WALK_FORWARD_WINDOWS = 1
MIN_OOS_SHARPE = 0.0


def build_run_tags(
    *,
    model_name: str,
    pipeline_run_id: str,
    data_version: str,
    feature_version: str,
    environment: str,
    model_version: Optional[str] = None,
    extra_tags: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    tags = {
        "model_name": str(model_name),
        "pipeline_run_id": str(pipeline_run_id),
        "data_version": str(data_version),
        "feature_version": str(feature_version),
        "environment": str(environment),
    }
    if model_version:
        tags["model_version"] = str(model_version)
    if extra_tags:
        tags.update({str(k): str(v) for k, v in extra_tags.items()})
    return tags


def build_run_name(run_kind: str, model_name: str, environment: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{run_kind}_{model_name}_{environment}_{stamp}"


def validate_promotion_gate(
    *,
    metrics: Mapping[str, float],
    tags: Mapping[str, str],
) -> None:
    missing_tags = [key for key in REQUIRED_RUN_TAGS if not tags.get(key)]
    if missing_tags:
        raise ValueError(f"missing required tags: {', '.join(missing_tags)}")

    missing_metrics = [key for key in REQUIRED_PROMOTION_METRICS if key not in metrics]
    if missing_metrics:
        raise ValueError(f"missing required metrics: {', '.join(missing_metrics)}")

    if float(metrics.get("walk_forward_window_count", 0)) < MIN_WALK_FORWARD_WINDOWS:
        raise ValueError("walk-forward diagnostics missing or insufficient")

    if float(metrics.get("oos_sharpe", 0.0)) <= MIN_OOS_SHARPE:
        raise ValueError("oos_sharpe must be positive for promotion")

    if float(metrics.get("pbo", 1.0)) > MAX_PROMOTION_PBO:
        raise ValueError(f"pbo above allowed threshold: {metrics.get('pbo')}")


class MLflowTracker:
    def __init__(self, tracking_uri: Optional[str] = None):
        if not _MLFLOW_AVAILABLE:
            raise ImportError("mlflow is required. Install with: pip install mlflow")

        self.tracking_uri = tracking_uri or MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        self.client = MlflowClient()

    def start_run(self, run_name: Optional[str] = None, tags: Optional[dict] = None):
        return mlflow.start_run(run_name=run_name, tags=tags)

    def end_run(self):
        mlflow.end_run()

    @contextmanager
    def active_run(self, run_name: Optional[str] = None, tags: Optional[dict] = None):
        with mlflow.start_run(run_name=run_name, tags=tags) as run:
            yield run

    @contextmanager
    def managed_run(
        self,
        *,
        run_kind: str,
        model_name: str,
        pipeline_run_id: str,
        data_version: str,
        feature_version: str,
        environment: str,
        model_version: Optional[str] = None,
        extra_tags: Optional[dict[str, str]] = None,
        run_name: Optional[str] = None,
    ):
        tags = build_run_tags(
            model_name=model_name,
            pipeline_run_id=pipeline_run_id,
            data_version=data_version,
            feature_version=feature_version,
            environment=environment,
            model_version=model_version,
            extra_tags=extra_tags,
        )
        resolved_run_name = run_name or build_run_name(run_kind, model_name, environment)
        with mlflow.start_run(run_name=resolved_run_name, tags=tags) as run:
            yield run

    def log_params(self, params: dict):
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict):
        mlflow.log_metrics(metrics)

    def log_model(self, model, name: str, flavor: str = "sklearn"):
        mlflow.sklearn.log_model(model, name)

    def log_artifact(self, local_path: str):
        mlflow.log_artifact(local_path)

    def log_dict(self, dictionary: dict, artifact_file: str):
        mlflow.log_dict(dictionary, artifact_file)

    def register_model(
        self,
        model_uri: str,
        name: str,
        version: Optional[int] = None,
        stage: Optional[str] = None,
    ) -> dict:
        desc = f"Registered at {datetime.now().isoformat()}"
        mv = mlflow.register_model(model_uri, name)
        if stage:
            client = MlflowClient()
            client.transition_model_version_stage(name, mv.version, stage=stage)
        return {"name": mv.name, "version": mv.version, "stage": mv.current_stage}

    def get_latest_version(self, name: str, stage: Optional[str] = None) -> Optional[dict]:
        try:
            client = MlflowClient()
            if stage:
                mv = client.get_latest_version(name, stage=stage)
            else:
                versions = client.search_model_versions(f"name='{name}'")
                if not versions:
                    return None
                mv = max(versions, key=lambda v: v.version)
            return {"name": mv.name, "version": mv.version, "stage": mv.current_stage}
        except Exception:
            return None

    def load_model(self, name: str, version: Optional[int] = None, stage: Optional[str] = None):
        if version:
            model_uri = f"models:/{name}/{version}"
        elif stage:
            model_uri = f"models:/{name}/{stage}"
        else:
            latest = self.get_latest_version(name)
            if not latest:
                raise ValueError(f"No model found for {name}")
            model_uri = f"models:/{name}/{latest['version']}"
        return mlflow.pyfunc.load_model(model_uri)


def get_tracker() -> MLflowTracker:
    return MLflowTracker()


@contextmanager
def mlflow_start_run(run_name: Optional[str] = None, tags: Optional[dict] = None):
    if not _MLFLOW_AVAILABLE:
        yield None
        return

    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        yield run


def log_ic_metrics(
    ic_series: pd.Series,
    model_name: str,
    rolling_window: int = 60,
):
    if not _MLFLOW_AVAILABLE:
        return

    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    ic_rolling = ic_series.tail(rolling_window).mean() if len(ic_series) >= rolling_window else ic_series.mean()

    metrics = {
        f"{model_name}_ic_mean": ic_mean,
        f"{model_name}_ic_std": ic_std,
        f"{model_name}_ic_rolling_{rolling_window}d": ic_rolling,
        f"{model_name}_ic_count": len(ic_series),
    }
    mlflow.log_metrics(metrics)


def log_signal_metrics(
    signals: pd.DataFrame,
    model_name: str,
):
    if not _MLFLOW_AVAILABLE:
        return

    corr_matrix = signals.corr()
    upper_tri = corr_matrix.where(
        pd.Series(range(len(corr_matrix)))[:, None] > pd.Series(range(len(corr_matrix)))
    )
    max_corr = upper_tri.stack().max() if not upper_tri.stack().empty else 0

    metrics = {
        f"{model_name}_max_correlation": max_corr,
        f"{model_name}_mean_correlation": corr_matrix.values.mean(),
        f"{model_name}_signal_count": len(signals.columns),
    }
    mlflow.log_metrics(metrics)


def setup_mlflow_docker() -> dict:
    return {
        "mlflow": {
            "image": "ghcr.io/mlflow/mlflow:latest",
            "ports": ["5001:5000"],
            "volumes": ["./mlruns:/mlruns"],
            "command": "mlflow server "
            "--backend-store-uri postgresql://mlflow:mlflow@mlflow-db:5432/mlflow "
            "--default-artifact-root s3://mlflow-artifacts/ "
            "--host 0.0.0.0",
            "environment": [
                "AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}",
                "AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}",
            ],
        }
    }


def log_backtest_result(
    result: "BacktestResult",
    run_name: Optional[str] = None,
    *,
    pipeline_run_id: str = "manual-backtest",
    data_version: str = "unknown",
    feature_version: str = "unknown",
    environment: str = "paper",
    model_name: str = "council",
    model_version: Optional[str] = None,
):
    if not _MLFLOW_AVAILABLE:
        return

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    tags = build_run_tags(
        model_name=model_name,
        pipeline_run_id=pipeline_run_id,
        data_version=data_version,
        feature_version=feature_version,
        environment=environment,
        model_version=model_version,
        extra_tags={"run_kind": "backtest"},
    )
    with mlflow.start_run(
        run_name=run_name or build_run_name("backtest", model_name, environment),
        tags=tags,
    ):
        mlflow.log_params({
            "start_date": getattr(result, "start_date", "unknown"),
            "end_date": getattr(result, "end_date", "unknown"),
            "initial_capital": getattr(result, "initial_capital", 0),
        })

        metrics = {
            "sharpe": result.sharpe,
            "max_drawdown": result.max_drawdown,
            "n_trades": result.n_trades,
            "turnover": float(getattr(result, "stats", {}).get("turnover", 0.0)),
            "gross_sharpe": float(getattr(result, "stats", {}).get("gross_sharpe", 0.0)),
            "gross_max_drawdown": float(getattr(result, "stats", {}).get("gross_max_drawdown", 0.0)),
            "gross_cagr": float(getattr(result, "stats", {}).get("gross_cagr", 0.0)),
            "gross_calmar": float(getattr(result, "stats", {}).get("gross_calmar", 0.0)),
            "estimated_costs_usd": float(getattr(result, "stats", {}).get("estimated_costs_usd", 0.0)),
            "gross_final_equity": float(getattr(result, "stats", {}).get("gross_final_equity", 0.0)),
            "net_final_equity": float(getattr(result, "stats", {}).get("final_equity", 0.0)),
        }
        mlflow.log_metrics(metrics)
