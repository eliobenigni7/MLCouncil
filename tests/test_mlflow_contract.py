from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest


class _FakeRunRecord:
    def __init__(self, run_id: str, run_name: str | None, tags: dict[str, str] | None):
        self.info = SimpleNamespace(run_id=run_id, run_name=run_name)
        self.data = SimpleNamespace(tags=dict(tags or {}), metrics={}, params={})


class _FakeRunContext:
    def __init__(self, store: "_FakeMlflowStore", record: _FakeRunRecord):
        self.store = store
        self.record = record

    def __enter__(self):
        self.store.active_run_id = self.record.info.run_id
        return self.record

    def __exit__(self, exc_type, exc, tb):
        self.store.active_run_id = None
        return False


class _FakeMlflowStore:
    def __init__(self):
        self.active_run_id: str | None = None
        self.counter = 0
        self.runs: list[_FakeRunRecord] = []
        self.tracking_uri: str | None = None
        self.experiment_name: str | None = None
        self.sklearn = SimpleNamespace(log_model=lambda model, name: None)
        self.pyfunc = SimpleNamespace(load_model=lambda uri: {"uri": uri})

    def set_tracking_uri(self, uri: str):
        self.tracking_uri = uri

    def set_experiment(self, name: str):
        self.experiment_name = name

    def start_run(self, run_name: str | None = None, tags: dict[str, str] | None = None):
        self.counter += 1
        record = _FakeRunRecord(f"run-{self.counter}", run_name, tags)
        self.runs.append(record)
        return _FakeRunContext(self, record)

    def end_run(self):
        self.active_run_id = None

    def _current(self) -> _FakeRunRecord:
        for record in reversed(self.runs):
            if record.info.run_id == self.active_run_id:
                return record
        raise AssertionError("No active MLflow run")

    def log_metrics(self, metrics: dict):
        self._current().data.metrics.update(metrics)

    def log_params(self, params: dict):
        self._current().data.params.update(params)

    def log_artifact(self, local_path: str):
        return None

    def log_dict(self, dictionary: dict, artifact_file: str):
        return None

    def register_model(self, model_uri: str, name: str):
        return SimpleNamespace(name=name, version=1, current_stage="None")


class _FakeMlflowClient:
    def __init__(self, store: _FakeMlflowStore):
        self.store = store

    def get_run(self, run_id: str):
        for record in self.store.runs:
            if record.info.run_id == run_id:
                return record
        raise KeyError(run_id)

    def get_experiment_by_name(self, name: str):
        return SimpleNamespace(experiment_id="exp-1", name=name)

    def search_runs(self, experiment_ids: list[str], order_by: list[str] | None = None):
        return list(reversed(self.store.runs))

    def search_model_versions(self, query: str):
        return []

    def transition_model_version_stage(self, name: str, version: int, stage: str):
        return None


def _install_fake_mlflow(monkeypatch):
    from council import mlflow_utils

    store = _FakeMlflowStore()
    monkeypatch.setattr(mlflow_utils, "_MLFLOW_AVAILABLE", True)
    monkeypatch.setattr(mlflow_utils, "mlflow", store, raising=False)
    monkeypatch.setattr(mlflow_utils, "MlflowClient", lambda: _FakeMlflowClient(store), raising=False)
    return mlflow_utils, store


def test_build_run_tags_includes_required_lineage_fields():
    from council.mlflow_utils import build_run_tags

    tags = build_run_tags(
        model_name="lgbm",
        pipeline_run_id="run-001",
        data_version="data-v1",
        feature_version="feat-v2",
        environment="paper",
        model_version="model-v3",
        extra_tags={"stage": "training"},
    )

    assert tags["model_name"] == "lgbm"
    assert tags["pipeline_run_id"] == "run-001"
    assert tags["data_version"] == "data-v1"
    assert tags["feature_version"] == "feat-v2"
    assert tags["environment"] == "paper"
    assert tags["model_version"] == "model-v3"
    assert tags["stage"] == "training"


def test_promotion_gate_rejects_missing_required_metrics():
    from council.mlflow_utils import validate_promotion_gate

    with pytest.raises(ValueError, match="missing required metrics"):
        validate_promotion_gate(
            metrics={"sharpe": 1.1},
            tags={
                "model_name": "lgbm",
                "pipeline_run_id": "run-002",
                "data_version": "data-v2",
                "feature_version": "feat-v2",
                "environment": "paper",
            },
        )


def test_promotion_gate_rejects_high_pbo():
    from council.mlflow_utils import validate_promotion_gate

    with pytest.raises(ValueError, match="pbo"):
        validate_promotion_gate(
            metrics={
                "sharpe": 1.2,
                "max_drawdown": -0.09,
                "turnover": 0.18,
                "oos_sharpe": 0.6,
                "oos_max_drawdown": -0.08,
                "oos_turnover": 0.16,
                "walk_forward_window_count": 4,
                "pbo": 0.75,
            },
            tags={
                "model_name": "lgbm",
                "pipeline_run_id": "run-004",
                "data_version": "data-v4",
                "feature_version": "feat-v4",
                "environment": "paper",
            },
        )


def test_managed_run_logs_required_tags(monkeypatch):
    mlflow_utils, _ = _install_fake_mlflow(monkeypatch)

    tracker = mlflow_utils.MLflowTracker(tracking_uri="file:///tmp/fake-mlruns")

    with tracker.managed_run(
        run_kind="training",
        model_name="lgbm",
        pipeline_run_id="run-003",
        data_version="data-v3",
        feature_version="feat-v3",
        environment="paper",
        model_version="model-v3",
    ) as run:
        tracker.log_metrics({"sharpe": 1.23, "max_drawdown": -0.08, "turnover": 0.21})

    stored = tracker.client.get_run(run.info.run_id)
    assert stored.data.tags["model_name"] == "lgbm"
    assert stored.data.tags["pipeline_run_id"] == "run-003"
    assert stored.data.tags["data_version"] == "data-v3"
    assert stored.data.tags["feature_version"] == "feat-v3"
    assert stored.data.tags["environment"] == "paper"
    assert stored.data.tags["model_version"] == "model-v3"


def test_log_backtest_result_logs_gross_and_net_metrics(monkeypatch):
    from backtest.runner import BacktestResult

    mlflow_utils, store = _install_fake_mlflow(monkeypatch)

    result = BacktestResult(
        fills=pd.DataFrame(
            [{"ts_event": int(pd.Timestamp("2024-01-03T00:00:00Z").value), "last_px": 100.0, "last_qty": 10.0}]
        ),
        equity_curve=pd.Series(
            [100_000.0, 100_485.0],
            index=pd.bdate_range("2024-01-02", periods=2),
            name="equity_net",
        ),
        stats={
            "sharpe": 1.05,
            "gross_sharpe": 1.12,
            "max_drawdown": -0.08,
            "gross_max_drawdown": -0.07,
            "turnover": 0.20,
            "estimated_costs_usd": 15.0,
            "final_equity": 100_485.0,
            "gross_final_equity": 100_500.0,
        },
    )
    result.start_date = "2024-01-02"
    result.end_date = "2024-01-03"
    result.initial_capital = 100_000.0

    mlflow_utils.log_backtest_result(
        result,
        pipeline_run_id="backtest-001",
        data_version="data-v1",
        feature_version="feat-v1",
        environment="paper",
        model_name="council",
    )

    assert store.runs, "Expected one MLflow run for the backtest"
    logged = store.runs[-1].data.metrics
    assert logged["sharpe"] == pytest.approx(1.05)
    assert logged["gross_sharpe"] == pytest.approx(1.12)
    assert logged["gross_max_drawdown"] == pytest.approx(-0.07)
    assert logged["estimated_costs_usd"] == pytest.approx(15.0)


def test_log_intraday_decision_records_runtime_and_agent_metadata(monkeypatch):
    mlflow_utils, store = _install_fake_mlflow(monkeypatch)

    trace = {
        "agent_name": "rule-based-agent",
        "summary": "AAPL favored over MSFT.",
        "confidence": 0.77,
        "prompt_version": "fallback-v1",
        "model_version": "rule-based-v1",
        "provider": "openai",
        "request_id": "resp_123",
    }
    decision = {
        "decision_id": "decision-123",
        "market_session": "regular",
        "schedule_minutes": 15,
        "market_snapshot_version": "polygon-202604091445",
        "tickers": ["AAPL", "MSFT"],
        "execution_intents": [
            {"ticker": "AAPL", "side": "buy", "confidence": 0.77},
            {"ticker": "MSFT", "side": "sell", "confidence": 0.22},
        ],
    }

    mlflow_utils.log_intraday_decision(
        decision=decision,
        agent_trace=trace,
        pipeline_run_id="intraday-runtime-001",
        data_version="market-snapshot-v1",
        feature_version="intraday-features-v1",
        environment="paper",
        model_name="intraday-council",
    )

    assert store.runs, "Expected an MLflow run for intraday decision logging"
    logged = store.runs[-1]
    assert logged.data.tags["run_kind"] == "intraday_runtime"
    assert logged.data.tags["agent_name"] == "rule-based-agent"
    assert logged.data.tags["agent_provider"] == "openai"
    assert logged.data.tags["market_data_provider"] == "polygon-202604091445"
    assert logged.data.tags["agent_request_id"] == "resp_123"
    assert logged.data.metrics["agent_confidence"] == pytest.approx(0.77)
    assert logged.data.metrics["execution_intent_count"] == pytest.approx(2)
