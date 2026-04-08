from __future__ import annotations

import pytest


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


def test_managed_run_logs_required_tags(tmp_path):
    from council.mlflow_utils import MLflowTracker

    tracking_uri = (tmp_path / "mlruns").resolve().as_uri()
    tracker = MLflowTracker(tracking_uri=tracking_uri)

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
