from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_docker_compose_ports_are_overridable():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert '${ADMIN_API_PORT:-8000}:8000' in compose
    assert '${DASHBOARD_PORT:-8501}:8501' in compose
    assert '${DAGSTER_PORT:-3000}:3000' in compose
    assert '${MLFLOW_PORT:-5000}:5000' in compose


def test_dagster_service_has_mlflow_tracking_uri():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert 'MLFLOW_TRACKING_URI=http://mlflow:5000' in compose


def test_mlflow_service_allows_internal_service_host_header():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert '--allowed-hosts localhost,localhost:5000,localhost:${MLFLOW_PORT:-5000},127.0.0.1,127.0.0.1:5000,127.0.0.1:${MLFLOW_PORT:-5000},mlflow,mlflow:5000' in compose


def test_mlflow_service_runs_single_worker():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "python -m mlflow server --workers 1" in compose
