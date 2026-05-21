from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_compose(name: str) -> dict:
    text = (ROOT / name).read_text(encoding="utf-8")
    return yaml.safe_load(text)


def test_docker_compose_ports_are_overridable():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "${ADMIN_API_PORT:-8000}:8000" in compose
    assert "${DASHBOARD_PORT:-8501}:8501" in compose
    assert "${DAGSTER_PORT:-3000}:3000" in compose
    assert "${MLFLOW_PORT:-5000}:5000" in compose


def test_dagster_service_has_mlflow_tracking_uri():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "MLFLOW_TRACKING_URI: http://mlflow:5000" in compose


def test_mlflow_service_allows_internal_service_host_header():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert '--allowed-hosts "*"' in compose


def test_mlflow_service_runs_single_worker():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "python -m mlflow server --workers 1" in compose


def test_core_compose_wires_otel_and_manifest():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "MLCOUNCIL_USE_PRODUCTION_MANIFEST" in compose
    assert "MLCOUNCIL_OTEL_ENABLED" in compose
    assert "otel-collector:4318" in compose
    assert "mlcouncil-net" in compose
    assert "models/checkpoints" in compose


def test_observability_compose_uses_shared_network_and_profile():
    data = _load_compose("docker-compose.observability.yml")

    assert data["networks"]["mlcouncil"]["name"] == "mlcouncil-net"
    assert data["services"]["otel-collector"]["profiles"] == ["observability"]
    assert data["services"]["grafana"]["profiles"] == ["observability"]


def test_dockerfile_exposes_dagster_and_checkpoints():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "models/checkpoints" in dockerfile
    assert "MLCOUNCIL_USE_PRODUCTION_MANIFEST" in dockerfile
    assert "3000" in dockerfile
