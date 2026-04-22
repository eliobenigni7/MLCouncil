from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_write_artifact_manifest_records_sha256(tmp_path, monkeypatch):
    from council import artifacts as artifact_mod
    from runtime_env import get_config_hash

    payload_path = tmp_path / "daily_orders.parquet"
    payload_path.write_bytes(b"parquet-bytes-for-test")

    monkeypatch.setenv("MLCOUNCIL_ENV_PROFILE", "paper")
    manifest_path = artifact_mod.write_artifact_manifest(
        payload_path,
        artifact_type="execution_orders",
        lineage={"pipeline_run_id": "run-001", "model_version": "model-v1"},
        metadata={"partition_date": "2026-04-22"},
    )

    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text())
    expected_sha = hashlib.sha256(payload_path.read_bytes()).hexdigest()

    assert payload["artifact_type"] == "execution_orders"
    assert payload["sha256"] == expected_sha
    assert payload["runtime_profile"] == "paper"
    assert payload["config_hash"] == get_config_hash()
    assert payload["lineage"]["pipeline_run_id"] == "run-001"
    assert payload["metadata"]["partition_date"] == "2026-04-22"
    assert payload["size_bytes"] == payload_path.stat().st_size


def test_config_hash_changes_with_runtime_env_values(tmp_path, monkeypatch):
    from runtime_env import get_config_hash

    runtime_env_path = tmp_path / "runtime.env"
    runtime_env_path.write_text(
        "ALPACA_BASE_URL=https://paper-api.alpaca.markets\n"
        "MLCOUNCIL_MAX_DAILY_ORDERS=20\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MLCOUNCIL_RUNTIME_ENV_PATH", str(runtime_env_path))

    baseline = get_config_hash()

    runtime_env_path.write_text(
        "ALPACA_BASE_URL=https://paper-api.alpaca.markets\n"
        "MLCOUNCIL_MAX_DAILY_ORDERS=25\n",
        encoding="utf-8",
    )

    assert get_config_hash() != baseline


def test_config_hash_ignores_runtime_env_path_identity(tmp_path, monkeypatch):
    from runtime_env import get_config_hash

    runtime_env_a = tmp_path / "runtime-a.env"
    runtime_env_b = tmp_path / "nested" / "runtime-b.env"
    runtime_env_b.parent.mkdir()
    contents = "ALPACA_BASE_URL=https://paper-api.alpaca.markets\nMLCOUNCIL_MAX_DAILY_ORDERS=20\n"
    runtime_env_a.write_text(contents, encoding="utf-8")
    runtime_env_b.write_text(contents, encoding="utf-8")

    monkeypatch.setenv("MLCOUNCIL_RUNTIME_ENV_PATH", str(runtime_env_a))
    baseline = get_config_hash()

    monkeypatch.setenv("MLCOUNCIL_RUNTIME_ENV_PATH", str(runtime_env_b))
    assert get_config_hash() == baseline


def test_dependency_fingerprint_includes_requirements_lock(tmp_path, monkeypatch):
    from council import artifacts as artifact_mod

    req_dir = tmp_path
    (req_dir / "requirements.txt").write_text("alpha==1\n", encoding="utf-8")
    (req_dir / "requirements_api.txt").write_text("beta==1\n", encoding="utf-8")
    (req_dir / "requirements_ci.txt").write_text("gamma==1\n", encoding="utf-8")
    lock_path = req_dir / "requirements_lock.txt"
    lock_path.write_text("lock==1\n", encoding="utf-8")

    monkeypatch.setattr(artifact_mod, "_ROOT", req_dir)
    monkeypatch.setattr(
        artifact_mod,
        "_REQUIREMENT_FILES",
        (
            req_dir / "requirements.txt",
            req_dir / "requirements_api.txt",
            req_dir / "requirements_ci.txt",
            req_dir / "requirements_lock.txt",
        ),
    )

    baseline = artifact_mod.dependency_fingerprint()
    lock_path.write_text("lock==2\n", encoding="utf-8")

    assert artifact_mod.dependency_fingerprint() != baseline


def test_write_artifact_manifest_raises_when_artifact_missing(tmp_path):
    from council.artifacts import write_artifact_manifest

    missing = tmp_path / "missing.bin"
    try:
        write_artifact_manifest(missing, artifact_type="missing")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_manifest_sidecar_suffix_is_stable():
    from council.artifacts import manifest_path_for

    payload = Path("data/operations/2026-04-22.json")
    manifest = manifest_path_for(payload)

    assert str(manifest).endswith(".json.manifest")
