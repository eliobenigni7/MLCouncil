from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _clear_runtime_env(monkeypatch):
    try:
        from runtime_env import _LOADED_ENV_VALUES

        _LOADED_ENV_VALUES.clear()
    except Exception:
        pass
    for key in [
        "MLCOUNCIL_ENV_PROFILE",
        "MLCOUNCIL_RUNTIME_ENV_PATH",
        "MLCOUNCIL_DOTENV_PATH",
        "ALPACA_API_KEY",
        "ALPACA_PAPER_KEY",
        "ALPACA_SECRET_KEY",
        "ALPACA_PAPER_SECRET",
        "ALPACA_BASE_URL",
        "MLCOUNCIL_MAX_DAILY_ORDERS",
        "MLCOUNCIL_MAX_TURNOVER",
        "MLCOUNCIL_MAX_POSITION_SIZE",
        "MLCOUNCIL_MAX_SECTOR_EXPOSURE",
        "MLCOUNCIL_AUTOMATION_PAUSED",
    ]:
        monkeypatch.delenv(key, raising=False)


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

    _clear_runtime_env(monkeypatch)
    runtime_env_path = tmp_path / "runtime.env"
    runtime_env_path.write_text(
        "ALPACA_BASE_URL=https://paper-api.alpaca.markets\n"
        "MLCOUNCIL_MAX_DAILY_ORDERS=20\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MLCOUNCIL_RUNTIME_ENV_PATH", str(runtime_env_path))
    monkeypatch.delenv("MLCOUNCIL_MAX_DAILY_ORDERS", raising=False)

    baseline = get_config_hash()

    runtime_env_path.write_text(
        "ALPACA_BASE_URL=https://paper-api.alpaca.markets\n"
        "MLCOUNCIL_MAX_DAILY_ORDERS=25\n",
        encoding="utf-8",
    )

    assert get_config_hash() != baseline


def test_config_hash_ignores_runtime_env_path_identity(tmp_path, monkeypatch):
    from runtime_env import get_config_hash

    _clear_runtime_env(monkeypatch)
    runtime_env_a = tmp_path / "runtime-a.env"
    runtime_env_b = tmp_path / "nested" / "runtime-b.env"
    runtime_env_b.parent.mkdir()
    contents = "ALPACA_BASE_URL=https://paper-api.alpaca.markets\nMLCOUNCIL_MAX_DAILY_ORDERS=20\n"
    runtime_env_a.write_text(contents, encoding="utf-8")
    runtime_env_b.write_text(contents, encoding="utf-8")

    monkeypatch.setenv("MLCOUNCIL_RUNTIME_ENV_PATH", str(runtime_env_a))
    monkeypatch.delenv("MLCOUNCIL_MAX_DAILY_ORDERS", raising=False)
    baseline = get_config_hash()

    monkeypatch.setenv("MLCOUNCIL_RUNTIME_ENV_PATH", str(runtime_env_b))
    assert get_config_hash() == baseline


def test_config_hash_changes_with_effective_env_and_secret_values(tmp_path, monkeypatch):
    from runtime_env import get_config_hash

    _clear_runtime_env(monkeypatch)
    runtime_env_path = tmp_path / "runtime.env"
    runtime_env_path.write_text(
        "ALPACA_BASE_URL=https://paper-api.alpaca.markets\n"
        "MLCOUNCIL_MAX_DAILY_ORDERS=20\n"
        "ALPACA_PAPER_KEY=file-key\n",
        encoding="utf-8",
    )
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()
    (secrets_dir / "alpaca_api_key").write_text("secret-one", encoding="utf-8")

    monkeypatch.setenv("MLCOUNCIL_RUNTIME_ENV_PATH", str(runtime_env_path))
    monkeypatch.setenv("MLCOUNCIL_ENV_PROFILE", "paper")
    monkeypatch.setattr("runtime_env._DOCKER_SECRETS_DIR", secrets_dir)

    baseline = get_config_hash()

    monkeypatch.setenv("MLCOUNCIL_MAX_DAILY_ORDERS", "25")
    assert get_config_hash() != baseline

    monkeypatch.setenv("MLCOUNCIL_MAX_DAILY_ORDERS", "20")
    (secrets_dir / "alpaca_api_key").write_text("secret-two", encoding="utf-8")
    assert get_config_hash() != baseline


def test_validate_runtime_profile_accepts_docker_secrets_for_paper_profile(tmp_path, monkeypatch):
    from runtime_env import validate_runtime_profile

    _clear_runtime_env(monkeypatch)
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()
    (secrets_dir / "alpaca_api_key").write_text("paper-key", encoding="utf-8")
    (secrets_dir / "alpaca_secret_key").write_text("paper-secret", encoding="utf-8")

    runtime_env_path = tmp_path / "runtime.env"
    runtime_env_path.write_text(
        "ALPACA_BASE_URL=https://paper-api.alpaca.markets\n"
        "MLCOUNCIL_MAX_DAILY_ORDERS=20\n"
        "MLCOUNCIL_MAX_TURNOVER=0.30\n"
        "MLCOUNCIL_MAX_POSITION_SIZE=0.10\n"
        "MLCOUNCIL_AUTOMATION_PAUSED=false\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("MLCOUNCIL_ENV_PROFILE", "paper")
    monkeypatch.setenv("MLCOUNCIL_RUNTIME_ENV_PATH", str(runtime_env_path))
    monkeypatch.setattr("runtime_env._DOCKER_SECRETS_DIR", secrets_dir)

    result = validate_runtime_profile()

    assert result["valid"] is True
    assert result["status"] == "valid"
    assert result["missing"] == []


def test_config_hash_includes_project_dotenv_and_aliases(tmp_path, monkeypatch):
    from runtime_env import get_config_hash

    _clear_runtime_env(monkeypatch)
    project_dotenv = tmp_path / ".env"
    project_dotenv.write_text(
        "ALPACA_PAPER_KEY=paper-key\n"
        "ALPACA_PAPER_SECRET=paper-secret\n"
        "ALPACA_BASE_URL=https://paper-api.alpaca.markets\n",
        encoding="utf-8",
    )
    runtime_env_path = tmp_path / "runtime.env"
    runtime_env_path.write_text("MLCOUNCIL_MAX_DAILY_ORDERS=20\n", encoding="utf-8")

    monkeypatch.setenv("MLCOUNCIL_DOTENV_PATH", str(project_dotenv))
    monkeypatch.setenv("MLCOUNCIL_RUNTIME_ENV_PATH", str(runtime_env_path))
    monkeypatch.setenv("MLCOUNCIL_ENV_PROFILE", "paper")

    baseline = get_config_hash()

    project_dotenv.write_text(
        "ALPACA_API_KEY=paper-key\n"
        "ALPACA_SECRET_KEY=paper-secret\n"
        "ALPACA_BASE_URL=https://paper-api.alpaca.markets\n",
        encoding="utf-8",
    )

    assert get_config_hash() == baseline


def test_config_hash_reflects_process_env_over_files(tmp_path, monkeypatch):
    from runtime_env import get_config_hash

    _clear_runtime_env(monkeypatch)
    project_dotenv = tmp_path / ".env"
    project_dotenv.write_text("MLCOUNCIL_MAX_TURNOVER=0.30\n", encoding="utf-8")
    runtime_env_path = tmp_path / "runtime.env"
    runtime_env_path.write_text("MLCOUNCIL_MAX_TURNOVER=0.30\n", encoding="utf-8")

    monkeypatch.setenv("MLCOUNCIL_DOTENV_PATH", str(project_dotenv))
    monkeypatch.setenv("MLCOUNCIL_RUNTIME_ENV_PATH", str(runtime_env_path))

    baseline = get_config_hash()
    monkeypatch.setenv("MLCOUNCIL_MAX_TURNOVER", "0.40")

    assert get_config_hash() != baseline


def test_config_hash_ignores_stale_loaded_env_values(tmp_path, monkeypatch):
    from runtime_env import get_config_hash, load_runtime_env

    _clear_runtime_env(monkeypatch)
    project_dotenv = tmp_path / ".env"
    project_dotenv.write_text("ALPACA_BASE_URL=https://paper-api.alpaca.markets\n", encoding="utf-8")
    runtime_env_path = tmp_path / "runtime.env"
    runtime_env_path.write_text("MLCOUNCIL_MAX_DAILY_ORDERS=20\n", encoding="utf-8")

    monkeypatch.setenv("MLCOUNCIL_DOTENV_PATH", str(project_dotenv))
    monkeypatch.setenv("MLCOUNCIL_RUNTIME_ENV_PATH", str(runtime_env_path))

    load_runtime_env()

    baseline = get_config_hash()

    runtime_env_path.write_text("MLCOUNCIL_MAX_DAILY_ORDERS=25\n", encoding="utf-8")

    assert get_config_hash() != baseline


def test_config_hash_ignores_stale_loaded_legacy_alias_values(tmp_path, monkeypatch):
    from runtime_env import get_config_hash, load_runtime_env

    _clear_runtime_env(monkeypatch)
    project_dotenv = tmp_path / ".env"
    project_dotenv.write_text(
        "ALPACA_PAPER_KEY=legacy-key-1\n"
        "ALPACA_PAPER_SECRET=legacy-secret-1\n"
        "ALPACA_BASE_URL=https://paper-api.alpaca.markets\n",
        encoding="utf-8",
    )
    runtime_env_path = tmp_path / "runtime.env"
    runtime_env_path.write_text("MLCOUNCIL_MAX_DAILY_ORDERS=20\n", encoding="utf-8")

    monkeypatch.setenv("MLCOUNCIL_DOTENV_PATH", str(project_dotenv))
    monkeypatch.setenv("MLCOUNCIL_RUNTIME_ENV_PATH", str(runtime_env_path))

    load_runtime_env()

    baseline = get_config_hash()

    project_dotenv.write_text(
        "ALPACA_PAPER_KEY=legacy-key-2\n"
        "ALPACA_PAPER_SECRET=legacy-secret-2\n"
        "ALPACA_BASE_URL=https://paper-api.alpaca.markets\n",
        encoding="utf-8",
    )

    assert get_config_hash() != baseline


def test_config_hash_ignores_unrelated_environment_noise(tmp_path, monkeypatch):
    from runtime_env import get_config_hash

    _clear_runtime_env(monkeypatch)
    runtime_env_path = tmp_path / "runtime.env"
    runtime_env_path.write_text(
        "ALPACA_BASE_URL=https://paper-api.alpaca.markets\n"
        "MLCOUNCIL_MAX_DAILY_ORDERS=20\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MLCOUNCIL_RUNTIME_ENV_PATH", str(runtime_env_path))

    baseline = get_config_hash()
    monkeypatch.setenv("UNRELATED_TEST_FLAG", "1")
    monkeypatch.setenv("ANOTHER_NOISE_VAR", "abc")

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
