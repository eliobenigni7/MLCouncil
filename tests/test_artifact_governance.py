from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_write_artifact_manifest_records_sha256(tmp_path, monkeypatch):
    from council import artifacts as artifact_mod

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
    assert payload["lineage"]["pipeline_run_id"] == "run-001"
    assert payload["metadata"]["partition_date"] == "2026-04-22"
    assert payload["size_bytes"] == payload_path.stat().st_size


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
