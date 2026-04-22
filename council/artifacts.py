from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from runtime_env import get_runtime_profile

_ROOT = Path(__file__).resolve().parents[1]
_REQUIREMENT_FILES = (
    _ROOT / "requirements.txt",
    _ROOT / "requirements_api.txt",
    _ROOT / "requirements_ci.txt",
)


def manifest_path_for(path: Path | str) -> Path:
    artifact_path = Path(path)
    return Path(f"{artifact_path}.manifest")


def artifact_sha256(path: Path | str) -> str:
    artifact_path = Path(path)
    digest = hashlib.sha256()
    with artifact_path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def dependency_fingerprint() -> str:
    digest = hashlib.sha256()
    for req_path in _REQUIREMENT_FILES:
        if not req_path.exists():
            continue
        digest.update(str(req_path.name).encode("utf-8"))
        digest.update(b"\n")
        digest.update(req_path.read_bytes())
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def _git_sha() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return completed.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _normalize_mapping(values: Mapping[str, Any] | None) -> dict[str, Any]:
    if not values:
        return {}
    normalized: dict[str, Any] = {}
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            normalized[str(key)] = value
        else:
            normalized[str(key)] = str(value)
    return normalized


def build_artifact_manifest(
    path: Path | str,
    *,
    artifact_type: str,
    lineage: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")
    if artifact_path.is_dir():
        raise IsADirectoryError(f"Artifact path points to a directory: {artifact_path}")

    return {
        "artifact_type": str(artifact_type),
        "artifact_path": str(artifact_path),
        "sha256": artifact_sha256(artifact_path),
        "size_bytes": int(artifact_path.stat().st_size),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_profile": get_runtime_profile(),
        "python_version": sys.version.split()[0],
        "git_sha": _git_sha(),
        "dependency_fingerprint": dependency_fingerprint(),
        "lineage": _normalize_mapping(lineage),
        "metadata": _normalize_mapping(metadata),
    }


def write_artifact_manifest(
    path: Path | str,
    *,
    artifact_type: str,
    lineage: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    artifact_path = Path(path)
    manifest = build_artifact_manifest(
        artifact_path,
        artifact_type=artifact_type,
        lineage=lineage,
        metadata=metadata,
    )
    manifest_path = manifest_path_for(artifact_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path
