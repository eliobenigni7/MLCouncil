from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from data.contracts import LINEAGE_COLUMNS, version_payload


def build_pipeline_run_id(context: Any, partition_date: str) -> str:
    run_id = getattr(context, "run_id", None)
    if run_id:
        return str(run_id)
    return f"manual-{partition_date}"


def checkpoint_version(checkpoint: Path, fallback: str) -> str:
    if not checkpoint.exists():
        return fallback
    digest = sha256(checkpoint.read_bytes()).hexdigest()
    return f"{checkpoint.stem}-{digest[:12]}"


def merge_versions(*versions: str) -> str:
    unique_versions = [version for version in versions if version]
    if not unique_versions:
        return "unknown"
    digest = sha256("|".join(sorted(set(unique_versions))).encode("utf-8")).hexdigest()
    return digest[:12]


def attach_lineage(series: pd.Series, **lineage: str) -> pd.Series:
    normalized = {key: str(value) for key, value in lineage.items() if key in LINEAGE_COLUMNS}
    series.attrs.update(normalized)
    return series


def extract_lineage(payload: Any) -> dict[str, str]:
    attrs = getattr(payload, "attrs", {}) or {}
    return {
        key: str(attrs[key])
        for key in LINEAGE_COLUMNS
        if key in attrs and attrs[key] is not None
    }


def dataframe_lineage_columns(lineage: Mapping[str, str], n_rows: int) -> dict[str, list[str]]:
    normalized = {key: str(lineage.get(key, "unknown")) for key in LINEAGE_COLUMNS}
    return {key: [value] * n_rows for key, value in normalized.items()}


def build_feature_lineage(
    *,
    asset_name: str,
    payload: Any,
    context: Any,
    partition_date: str,
    model_version: str,
    data_payload: Any | None = None,
) -> dict[str, str]:
    data_source = payload if data_payload is None else data_payload
    return {
        "pipeline_run_id": build_pipeline_run_id(context, partition_date),
        "data_version": version_payload(f"{asset_name}-data", data_source, partition_date),
        "feature_version": version_payload(asset_name, payload, partition_date),
        "model_version": model_version,
    }


def merge_lineage(*payloads: Any, context: Any, partition_date: str, model_version: str) -> dict[str, str]:
    sources = [extract_lineage(payload) for payload in payloads if payload is not None]
    return {
        "pipeline_run_id": build_pipeline_run_id(context, partition_date),
        "data_version": merge_versions(*[source.get("data_version", "") for source in sources]),
        "feature_version": merge_versions(*[source.get("feature_version", "") for source in sources]),
        "model_version": merge_versions(model_version, *[source.get("model_version", "") for source in sources]),
    }


def lineage_metadata(lineage: Mapping[str, str]) -> dict[str, str]:
    return {key: str(lineage.get(key, "unknown")) for key in LINEAGE_COLUMNS}


def lineage_artifact_payload(lineage: Mapping[str, str], **extra: Any) -> dict[str, Any]:
    payload = {**lineage_metadata(lineage), **extra}
    payload["lineage_signature"] = sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    return payload
