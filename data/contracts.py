from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any

import pandas as pd
import polars as pl


LINEAGE_COLUMNS = (
    "pipeline_run_id",
    "data_version",
    "feature_version",
    "model_version",
)

_META_COLUMNS = {"ticker", "valid_time", "transaction_time"}


@dataclass(frozen=True)
class AssetContract:
    required_columns: tuple[str, ...]
    unique_key_columns: tuple[str, ...] = ()
    non_null_columns: tuple[str, ...] = ()
    min_rows: int = 0
    allow_empty: bool = False
    min_feature_columns: int = 0


ASSET_CONTRACTS: dict[str, AssetContract] = {
    "raw_ohlcv": AssetContract(
        required_columns=(
            "ticker",
            "valid_time",
            "transaction_time",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
        ),
        unique_key_columns=("ticker", "valid_time"),
        non_null_columns=("ticker", "valid_time", "close", "adj_close"),
        min_rows=1,
    ),
    "raw_news": AssetContract(
        required_columns=(
            "ticker",
            "valid_time",
            "transaction_time",
            "title",
            "published",
            "source",
            "url",
        ),
        unique_key_columns=("ticker", "url"),
        non_null_columns=("ticker", "valid_time", "title", "url"),
        allow_empty=True,
    ),
    "raw_macro": AssetContract(
        required_columns=("valid_time",),
        unique_key_columns=("valid_time",),
        non_null_columns=("valid_time",),
        min_rows=1,
    ),
    "alpha158_features": AssetContract(
        required_columns=("ticker", "valid_time"),
        unique_key_columns=("ticker", "valid_time"),
        non_null_columns=("ticker", "valid_time"),
        min_rows=1,
        min_feature_columns=50,
    ),
    "sentiment_features": AssetContract(
        required_columns=("ticker", "valid_time", "sentiment_score"),
        unique_key_columns=("ticker", "valid_time"),
        non_null_columns=("ticker", "valid_time", "sentiment_score"),
        allow_empty=True,
    ),
    "daily_orders": AssetContract(
        required_columns=(
            "ticker",
            "direction",
            "quantity",
            "target_weight",
            *LINEAGE_COLUMNS,
        ),
        non_null_columns=(
            "ticker",
            "direction",
            "quantity",
            "target_weight",
            *LINEAGE_COLUMNS,
        ),
        allow_empty=True,
    ),
}


def _to_pandas(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, pd.Series):
        return data.to_frame()
    if isinstance(data, pl.DataFrame):
        return data.to_pandas()
    if isinstance(data, dict):
        return pd.DataFrame([data])
    raise TypeError(f"Unsupported contract payload type: {type(data)!r}")


def _count_duplicate_rows(frame: pd.DataFrame, key_columns: tuple[str, ...]) -> int:
    if not key_columns or frame.empty:
        return 0
    return int(frame.duplicated(subset=list(key_columns)).sum())


def _count_nulls(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in columns:
        if column not in frame.columns:
            continue
        counts[column] = int(frame[column].isna().sum())
    return {column: count for column, count in counts.items() if count > 0}


def _ensure_partition_alignment(frame: pd.DataFrame, partition_date: str | None) -> None:
    if partition_date is None or frame.empty or "valid_time" not in frame.columns:
        return

    valid_series = pd.to_datetime(frame["valid_time"], utc=False, errors="coerce")
    valid_dates = valid_series.dt.date.dropna().unique()
    if len(valid_dates) == 0:
        return

    expected = pd.Timestamp(partition_date).date()
    if expected not in set(valid_dates):
        raise ValueError(
            f"valid_time does not include partition_date={partition_date}"
        )


def validate_asset_contract(
    asset_name: str,
    data: Any,
    partition_date: str | None = None,
) -> dict[str, Any]:
    if asset_name not in ASSET_CONTRACTS:
        raise KeyError(f"Unknown asset contract: {asset_name}")

    contract = ASSET_CONTRACTS[asset_name]
    frame = _to_pandas(data)

    missing_columns = [
        column for column in contract.required_columns if column not in frame.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{asset_name}: missing required columns: {missing}")

    if frame.empty and not contract.allow_empty:
        raise ValueError(f"{asset_name}: empty payload is not allowed")

    if len(frame) < contract.min_rows and not (frame.empty and contract.allow_empty):
        raise ValueError(
            f"{asset_name}: expected at least {contract.min_rows} rows, got {len(frame)}"
        )

    null_counts = _count_nulls(frame, contract.non_null_columns)
    if null_counts:
        raise ValueError(f"{asset_name}: null values found in required columns: {null_counts}")

    duplicate_count = _count_duplicate_rows(frame, contract.unique_key_columns)
    if duplicate_count:
        raise ValueError(
            f"{asset_name}: duplicate rows found for keys {contract.unique_key_columns}"
        )

    if contract.min_feature_columns:
        feature_columns = [
            column for column in frame.columns if column not in _META_COLUMNS
        ]
        if len(feature_columns) < contract.min_feature_columns:
            raise ValueError(
                f"{asset_name}: expected at least {contract.min_feature_columns} feature columns, "
                f"got {len(feature_columns)}"
            )

    _ensure_partition_alignment(frame, partition_date)

    return {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": list(frame.columns),
    }


def version_payload(asset_name: str, data: Any, partition_date: str | None = None) -> str:
    frame = _to_pandas(data)
    serializable = {
        "asset": asset_name,
        "partition_date": partition_date,
        "columns": list(frame.columns),
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "preview": frame.head(10).astype(str).to_dict(orient="records"),
    }
    digest = sha256(json.dumps(serializable, sort_keys=True).encode("utf-8")).hexdigest()
    return f"{asset_name}-{digest[:12]}"
