from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

ORDERS_DIR = Path("data/orders")

_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_date_str(date_str: str) -> str:
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")
    return date_str


def _safe_path_join(base: Path, date_str: str, extension: str) -> Path:
    validated = _validate_date_str(date_str)
    safe_name = f"{validated}{extension}"
    resolved = (base / safe_name).resolve()
    if not str(resolved).startswith(str(base.resolve())):
        raise ValueError("Path traversal attempt detected")
    return resolved


def get_latest_orders() -> pd.DataFrame | None:
    if not ORDERS_DIR.exists():
        return None
    files = sorted(ORDERS_DIR.glob("*.parquet"))
    if not files:
        return None
    return pd.read_parquet(files[-1])


def get_orders_for_date(date_str: str) -> pd.DataFrame | None:
    path = _safe_path_join(ORDERS_DIR, date_str, ".parquet")
    if not path.exists():
        return None
    return pd.read_parquet(path)


def get_order_dates() -> list[str]:
    if not ORDERS_DIR.exists():
        return []
    dates = []
    for p in ORDERS_DIR.glob("*.parquet"):
        stem = p.stem
        try:
            _validate_date_str(stem)
            dates.append(stem)
        except ValueError:
            continue
    return sorted(dates)


def get_current_weights() -> dict[str, float]:
    orders = get_latest_orders()
    if orders is None or "target_weight" not in orders.columns:
        return {}
    return (
        orders[["ticker", "target_weight"]]
        .set_index("ticker")["target_weight"]
        .to_dict()
    )
