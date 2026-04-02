from __future__ import annotations

from pathlib import Path

import pandas as pd

ORDERS_DIR = Path("data/orders")


def get_latest_orders() -> pd.DataFrame | None:
    if not ORDERS_DIR.exists():
        return None
    files = sorted(ORDERS_DIR.glob("*.parquet"))
    if not files:
        return None
    return pd.read_parquet(files[-1])


def get_orders_for_date(date_str: str) -> pd.DataFrame | None:
    path = ORDERS_DIR / f"{date_str}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def get_order_dates() -> list[str]:
    if not ORDERS_DIR.exists():
        return []
    return sorted(p.stem for p in ORDERS_DIR.glob("*.parquet"))


def get_current_weights() -> dict[str, float]:
    orders = get_latest_orders()
    if orders is None or "target_weight" not in orders.columns:
        return {}
    return (
        orders[["ticker", "target_weight"]]
        .set_index("ticker")["target_weight"]
        .to_dict()
    )
