from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

from fastapi import APIRouter
from runtime_env import load_runtime_env

from api.services.dagster_client import DagsterClient

router = APIRouter(tags=["health"])

load_runtime_env()
_dagster_client = DagsterClient()


def _data_freshness() -> dict:
    orders_dir = Path("data/orders")
    if not orders_dir.exists():
        return {"status": "no_data", "last_order_date": None}
    files = sorted(orders_dir.glob("*.parquet"))
    if not files:
        return {"status": "no_data", "last_order_date": None}
    last = files[-1].stem
    try:
        last_dt = datetime.strptime(last, "%Y-%m-%d")
        days_ago = (datetime.now() - last_dt).days
        status = "fresh" if days_ago <= 2 else "stale"
        return {"status": status, "last_order_date": last, "days_ago": days_ago}
    except ValueError:
        return {"status": "unknown", "last_order_date": last}


def _arctic_store_status() -> str:
    try:
        from data.store.arctic_store import FeatureStore
        fs = FeatureStore(uri=os.getenv("ARCTICDB_URI", "lmdb://data/arctic/"))
        symbols = fs.list_symbols()
        return "ok" if symbols else "empty"
    except ImportError:
        return "unavailable"
    except Exception:
        return "error"


@router.get("/health")
async def health():
    data_fresh = _data_freshness()
    arctic = _arctic_store_status()
    alerts_path = Path("data/monitoring/current_alerts.json")
    has_alerts = alerts_path.exists()

    components = {
        "data_freshness": data_fresh["status"],
        "arctic_store": arctic,
        "monitoring": "active" if has_alerts else "idle",
    }

    overall = "ok"
    if data_fresh["status"] in ("stale", "no_data") or arctic in ("unavailable", "error"):
        overall = "degraded"

    return {
        "status": overall,
        "version": "0.1.0",
        "components": components,
        "data_freshness": data_fresh,
    }


@router.get("/health/dagster")
async def dagster_health():
    ps = await _dagster_client.get_last_status()
    return {
        "reachable": ps.status != "unreachable",
        "last_run_id": ps.last_run_id,
        "last_status": ps.status,
        "partition": ps.partition,
        "start_time": ps.start_time,
        "end_time": ps.end_time,
    }
