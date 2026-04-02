from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter

from api.services.dagster_client import DagsterClient

router = APIRouter(tags=["health"])

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
        fs = FeatureStore()
        symbols = fs.list_symbols()
        return "ok" if symbols else "empty"
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
        "monitoring": "active" if has_alerts else "no_file",
    }

    overall = "ok"
    if data_fresh["status"] == "stale" or arctic == "error":
        overall = "degraded"
    if data_fresh["status"] in ("no_data",) and arctic in ("error", "empty"):
        overall = "unhealthy"

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
