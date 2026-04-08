from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter

from api.services.dagster_client import DagsterClient
from runtime_env import load_runtime_env, validate_runtime_profile

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


def _monitoring_status() -> str:
    alerts_path = Path("data/monitoring/current_alerts.json")
    return "active" if alerts_path.exists() else "idle"


def _runtime_env_status() -> dict:
    return validate_runtime_profile()


def _latest_operations_status() -> dict:
    operations_dir = Path("data/operations")
    if not operations_dir.exists():
        return {"status": "no_runs", "last_date": None}

    files = sorted(operations_dir.glob("*.json"))
    if not files:
        return {"status": "no_runs", "last_date": None}

    latest = files[-1]
    try:
        payload = json.loads(latest.read_text())
    except (OSError, json.JSONDecodeError):
        return {"status": "error", "last_date": latest.stem}

    trade_status = payload.get("trade_status", "unknown")
    return {
        "status": trade_status,
        "last_date": payload.get("date", latest.stem),
        "paused": payload.get("paused", False),
        "paper": payload.get("paper", False),
    }


@router.get("/health")
async def health():
    data_fresh = _data_freshness()
    arctic = _arctic_store_status()
    monitoring = _monitoring_status()
    runtime = _runtime_env_status()
    operations = _latest_operations_status()

    components = {
        "data_freshness": data_fresh["status"],
        "arctic_store": arctic,
        "monitoring": monitoring,
        "runtime_env": runtime["status"],
        "trading_operations": operations["status"],
    }

    overall = "ok"
    if (
        data_fresh["status"] in ("stale", "no_data")
        or arctic in ("unavailable", "error")
        or runtime["status"] == "invalid"
        or operations["status"] in ("blocked", "degraded", "error")
    ):
        overall = "degraded"

    return {
        "status": overall,
        "version": "0.1.0",
        "components": components,
        "data_freshness": data_fresh,
        "runtime_env": runtime,
        "trading_operations": operations,
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
