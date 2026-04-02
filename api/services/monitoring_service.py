from __future__ import annotations

import json
from pathlib import Path

ALERTS_DIR = Path("data/alerts")
CURRENT_ALERTS_PATH = Path("data/monitoring/current_alerts.json")


def get_current_alerts() -> list[dict]:
    try:
        from council.alerts import load_current_alerts
        return load_current_alerts()
    except FileNotFoundError:
        return []
    except ImportError:
        if CURRENT_ALERTS_PATH.exists():
            return json.loads(CURRENT_ALERTS_PATH.read_text())
        return []


def get_alert_history(limit: int = 30) -> list[dict]:
    if not ALERTS_DIR.exists():
        return []
    files = sorted(ALERTS_DIR.glob("*.json"), reverse=True)[:limit]
    all_alerts = []
    for f in files:
        try:
            data = json.loads(f.read_text())
            all_alerts.extend(data if isinstance(data, list) else [data])
        except (json.JSONDecodeError, KeyError):
            continue
    return all_alerts
