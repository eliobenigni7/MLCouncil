"""Load monitoring thresholds from config/monitoring.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MONITORING_PATH = _ROOT / "config" / "monitoring.yaml"

_DEFAULTS = {
    "ic_monitoring": {
        "threshold": 0.005,
        "window_days": 60,
        "consecutive_days": 20,
    },
    "cost_calibration": {
        "divergence_warning_bps": 5.0,
        "divergence_critical_bps": 15.0,
        "consecutive_sessions": 5,
        "min_fills_per_tier": 30,
    },
}


def load_monitoring_config(path: Path = DEFAULT_MONITORING_PATH) -> dict[str, Any]:
    if not path.exists():
        return dict(_DEFAULTS)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    merged = dict(_DEFAULTS)
    for key, value in data.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged
