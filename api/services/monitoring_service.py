from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import dotenv_values

from runtime_env import LEGACY_ENV_ALIASES, get_runtime_env_path, load_runtime_env

ALERTS_DIR = Path("data/alerts")
CURRENT_ALERTS_PATH = Path("data/monitoring/current_alerts.json")
RUNTIME_ENV_PATH = get_runtime_env_path()

SETTINGS_FIELDS = [
    {
        "key": "OPENAI_API_KEY",
        "label": "OpenAI API Key",
        "description": "Needed for LLM-backed components and research flows.",
        "secret": True,
        "placeholder": "sk-...",
    },
    {
        "key": "POLYGON_API_KEY",
        "label": "Polygon API Key",
        "description": "Market data provider for live and historical integrations.",
        "secret": True,
        "placeholder": "polygon-api-key",
    },
    {
        "key": "ALPACA_PAPER_KEY",
        "label": "Alpaca Paper Key",
        "description": "Paper trading API key used by execution services.",
        "secret": True,
        "placeholder": "alpaca-paper-key",
    },
    {
        "key": "ALPACA_PAPER_SECRET",
        "label": "Alpaca Paper Secret",
        "description": "Paper trading secret used by execution services.",
        "secret": True,
        "placeholder": "alpaca-paper-secret",
    },
    {
        "key": "ALPACA_LIVE_KEY",
        "label": "Alpaca Live Key",
        "description": "Optional live trading API key.",
        "secret": True,
        "placeholder": "alpaca-live-key",
    },
    {
        "key": "ALPACA_LIVE_SECRET",
        "label": "Alpaca Live Secret",
        "description": "Optional live trading secret.",
        "secret": True,
        "placeholder": "alpaca-live-secret",
    },
    {
        "key": "ALPACA_BASE_URL",
        "label": "Alpaca Base URL",
        "description": "Paper or live endpoint base URL.",
        "secret": False,
        "placeholder": "https://paper-api.alpaca.markets",
    },
    {
        "key": "ALERT_EMAIL",
        "label": "Alert Email",
        "description": "Sender email used for monitoring notifications.",
        "secret": False,
        "placeholder": "alerts@example.com",
    },
    {
        "key": "SMTP_PASSWORD",
        "label": "SMTP Password",
        "description": "SMTP or app password for alert emails.",
        "secret": True,
        "placeholder": "smtp-password",
    },
    {
        "key": "ARCTICDB_URI",
        "label": "ArcticDB URI",
        "description": "Shared feature store location.",
        "secret": False,
        "placeholder": "lmdb://data/arctic/",
    },
    {
        "key": "MLFLOW_TRACKING_URI",
        "label": "MLflow Tracking URI",
        "description": "MLflow endpoint for experiment tracking.",
        "secret": False,
        "placeholder": "http://localhost:5000",
    },
    {
        "key": "MLCOUNCIL_API_KEY",
        "label": "Admin API Key",
        "description": "Optional key required by private admin endpoints.",
        "secret": True,
        "placeholder": "admin-api-key",
    },
]

_LEGACY_WRITE_ALIASES = {
    "ALPACA_PAPER_KEY": "ALPACA_API_KEY",
    "ALPACA_PAPER_SECRET": "ALPACA_SECRET_KEY",
}


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


def get_runtime_settings() -> dict:
    load_runtime_env()
    values = _read_runtime_env_file()
    settings = []

    for field in SETTINGS_FIELDS:
        value = _resolve_setting_value(field["key"], values)
        settings.append(
            {
                **field,
                "value": value,
                "configured": bool(value),
            }
        )

    return {
        "path": str(RUNTIME_ENV_PATH),
        "settings": settings,
    }


# Keys that must not be overwritten at runtime via the API — changing them
# mid-process would bypass auth or break in-flight requests.
_IMMUTABLE_KEYS = frozenset({"MLCOUNCIL_API_KEY"})


def update_runtime_settings(updates: dict[str, str | None]) -> dict:
    current = _read_runtime_env_file()

    for field in SETTINGS_FIELDS:
        key = field["key"]
        if key not in updates:
            continue
        if key in _IMMUTABLE_KEYS:
            continue  # Silently skip — callers should not change auth keys at runtime.

        value = (updates.get(key) or "").strip()
        if value:
            current[key] = value
            os.environ[key] = value
        else:
            current.pop(key, None)
            os.environ.pop(key, None)

        legacy_key = _LEGACY_WRITE_ALIASES.get(key)
        if legacy_key:
            if value:
                current[legacy_key] = value
                os.environ[legacy_key] = value
            else:
                current.pop(legacy_key, None)
                os.environ.pop(legacy_key, None)

    _write_runtime_env_file(current)
    load_runtime_env()
    return get_runtime_settings()


def _read_runtime_env_file() -> dict[str, str]:
    if not RUNTIME_ENV_PATH.exists():
        return {}
    return {
        key: value
        for key, value in dotenv_values(RUNTIME_ENV_PATH).items()
        if value is not None
    }


def _resolve_setting_value(key: str, values: dict[str, str]) -> str:
    if key in values:
        return values[key]

    legacy_key = LEGACY_ENV_ALIASES.get(key)
    if legacy_key and legacy_key in values:
        return values[legacy_key]

    if os.getenv(key):
        return os.environ[key]

    if legacy_key and os.getenv(legacy_key):
        return os.environ[legacy_key]

    return ""


def _write_runtime_env_file(values: dict[str, str]) -> None:
    RUNTIME_ENV_PATH.parent.mkdir(parents=True, exist_ok=True)

    ordered_keys = [field["key"] for field in SETTINGS_FIELDS]
    extras = sorted(key for key in values.keys() if key not in ordered_keys)
    lines: list[str] = []

    for key in ordered_keys + extras:
        if key not in values:
            continue
        lines.append(f"{key}={_serialize_env_value(values[key])}")

    RUNTIME_ENV_PATH.write_text("\n".join(lines) + ("\n" if lines else ""))


def _serialize_env_value(value: str) -> str:
    if any(ch.isspace() or ch in {'"', "#"} for ch in value):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value
