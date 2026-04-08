from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_DIR = Path("config")


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _flatten_tickers(universe: dict) -> list[str]:
    tickers = universe.get("tickers")
    if isinstance(tickers, list):
        return tickers

    flattened: list[str] = []
    for key, value in universe.items():
        if key == "settings" or not isinstance(value, list):
            continue
        flattened.extend(str(ticker) for ticker in value)

    # Deduplicate while preserving order.
    return list(dict.fromkeys(flattened))


def _normalize_universe_config(data: dict) -> dict:
    universe = data.get("universe", {})
    settings = data.get("settings")
    if settings is None and isinstance(universe, dict):
        settings = universe.get("settings", {})

    normalized = dict(data)
    normalized["universe"] = {"tickers": _flatten_tickers(universe)}
    normalized["settings"] = settings or {}
    return normalized


def get_universe() -> dict:
    return _normalize_universe_config(_read_yaml(CONFIG_DIR / "universe.yaml"))


def update_universe(data: dict) -> dict:
    if "universe" not in data:
        raise ValueError("Payload must contain a 'universe' key.")
    universe_val = data["universe"]
    if not isinstance(universe_val, dict):
        raise ValueError("'universe' must be a dict (with bucket keys or a 'tickers' list).")
    # Ensure at least one valid ticker list is present.
    tickers = _flatten_tickers(universe_val)
    if not tickers:
        raise ValueError("No tickers found in 'universe' payload.")
    if not all(isinstance(t, str) and t.isalnum() for t in tickers):
        raise ValueError("All tickers must be alphanumeric strings.")
    _write_yaml(CONFIG_DIR / "universe.yaml", data)
    return data


def get_models() -> dict:
    return _read_yaml(CONFIG_DIR / "models.yaml")


def get_regime_weights() -> dict:
    return _read_yaml(CONFIG_DIR / "regime_weights.yaml")


def update_regime_weights(data: dict) -> dict:
    _write_yaml(CONFIG_DIR / "regime_weights.yaml", data)
    return data
