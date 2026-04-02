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


def get_universe() -> dict:
    return _read_yaml(CONFIG_DIR / "universe.yaml")


def update_universe(data: dict) -> dict:
    _write_yaml(CONFIG_DIR / "universe.yaml", data)
    return data


def get_models() -> dict:
    return _read_yaml(CONFIG_DIR / "models.yaml")


def get_regime_weights() -> dict:
    return _read_yaml(CONFIG_DIR / "regime_weights.yaml")


def update_regime_weights(data: dict) -> dict:
    _write_yaml(CONFIG_DIR / "regime_weights.yaml", data)
    return data
