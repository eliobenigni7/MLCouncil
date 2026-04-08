from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

_ROOT = Path(__file__).resolve().parent
_CONFIG_DIR = _ROOT / "config"
_DEFAULT_RUNTIME_ENV_PATH = _CONFIG_DIR / "runtime.env"

LEGACY_ENV_ALIASES = {
    "ALPACA_PAPER_KEY": "ALPACA_API_KEY",
    "ALPACA_PAPER_SECRET": "ALPACA_SECRET_KEY",
}


def get_runtime_profile() -> str:
    profile = os.getenv("MLCOUNCIL_ENV_PROFILE", "local").strip().lower()
    return profile or "local"


def get_runtime_env_path() -> Path:
    override = os.getenv("MLCOUNCIL_RUNTIME_ENV_PATH")
    if override:
        return Path(override)

    profile_path = _CONFIG_DIR / f"runtime.{get_runtime_profile()}.env"
    if profile_path.exists():
        return profile_path
    return _DEFAULT_RUNTIME_ENV_PATH


def load_runtime_env(*, override: bool = False) -> dict[str, str]:
    path = get_runtime_env_path()
    loaded: dict[str, str] = {}

    if path.exists():
        loaded = {
            key: value
            for key, value in dotenv_values(path).items()
            if value is not None
        }
        for key, value in loaded.items():
            if override or key not in os.environ:
                os.environ[key] = value

    _apply_legacy_aliases()
    return loaded


def validate_required_env(*keys: str) -> list[str]:
    missing = [key for key in keys if not os.getenv(key)]
    return missing


def _apply_legacy_aliases() -> None:
    for canonical_key, legacy_key in LEGACY_ENV_ALIASES.items():
        if not os.getenv(canonical_key) and os.getenv(legacy_key):
            os.environ[canonical_key] = os.environ[legacy_key]
