from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

_ROOT = Path(__file__).resolve().parent
_DEFAULT_RUNTIME_ENV_PATH = _ROOT / "config" / "runtime.env"

LEGACY_ENV_ALIASES = {
    "ALPACA_PAPER_KEY": "ALPACA_API_KEY",
    "ALPACA_PAPER_SECRET": "ALPACA_SECRET_KEY",
}


def get_runtime_env_path() -> Path:
    override = os.getenv("MLCOUNCIL_RUNTIME_ENV_PATH")
    return Path(override) if override else _DEFAULT_RUNTIME_ENV_PATH


def load_runtime_env(*, override: bool = False) -> dict[str, str]:
    path = get_runtime_env_path()
    
    if path.exists():
        loaded = {
            key: value
            for key, value in dotenv_values(path).items()
            if value is not None
        }
        for key, value in loaded.items():
            if key not in os.environ:
                os.environ[key] = value

    _apply_legacy_aliases()
    return {}


def _apply_legacy_aliases() -> None:
    for canonical_key, legacy_key in LEGACY_ENV_ALIASES.items():
        if not os.getenv(canonical_key) and os.getenv(legacy_key):
            os.environ[canonical_key] = os.environ[legacy_key]
