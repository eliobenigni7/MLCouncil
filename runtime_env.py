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

SUPPORTED_RUNTIME_PROFILES = {"local", "paper"}

PROFILE_REQUIRED_KEYS = {
    "local": [],
    "paper": [
        "ALPACA_BASE_URL",
        "ALPACA_PAPER_KEY",
        "ALPACA_PAPER_SECRET",
        "MLCOUNCIL_MAX_DAILY_ORDERS",
        "MLCOUNCIL_MAX_TURNOVER",
        "MLCOUNCIL_MAX_POSITION_SIZE",
        "MLCOUNCIL_AUTOMATION_PAUSED",
    ],
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


def validate_runtime_profile(profile: str | None = None) -> dict[str, object]:
    active_profile = (profile or get_runtime_profile()).strip().lower() or "local"
    env_path = get_runtime_env_path()
    load_runtime_env()
    exists = env_path.exists()
    errors: list[str] = []
    warnings: list[str] = []

    if active_profile not in SUPPORTED_RUNTIME_PROFILES:
        errors.append(
            f"Unsupported runtime profile '{active_profile}'. "
            f"Supported profiles: {', '.join(sorted(SUPPORTED_RUNTIME_PROFILES))}."
        )

    required = PROFILE_REQUIRED_KEYS.get(active_profile, [])
    missing = validate_required_env(*required)

    paper_guard_ok = True
    if active_profile == "paper":
        base_url = os.getenv("ALPACA_BASE_URL", "").strip()
        if base_url and "paper-api.alpaca.markets" not in base_url:
            paper_guard_ok = False
            errors.append("Paper profile must point to Alpaca paper endpoint.")

    if not exists:
        warnings.append(f"Runtime env file not found: {env_path}")

    valid = not errors and not missing
    if active_profile == "paper":
        valid = valid and paper_guard_ok

    return {
        "profile": active_profile,
        "env_path": str(env_path),
        "exists": exists,
        "missing": missing,
        "errors": errors,
        "warnings": warnings,
        "paper_guard_ok": paper_guard_ok,
        "valid": valid,
        "status": "valid" if valid else "invalid",
    }


def _apply_legacy_aliases() -> None:
    for canonical_key, legacy_key in LEGACY_ENV_ALIASES.items():
        if not os.getenv(canonical_key) and os.getenv(legacy_key):
            os.environ[canonical_key] = os.environ[legacy_key]
