from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
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

PLACEHOLDER_ENV_VALUES = {
    "",
    "replace-me",
    "changeme",
    "change-me",
    "your-api-key",
    "your-secret",
}

_DOCKER_SECRETS_DIR = Path("/run/secrets")

# Mapping from env-var name to Docker secret filename
_SECRET_FILE_NAMES: dict[str, str] = {
    "ALPACA_API_KEY": "alpaca_api_key",
    "ALPACA_PAPER_KEY": "alpaca_api_key",
    "ALPACA_SECRET_KEY": "alpaca_secret_key",
    "ALPACA_PAPER_SECRET": "alpaca_secret_key",
    "POLYGON_API_KEY": "polygon_api_key",
    "SMTP_PASSWORD": "smtp_password",
}


def get_secret(env_name: str, default: str = "") -> str:
    """Read a secret, preferring Docker secret files over env vars.

    Lookup order:
      1. ``/run/secrets/<mapped_name>`` (Docker secret mount)
      2. ``os.environ[env_name]``
      3. *default*

    This allows the same application code to work both in Docker
    (with secrets mounted) and locally (with ``.env`` / env vars).
    """
    secret_filename = _SECRET_FILE_NAMES.get(env_name, env_name.lower())
    secret_path = _DOCKER_SECRETS_DIR / secret_filename
    if secret_path.is_file():
        return secret_path.read_text().strip()
    return os.environ.get(env_name, default)


@dataclass(frozen=True)
class TradingSettings:
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    automation_paused: bool = False
    max_daily_orders: int = 20
    max_turnover: float = 0.30
    max_position_size: float = 0.10
    max_sector_exposure: float = 0.25


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


def get_config_hash() -> str:
    digest = hashlib.sha256()
    digest.update(get_runtime_profile().encode("utf-8"))
    digest.update(b"\n")
    env_path = get_runtime_env_path()
    if env_path.exists():
        digest.update(env_path.read_bytes())
    return digest.hexdigest()[:16]


def get_project_dotenv_path() -> Path:
    override = os.getenv("MLCOUNCIL_DOTENV_PATH")
    if override:
        return Path(override)
    return _ROOT / ".env"


def load_runtime_env(*, override: bool = False) -> dict[str, str]:
    path = get_runtime_env_path()
    loaded: dict[str, str] = {}
    dotenv_path = get_project_dotenv_path()

    _apply_legacy_aliases()

    if dotenv_path.exists():
        for key, value in dotenv_values(dotenv_path).items():
            if value is not None and _should_set_env_value(
                key=key,
                value=value,
                override=override,
            ):
                os.environ[key] = value
        _apply_legacy_aliases()

    if path.exists():
        loaded = {
            key: value
            for key, value in dotenv_values(path).items()
            if value is not None
        }
        for key, value in loaded.items():
            if _should_set_env_value(key=key, value=value, override=override):
                os.environ[key] = value

    _apply_legacy_aliases()
    return loaded


def validate_required_env(*keys: str) -> list[str]:
    missing = [key for key in keys if is_placeholder_env_value(os.getenv(key))]
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


def get_trading_settings() -> TradingSettings:
    load_runtime_env()
    return TradingSettings(
        alpaca_base_url=(os.getenv("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets").strip(),
        automation_paused=_parse_bool_env("MLCOUNCIL_AUTOMATION_PAUSED", default=False),
        max_daily_orders=_parse_int_env("MLCOUNCIL_MAX_DAILY_ORDERS", default=20),
        max_turnover=_parse_float_env("MLCOUNCIL_MAX_TURNOVER", default=0.30),
        max_position_size=_parse_float_env("MLCOUNCIL_MAX_POSITION_SIZE", default=0.10),
        max_sector_exposure=_parse_float_env("MLCOUNCIL_MAX_SECTOR_EXPOSURE", default=0.25),
    )


def _apply_legacy_aliases() -> None:
    for canonical_key, legacy_key in LEGACY_ENV_ALIASES.items():
        canonical_value = os.getenv(canonical_key)
        legacy_value = os.getenv(legacy_key)
        if (
            is_placeholder_env_value(canonical_value)
            and not is_placeholder_env_value(legacy_value)
        ):
            os.environ[canonical_key] = os.environ[legacy_key]


def is_placeholder_env_value(value: str | None) -> bool:
    if value is None:
        return True
    return value.strip().lower() in PLACEHOLDER_ENV_VALUES


def _should_set_env_value(*, key: str, value: str, override: bool) -> bool:
    current_value = os.getenv(key)
    loaded_is_placeholder = is_placeholder_env_value(value)
    current_is_placeholder = is_placeholder_env_value(current_value)

    if loaded_is_placeholder and not current_is_placeholder:
        return False

    if not loaded_is_placeholder:
        return True

    if override:
        return True

    return current_value is None or current_is_placeholder


def _parse_bool_env(key: str, *, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int_env(key: str, *, default: int) -> int:
    value = os.getenv(key)
    if value is None or not value.strip():
        return default
    return int(value)


def _parse_float_env(key: str, *, default: float) -> float:
    value = os.getenv(key)
    if value is None or not value.strip():
        return default
    return float(value)
