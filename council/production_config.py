"""Production manifest — single source of truth after walk-forward promotion.

When ``MLCOUNCIL_USE_PRODUCTION_MANIFEST=true`` (default for ``prod`` profile),
council/portfolio env flags are derived from ``config/production_manifest.yaml``
instead of ad-hoc ``MLCOUNCIL_*`` overrides. Challengers enter production only
after ``scripts/promote_model.py`` updates this file.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST_PATH = _ROOT / "config" / "production_manifest.yaml"


def manifest_enabled() -> bool:
    raw = os.getenv("MLCOUNCIL_USE_PRODUCTION_MANIFEST", "").strip().lower()
    if raw in _TRUTHY:
        return True
    profile = os.getenv("MLCOUNCIL_ENV_PROFILE", "").strip().lower()
    return profile in ("prod", "production")


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    p = path or _MANIFEST_PATH
    if not p.exists():
        logger.warning(f"production manifest missing at {p}; using built-in defaults")
        return _default_manifest()
    with open(p, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def _default_manifest() -> dict[str, Any]:
    with open(_MANIFEST_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def save_manifest(data: dict[str, Any], path: Path | None = None) -> Path:
    p = path or _MANIFEST_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False)
    return p


def council_setting(key: str, default: str | None = None) -> str:
    """Read council.* from manifest when manifest mode is on, else env."""
    if manifest_enabled():
        manifest = load_manifest()
        council = manifest.get("council") or {}
        val = council.get(key)
        if val is not None:
            return str(val).strip().lower()
    if default is not None:
        return default
    return ""


def feature_enabled(key: str) -> bool:
    if manifest_enabled():
        manifest = load_manifest()
        features = manifest.get("features") or {}
        return str(features.get(key, False)).lower() in _TRUTHY or features.get(key) is True
    raw = os.getenv(f"MLCOUNCIL_{key.upper()}", "").strip().lower()
    return raw in _TRUTHY


def expert_enabled(name: str) -> bool:
    if manifest_enabled():
        manifest = load_manifest()
        experts = manifest.get("experts") or {}
        block = experts.get(name) or {}
        return bool(block.get("enabled", False))
    return False


def get_aggregator_mode() -> str:
    if manifest_enabled():
        mode = council_setting("aggregator_mode", "linear")
        return mode if mode in ("linear", "moe") else "linear"
    from council.moe_gating import aggregator_mode

    return aggregator_mode()


def get_position_sizing_mode() -> str:
    if manifest_enabled():
        mode = council_setting("position_sizing", "conformal")
        return mode if mode in ("conformal", "cqr") else "conformal"
    from council.cqr import position_sizing_mode

    return position_sizing_mode()


def get_covariance_estimator() -> str:
    if manifest_enabled():
        mode = council_setting("covariance_estimator", "ledoit")
        return mode if mode in ("ledoit", "dcc", "factor") else "ledoit"
    from council.covariance_dynamic import covariance_estimator_mode

    return covariance_estimator_mode()


def get_portfolio_mode() -> str:
    if manifest_enabled():
        mode = council_setting("portfolio_mode", "cvxpy")
        return mode if mode in ("cvxpy", "diff", "hrp_blend") else "cvxpy"
    from council.portfolio_diff import portfolio_constructor_mode

    return portfolio_constructor_mode()


def use_stacked_council() -> bool:
    if manifest_enabled():
        raw = council_setting("use_stacked_council", "false")
        return raw in _TRUTHY
    from council.frontier import use_stacked_council_signal

    return use_stacked_council_signal()


def get_regime_mode() -> str:
    if manifest_enabled():
        mode = council_setting("regime_mode", "label")
        return mode if mode in ("label", "embedding") else "label"
    from council.aggregator import regime_mode

    return regime_mode()


def apply_manifest_to_environ() -> None:
    """Sync manifest council flags into os.environ for legacy call sites."""
    if not manifest_enabled():
        return
    os.environ["MLCOUNCIL_AGGREGATOR_MODE"] = get_aggregator_mode()
    os.environ["MLCOUNCIL_POSITION_SIZING"] = get_position_sizing_mode()
    os.environ["MLCOUNCIL_COVARIANCE_ESTIMATOR"] = get_covariance_estimator()
    os.environ["MLCOUNCIL_PORTFOLIO_MODE"] = get_portfolio_mode()
    os.environ["MLCOUNCIL_REGIME_MODE"] = get_regime_mode()
    if feature_enabled("online_learning"):
        os.environ["MLCOUNCIL_ONLINE_LEARNING"] = "true"
    else:
        os.environ.pop("MLCOUNCIL_ONLINE_LEARNING", None)
    if feature_enabled("otel_enabled"):
        os.environ["MLCOUNCIL_OTEL_ENABLED"] = "true"
    if feature_enabled("hrp_soft_prior"):
        os.environ["MLCOUNCIL_HRP_SOFT_PRIOR"] = "true"


def record_promotion(
    model: str,
    *,
    gate_report_path: str,
    promoted_by: str = "promote_model.py",
    manifest_path: Path | None = None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append audit entry and refresh manifest metadata."""
    data = manifest if manifest is not None else load_manifest(manifest_path)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    data["updated_by"] = promoted_by
    history = list(data.get("promotion_history") or [])
    history.append(
        {
            "model": model,
            "at": data["updated_at"],
            "gate_report": gate_report_path,
            "by": promoted_by,
        }
    )
    data["promotion_history"] = history[-50:]
    save_manifest(data, manifest_path)
    return data


def promote_technical_to_tft(manifest: dict[str, Any], root: Path) -> None:
    manifest["models"]["technical"] = {
        "family": "tft",
        "checkpoint": "models/checkpoints/tft_challenger.pkl",
    }
    experts = manifest.setdefault("experts", {})
    tft = experts.setdefault("tft", {})
    tft["enabled"] = True


def copy_checkpoint(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    hash_src = src.with_suffix(src.suffix + ".hash")
    if hash_src.exists():
        shutil.copy2(hash_src, dst.with_suffix(dst.suffix + ".hash"))
