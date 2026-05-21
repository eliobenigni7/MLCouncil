"""Walk-forward champion/challenger promotion gate (ADR T1.1).

Challengers are trained and evaluated in shadow mode; daily pipeline keeps
using champion checkpoints until promotion criteria pass (including three
consecutive CI passes before auto-promote).
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import pandas as pd

from backtest.validation import (
    ModelPromotionResult,
    run_walk_forward_analysis,
    validate_model_promotion,
)

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_OPERATIONS = _ROOT / "data" / "operations"
_RESULTS = _ROOT / "data" / "results"

SUPPORTED_MODELS = frozenset({"lightgbm", "sentiment", "hmm", "tft"})

DEFAULT_TRAIN_WINDOW = 252
DEFAULT_TEST_WINDOW = 63
DEFAULT_PURGE_PERIOD = 1
DEFAULT_EMBARGO_PERIOD = 1

_MODEL_CONFIG: dict[str, dict[str, str]] = {
    "lightgbm": {
        "champion_checkpoint": "models/checkpoints/lgbm_latest.pkl",
        "challenger_checkpoint": "models/checkpoints/lgbm_challenger.pkl",
        "champion_metrics": "data/operations/walkforward_champion_lightgbm.json",
        "signals_cache": "data/results/walkforward_signals_lightgbm.parquet",
        "returns_cache": "data/results/walkforward_forward_returns.parquet",
        "train_script": "scripts/train_lgbm_standalone.py",
        "streak_file": "data/operations/walkforward_streak_lightgbm.json",
    },
    "sentiment": {
        "champion_checkpoint": "models/checkpoints/sentiment_latest.pkl",
        "challenger_checkpoint": "models/checkpoints/sentiment_challenger.pkl",
        "champion_metrics": "data/operations/walkforward_champion_sentiment.json",
        "signals_cache": "data/results/walkforward_signals_sentiment.parquet",
        "returns_cache": "data/results/walkforward_forward_returns.parquet",
        "train_script": "",
        "streak_file": "data/operations/walkforward_streak_sentiment.json",
    },
    "hmm": {
        "champion_checkpoint": "models/checkpoints/hmm_latest.pkl",
        "challenger_checkpoint": "models/checkpoints/hmm_challenger.pkl",
        "champion_metrics": "data/operations/walkforward_champion_hmm.json",
        "signals_cache": "data/results/walkforward_signals_hmm.parquet",
        "returns_cache": "data/results/walkforward_forward_returns.parquet",
        "train_script": "",
        "streak_file": "data/operations/walkforward_streak_hmm.json",
    },
    "tft": {
        "champion_checkpoint": "models/checkpoints/lgbm_latest.pkl",
        "challenger_checkpoint": "models/checkpoints/tft_challenger.pkl",
        "champion_metrics": "data/operations/walkforward_champion_lightgbm.json",
        "signals_cache": "data/results/walkforward_signals_tft.parquet",
        "returns_cache": "data/results/walkforward_forward_returns.parquet",
        "train_script": "scripts/train_tft.py",
        "streak_file": "data/operations/walkforward_streak_tft.json",
        "shadow_signals": "data/results/tft_shadow_signals.parquet",
    },
}

CONSECUTIVE_PASSES_REQUIRED = 3


def model_config(model: str) -> dict[str, str]:
    key = model.lower().strip()
    if key not in _MODEL_CONFIG:
        raise ValueError(f"Unsupported model {model!r}; choose from {sorted(SUPPORTED_MODELS)}")
    return _MODEL_CONFIG[key]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_champion_metrics(model: str, root: Path | None = None) -> dict[str, float]:
    """Load persisted champion walk-forward summary metrics."""
    cfg = model_config(model)
    base = root or _ROOT
    data = _read_json(base / cfg["champion_metrics"])
    if not data:
        return {
            "oos_sharpe": 0.0,
            "pbo": 0.0,
            "walk_forward_window_count": 0,
        }
    return {
        "oos_sharpe": float(data.get("oos_sharpe", 0.0)),
        "pbo": float(data.get("pbo", 0.0)),
        "walk_forward_window_count": int(data.get("walk_forward_window_count", 0)),
    }


def load_signal_frames(
    model: str,
    root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Load cached challenger signal matrix and forward returns for walk-forward."""
    cfg = model_config(model)
    base = root or _ROOT
    signals_path = base / cfg["signals_cache"]
    returns_path = base / cfg["returns_cache"]
    if not signals_path.exists() or not returns_path.exists():
        return None

    signals = pd.read_parquet(signals_path)
    returns = pd.read_parquet(returns_path)
    if not isinstance(signals.index, pd.DatetimeIndex):
        signals.index = pd.to_datetime(signals.index)
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns.index = pd.to_datetime(returns.index)
    return signals.sort_index(), returns.sort_index()


def retrain_challenger(
    model: str,
    *,
    root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Retrain challenger checkpoint (shadow). Skipped in dry-run or when no train script."""
    cfg = model_config(model)
    base = root or _ROOT
    if dry_run:
        return {"status": "skipped_dry_run", "checkpoint": str(base / cfg["challenger_checkpoint"])}

    script = cfg.get("train_script") or ""
    if not script:
        return {
            "status": "skipped_no_train_script",
            "detail": f"No weekly train script configured for {model}",
        }

    script_path = base / script
    if not script_path.exists():
        return {"status": "failed", "detail": f"Train script missing: {script_path}"}

    if model in ("lightgbm", "tft"):
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(base),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return {
                "status": "failed",
                "detail": proc.stderr or proc.stdout or "train script failed",
            }
        challenger = base / cfg["challenger_checkpoint"]
        if model == "lightgbm":
            champion = base / cfg["champion_checkpoint"]
            challenger.parent.mkdir(parents=True, exist_ok=True)
            if champion.exists():
                challenger.write_bytes(champion.read_bytes())
                return {"status": "ok", "checkpoint": str(challenger)}
            return {"status": "failed", "detail": f"Champion checkpoint not found: {champion}"}
        if challenger.exists():
            return {"status": "ok", "checkpoint": str(challenger)}
        return {"status": "failed", "detail": f"TFT challenger checkpoint not found: {challenger}"}

    return {"status": "skipped_not_implemented", "detail": f"Retrain for {model} not wired yet"}


def evaluate_walk_forward(
    signals: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    train_window: int = DEFAULT_TRAIN_WINDOW,
    test_window: int = DEFAULT_TEST_WINDOW,
    purge_period: int = DEFAULT_PURGE_PERIOD,
    embargo_period: int = DEFAULT_EMBARGO_PERIOD,
) -> dict[str, float]:
    """Run purged walk-forward analysis and return summary metrics."""
    common = signals.index.intersection(forward_returns.index)
    if len(common) < train_window + test_window:
        return {
            "oos_sharpe": 0.0,
            "pbo": 1.0,
            "walk_forward_window_count": 0,
        }

    sig = signals.loc[common].astype(float)
    fwd = forward_returns.loc[common].astype(float)
    result = run_walk_forward_analysis(
        signals=sig,
        forward_returns=fwd,
        train_window=train_window,
        test_window=test_window,
        step=test_window,
        purge_period=purge_period,
        embargo_period=embargo_period,
    )
    summary = dict(result.get("summary", {}))
    return {
        "oos_sharpe": float(summary.get("oos_sharpe", 0.0)),
        "pbo": float(summary.get("pbo", 1.0)),
        "walk_forward_window_count": int(summary.get("walk_forward_window_count", 0)),
        "oos_max_drawdown": float(summary.get("oos_max_drawdown", 0.0)),
        "oos_turnover": float(summary.get("oos_turnover", 0.0)),
    }


def _update_promotion_streak(
    model: str,
    passed: bool,
    *,
    root: Path,
    dry_run: bool,
) -> dict[str, Any]:
    cfg = model_config(model)
    streak_path = root / cfg["streak_file"]
    streak = _read_json(streak_path)
    consecutive = int(streak.get("consecutive_passes", 0))
    if passed and not dry_run:
        consecutive += 1
    elif not passed:
        consecutive = 0
    payload = {
        "consecutive_passes": consecutive,
        "required_for_auto_promote": CONSECUTIVE_PASSES_REQUIRED,
        "auto_promote_eligible": consecutive >= CONSECUTIVE_PASSES_REQUIRED,
    }
    if not dry_run:
        _write_json(streak_path, payload)
    return payload


def run_model_promotion_gate(
    model: str,
    *,
    root: Path | None = None,
    dry_run: bool = False,
    retrain_fn: Callable[..., dict[str, Any]] | None = None,
    evaluate_fn: Callable[..., dict[str, float]] | None = None,
    validate_fn: Callable[..., ModelPromotionResult] | None = None,
) -> dict[str, Any]:
    """Orchestrate challenger retrain, walk-forward gate, and promotion report."""
    base = root or _ROOT
    cfg = model_config(model)
    report: dict[str, Any] = {
        "model": model,
        "dry_run": dry_run,
        "status": "skipped",
        "champion_metrics": {},
        "challenger_metrics": {},
        "promotion_passed": None,
        "reasons": [],
        "retrain": {},
        "streak": {},
        "auto_promote_eligible": False,
        "shadow_mode": True,
        "report_path": str(base / "data" / "operations" / f"walkforward_promotion_{model}.json"),
    }

    champion_metrics = load_champion_metrics(model, base)
    report["champion_metrics"] = champion_metrics

    retrain = (retrain_fn or retrain_challenger)(model, root=base, dry_run=dry_run)
    report["retrain"] = retrain

    frames = load_signal_frames(model, base)
    if frames is None:
        report["status"] = "skipped_no_signal_cache"
        report["reasons"] = [
            f"Missing cached signals at {cfg['signals_cache']} and/or "
            f"{cfg['returns_cache']}; populate via backtest or shadow logging."
        ]
        _write_json(Path(report["report_path"]), report)
        return report

    signals, forward_returns = frames
    challenger_metrics = (evaluate_fn or evaluate_walk_forward)(signals, forward_returns)
    report["challenger_metrics"] = challenger_metrics

    validator = validate_fn or validate_model_promotion
    result = validator(champion_metrics, challenger_metrics)
    report["promotion_passed"] = result.passed
    report["reasons"] = result.reasons
    report["streak"] = _update_promotion_streak(
        model,
        result.passed,
        root=base,
        dry_run=dry_run,
    )
    report["auto_promote_eligible"] = bool(report["streak"].get("auto_promote_eligible"))

    if result.passed:
        report["status"] = "gate_passed_shadow"
        logger.info("Walk-forward promotion gate PASSED for %s (shadow; not wired to pipeline)", model)
    else:
        report["status"] = "gate_failed"
        logger.warning("Walk-forward promotion gate FAILED for %s: %s", model, result.reasons)

    _write_json(Path(report["report_path"]), report)
    return report


def promote_model_to_production(
    model: str,
    *,
    root: Path | None = None,
    force: bool = False,
    promoted_by: str = "promote_model.py",
) -> dict[str, Any]:
    """Promote challenger to champion after gate pass + consecutive streak.

    Updates checkpoints, champion metrics cache, and ``production_manifest.yaml``.
    """
    import shutil

    from council.production_config import (
        copy_checkpoint,
        load_manifest,
        record_promotion,
        save_manifest,
    )

    base = root or _ROOT
    cfg = model_config(model)
    report_path = base / "data" / "operations" / f"walkforward_promotion_{model}.json"
    gate_report = _read_json(report_path)

    if not gate_report:
        raise RuntimeError(
            f"No gate report at {report_path}. Run run_walkforward_promotion.py first."
        )

    streak_path = base / cfg["streak_file"]
    streak = _read_json(streak_path)
    consecutive = int(streak.get("consecutive_passes", 0))
    eligible = bool(streak.get("auto_promote_eligible")) or consecutive >= CONSECUTIVE_PASSES_REQUIRED

    if not gate_report.get("promotion_passed") and not force:
        raise RuntimeError(
            f"Gate did not pass for {model}: {gate_report.get('reasons')}. Use --force to override."
        )
    if not eligible and not force:
        raise RuntimeError(
            f"Need {CONSECUTIVE_PASSES_REQUIRED} consecutive passes (have {consecutive}). "
            "Use --force to override."
        )

    manifest = load_manifest(base / "config" / "production_manifest.yaml")
    promotion_result: dict[str, Any] = {
        "model": model,
        "promoted_at": None,
        "checkpoint": "",
        "manifest_updates": [],
        "status": "ok",
    }

    challenger = base / cfg["challenger_checkpoint"]
    champion = base / cfg["champion_checkpoint"]

    if model == "lightgbm":
        if challenger.exists():
            copy_checkpoint(challenger, champion)
        promotion_result["checkpoint"] = str(champion)
        _write_json(
            base / cfg["champion_metrics"],
            gate_report.get("challenger_metrics") or {},
        )
    elif model == "tft":
        if not challenger.exists():
            raise FileNotFoundError(f"TFT challenger missing: {challenger}")
        manifest.setdefault("models", {})["technical"] = {
            "family": "tft",
            "checkpoint": str(cfg["challenger_checkpoint"]),
        }
        manifest.setdefault("experts", {}).setdefault("tft", {})["enabled"] = True
        promotion_result["manifest_updates"].append("models.technical=tft")
        promotion_result["checkpoint"] = str(challenger)
        _write_json(base / cfg["champion_metrics"], gate_report.get("challenger_metrics") or {})
    elif model in ("sentiment", "hmm"):
        if challenger.exists():
            copy_checkpoint(challenger, champion)
        elif champion.exists() and not force:
            raise FileNotFoundError(f"Challenger missing: {challenger}")
        key = "sentiment" if model == "sentiment" else "regime"
        manifest.setdefault("models", {})[key] = {
            "family": model,
            "checkpoint": str(cfg["champion_checkpoint"]),
        }
        promotion_result["checkpoint"] = str(champion)
        _write_json(base / cfg["champion_metrics"], gate_report.get("challenger_metrics") or {})
    else:
        raise ValueError(f"Promotion not implemented for model={model}")

    manifest_path = base / "config" / "production_manifest.yaml"
    manifest = record_promotion(
        model,
        gate_report_path=str(report_path),
        promoted_by=promoted_by,
        manifest_path=manifest_path,
        manifest=manifest,
    )

    promotion_result["promoted_at"] = manifest.get("updated_at")
    logger.info("Promoted %s to production champion", model)
    return promotion_result
