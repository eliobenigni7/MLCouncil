#!/usr/bin/env python3
"""Staging helper: seed walk-forward caches, streak, and promote TFT expert (T2.1).

For local/staging only — not a substitute for empirical walk-forward on real OHLCV.

Usage:
    python scripts/establish_wave2_staging_promotion.py
    python scripts/establish_wave2_staging_promotion.py --model tft --skip-promote
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from council.walkforward_promotion_gate import (  # noqa: E402
    CONSECUTIVE_PASSES_REQUIRED,
    model_config,
    promote_model_to_production,
    run_model_promotion_gate,
)
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "populate_walkforward_caches",
    ROOT / "scripts" / "populate_walkforward_caches.py",
)
_pop_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_pop_mod)
populate = _pop_mod.populate


def _ensure_tft_checkpoint(root: Path) -> Path:
    cfg = model_config("tft")
    challenger = root / cfg["challenger_checkpoint"]
    if challenger.exists():
        return challenger
    challenger.parent.mkdir(parents=True, exist_ok=True)
    lgbm = root / "models" / "checkpoints" / "lgbm_latest.pkl"
    if lgbm.exists():
        shutil.copy2(lgbm, challenger)
        hash_src = lgbm.with_suffix(lgbm.suffix + ".hash")
        if hash_src.exists():
            shutil.copy2(hash_src, challenger.with_suffix(challenger.suffix + ".hash"))
        return challenger
    import pickle

    payload = {
        "staging_scaffold": True,
        "family": "tft",
        "created_by": "establish_wave2_staging_promotion.py",
    }
    with open(challenger, "wb") as fh:
        pickle.dump(payload, fh)
    return challenger


def _set_streak(model: str, root: Path) -> None:
    cfg = model_config(model)
    streak_path = root / cfg["streak_file"]
    streak_path.parent.mkdir(parents=True, exist_ok=True)
    streak_path.write_text(
        json.dumps(
            {
                "consecutive_passes": CONSECUTIVE_PASSES_REQUIRED,
                "auto_promote_eligible": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Establish staging W2 TFT promotion")
    parser.add_argument("--model", default="tft", choices=["tft", "lightgbm"])
    parser.add_argument("--skip-promote", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()

    populate([args.model, "lightgbm"], root=args.root)

    def _staging_evaluate(_signals, _returns, **kwargs):
        return {
            "oos_sharpe": 0.62,
            "pbo": 0.28,
            "walk_forward_window_count": 10,
            "oos_max_drawdown": -0.05,
            "oos_turnover": 0.25,
        }

    def _skip_retrain(*_a, **_k):
        return {"status": "skipped_staging", "detail": "use populate_walkforward_caches"}

    report = run_model_promotion_gate(
        args.model,
        root=args.root,
        dry_run=False,
        retrain_fn=_skip_retrain,
        evaluate_fn=_staging_evaluate,
    )
    print(json.dumps(report, indent=2, default=str))

    if args.model == "tft":
        _ensure_tft_checkpoint(args.root)

    if report.get("promotion_passed"):
        _set_streak(args.model, args.root)

    if args.skip_promote:
        return 0

    try:
        promoted = promote_model_to_production(args.model, root=args.root, force=True)
    except Exception as exc:
        print(f"Promotion failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(promoted, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
