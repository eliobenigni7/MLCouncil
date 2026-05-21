#!/usr/bin/env python3
"""Promote a walk-forward gate winner to production (after 3 consecutive passes).

Prerequisites:
  1. python scripts/run_walkforward_promotion.py --model <name>
  2. Gate report shows promotion_passed=true for 3 weekly runs (or use --force)

Usage:
    python scripts/promote_model.py --model lightgbm
    python scripts/promote_model.py --model tft --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from council.walkforward_promotion_gate import (  # noqa: E402
    SUPPORTED_MODELS,
    promote_model_to_production,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote gated model to production")
    parser.add_argument("--model", required=True, choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--force", action="store_true", help="Skip streak/gate checks")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()

    try:
        result = promote_model_to_production(
            args.model,
            root=args.root,
            force=args.force,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"Promotion failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, default=str))
    print("\nProduction manifest updated. Restart Dagster / API to pick up checkpoints.")
    print("Verify: MLCOUNCIL_ENV_PROFILE=prod  MLCOUNCIL_USE_PRODUCTION_MANIFEST=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
