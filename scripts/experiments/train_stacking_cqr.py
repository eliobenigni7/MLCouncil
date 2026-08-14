"""Fit CQR position sizer and optional stacking meta-learner (T3.2).

Writes:
  - models/checkpoints/cqr_sizer.pkl (when --cqr)
  - models/checkpoints/stacking_meta.pkl (when --stacking)

Usage:
    python scripts/train_stacking_cqr.py --synthetic
    python scripts/train_stacking_cqr.py --cqr --stacking
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from council.sizing.cqr import (
    CQRPositionSizer,
    DEFAULT_CQR_CHECKPOINT,
    DEFAULT_STACKING_CHECKPOINT,
    StackingMetaLearner,
)


def _synthetic_xy(n: int = 400, p: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = 1.2 * X[:, 0] - 0.8 * X[:, 1] + rng.normal(0, 0.25, n)
    return X, y


def _synthetic_expert_panel(n: int = 200, seed: int = 1) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = [f"T{i}" for i in range(n)]
    base = pd.DataFrame(
        {
            "lgbm": rng.standard_normal(n),
            "sentiment": rng.standard_normal(n),
        },
        index=idx,
    )
    y = base["lgbm"] * 0.55 + base["sentiment"] * 0.35 + rng.normal(0, 0.1, n)
    return base, pd.Series(y, index=idx)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CQR / stacking checkpoints")
    parser.add_argument("--cqr", action="store_true", help="Fit and save CQR sizer")
    parser.add_argument("--stacking", action="store_true", help="Fit and save stacking meta-learner")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--coverage", type=float, default=0.85)
    args = parser.parse_args()

    if not args.cqr and not args.stacking:
        args.cqr = True
        args.stacking = True

    if args.cqr:
        X, y = _synthetic_xy()
        sizer = CQRPositionSizer(coverage=args.coverage)
        sizer.fit(X, y)
        DEFAULT_CQR_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        with open(DEFAULT_CQR_CHECKPOINT, "wb") as fh:
            pickle.dump(sizer, fh)
        print(f"CQR sizer saved -> {DEFAULT_CQR_CHECKPOINT}")

    if args.stacking:
        base, y = _synthetic_expert_panel()
        meta = StackingMetaLearner(use_xgb=False)
        meta.fit(base, y)
        meta.save(DEFAULT_STACKING_CHECKPOINT)
        print(f"Stacking meta saved -> {DEFAULT_STACKING_CHECKPOINT}")


if __name__ == "__main__":
    main()
