#!/usr/bin/env python
"""Compatibility wrapper for the experimental train_alpha_portfolio_end2end.py trainer.

Implementation lives in `scripts/experiments/train_alpha_portfolio_end2end.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.experiments.train_alpha_portfolio_end2end import main


if __name__ == "__main__":
    raise SystemExit(main())
