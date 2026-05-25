#!/usr/bin/env python
"""Compatibility wrapper for the experimental run_sharpe_ablation.py tool.

Implementation lives in `scripts/experiments/run_sharpe_ablation.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.experiments.run_sharpe_ablation import main


if __name__ == "__main__":
    raise SystemExit(main())
