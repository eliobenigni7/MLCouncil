#!/usr/bin/env python
"""Compatibility wrapper for the experimental train_tft.py trainer.

Implementation lives in `scripts/experiments/train_tft.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.experiments.train_tft import main


if __name__ == "__main__":
    raise SystemExit(main())
