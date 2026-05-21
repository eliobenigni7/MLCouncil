#!/usr/bin/env python3
"""Bootstrap checkpoints required for MLCOUNCIL_ENV_PROFILE=frontier.

Run once after pip install -r requirements.txt:

    python scripts/bootstrap_frontier.py

Then set in `.env`:

    MLCOUNCIL_ENV_PROFILE=frontier
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], label: str) -> None:
    print(f"\n==> {label}")
    print("    ", " ".join(cmd))
    subprocess.run(cmd, cwd=_ROOT, check=False)


def main() -> None:
    py = sys.executable
    steps = [
        ([py, "scripts/train_moe_gating.py"], "MoE gate"),
        ([py, "scripts/train_stacking_cqr.py", "--cqr", "--stacking"], "CQR + stacking"),
        ([py, "scripts/train_regime_dss.py", "--epochs", "30"], "Deep regime DSS"),
        ([py, "scripts/train_alpha_portfolio_end2end.py", "--epochs", "10"], "E2E portfolio scaffold"),
    ]
    for cmd, label in steps:
        _run(cmd, label)

    print("\nFrontier bootstrap finished.")
    print("Optional (GPU/slow):")
    print("  python scripts/train_tft.py --start 2021-01-01 --end 2024-12-31")
    print("\nEnable frontier in .env:")
    print("  MLCOUNCIL_ENV_PROFILE=frontier")
    print("\nStart observability:")
    print("  docker compose -f docker-compose.observability.yml up -d")


if __name__ == "__main__":
    main()
