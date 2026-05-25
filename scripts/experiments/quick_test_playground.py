#!/usr/bin/env python3
"""
Quick test to verify the playground works end-to-end.
"""
import os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["MLCOUNCIL_PORTFOLIO_SHRINK_COV"] = "true"

from backtest.playground import PlaygroundParams, run_playground_backtest, load_available_universe

UNIVERSE = load_available_universe()
print(f"Universe: {len(UNIVERSE)} tickers")

for n in [5, 10, 20]:
    params = PlaygroundParams(
        start_date="2021-01-01",
        end_date="2025-12-31",
        universe=UNIVERSE[:n],
        initial_capital=1_000_000.0,
    )
    t0 = time.time()
    try:
        result = run_playground_backtest(params)
        s = result.stats
        print(f"n={n}: Sharpe={s.get('sharpe'):.3f} CAGR={s.get('cagr'):.3f} DD={s.get('max_drawdown'):.3f} ({time.time()-t0:.1f}s)")
    except Exception as e:
        import traceback
        print(f"n={n}: FAILED - {e}")
        traceback.print_exc()
        break
