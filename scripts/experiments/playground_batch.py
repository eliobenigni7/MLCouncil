#!/usr/bin/env python3
"""
Batch backtest playground experiment runner.
Tests 30+ parameter combinations, logs results, finds best config.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["MLCOUNCIL_PORTFOLIO_SHRINK_COV"] = "true"

from backtest.playground import PlaygroundParams, run_playground_backtest, load_available_universe

RESULTS_FILE = ROOT / "data" / "results_playground" / "batch_results.json"

# Use all available tickers for a robust test
UNIVERSE = load_available_universe()
START = "2021-01-01"
END = "2025-12-31"
CAPITAL = 1_000_000.0


def run_and_score(params: PlaygroundParams, tag: str) -> dict:
    """Run a single backtest and extract key metrics."""
    t0 = time.time()
    try:
        result = run_playground_backtest(params)
    except Exception as e:
        return {"tag": tag, "error": str(e), "elapsed": time.time() - t0}
    
    s = result.stats
    if s is None:
        return {"tag": tag, "error": "No stats", "elapsed": time.time() - t0}
    
    def _safe_float(v, default=0.0) -> float:
        try:
            if isinstance(v, (np.floating, np.integer)):
                return float(v)
            if isinstance(v, pd.Series):
                return float(v.iloc[-1]) if not v.empty else default
            return float(v or default)
        except (ValueError, TypeError, IndexError):
            return default
    
    sharpe = _safe_float(s.get("sharpe"))
    cagr = _safe_float(s.get("cagr"))
    max_dd = _safe_float(s.get("max_drawdown"))
    calmar = _safe_float(s.get("calmar"))
    sortino = _safe_float(s.get("sortino"))
    total_return = _safe_float(s.get("total_return_pct"))
    n_rebalances = int(_safe_float(s.get("n_trades"), 0))
    win_rate = _safe_float(s.get("win_rate"))
    
    # Composite score: Sharpe is primary, with penalties for high drawdown
    dd_penalty = 1.0 - min(abs(max_dd) / 0.40, 0.5)  # -30% DD halves the score
    score = sharpe * dd_penalty * (1.0 + 0.2 * calmar)
    
    elapsed = time.time() - t0
    return {
        "tag": tag,
        "sharpe": round(sharpe, 4),
        "cagr": round(cagr, 4),
        "max_drawdown": round(max_dd, 4),
        "calmar": round(calmar, 4),
        "sortino": round(sortino, 4),
        "total_return_pct": round(total_return, 2),
        "n_trades": n_rebalances,
        "win_rate": round(win_rate, 4),
        "score": round(score, 4),
        "elapsed": round(elapsed, 1),
    }


def make_params(**overrides) -> PlaygroundParams:
    """Build default params with overrides."""
    defaults = {
        "start_date": START,
        "end_date": END,
        "universe": UNIVERSE,
        "initial_capital": CAPITAL,
        "slippage_bps": 3.0,
        "commission_bps": 0.5,
        "regime_weights": {
            "bull":       {"lgbm": 0.55, "sentiment": 0.25, "hmm": 0.20},
            "bear":       {"lgbm": 0.35, "sentiment": 0.15, "hmm": 0.50},
            "transition": {"lgbm": 0.45, "sentiment": 0.20, "hmm": 0.35},
        },
        "weight_clip_min": 0.05,
        "weight_clip_max": 0.60,
        "ic_rolling_window": 60,
        "sharpe_rolling_window": 120,
        "use_orthogonality": True,
        "max_correlation": 0.65,
        "max_position": 0.08,
        "max_turnover": 0.20,
        "max_vol_ann": 0.30,
        "sector_cap": 0.45,
        "min_signal_strength": 0.20,
        "note": "",
    }
    defaults.update(overrides)
    return PlaygroundParams(**defaults)


# =====================================================================
# EXPERIMENT DESIGN
# =====================================================================

experiments = []
exp_id = 0

# --- Baseline: the old defaults ---
exp_id += 1
experiments.append(("bl_baseline_old", make_params(note="Baseline old defaults")))

# --- NEW regime weights (from Phase 1) ---
new_weights = {
    "bull":       {"lgbm": 0.70, "sentiment": 0.12, "hmm": 0.18},
    "bear":       {"lgbm": 0.65, "sentiment": 0.10, "hmm": 0.25},
    "transition": {"lgbm": 0.68, "sentiment": 0.12, "hmm": 0.20},
}
new_weights_wider = {
    "bull":       {"lgbm": 0.65, "sentiment": 0.15, "hmm": 0.20},
    "bear":       {"lgbm": 0.55, "sentiment": 0.15, "hmm": 0.30},
    "transition": {"lgbm": 0.60, "sentiment": 0.15, "hmm": 0.25},
}

exp_id += 1
experiments.append(("a1_new_regime_weights", make_params(regime_weights=new_weights, note="Phase1 regime weights")))
exp_id += 1
experiments.append(("a2_new_weights_wider", make_params(regime_weights=new_weights_wider, note="Phase1 moderate")))

# --- Weight clip variations (wider max) on old weights ---
for clip_max in [0.60, 0.70, 0.75, 0.85]:
    exp_id += 1
    experiments.append((f"b_clip_max_{clip_max:.0f}", make_params(weight_clip_max=clip_max)))

# --- Weight clip variations on NEW weights ---
for clip_max in [0.70, 0.75, 0.85]:
    exp_id += 1
    tag = f"c_new_clip_{clip_max:.0f}"
    experiments.append((tag, make_params(regime_weights=new_weights, weight_clip_max=clip_max)))

# --- Orthogonality on/off ---
exp_id += 1
experiments.append(("d_ortho_off", make_params(use_orthogonality=False)))
exp_id += 1
experiments.append(("d_ortho_new_weights", make_params(regime_weights=new_weights, use_orthogonality=False)))

# --- Max position (position sizing) ---
for max_pos in [0.05, 0.08, 0.10, 0.15]:
    exp_id += 1
    experiments.append((f"e_maxpos_{max_pos:.0f}pct", make_params(max_position=max_pos)))

# --- Max turnover ---
for turnover in [0.15, 0.20, 0.30, 0.40]:
    exp_id += 1
    experiments.append((f"f_turn_{turnover:.0f}pct", make_params(max_turnover=turnover)))

# --- Min signal strength ---
for min_sig in [0.05, 0.10, 0.20, 0.30]:
    exp_id += 1
    experiments.append((f"g_minsig_{min_sig:.0f}pct", make_params(min_signal_strength=min_sig)))

# --- Max vol ann ---
for vol in [0.20, 0.30, 0.40]:
    exp_id += 1
    experiments.append((f"h_vol_{vol:.0f}pct", make_params(max_vol_ann=vol)))

# --- Costs ---
for slip, comm in [(3.0, 0.5), (1.5, 0.3), (5.0, 1.0), (8.0, 2.0), (1.0, 0.1)]:
    exp_id += 1
    experiments.append((f"i_cost_{slip:.0f}sl_{comm:.1f}cm", make_params(slippage_bps=slip, commission_bps=comm)))

# --- IC rolling window ---
for icw in [30, 60, 90, 120]:
    exp_id += 1
    experiments.append((f"j_icw_{icw}", make_params(ic_rolling_window=icw)))

# --- Sharpe rolling window ---
for srw in [60, 120, 180]:
    exp_id += 1
    experiments.append((f"k_srw_{srw}", make_params(sharpe_rolling_window=srw)))

# --- HMM-heavy regime weights (bear market hedge) ---
hmm_heavy = {
    "bull":       {"lgbm": 0.60, "sentiment": 0.15, "hmm": 0.25},
    "bear":       {"lgbm": 0.20, "sentiment": 0.10, "hmm": 0.70},
    "transition": {"lgbm": 0.40, "sentiment": 0.15, "hmm": 0.45},
}
exp_id += 1
experiments.append(("l_hmm_heavy", make_params(regime_weights=hmm_heavy)))

# --- Sentiment-heavy ---
sent_heavy = {
    "bull":       {"lgbm": 0.40, "sentiment": 0.40, "hmm": 0.20},
    "bear":       {"lgbm": 0.25, "sentiment": 0.30, "hmm": 0.45},
    "transition": {"lgbm": 0.30, "sentiment": 0.35, "hmm": 0.35},
}
exp_id += 1
experiments.append(("m_sent_heavy", make_params(regime_weights=sent_heavy)))

# --- Equal weights across regimes ---
equal_w = {
    "bull":       {"lgbm": 0.34, "sentiment": 0.33, "hmm": 0.33},
    "bear":       {"lgbm": 0.34, "sentiment": 0.33, "hmm": 0.33},
    "transition": {"lgbm": 0.34, "sentiment": 0.33, "hmm": 0.33},
}
exp_id += 1
experiments.append(("n_equal_weights", make_params(regime_weights=equal_w)))

# --- Best guesses: combinations of what works ---
exp_id += 1
experiments.append(("o_bestguess1", make_params(
    regime_weights=new_weights_wider,
    weight_clip_max=0.75,
    max_position=0.10,
    max_turnover=0.30,
    slippage_bps=2.0,
    commission_bps=0.3,
    max_vol_ann=0.30,
)))

exp_id += 1
experiments.append(("o_bestguess2", make_params(
    regime_weights=new_weights,
    weight_clip_max=0.75,
    max_position=0.12,
    max_turnover=0.25,
    min_signal_strength=0.10,
    slippage_bps=1.5,
    commission_bps=0.3,
    max_vol_ann=0.35,
)))

exp_id += 1
experiments.append(("o_bestguess3", make_params(
    regime_weights=new_weights_wider,
    weight_clip_max=0.70,
    max_position=0.10,
    max_turnover=0.25,
    min_signal_strength=0.10,
    slippage_bps=1.5,
    commission_bps=0.3,
    max_vol_ann=0.30,
    use_orthogonality=True,
)))

exp_id += 1
experiments.append(("o_bestguess4_ortho_off", make_params(
    regime_weights=new_weights_wider,
    weight_clip_max=0.75,
    max_position=0.10,
    max_turnover=0.30,
    min_signal_strength=0.10,
    slippage_bps=1.5,
    commission_bps=0.3,
    max_vol_ann=0.30,
    use_orthogonality=False,
)))

exp_id += 1
experiments.append(("o_bestguess5_bear_hedge", make_params(
    regime_weights={
        "bull":       {"lgbm": 0.65, "sentiment": 0.15, "hmm": 0.20},
        "bear":       {"lgbm": 0.40, "sentiment": 0.10, "hmm": 0.50},
        "transition": {"lgbm": 0.55, "sentiment": 0.15, "hmm": 0.30},
    },
    weight_clip_max=0.75,
    max_position=0.10,
    max_turnover=0.30,
    min_signal_strength=0.05,
    slippage_bps=1.5,
    commission_bps=0.3,
    max_vol_ann=0.30,
    use_orthogonality=True,
)))

exp_id += 1
experiments.append(("o_bestguess6_aggressive", make_params(
    regime_weights={
        "bull":       {"lgbm": 0.75, "sentiment": 0.10, "hmm": 0.15},
        "bear":       {"lgbm": 0.50, "sentiment": 0.10, "hmm": 0.40},
        "transition": {"lgbm": 0.70, "sentiment": 0.10, "hmm": 0.20},
    },
    weight_clip_max=0.80,
    max_position=0.15,
    max_turnover=0.35,
    min_signal_strength=0.05,
    slippage_bps=1.5,
    commission_bps=0.3,
    max_vol_ann=0.40,
    use_orthogonality=False,
    ic_rolling_window=90,
)))

exp_id += 1
experiments.append(("o_bestguess7_conservative", make_params(
    regime_weights=new_weights_wider,
    weight_clip_max=0.65,
    max_position=0.07,
    max_turnover=0.20,
    min_signal_strength=0.15,
    slippage_bps=3.0,
    commission_bps=0.5,
    max_vol_ann=0.25,
    use_orthogonality=True,
)))


# =====================================================================
# RUN ALL EXPERIMENTS
# =====================================================================

print(f"MLCouncil Playground Batch — {len(experiments)} experiments")
print(f"Universe: {len(UNIVERSE)} tickers | Window: {START} → {END}")
print("=" * 80)

results = []
errors = []

for i, (tag, params) in enumerate(experiments):
    print(f"\n[{i+1}/{len(experiments)}] {tag}...", end=" ", flush=True)
    r = run_and_score(params, tag)
    if "error" in r and r.get("error"):
        errors.append(r)
        print(f"❌ {r['error']}")
    else:
        results.append(r)
        print(f"Sharpe={r['sharpe']:.3f} CAGR={r['cagr']:.3f} DD={r['max_drawdown']:.3f} Score={r['score']:.3f} ({r['elapsed']}s)")

# Sort by score descending
results.sort(key=lambda x: x["score"], reverse=True)

# Save
out = {
    "n_experiments": len(experiments),
    "n_succeeded": len(results),
    "n_failed": len(errors),
    "window": {"start": START, "end": END},
    "universe_size": len(UNIVERSE),
    "top_10": results[:10],
    "all_results": results,
    "errors": errors,
}
with open(RESULTS_FILE, "w") as f:
    json.dump(out, f, indent=2, default=str)

print("\n" + "=" * 80)
print(f"\n🏆 TOP 10 RESULTS:")
print(f"{'Rank':<6} {'Tag':<30} {'Sharpe':<8} {'CAGR':<8} {'DD':<8} {'Calmar':<8} {'Score':<8} {'WinRate':<8}")
print("-" * 85)
for rank, r in enumerate(results[:10], 1):
    print(f"{rank:<6} {r['tag']:<30} {r['sharpe']:<8.3f} {r['cagr']:<8.3f} {r['max_drawdown']:<8.3f} {r.get('calmar',0):<8.3f} {r['score']:<8.3f} {r.get('win_rate',0):<8.3f}")

print(f"\n📊 Full results: {RESULTS_FILE}")
print(f"✅ Succeeded: {len(results)}/{len(experiments)} ❌ Failed: {len(errors)}")
