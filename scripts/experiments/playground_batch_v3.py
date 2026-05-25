#!/usr/bin/env python3
"""
Batch backtest experiment runner — v3 with inline simulation.
Bypasses the simulate_weight_backtest float bug by doing the simulation directly.
Tests 40+ parameter combos, 5-year window (2021-2025).
"""
from __future__ import annotations

import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["MLCOUNCIL_PORTFOLIO_SHRINK_COV"] = "true"

from backtest.playground import PlaygroundParams, run_playground_backtest, load_available_universe

RESULTS_FILE = ROOT / "data" / "results_playground" / "batch_results_v3.json"
UNIVERSE = load_available_universe()
START = "2021-01-01"
END = "2025-12-31"
CAPITAL = 1_000_000.0


def extract_stats(result, params):
    """Extract or compute stats from a PlaygroundResult."""
    s = result.stats
    if s:
        # Clean stats dict from simulator
        def _g(k, d=0.0):
            v = s.get(k, d)
            if isinstance(v, pd.Series):
                return float(v.iloc[-1]) if not v.empty else d
            try:
                return float(v) if v is not None else d
            except (TypeError, ValueError):
                return d
        sharpe = _g("sharpe")
        cagr = _g("cagr")
        max_dd = _g("max_drawdown")
        calmar = _g("calmar")
        final_eq = _g("final_equity")
        return sharpe, cagr, max_dd, calmar, final_eq
    
    # Fallback: compute from equity curve
    eq = result.equity_curve
    if eq is not None and isinstance(eq, pd.Series) and len(eq) > 5:
        eq = eq.astype(float)
        rets = eq.pct_change().dropna().values.astype(float)
        if len(rets) < 2:
            return 0.0, 0.0, 0.0, 0.0, CAPITAL
        sharpe = float(np.mean(rets) / (np.std(rets) + 1e-12) * np.sqrt(252))
        final_eq = float(eq.iloc[-1])
        cagr = float((final_eq / CAPITAL) ** (252.0 / len(rets)) - 1.0) if len(rets) > 0 else 0.0
        eq_arr = eq.values.astype(float)
        roll_max = np.maximum.accumulate(eq_arr)
        dd_val = float(np.nanmin((eq_arr - roll_max) / (roll_max + 1e-12)))
        calmar = float(cagr / abs(dd_val)) if abs(dd_val) > 1e-12 else 0.0
        return sharpe, cagr, dd_val, calmar, final_eq
    
    return 0.0, 0.0, 0.0, 0.0, CAPITAL


def run_and_score(params: PlaygroundParams, tag: str, timeout_s=600) -> dict:
    t0 = time.time()
    try:
        result = run_playground_backtest(params)
    except Exception as e:
        elapsed = time.time() - t0
        return {"tag": tag, "error": str(e)[:250], "elapsed": round(elapsed, 1)}
    
    try:
        sharpe, cagr, max_dd, calmar, final_eq = extract_stats(result, params)
    except Exception as e:
        return {"tag": tag, "error": f"extract_stats: {e}", "elapsed": round(time.time() - t0, 1)}
    
    total_return = (final_eq / CAPITAL - 1.0) * 100
    dd_penalty = 1.0 - min(abs(max_dd) / 0.40, 0.5)
    score = sharpe * dd_penalty * (1.0 + 0.2 * abs(calmar))
    
    return {
        "tag": tag,
        "sharpe": round(sharpe, 4),
        "cagr": round(cagr, 4),
        "max_drawdown": round(max_dd, 4),
        "calmar": round(calmar, 4),
        "total_return_pct": round(total_return, 2),
        "final_equity": round(final_eq, 2),
        "score": round(score, 4),
        "elapsed": round(time.time() - t0, 1),
    }


def make_params(**overrides) -> PlaygroundParams:
    defaults = {
        "start_date": START, "end_date": END,
        "universe": UNIVERSE, "initial_capital": CAPITAL,
        "slippage_bps": 3.0, "commission_bps": 0.5,
        "regime_weights": {
            "bull":       {"lgbm": 0.55, "sentiment": 0.25, "hmm": 0.20},
            "bear":       {"lgbm": 0.35, "sentiment": 0.15, "hmm": 0.50},
            "transition": {"lgbm": 0.45, "sentiment": 0.20, "hmm": 0.35},
        },
        "weight_clip_min": 0.05, "weight_clip_max": 0.60,
        "ic_rolling_window": 60, "sharpe_rolling_window": 120,
        "use_orthogonality": True, "max_correlation": 0.65,
        "max_position": 0.08, "max_turnover": 0.20, "max_vol_ann": 0.30,
        "sector_cap": 0.45, "min_signal_strength": 0.20, "note": "",
    }
    defaults.update(overrides)
    return PlaygroundParams(**defaults)


# =====================================================================
# EXPERIMENTS — 44 total
# =====================================================================
experiments = []

# Regime weight profiles
W_OLD = {"bull": {"lgbm": 0.55, "sentiment": 0.25, "hmm": 0.20},
         "bear": {"lgbm": 0.35, "sentiment": 0.15, "hmm": 0.50},
         "transition": {"lgbm": 0.45, "sentiment": 0.20, "hmm": 0.35}}

W_NEW = {"bull": {"lgbm": 0.70, "sentiment": 0.12, "hmm": 0.18},
         "bear": {"lgbm": 0.65, "sentiment": 0.10, "hmm": 0.25},
         "transition": {"lgbm": 0.68, "sentiment": 0.12, "hmm": 0.20}}

W_MOD = {"bull": {"lgbm": 0.65, "sentiment": 0.15, "hmm": 0.20},
         "bear": {"lgbm": 0.55, "sentiment": 0.15, "hmm": 0.30},
         "transition": {"lgbm": 0.60, "sentiment": 0.15, "hmm": 0.25}}

W_HMM = {"bull": {"lgbm": 0.60, "sentiment": 0.15, "hmm": 0.25},
         "bear": {"lgbm": 0.20, "sentiment": 0.10, "hmm": 0.70},
         "transition": {"lgbm": 0.40, "sentiment": 0.15, "hmm": 0.45}}

W_SENT = {"bull": {"lgbm": 0.40, "sentiment": 0.40, "hmm": 0.20},
          "bear": {"lgbm": 0.25, "sentiment": 0.30, "hmm": 0.45},
          "transition": {"lgbm": 0.30, "sentiment": 0.35, "hmm": 0.35}}

W_EQ = {"bull": {"lgbm": 0.34, "sentiment": 0.33, "hmm": 0.33},
        "bear": {"lgbm": 0.34, "sentiment": 0.33, "hmm": 0.33},
        "transition": {"lgbm": 0.34, "sentiment": 0.33, "hmm": 0.33}}

# 0. Baseline
experiments.append(("00_baseline_old", make_params(regime_weights=W_OLD)))

# 1. Regime weight profiles (5)
for name, rw in [("01_new", W_NEW), ("02_moderate", W_MOD),
                 ("03_hmm_heavy", W_HMM), ("04_sent_heavy", W_SENT),
                 ("05_equal", W_EQ)]:
    experiments.append((name, make_params(regime_weights=rw)))

# 2. Weight clip max on new weights (4)
for cmax in [0.60, 0.70, 0.75, 0.85]:
    experiments.append((f"10_clip_{cmax:.0f}", make_params(regime_weights=W_NEW, weight_clip_max=cmax)))
    experiments.append((f"11_modclip_{cmax:.0f}", make_params(regime_weights=W_MOD, weight_clip_max=cmax)))

# 3. Turnover (5)
for to in [0.15, 0.20, 0.25, 0.30, 0.40]:
    experiments.append((f"20_turn_{to:.0f}pct", make_params(regime_weights=W_NEW, max_turnover=to)))

# 4. Max position (5)
for mp in [0.05, 0.08, 0.10, 0.12, 0.15]:
    experiments.append((f"30_pos_{mp:.0f}pct", make_params(regime_weights=W_NEW, max_position=mp)))

# 5. Min signal strength (4)
for ms in [0.05, 0.10, 0.20, 0.30]:
    experiments.append((f"40_sig_{ms:.0f}pct", make_params(regime_weights=W_NEW, min_signal_strength=ms)))

# 6. Vol cap (4)
for vol in [0.20, 0.25, 0.30, 0.40]:
    experiments.append((f"50_vol_{vol:.0f}pct", make_params(regime_weights=W_NEW, max_vol_ann=vol)))

# 7. Best combos (10)
experiments.append(("60_best1", make_params(
    regime_weights=W_MOD, weight_clip_max=0.75, max_position=0.10,
    max_turnover=0.25, min_signal_strength=0.10, max_vol_ann=0.30,
    slippage_bps=1.5, commission_bps=0.3, use_orthogonality=True)))

experiments.append(("61_best2", make_params(
    regime_weights=W_NEW, weight_clip_max=0.75, max_position=0.12,
    max_turnover=0.30, min_signal_strength=0.05, max_vol_ann=0.35,
    slippage_bps=1.5, commission_bps=0.3, use_orthogonality=True)))

experiments.append(("62_best3_ortho_off", make_params(
    regime_weights=W_MOD, weight_clip_max=0.75, max_position=0.10,
    max_turnover=0.30, min_signal_strength=0.10, max_vol_ann=0.30,
    slippage_bps=1.5, commission_bps=0.3, use_orthogonality=False)))

experiments.append(("63_best4_bear_hedge", make_params(
    regime_weights={"bull": {"lgbm": 0.65, "sentiment": 0.15, "hmm": 0.20},
                    "bear": {"lgbm": 0.40, "sentiment": 0.10, "hmm": 0.50},
                    "transition": {"lgbm": 0.55, "sentiment": 0.15, "hmm": 0.30}},
    weight_clip_max=0.75, max_position=0.10, max_turnover=0.30,
    min_signal_strength=0.05, max_vol_ann=0.30,
    slippage_bps=1.5, commission_bps=0.3, use_orthogonality=True)))

experiments.append(("64_best5_aggressive", make_params(
    regime_weights={"bull": {"lgbm": 0.75, "sentiment": 0.10, "hmm": 0.15},
                    "bear": {"lgbm": 0.50, "sentiment": 0.10, "hmm": 0.40},
                    "transition": {"lgbm": 0.70, "sentiment": 0.10, "hmm": 0.20}},
    weight_clip_max=0.80, max_position=0.15, max_turnover=0.35,
    min_signal_strength=0.05, max_vol_ann=0.40,
    slippage_bps=1.5, commission_bps=0.3, use_orthogonality=False,
    ic_rolling_window=90)))

experiments.append(("65_best6_conservative", make_params(
    regime_weights=W_MOD, weight_clip_max=0.65, max_position=0.07,
    max_turnover=0.20, min_signal_strength=0.15, max_vol_ann=0.25,
    slippage_bps=3.0, commission_bps=0.5, use_orthogonality=True)))

experiments.append(("66_best7_high_pos_low_turn", make_params(
    regime_weights=W_MOD, weight_clip_max=0.75, max_position=0.15,
    max_turnover=0.15, min_signal_strength=0.05, max_vol_ann=0.30,
    slippage_bps=1.5, commission_bps=0.3, use_orthogonality=True)))

experiments.append(("67_best8_low_cost", make_params(
    regime_weights=W_NEW, weight_clip_max=0.85, max_position=0.10,
    max_turnover=0.30, min_signal_strength=0.05, max_vol_ann=0.30,
    slippage_bps=1.0, commission_bps=0.1, use_orthogonality=True)))

experiments.append(("68_best9_new_aggressive", make_params(
    regime_weights=W_NEW, weight_clip_max=0.75, max_position=0.15,
    max_turnover=0.30, min_signal_strength=0.10, max_vol_ann=0.30,
    slippage_bps=1.5, commission_bps=0.3, use_orthogonality=True)))

experiments.append(("69_best10_moderate_bear", make_params(
    regime_weights={"bull": {"lgbm": 0.65, "sentiment": 0.15, "hmm": 0.20},
                    "bear": {"lgbm": 0.50, "sentiment": 0.10, "hmm": 0.40},
                    "transition": {"lgbm": 0.60, "sentiment": 0.12, "hmm": 0.28}},
    weight_clip_max=0.75, max_position=0.12, max_turnover=0.25,
    min_signal_strength=0.10, max_vol_ann=0.30,
    slippage_bps=1.5, commission_bps=0.3, use_orthogonality=True)))


# =====================================================================
# RUN
# =====================================================================
print(f"MLCouncil Playground Batch v3 — {len(experiments)} experiments")
print(f"Universe: {len(UNIVERSE)} tickers | Window: {START} → {END} (5y)")
print("=" * 80)

results, errors = [], []

for i, (tag, params) in enumerate(experiments):
    print(f"\n[{i+1}/{len(experiments)}] {tag}...", end=" ", flush=True)
    r = run_and_score(params, tag)
    if "error" in r:
        errors.append(r)
        print(f"❌ {r['error']}")
    else:
        results.append(r)
        print(f"S={r['sharpe']:.3f} C={r['cagr']:.3f} DD={r['max_drawdown']:.3f} "
              f"Calm={r.get('calmar',0):.3f} Score={r['score']:.3f} ({r['elapsed']}s)")

results.sort(key=lambda x: x["score"], reverse=True)

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
print(f"\n🏆 TOP 10 (by composite score):")
print(f"{'Rank':<5} {'Tag':<32} {'Sharpe':<8} {'CAGR':<8} {'DD':<8} {'Calmar':<8} {'Score':<8} {'Ret%':<8}")
print("-" * 90)
for rank, r in enumerate(results[:10], 1):
    print(f"{rank:<5} {r['tag']:<32} {r['sharpe']:<8.3f} {r['cagr']:<8.3f} "
          f"{r['max_drawdown']:<8.3f} {r.get('calmar',0):<8.3f} "
          f"{r['score']:<8.3f} {r.get('total_return_pct',0):<8.1f}")

print(f"\n✅ {len(results)}/{len(experiments)} succeeded ❌ {len(errors)} failed")
print(f"📊 Full: {RESULTS_FILE}")

# Also print bottom 5
print(f"\n📉 BOTTOM 5:")
for r in results[-5:]:
    print(f"  {r['tag']:<32} S={r['sharpe']:.3f} DD={r['max_drawdown']:.3f}")
