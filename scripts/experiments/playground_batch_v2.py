#!/usr/bin/env python3
"""
Batch backtest playground experiment runner.
Bypasses simulate_weight_backtest float bug by doing the simulation inline.
Tests 30+ combos with 5-year window (2021-2025).
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
from council.transaction_costs import TransactionCostModel

RESULTS_FILE = ROOT / "data" / "results_playground" / "batch_results_v2.json"
UNIVERSE = load_available_universe()
START = "2021-01-01"
END = "2025-12-31"
CAPITAL = 1_000_000.0


def _safe_float(v, default=0.0) -> float:
    try:
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        if isinstance(v, pd.Series):
            return float(v.iloc[-1]) if not v.empty else default
        return float(v or default)
    except (ValueError, TypeError, IndexError):
        return default


def simulate_safe(equity_curve, stats_dict):
    """Convert equity_series to stats safely."""
    if equity_curve is None or (isinstance(equity_curve, pd.Series) and equity_curve.empty):
        return None
    
    # If we got here, the run succeeded. Read stats from result.stats directly.
    return stats_dict


def run_and_score(params: PlaygroundParams, tag: str) -> dict:
    t0 = time.time()
    try:
        result = run_playground_backtest(params)
    except Exception as e:
        return {"tag": tag, "error": str(e)[:200], "elapsed": round(time.time() - t0, 1)}
    
    s = result.stats
    if s is None or not s.get("sharpe", 0):
        # Fall back: compute from equity curve ourselves
        try:
            eq = result.equity_curve
            if eq is not None and len(eq) > 1:
                rets = eq.pct_change().dropna()
                sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 1e-12 else 0.0
                final = float(eq.iloc[-1])
                cagr = (final / CAPITAL) ** (252 / len(rets)) - 1 if len(rets) > 0 else 0.0
                roll_max = eq.cummax()
                dd = float(((eq - roll_max) / roll_max).min())
                s = {"sharpe": sharpe, "cagr": cagr, "max_drawdown": dd, "final_equity": final}
            else:
                return {"tag": tag, "error": "Empty equity curve", "elapsed": round(time.time() - t0, 1)}
        except Exception as e:
            return {"tag": tag, "error": f"Stat fallback failed: {e}", "elapsed": round(time.time() - t0, 1)}
    
    sharpe = _safe_float(s.get("sharpe"))
    cagr = _safe_float(s.get("cagr"))
    max_dd = _safe_float(s.get("max_drawdown"))
    calmar = _safe_float(s.get("calmar"))
    final_eq = _safe_float(s.get("final_equity"))
    total_return = (final_eq / CAPITAL - 1.0) * 100 if CAPITAL > 0 else 0.0
    
    if abs(max_dd) < 1e-12:
        calmar = 0.0
    
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
# EXPERIMENTS
# =====================================================================
experiments = []

new_w = {"bull": {"lgbm": 0.70, "sentiment": 0.12, "hmm": 0.18},
         "bear": {"lgbm": 0.65, "sentiment": 0.10, "hmm": 0.25},
         "transition": {"lgbm": 0.68, "sentiment": 0.12, "hmm": 0.20}}

new_w_mod = {"bull": {"lgbm": 0.65, "sentiment": 0.15, "hmm": 0.20},
             "bear": {"lgbm": 0.55, "sentiment": 0.15, "hmm": 0.30},
             "transition": {"lgbm": 0.60, "sentiment": 0.15, "hmm": 0.25}}

hmm_heavy = {"bull": {"lgbm": 0.60, "sentiment": 0.15, "hmm": 0.25},
             "bear": {"lgbm": 0.20, "sentiment": 0.10, "hmm": 0.70},
             "transition": {"lgbm": 0.40, "sentiment": 0.15, "hmm": 0.45}}

sent_heavy = {"bull": {"lgbm": 0.40, "sentiment": 0.40, "hmm": 0.20},
              "bear": {"lgbm": 0.25, "sentiment": 0.30, "hmm": 0.45},
              "transition": {"lgbm": 0.30, "sentiment": 0.35, "hmm": 0.35}}

equal_w = {"bull": {"lgbm": 0.34, "sentiment": 0.33, "hmm": 0.33},
           "bear": {"lgbm": 0.34, "sentiment": 0.33, "hmm": 0.33},
           "transition": {"lgbm": 0.34, "sentiment": 0.33, "hmm": 0.33}}

# Baseline
experiments.append(("bl_baseline", make_params()))

# === GROUP A: Regime weights ===
for name, rw in [("a1_new_weights", new_w), ("a2_new_mod", new_w_mod),
                 ("a3_hmm_heavy", hmm_heavy), ("a4_sent_heavy", sent_heavy),
                 ("a5_equal", equal_w)]:
    experiments.append((name, make_params(regime_weights=rw)))

# === GROUP B: Weight clip ===
for cmax in [0.60, 0.70, 0.75, 0.85]:
    experiments.append((f"b_clip_{cmax:.0f}", make_params(weight_clip_max=cmax)))

# === GROUP C: New weights + clip ===
for cmax in [0.70, 0.75, 0.85]:
    experiments.append((f"c_nw_clip_{cmax:.0f}", make_params(regime_weights=new_w, weight_clip_max=cmax)))

# === GROUP D: Orthogonality ===
for ortho in [True, False, False]:
    experiments.append((f"d_ortho_new_{ortho}", make_params(regime_weights=new_w, use_orthogonality=ortho)))

# === GROUP E: Max position ===
for mp in [0.05, 0.08, 0.10, 0.12, 0.15]:
    experiments.append((f"e_pos_{mp:.0f}pct", make_params(max_position=mp)))

# === GROUP F: Turnover ===
for to in [0.15, 0.20, 0.25, 0.30, 0.40]:
    experiments.append((f"f_turn_{to:.0f}pct", make_params(max_turnover=to)))

# === GROUP G: Min signal strength ===
for ms in [0.05, 0.10, 0.20, 0.30]:
    experiments.append((f"g_sig_{ms:.0f}pct", make_params(min_signal_strength=ms)))

# === GROUP H: Vol cap ===
for vol in [0.20, 0.25, 0.30, 0.35, 0.40]:
    experiments.append((f"h_vol_{vol:.0f}pct", make_params(max_vol_ann=vol)))

# === GROUP I: Costs ===
for slip, comm in [(1.0, 0.1), (1.5, 0.3), (3.0, 0.5), (5.0, 1.0), (8.0, 2.0)]:
    experiments.append((f"i_cost_{slip:.0f}sl_{comm:.1f}cm", make_params(slippage_bps=slip, commission_bps=comm)))

# === GROUP J: IC window ===
for icw in [30, 60, 90, 120]:
    experiments.append((f"j_icw_{icw}", make_params(ic_rolling_window=icw)))

# === GROUP K: Sharpe window ===
for srw in [60, 90, 120, 180]:
    experiments.append((f"k_srw_{srw}", make_params(sharpe_rolling_window=srw)))

# === GROUP L: Best guesses ===
experiments.append(("l_best1", make_params(
    regime_weights=new_w_mod, weight_clip_max=0.75, max_position=0.10,
    max_turnover=0.25, min_signal_strength=0.10, slippage_bps=1.5,
    commission_bps=0.3, max_vol_ann=0.30, use_orthogonality=True)))

experiments.append(("l_best2", make_params(
    regime_weights=new_w, weight_clip_max=0.75, max_position=0.12,
    max_turnover=0.30, min_signal_strength=0.05, slippage_bps=1.5,
    commission_bps=0.3, max_vol_ann=0.35, use_orthogonality=True)))

experiments.append(("l_best3_ortho_off", make_params(
    regime_weights=new_w_mod, weight_clip_max=0.75, max_position=0.10,
    max_turnover=0.30, min_signal_strength=0.10, slippage_bps=1.5,
    commission_bps=0.3, max_vol_ann=0.30, use_orthogonality=False)))

experiments.append(("l_best4_bear_hedge", make_params(
    regime_weights={"bull": {"lgbm": 0.65, "sentiment": 0.15, "hmm": 0.20},
                    "bear": {"lgbm": 0.40, "sentiment": 0.10, "hmm": 0.50},
                    "transition": {"lgbm": 0.55, "sentiment": 0.15, "hmm": 0.30}},
    weight_clip_max=0.75, max_position=0.10, max_turnover=0.30,
    min_signal_strength=0.05, slippage_bps=1.5, commission_bps=0.3,
    max_vol_ann=0.30, use_orthogonality=True)))

experiments.append(("l_best5_aggressive", make_params(
    regime_weights={"bull": {"lgbm": 0.75, "sentiment": 0.10, "hmm": 0.15},
                    "bear": {"lgbm": 0.50, "sentiment": 0.10, "hmm": 0.40},
                    "transition": {"lgbm": 0.70, "sentiment": 0.10, "hmm": 0.20}},
    weight_clip_max=0.80, max_position=0.15, max_turnover=0.35,
    min_signal_strength=0.05, slippage_bps=1.5, commission_bps=0.3,
    max_vol_ann=0.40, use_orthogonality=False, ic_rolling_window=90)))

experiments.append(("l_best6_conservative", make_params(
    regime_weights=new_w_mod, weight_clip_max=0.65, max_position=0.07,
    max_turnover=0.20, min_signal_strength=0.15, slippage_bps=3.0,
    commission_bps=0.5, max_vol_ann=0.25, use_orthogonality=True)))

experiments.append(("l_best7_high_pos_low_turn", make_params(
    regime_weights=new_w_mod, weight_clip_max=0.75, max_position=0.15,
    max_turnover=0.15, min_signal_strength=0.05, slippage_bps=1.5,
    commission_bps=0.3, max_vol_ann=0.30, use_orthogonality=True)))

experiments.append(("l_best8_low_cost_wide_clip", make_params(
    regime_weights=new_w, weight_clip_max=0.85, max_position=0.10,
    max_turnover=0.30, min_signal_strength=0.05, slippage_bps=1.0,
    commission_bps=0.1, max_vol_ann=0.30, use_orthogonality=True)))

# =====================================================================
# RUN
# =====================================================================
print(f"MLCouncil Playground Batch v2 — {len(experiments)} experiments")
print(f"Universe: {len(UNIVERSE)} tickers | Window: {START} → {END} (5 years)")
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
        print(f"Sharpe={r['sharpe']:.3f} CAGR={r['cagr']:.3f} DD={r['max_drawdown']:.3f} Score={r['score']:.3f} ({r['elapsed']}s)")

results.sort(key=lambda x: x["score"], reverse=True)

out = {
    "n_experiments": len(experiments),
    "n_succeeded": len(results),
    "n_failed": len(errors),
    "window": {"start": START, "end": END},
    "universe_size": len(UNIVERSE),
    "top_results": results[:10],
    "all_results": results,
    "errors": errors,
}
with open(RESULTS_FILE, "w") as f:
    json.dump(out, f, indent=2, default=str)

print("\n" + "=" * 80)
print(f"\n🏆 TOP 10 RESULTS (by composite score):")
print(f"{'Rank':<6} {'Tag':<30} {'Sharpe':<8} {'CAGR':<8} {'DD':<8} {'Calmar':<8} {'Score':<8} {'Return%':<8}")
print("-" * 85)
for rank, r in enumerate(results[:10], 1):
    print(f"{rank:<6} {r['tag']:<30} {r['sharpe']:<8.3f} {r['cagr']:<8.3f} {r['max_drawdown']:<8.3f} {r.get('calmar',0):<8.3f} {r['score']:<8.3f} {r.get('total_return_pct',0):<8.1f}")

print(f"\n✅ Succeeded: {len(results)}/{len(experiments)} ❌ Failed: {len(errors)}")
print(f"📊 Full results: {RESULTS_FILE}")
