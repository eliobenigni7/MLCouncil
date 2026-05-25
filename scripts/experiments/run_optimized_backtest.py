#!/usr/bin/env python3
"""
Ottimizzazione MLCouncil — backtest reale con modelli LGBM+HMM+FinBERT
e le migliori env flags trovate dal playground batch.

Attiva:
- Phase 1: variance penalty, covariance shrinkage, nuovi regime weights
- HRP soft prior blend (30%)
- EWM IC-Sharpe halflife 60gg
- TC lambda 2.5
- Parametri portfolio: pos=0.10, turn=0.25, sig=0.10
"""
from __future__ import annotations

import json, os, sys, time, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# =============================================================================
# ENV FLAGS — Phase 1 attivate
# =============================================================================
os.environ["MLCOUNCIL_PORTFOLIO_SHRINK_COV"] = "true"
# Non settare RISK_LAMBDA — usa il default automatico (1/max_vol_daily²)
if "MLCOUNCIL_RISK_LAMBDA" in os.environ:
    del os.environ["MLCOUNCIL_RISK_LAMBDA"]
os.environ["MLCOUNCIL_IC_SHARPE_HALFLIFE"] = "60"
os.environ["MLCOUNCIL_TC_LAMBDA"] = "2.5"
os.environ["MLCOUNCIL_HRP_SOFT_PRIOR"] = "true"
os.environ["MLCOUNCIL_HRP_BLEND"] = "0.30"
os.environ["MLCOUNCIL_COMMISSION_BPS"] = "0.5"
os.environ["MLCOUNCIL_SLIPPAGE_BPS"] = "3.0"
os.environ["MLCOUNCIL_AGGREGATOR_MODE"] = "linear"

# =============================================================================
# Nuovi regime weights — sovrascrivo config/regime_weights.yaml
# =============================================================================
import yaml

NEW_REGIME_WEIGHTS = {
    "regime_weights": {
        "bull":       {"lgbm": 0.65, "sentiment": 0.15, "hmm": 0.20},
        "bear":       {"lgbm": 0.55, "sentiment": 0.15, "hmm": 0.30},
        "transition": {"lgbm": 0.60, "sentiment": 0.15, "hmm": 0.25},
    },
    "weight_clip": {"min": 0.05, "max": 0.75},
    "performance": {
        "min_history_days": 60,
        "ic_rolling_window": 60,
        "sharpe_rolling_window": 120,
    },
    "orthogonality": {
        "max_correlation": 0.65,
        "correlation_window": 90,
        "auto_downweight": True,
        "downweight_factor": 0.5,
    },
}

REGIME_FILE = ROOT / "config" / "regime_weights.yaml"
REGIME_BAK = ROOT / "config" / "regime_weights.yaml.optimized_bak"

# Backup
if not REGIME_BAK.exists():
    shutil.copy2(REGIME_FILE, REGIME_BAK)

with open(REGIME_FILE, "w") as f:
    yaml.safe_dump(NEW_REGIME_WEIGHTS, f, sort_keys=False)

print("=" * 70)
print("MLCouncil — Backtest Ottimizzato (modelli reali)")
print("=" * 70)
print(f"Regime weights: {NEW_REGIME_WEIGHTS['regime_weights']}")
print(f"Clip:           {NEW_REGIME_WEIGHTS['weight_clip']}")
print(f"Shrink cov:     true")
print(f"IC Halflife:    60")
print(f"TC Lambda:      2.5")
print(f"HRP blend:      0.30")
print(f"Slippage:       3.0 bps")
print()

from scripts.one_year_backtest import run_one_year_backtest

WINDOWS = [
    ("2025-01-01", "2025-12-31", "2025"),
    ("2024-01-01", "2024-12-31", "2024"),
    ("2023-01-01", "2023-12-31", "2023"),
    ("2022-01-01", "2022-12-31", "2022"),
    ("2021-01-01", "2021-12-31", "2021"),
    ("2021-01-01", "2025-12-31", "2021-2025"),
]

all_results = {}

for start, end, label in WINDOWS:
    print(f"\n{'='*60}")
    print(f"📅 {label}: {start} → {end}")
    print('=' * 60)
    
    t0 = time.time()
    
    result = run_one_year_backtest(
        year_start=start,
        year_end=end,
        train_window_months=6,
        force_linear=True,
        rebalance_every=5,
        vol_daily=0.0095,
        max_pos=0.10,
        max_turnover_env=0.25,
    )
    
    elapsed = time.time() - t0
    print(f"⏱  {elapsed:.0f}s")
    
    if "error" in result:
        print(f"  ❌ ERROR: {result['error']}")
        all_results[label] = {"error": result["error"], "elapsed": round(elapsed, 1)}
    else:
        sharpe = result.get("sharpe", 0)
        oos_sharpe = result.get("oos_sharpe", 0)
        cagr = result.get("cagr", 0) * 100
        dd = result.get("max_drawdown", 0) * 100
        turnover = result.get("turnover", 0) * 100
        pbo = result.get("pbo", 0) * 100
        
        print(f"  Sharpe:       {sharpe:.4f}")
        print(f"  OOS Sharpe:   {oos_sharpe:.4f}")
        print(f"  CAGR:         {cagr:.2f}%")
        print(f"  Max DD:       {dd:.2f}%")
        print(f"  Turnover:     {turnover:.2f}%")
        print(f"  PBO:          {pbo:.1f}%")
        
        if sharpe >= 1.0:
            print(f"  ✅ SHARPE ≥ 1.0 — TARGET RAGGIUNTO!")
        else:
            print(f"  ❌ Sharpe < 1.0")
        
        all_results[label] = {
            "sharpe": round(sharpe, 4),
            "oos_sharpe": round(oos_sharpe, 4),
            "cagr_pct": round(cagr, 2),
            "max_dd_pct": round(dd, 2),
            "turnover_pct": round(turnover, 2),
            "pbo_pct": round(pbo, 1),
            "n_windows": result.get("windows", 0),
            "elapsed_s": round(elapsed, 1),
        }

# Report finale
print("\n\n" + "=" * 70)
print("📊 REPORT FINALE — Backtest Ottimizzato (modelli reali)")
print("=" * 70)

for label, r in all_results.items():
    if "error" in r:
        print(f"  {label:<12} ❌ {r['error']}")
    else:
        sharpe = r["sharpe"]
        target = "✅" if sharpe >= 1.0 else "❌"
        print(f"  {label:<12} S={sharpe:.3f}  OOS={r['oos_sharpe']:.3f}  "
              f"C={r['cagr_pct']:.1f}%  DD={r['max_dd_pct']:.1f}%  "
              f"T={r['turnover_pct']:.1f}%  {target}")

# Salva
OUT = ROOT / "data" / "results" / "optimized_backtest_results.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n📁 Risultati: {OUT}")

# Ripristina regime_weights originali
if REGIME_BAK.exists():
    shutil.copy2(REGIME_BAK, REGIME_FILE)
    REGIME_BAK.unlink()
    print("↩ Regime weights ripristinati.")
