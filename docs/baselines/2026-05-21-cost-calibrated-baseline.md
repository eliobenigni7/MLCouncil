# Cost-Calibrated Baseline — 2026-05-21

Side-by-side reference for static lookup vs self-calibrated transaction costs (ADR-0003). Compare against [2026-05-21-clean-baseline.md](2026-05-21-clean-baseline.md) under the same walk-forward protocol.

**Status:** Template — run `python scripts/run_strategy_backtest.py --cost-mode=both` and paste metrics below.

---

## Provenance

| Field | Value |
|---|---|
| **Date** | 2026-05-21 |
| **Protocol** | `scripts/run_strategy_backtest.py --cost-mode=both` |
| **Artifacts** | `data/results/cost_ab.json`, `walk_forward_summary.json` |

---

## Cost A/B metrics

| Metric | Static lookup | Calibrated | Delta |
|---|---:|---:|---:|
| Net Sharpe | _TBD_ | _TBD_ | _TBD_ |
| Net CAGR | _TBD_ | _TBD_ | _TBD_ |
| Max drawdown | _TBD_ | _TBD_ | _TBD_ |
| Mean turnover | _TBD_ | _TBD_ | _TBD_ |
| Estimated costs (USD) | _TBD_ | _TBD_ | _TBD_ |
| Calibration version (SHA-256) | — | _TBD_ | — |

---

## Promotion gate

Run after backtest:

```powershell
python -c "
from pathlib import Path
import json
from backtest.validation import validate_cost_calibration_promotion
from council.cost_calibration import load_calibration, DEFAULT_CALIBRATION_PATH

ab = json.loads(Path('data/results/cost_ab.json').read_text())
artifact = None
if DEFAULT_CALIBRATION_PATH.exists():
    artifact = load_calibration(DEFAULT_CALIBRATION_PATH)
result = validate_cost_calibration_promotion(
    ab['static_stats'], ab['calibrated_stats'], artifact=artifact
)
print('passed:', result.passed)
print('reasons:', result.reasons)
"
```

---

## Notes

- Promotion requires calibrated net Sharpe within 0.1 of static, turnover within ±10%, and tier fill counts ≥ 30.
- On failure, `revert_to_static_cost_calibration()` writes `config/runtime_override.env` with `MLCOUNCIL_COST_CALIBRATION_PATH=`.
