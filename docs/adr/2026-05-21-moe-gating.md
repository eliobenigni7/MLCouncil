# ADR: Mixture-of-Experts Council Gating (T3.1 Shadow)

- Date: 2026-05-21
- Status: Accepted (shadow scaffold)
- Related: `docs/disruptive-roadmap-2026-05-21.md` Wave 3 T3.1

## Context

Linear regime-conditional weights in `CouncilAggregator` cannot express
non-linear expert selection when multiple alphas are active. MoE gating
learns $g_k(x_t, \text{regime})$ with $\sum_k g_k = 1$.

## Decision

1. **`council/moe_gating.py`** — `MoEGatingNetwork` (numpy softmax scaffold).
2. **`MLCOUNCIL_AGGREGATOR_MODE`** — `linear` (default) or `moe`.
3. **`scripts/train_moe_gating.py`** fits gate weights (IC proxy) →
   `models/checkpoints/moe_gate.pkl`; aggregator loads via `MoEGatingNetwork.load_or_create()`.
4. REINFORCE/PPO training deferred until walk-forward IC gate passes (≥ +0.01 IC delta).

## Consequences

- Positive: shadow path for non-linear blending without changing production default.
- Trade-off: untrained MoE must not run in production without promotion.

## Rollback

Unset `MLCOUNCIL_AGGREGATOR_MODE` or set `linear`.

## Verification

```bash
python scripts/train_moe_gating.py
python -m pytest tests/test_moe_gating.py -v
```
