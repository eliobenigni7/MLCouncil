# ADR-0006: Deep State-Space Regime Challenger (Shadow)

- Date: 2026-05-21
- Status: Accepted
- Decision owners: MLCouncil quant platform
- Related PR/Issue: Wave 2 track T2.3 (`docs/internal/disruptive-roadmap-2026-05-21.md`)

## Context

The daily council conditions portfolio weights on a discrete HMM regime label
(`bull` / `bear` / `transition`) from `models/regime.py`. A continuous latent
regime \(z_t \in \mathbb{R}^d\) from a deep state-space model (S4/Mamba-class)
may capture smoother transitions, but must not replace HMM in production until
walk-forward gating (T1.1) passes.

## Decision

1. **`models/regime_dss.py`** — `DeepRegimeModel` with `fit()`, `predict_embedding()`,
   `predict_regime()` (prototype softmax), `shadow_record()`, and ELBO helpers.
   Backend: `mamba-ssm` when importable (stub falls back to NumPy VAE); otherwise
   CPU NumPy recurrent VAE via `scipy.optimize`.
2. **`scripts/train_regime_dss.py`** — offline trainer writing
   `models/checkpoints/regime_dss_latest.pkl` (+ hash). Shadow only.
3. **`council/aggregator.py`** — `MLCOUNCIL_REGIME_MODE=label|embedding` (default
   `label`). Embedding mode blends `regime_weights` buckets via softmax over
   distance to centroids; optional `regime_embedding` kwarg on `aggregate()`.
4. **Pipeline** — HMM remains default; DSS runs only when
   `MLCOUNCIL_REGIME_DSS_SHADOW=true` (logging / comparison, no council promotion).

## Gating (T1.1, not enabled here)

- ELBO improvement vs HMM baseline ≥ 5% on validation macro window
- Transition sanity: no abrupt bull→bear without intermediate mass
- Council IC with `MLCOUNCIL_REGIME_MODE=embedding` within ±0.005 of label mode

## Consequences

- Positive: continuous regime signal for MoE/stacking (T3.1) without HMM relabel noise.
- Trade-off: shadow training adds ops steps; mamba-ssm CUDA stack optional.
- Operations: disabled by default; no Dagster asset change in this track.

## Alternatives Considered

1. **Replace HMM in `data/pipeline.py`** — violates champion/challenger policy (rejected).
2. **Pyro/NumPyro full VI** — heavier deps for v1; NumPy ELBO stub sufficient for shadow (deferred).
3. **Hard-code embedding→weights MLP** — deferred to T3.1 MoE gating (rejected for T2.3).

## Rollout Plan

1. Land model, train script, aggregator embedding mode, tests, ADR.
2. Weekly shadow train: `python scripts/train_regime_dss.py --compare-hmm`.
3. Paper experiment: `MLCOUNCIL_REGIME_MODE=embedding` + DSS centroids in council backtest.
4. Promote only after T1.1 walk-forward gate passes.

## Verification

```bash
python scripts/train_regime_dss.py --epochs 30
python -m pytest tests/test_regime_dss.py -v
```

## Rollback

- `MLCOUNCIL_REGIME_MODE=label` (default) — aggregator uses discrete HMM label only.
- `MLCOUNCIL_REGIME_DSS_SHADOW=false` (default) — skip DSS inference in scripts.
- Delete `models/checkpoints/regime_dss_latest.pkl` to drop shadow checkpoint.
