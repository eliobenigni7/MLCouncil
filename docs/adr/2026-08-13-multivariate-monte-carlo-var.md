# ADR: Multivariate Monte Carlo VaR Upgrade (F-0.1)

- Date: 2026-08-13
- Status: Accepted
- Decision owners: Risk lane
- Related: `docs/math-drilldown-2026-2030-autonomous-council.md` section 1
  (M10 multivariate VaR, roadmap F-0.1), `docs/roadmap-2026-2030-autonomous-council.md`

## Context

`RiskEngine.compute_var_monte_carlo` (`council/risk_engine.py`) sampled a
**single-step** multivariate Gaussian draw `R ~ N(mu·h, Sigma·h)` and took the
empirical quantile of the horizon-scaled P&L. Three mathematical weaknesses
(drill-down §1.1–1.2):

1. **Hidden Gaussianity**: the distributional model is univariate Gaussian for
   the portfolio P&L — no paths, no vol clustering, no tail dependence. The
   covariance was regularized by an arbitrary ridge + eigenvalue-clipping
   (`_regularize_covariance`) instead of Ledoit–Wolf shrinkage.
2. **No tail dependence**: the Gaussian copula has zero lower-tail dependence
   (λ_L = 0); it cannot model co-crash. A t-copula with ν=5, ρ=0.5 has
   λ_L ≈ 0.21.
3. **√h scaling is wrong under vol clustering**: with GARCH persistence
   (α+β ≈ 0.99 as coded in `covariance_dynamic.py:68`), the 10-day VaR via
   √10 can be off by 20–40% depending on regime.

Related hidden bug (finding #1): `method="generative"` computed
`cvar_1d = var_1d * 1.25`, which is *exactly* the Gaussian ES/VaR ratio at 95%
(φ(1.6449)/(0.05·1.6449) = 1.2539) — the 10⁴ sampled scenarios were thrown
away and replaced by the parametric Gaussian ratio. The simulation was
decorative.

## Decision

Rewrite the Monte Carlo math in `council/risk_engine.py` while keeping the
public API (`compute_var(returns, positions, portfolio_value, method, ...)`
with default `method="historical"`, `compute_var_monte_carlo`, `compute_full_risk`)
and the default behavior unchanged.

1. **Multi-step daily paths.** Simulate `n_simulations` paths of `horizon`
   daily portfolio returns and read VaR/ES from the empirical quantiles of the
   pathwise-compounded P&L (`Π(1+r_t) − 1`). Daily covariance:

   - **DCC(1,1)** (GARCH(1,1) vols + EWMA correlation, same recursion as
     `council/covariance_dynamic.DCCEstimator`) when `arch` is installed —
     wrapped in try/except, no new dependencies:
     `σ²_{i,t} = ω_i + α_i ε²_{i,t−1} + β_i σ²_{i,t−1}`,
     `Q_t = (1−a−b)Q̄ + a e_{t−1}e'_{t−1} + b Q_{t−1}`,
     `Σ_t = D_t R_t D_t`.
   - **Fallback**: GARCH(1,1) vol forecasts around the constant Ledoit–Wolf
     correlation, calibrated by method of moments (persistence φ = AR(1) of
     squared returns, `ω = σ̄²(1−φ)`, seed `σ²_0 = φ·recent + (1−φ)·σ̄²`).
     No `arch` needed; the vol cluster drives the √h break.
   - **Last resort**: constant Ledoit–Wolf covariance (multi-step compounding
     still applies).

2. **Ledoit–Wolf shrinkage.** `_ledoit_wolf_covariance` (sklearn `LedoitWolf`,
   already used in `covariance_dynamic.py:230–242`) replaces
   `_regularize_covariance` (ridge + eigenvalue clipping), which was removed.

3. **t-copula for tail dependence.** Optional parameter `tail_dof`
   (default 50 ≈ Gaussian; `None` or ≥ 10⁴ = exactly Gaussian; 5 = heavy
   tails). Sampling `Z ~ t_ν(0, R)` with `R` the Ledoit–Wolf correlation,
   `X = μ + σ·√((ν−2)/ν)·Z`. Lower-tail dependence
   `λ_L = 2·t_{ν+1}(−√((ν+1)(1−ρ)/(1+ρ)))` — 0 for Gaussian, ≈ 0.21 for
   (ν=5, ρ=0.5). `compute_var`/`compute_full_risk` default `tail_dof=None`
   (legacy Gaussian) to keep the default behavior bit-exact; the low-level
   `compute_var_monte_carlo` defaults to 50 per the roadmap.

4. **ES alongside VaR.** `compute_var_monte_carlo` now returns a
   `MonteCarloVaRResult` dataclass with `var_pct/var_dollar` and
   `es_pct/es_dollar` (ES = mean of the P&L tail beyond the VaR quantile);
   `cvar_*` fields are aliases of `es_*` and iteration yields
   `(var_dollar, cvar_dollar)` so legacy tuple unpacking keeps working.

5. **Stress replay** (`stress_replay=False` default): when enabled, every
   daily covariance gets (a) the correlation stress
   `Σ* = D(ρ*·11′ + (1−ρ*)I)D` with ρ* = 0.9, then (b) a top-k eigenvalue
   shock `s_k = 0.5`: `Σ* = V diag(λ(1+s))V′`.

6. **Generative CVaR fix** (finding #1): `method="generative"` computes CVaR
   from the empirical tail of the 10⁴ scenarios already sampled
   (`mean(scenarios ≤ VaR₉₅)`) instead of `1.25·VaR₉₅`. Method semantics
   (regime_scale stress, equal-weight portfolio, √h scaling for 5d/10d)
   unchanged.

## Consequences

- **Positive**: MC VaR/ES now reflect paths, vol clustering and tail
  dependence; ES is reported explicitly; the generative CVaR is empirical
  rather than a hidden Gaussian ratio; the arbitrary ridge/clipping
  regularizer is replaced by a statistically grounded shrinker.
- **Trade-offs**: the fallback GARCH vol dynamics are calibrated by method of
  moments (persistence from squared-return autocorrelation) rather than MLE —
  a documented approximation that activates only when `arch` is absent;
  DCC/GARCH forecasts use conditional expectations (no innovation feedback
  inside the horizon). `compute_var_monte_carlo`'s return type changed from a
  2-tuple to `MonteCarloVaRResult` — unpacking and field access preserved; no
  in-repo caller outside `risk_engine.py`/tests reads it.
- **Operational impacts**: default `compute_var`/`compute_full_risk` behavior
  (historical default, Gaussian MC when `tail_dof=None`) is unchanged; no
  dashboard/API contract changes (`VaRReport` untouched).

## Alternatives Considered

1. Pure constant Ledoit–Wolf covariance fallback with multi-step compounding
   (spec-literal). Rejected: for iid data the compounded 10-day VaR collapses
   back to √10 scaling within MC noise, so the vol-clustering requirement and
   its verification test could not be satisfied without `arch`.
2. Importing `DCCEstimator._garch_standardized_residuals` / `_univariate_vols`
   from `covariance_dynamic.py`. Rejected: it couples the risk engine to
   private state and does not expose per-day covariance matrices; the DCC
   recursion is replayed locally (same math, `a=0.03, b=0.96` defaults read
   from `DCCEstimator`).
3. Adding `es_*` fields to `VaRReport`. Rejected as redundant: `VaRReport`
   `cvar_*` fields already are the ES; the ES extension lives where the MC
   simulation happens (`MonteCarloVaRResult`).

## Rollout Plan

- Step 1: this change (risk engine + tests + ADR), default behavior intact.
- Step 2: opt-in evaluation via `tail_dof=5` / `stress_replay=True` in
  shadow risk reports; compare ES/VaR deltas against `historical` baseline.
- Step 3: if `arch` is installed in deployment environments, the DCC daily
  covariance path activates automatically (try/except); verify with the
  multi-step test.

## Verification

```bash
.venv\Scripts\python.exe -m pytest tests/test_risk_engine.py -v
.venv\Scripts\python.exe -c "import council.risk_engine; import council.covariance_dynamic"
```

New tests (all pass, 13/13):
- Kupiec POF + Christoffersen independence on synthetic data: correct model
  LR < 3.84 (χ²₁ critical), wrong vol / clustered violations LR > 3.84.
- ES sanity: Gaussian synthetic data → MC ES matches closed-form
  `μ − σ·φ(z)/(1−c)` within 5%.
- Tail dependence: empirical λ̂_L > 0.1 for t(ν=5, ρ=0.5) at p = 5·10⁻⁴,
  Gaussian λ̂_L < 0.05.
- Multi-step: 10d MC VaR > 1d MC VaR and deviates from √10·VaR₁d by > 8%
  (measured ≈ 12%) on vol-clustered data.
- Stress replay raises MC VaR (~+25% measured on the fixture).
- Existing tests untouched and green (MC reproducibility with seed, seed
  override, correlation structure).
