# MLCouncil Mathematics Drill-Down — Autonomous Council 2026–2030

Status: **Draft v1**
Date: 2026-08-13
Companion to: `docs/roadmap-2026-2030-autonomous-council.md`
Method: every section reports the **exact math currently in the code** (file:line), then the
mathematical critique, then the rigorous 2030 upgrade with its **verification statistic**.

---

## 0. The three structural mathematical sins

Everything below is a variation of three sins:

1. **Gaussianity.** VaR MC, parametric VaR, generative stress all assume MVN. Daily equity
   returns have excess kurtosis 2–7 (kurtosis 5–10 vs 3) and tail index α ≈ 3–4 (S&P 500 daily).
   P(|Z|>3) is 0.27% Gaussian; empirically it is 0.5–1.5%. Gaussian 99% VaR understates 1-in-100
   losses by roughly 2–3× versus EVT estimates.
2. **iid assumptions.** `sqrt(horizon)` scaling of VaR, Sharpe SEs without autocorrelation,
   IC SEs without overlap correction. Volatility clusters (GARCH persistence) make √h scaling
   wrong for multi-day risk.
3. **Correlation-as-dependence.** Pearson everywhere; no tail dependence. Correlations increase
   in down markets (Longin–Solnik), and multivariate Gaussian has **zero** tail dependence:
   it structurally cannot model co-crash.

---

## 1. Risk — M10 multivariate VaR (roadmap F-0.1)

### 1.1 Current math (`council/risk_engine.py`)

| Method | Formula (code) | Lines |
|---|---|---|
| Historical | `r_scaled = r·√h`; `VaR = percentile(r_scaled, (1−c)·100)`; `CVaR = mean(r_scaled ≤ VaR)`; dollar = `|VaR|·V`; guard `len(r) < 30 → (0,0)` | 217–234 |
| Parametric | `μ_h = μ·h`, `σ_h = σ·√h`, `z = Φ⁻¹(0.99)`; `VaR = μ_h − z·σ_h`; `CVaR = μ_h − σ_h·φ(z)/(1−c)` | 236–259 |
| Monte Carlo | `R ~ N(μ·h, Σ·h)` (n=10⁴, `multivariate_normal`); `P&L = (R @ w)·V`; percentile; Σ regularized by ridge `max(diag_mean·1e-6, 1e-12)` + eigenvalue clipping | 261–322 |
| Generative | `X ~ N(μ·s, Σ·s²)` (s = regime_scale, n=10⁴); **equal-weighted** `r_p = mean(X, axis=1)`; `VaR95 = q₀.₀₅(r_p)`; **`CVaR = 1.25·VaR95`** | 345–354 |
| Pre-trade | `VaR = 2.326·√(w'Σw)` (fixed z, 1-day) | 644–648 |

**Hidden-Gaussian finding #1:** `CVaR = 1.25·VaR₉₅` is *exactly* the Gaussian ES/VaR ratio at 95%:
φ(1.6449)/(0.05·1.6449) = 2.0627/1.6449 = **1.2539**. The "generative" method simulates 10⁴
scenarios and then throws away the empirical tail, replacing it with the parametric Gaussian
ratio. The simulation is decorative.

**Hidden-Gaussian finding #2:** MC VaR *is* multivariate in asset space (`R @ w`), but it is a
**single-step** Gaussian draw: no paths, no vol clustering, no tail dependence, and the
regularization is eigenvalue clipping (arbitrary) rather than shrinkage. The drift register's
"univariate" criticism is correct in spirit: the *distributional model* is univariate Gaussian
for the portfolio P&L.

### 1.2 Why the upgrade matters (quantified)

- **Tails**: Gaussian ES₉₉ = μ − 2.665σ. A Student-t with ν=5 (typical for equity returns):
  ES₉₉ = μ − 4.21σ (numerically: t₅ 99% quantile 3.365, ES ratio higher). Gap ≈ **1.6×**.
- **Co-crash**: Gaussian λ_L = 0. t-copula (ν=5, ρ=0.5): λ_L = 2·t₆(−√((ν+1)(1−ρ)/(1+ρ))) =
  2·t₆(−√2) ≈ 0.21. A 26-asset book with ρ=0.5 average correlation and Gaussian assumption
  will miss simultaneous 3σ moves that happen ~21% of the time when one asset is in its tail.
- **Multi-day**: √h scaling assumes iid. With GARCH(1,1) persistence (α+β ≈ 0.99, as coded in
  `covariance_dynamic.py:68`), 10-day VaR via √10 can be off by 20–40% depending on regime.

### 1.3 Proposed math (in order of priority)

**Step 1 — Multi-step simulation with time-varying covariance.**
Simulate daily paths t = 1..h, not one horizon draw. Daily Σ_t from the repo's own DCC(1,1)
(`covariance_dynamic.py`):
```
GARCH(1,1):   σ²_{i,t} = ω_i + α_i·ε²_{i,t−1} + β_i·σ²_{i,t−1}
DCC:          Q_t = (1−a−b)·Q̄ + a·(e_{t−1}·e'_{t−1}) + b·Q_{t−1}
              R_t = diag(Q_t)^(−1/2) · Q_t · diag(Q_t)^(−1/2)
              Σ_t = D_t · R_t · D_t          (D_t = diag(σ_{i,t}))
```
Compound returns pathwise; portfolio VaR/ES = empirical quantiles of the compounded P&L.
Replace ridge+eigenvalue-clipping with the Ledoit–Wolf shrinkage already imported in the repo
(`covariance_dynamic.py:230–242`).

**Step 2 — t-copula for tail dependence.**
```
Z ~ t_ν(0, R);  X_i = μ_i + σ_i·√((ν−2)/ν)·Z_i;   R = Ledoit–Wolf correlation
```
Calibrate ν by MLE on standardized residuals (or EVT margins + t-copula, the standard
quant-risk construction). Lower-tail dependence:
```
λ_L = 2·t_{ν+1}( −√( (ν+1)(1−ρ) / (1+ρ) ) )
```

**Step 3 — EVT for extreme quantiles (beyond 99%).**
Peaks-over-Threshold with GPD on exceedances above u:
```
VaR_α = u + (σ/ξ)·( (n/N_u)·(1−α) )^{−ξ} − 1 )      (ξ ≠ 0)
ES_α  = (VaR_α + σ − ξ·u) / (1−ξ)
```
ξ estimated by Hill: `ξ̂ = (1/k)·Σ_{i=1}^k ln(X_(i)/X_(k+1))`; threshold via mean-residual-life.

**Step 4 — Stress replay (the M10 ADR's "stress replay" requirement).**
- *Eigenvalue stress*: Σ* = V·diag(λ·(1+s_k))·V' on the top-k principal components.
- *Correlation stress*: Σ* = D·(ρ*·11' + (1−ρ*)·I)·D with ρ* a target (e.g., 0.9).
- *Narrative scenarios* from `generative_stress.py` (LLM-driven parameterization) become explicit
  shocks (μ_s, Σ_s), replacing the current single `regime_scale` scalar.

**Step 5 — Report ES, not just VaR.** VaR is not subadditive; ES is coherent:
```
ES_α = (1/α)·∫₀^α VaR_u du   ≈  mean of the α-tail order statistics
```

### 1.4 Verification math (the part that makes it 2030)

| Test | Statistic | Criterion |
|---|---|---|
| Kupiec POF (unconditional coverage) | LR_POF = −2·ln[ ((1−p)^{T−N}·p^N) / ((1−N/T)^{T−N}·(N/T)^N) ] | ~ χ²₁, reject if > 3.84 |
| Christoffersen (independence) | LR_ind = −2·ln[ (1−π)^{n₀₀+n₁₀}·π^{n₀₁+n₁₁} / ((1−π₀)^{n₀₀}·π₀^{n₀₁}·(1−π₁)^{n₁₀}·π₁^{n₁₁}) ] | ~ χ²₁ |
| Joint | LR_POF + LR_ind | ~ χ²₂ |
| ES backtest | Acerbi–Szekely: empirical ES from exceedances vs model ES, significance via simulation under H₀ | Z1 ≈ 0 within MC band |
| Coverage calibration | Empirical hit rate over rolling windows (already exists: `tests/test_council.py:312–337` pattern) | band around p |

Existing tests to extend: `tests/test_risk_engine.py:96–163` (MC reproducibility, correlation
structure) — add Kupiec on synthetic data with known non-Gaussian contamination.

---

## 2. The immune system (roadmap F-0.2)

### 2.1 ADWIN — current vs canonical (`council/drift.py:37–123`)

```
δ' = δ / log₂(max(2, n_buckets))
ε_cut = √(2·m·σ²_left·δ') + (2/(3·m))·δ'          (Bernstein-type bound)
m     = 1/(1/w_left + 1/w_right)  = w_l·w_r/(w_l+w_r)
drift ⟺ |μ̂_left − μ̂_right| > ε_cut
```
The `m` term is exactly Bifet–Gavaldà's n₀n₁/(n₀+n₁). Two deviations from the canonical ADWIN:
(a) the pooled variance is computed **only from the left window** (line 104) — asymmetric; the
canonical form uses both windows: `ε_cut = √( (1/(2m₀) + 1/(2m₁))·ln(4/δ') )`; (b) bucket variance
uses raw second moment (`E[x²] − mean²`, line 61) — fine, but the left-only pooling biases the
cut against right-window variance. **Upgrade**: symmetric pooling, and add the CUSUM alternative
for mean drift: `S_t = max(0, S_{t−1} + x_t − μ₀ − k)`, alarm at h; k = δ_target/2, h from ARL.

### 2.2 DDM — standard, correct (`drift.py:147–194`)

`p̂ ± ŝ` with ŝ = √(p̂(1−p̂)/n); warning at p_min + 2·s_min, drift at p_min + 3·s_min (Gama et al.).
No change needed; only note: binomial CI width 1/√n means with n=30 the 2σ band is ±0.18 —
warnings are noisy by construction; feed it rank-returns, not raw indicators, if false-alarm rate
matters.

### 2.3 "TDA" is not TDA (`council/tda_warning.py:52–67`)

Current: `β₁_proxy = mean(|ρ_ij|)` over the 30-day correlation matrix, alert if ≥ 0.35. This is
a mean-correlation monitor, not persistence homology. Calibration sanity: under the null of
independent returns, E|ρ̂| ≈ √(2/π)·1/√(T−2) ≈ 0.15 at T=30 (and Marcenko–Pastur gives
λ_max = (1+√(N/T))² ≈ 3.7 for N=26, T=30) — so 0.35 is above noise, but the statistic carries no
topological information (no Betti numbers, no filtration, no sliding embedding).

**Upgrade** (real TDA, cheap with `ripser`):
- Sliding-window point cloud: delay τ (first autocorrelation zero-crossing), embedding dim d
  (false-nearest-neighbors).
- Vietoris–Rips filtration; track **Betti-1 count** (loops = market "cycles"/disorder) and
  **persistence entropy**: `H = −Σ p_i·ln p_i`, `p_i = l_i/Σ l_j` (l_i = lifetime of feature i).
- Early-warning signal: Betti-1 / entropy increasing above a permutation-null quantile (block
  bootstrap of returns), instead of a fixed 0.35 threshold.

### 2.4 "Causal drift" is thresholded correlation (`council/causal_drift.py:58–94`)

Current: link if `|Pearson(ρ(feature, return))| ≥ 0.15`; drift = `(|added|+|removed|)/|base| ≥ 0.25`.
No temporal ordering, no conditioning → spurious links from common factors (a VIX link appears
whenever any macro factor moves).

**Upgrade** (PCMCI-lite, no tigramite needed):
- Partial correlation with Fisher z-transform:
```
ρ_{XY|Z} = (ρ_XY − ρ_XZ·ρ_ZY) / √((1−ρ²_XZ)(1−ρ²_ZY))
z = 0.5·ln((1+r)/(1−r))  ~  N(0, 1/(n − |Z| − 3))  under ρ = 0
```
- PC skeleton: start complete lagged graph, remove edges by CI tests at increasing |Z|;
  keep only **lagged** predictors of forward_return (temporal order gives causal direction).
- Drift metric: Jaccard distance on parent sets `d = |A Δ B| / |A ∪ B|` (instead of the
  asymmetric change_frac), significance from permutation null.

---

## 3. The promotion gate — statistics of autonomy (roadmap P-2.1/P-2.3)

### 3.1 Current gate (`backtest/validation.py:890–925`, `walkforward_promotion_gate.py`)

```
pass ⟺  oos_sharpe_challenger ≥ oos_sharpe_champion − 0.1
        ∧ PBO_proxy ≤ 0.5
        ∧ walk_forward_window_count ≥ 8
streak: 3 consecutive passes (weekly CI) → promote
oos_sharpe = mean over windows of (mean/std·√252)
PBO_proxy  = mean( (IS ≥ median(IS)) ∧ (OOS ≤ median(OOS)) )
```

### 3.2 Mathematical critique

1. **The 0.1-Sharpe tolerance is near noise.** With T = 63 OOS days per window,
   SE(SR̂) ≈ √((1 + 0.5·SR̂²)/T) ≈ √((1+0.5·0.36)/63) ≈ **0.14** (Lo 2002, iid). The tolerance
   (0.1) is ~0.7 SE: a challenger with *equal* true Sharpe passes with high probability by luck.
2. **Multiple testing is unaccounted.** The CI matrix is 4 models × 11 windows, run weekly →
   ~50+ strategy-window trials per quarter. Best-of-many Sharpe inflates; the gate needs the
   number of trials m in the decision. (This is the PBO machinery's own point, applied to the
   gate itself.)
3. **PBO_proxy is the simplified version.** The full Bailey–Borwein–López-de-Prado CSCV:
   split S trials into halves across all C(S, s) combinations; ω_c = fraction of IS-best
   strategies underperforming the OOS median; `PBO = (2/C)·Σ_c max(ω_c − 0.5, 0)`.
4. **Streak = 3 is a run-test.** Under a null pass-rate p, P(3 consecutive passes) = p³;
   p = 0.5 → 12.5%. Not a strong filter on its own; it buys temporal consistency, which is good,
   but its power should be computed.

### 3.3 Proposed math

**Deflated Sharpe Ratio (DSR)** as the promotion statistic:
```
DSR = Φ[ (SR̂ − SR₀)·√(T−1) / √(1 − γ₃·SR̂ + ((γ₄−1)/4)·SR̂²) ]
SR₀ = E[max_{n≤m} SR] ≈ (1−γ)·Φ⁻¹(1−1/m) + γ·Φ⁻¹(1−1/(m·e)),  γ = 0.5772 (Euler–Mascheroni)
```
where m = number of independent trials (hypotheses evaluated per quarter), γ₃, γ₄ = skewness,
kurtosis of OOS returns. Promote only if DSR ≥ 0.95. This *replaces* the ad-hoc −0.1 tolerance
with a proper inference that punishes trial multiplicity.

**IC tests with overlap correction.** For signal IC (Spearman, cross-sectional per day):
```
SE(IC̄) = σ_IC/√N · √(1 + 2·Σ_{k=1}^K (1 − k/(K+1))·ρ_k)     (Newey–West on daily ICs)
```
Reject H₀: IC = 0 at the family level with Holm–Benjamini–Hochberg across hypotheses, and
allocate an explicit per-hypothesis α budget (the "hypothesis spend" of P-2.1).

**Combining K windows.** Don't average Sharpes; combine at the z level:
```
Stouffer:  z̄ = (1/√K)·Σ_k z_k          Fisher:  χ² = −2·Σ_k ln(p_k) ~ χ²_{2K}
```
and require both the combined test *and* the streak (temporal consistency), each with its own
documented power.

**Paired comparison** (champion vs challenger on aligned returns): Ledoit–Wolf (2008)
two-sample Sharpe test on the daily return series, instead of comparing means of window Sharpe.

### 3.4 Alpha-decay / retirement (P-2.3)

Model IC decay: `IC(t) = IC₀·2^(−t/h)`; h estimated by OLS on ln(IC) (log-linear, robust to
outliers with Huber loss). Retirement rule:
```
retire ⟺ ĥ ≤ h_min  ∨  SPRT(IC̄ ≤ 0) crosses lower boundary
```
SPRT on daily ICs: likelihood-ratio boundaries for H₀: IC = 0 vs H₁: IC = δ, with error rates
α = β = 0.05 — this gives a *statistically principled* "retire after ~K days" latency instead of
a fixed halflife.

---

## 4. Online learning gate (T1.2) — the 0.05 threshold is noise-dominated

Current (`models/online.py:135–143, 315–322`): accept refit iff `ic_today ≥ ic_baseline − 0.05`,
IC = Spearman over a **10-day** eval slice.

**The math problem:** under H₀ (no skill change), SE(ρ̂_Spearman) ≈ 1/√(n−1) ≈ **0.33** for n = 10.
The 0.05 threshold is ~**0.15σ**: the gate accepts essentially every refit (it only rejects
catastrophic drops). It provides the *illusion* of an IC gate.

**Upgrade:** set the tolerance from the noise model: δ ≥ z_α·√((1−ρ²)/(n−1)) with power
analysis n ≥ ((z_α + z_β)/δ)²·σ²; or use a paired Wilcoxon signed-rank test on daily ICs
(refit vs baseline, same days); or SPRT with 10-day updates. Also: evaluate IC on **overlapping
days** (the current eval slice vs the refit window overlap — document the leakage boundary).

---

## 5. Sizing & portfolio math

### 5.1 Conformal sizing — guarantees and limits

Current (`council/conformal.py`): MAPIE `CrossConformalRegressor(method="plus", cv=10)`, Ridge,
coverage = 0.80, multiplier `mult = clip(exp(1 − width/median(width)), 0.3, 1.8)`;
`filter_low_confidence`: zero if width ≥ p85.

Math notes:
- **Coverage bound** (Barber–Candès–Ramdas–Tibshirani 2021, jackknife+/CV+):
  `P(Y ∈ C) ≥ 1 − 2α·n/(n+1)` → asymptotically **1 − 2α = 0.60** at α = 0.20. The nominal 0.80
  is not guaranteed; only 0.60 is. The empirical test (`tests/test_council.py:312–337` targets
  ≥0.85) suggests the real-world coverage is higher, but the guarantee is what it is. Either
  accept and document, or switch to split conformal (`1 − α` guarantee, α = 0.10 → 0.90) with a
  fresh calibration split.
- **Exchangeability is violated under drift.** Split conformal requires exchangeable
  calibration data; markets drift. Upgrade: **weighted conformal** with recency weights
  `w_i = λ^{T−i}` (λ ≈ 0.98): the quantile of the weighted score distribution restores
  validity under covariate shift (Tibshirani et al. 2019).
- The alpha multiplier `exp(1 − width_norm)` is log-linear shrinkage of signal by uncertainty —
  reasonable; document that it voids the coverage guarantee for the *sized* position (it is a
  decision layer above the interval, which is fine — just not a conformal claim).

### 5.2 CQR (`council/cqr.py`)

Pinball loss `ρ_τ(e) = max(τ·e, (τ−1)·e)`; split conformal on median residuals; interval
`[Xβ_lo + b_lo + q_lo, Xβ_hi + b_hi + q_hi]`. Correct split-conformal construction. Note:
residual quantiles from the *median* model (lines 107–110) — standard CQR uses quantile-model
residuals; median-residual calibration is the simpler "conformalized mean" variant with weaker
guarantees. Coverage by vol-quintile is already tested (`tests/test_cqr.py:53–62`) — good;
extend to weighted conformal under drift.

### 5.3 CVXPY objective — it is already the right shape (`council/portfolio.py:409–457`)

```
max  α_eff'w − 0.5·λ_risk·w'Σw − λ_tc·TC
s.t. 1'w = 0.90;  w ≤ cap_tier;  w'Σw ≤ σ_max²;  ‖w − w_curr‖₁ ≤ 2·max_turnover
     w ≥ 0;  sector: Σ_{i∈S} w_i ≤ cap_S;  beta: |w'β| ≤ 0.40
TC = (‖w − w_curr‖₁/2)·(comm + slipp)/10⁴
```
Note the elegant normalization: `λ_risk = 1/σ_max²` (line 422) makes the risk term
`0.5·(σ_p/σ_max)²` — order-1 at the cap. The turnover penalty λ_tc = 2.0 is a heuristic in
return units; the mathematically consistent value is λ_tc = 1 (both terms are already in
objective units), or calibrate λ_tc so the marginal TC equals marginal expected cost — this is a
one-line fix worth an ADR note.

### 5.4 Kelly — docstring/implementation mismatch (`council/fractional_kelly.py:116–132`)

Docstring claims rolling variance; code uses **cross-sectional variance of signals**:
`f* = clip(k·μ/σ²_cs, 0, max_position)`, k = 0.3. For iid Gaussian bets the exact Kelly is
`f* = μ/σ²` (maximizer of E[ln(1 + f·r)]); fractional 0.3 is a standard estimation-error
discount — fine. But cross-sectional σ² is not the P&L variance of the bet; the correct input is
the *time-series* variance of realized signal-aligned returns (σ²_ε from r = β·s + ε):
`f* = k·β·s/σ²_ε`. Document the intended semantic, or switch to the time-series estimator.

### 5.5 Drawdown breaker (`council/risk_rules.py:179–185`, portfolio drawdown scale)

`scale = max(0.25, 1 − excess/0.07)` is an ad-hoc linear de-risker. The correct calibration is
the drawdown distribution: for zero-drift Brownian motion over T days,
`E[MD_T] ≈ 0.63·σ·√T` (Magdon-Ismail et al. 2004), and the exact distribution is computable by
Monte Carlo under the portfolio model. Recommend: set the 0.07 threshold from the simulated
MD₉₅ quantile, and make the scale function `scale_t = σ_target/σ̂_t` (vol-targeting identity)
clipped to [0.25, 1] — same behavior, principled inputs.

---

## 6. Execution math

### 6.1 Cost model — already moving the right way (`council/transaction_costs.py`)

- √-law: `impact = base·η·√(notional/ADV)` — matches the empirical square-root law
  `Δp/σ = Y·√(Q/V)` (Lillo–Farmer 2004), Y ≈ 0.1–1.
- Blend: `α = min(1, n_fills/30)`; `cost = (1−α)·lookup + α·κ̂` — this is a **Bayesian
  posterior mean with a prior of effective sample size 30**: `κ̂_post = (n·κ̂ + 30·κ₀)/(n+30)`.
  Mathematically clean; upgrade to a proper Normal-inverse-gamma conjugate (per ticker/tier)
  and add regime conditioning (vol-scaled κ).
- Calibration source: `IS_bps = 10⁴·(fill − decision)/decision·sign(side)`, κ̂ = median per
  ticker/tier (`cost_calibration.py:114–119`). Median is robust; consider Huber/quantile
  regression with participation and vol covariates: `IS_bps = a + b·participation + c·σ̂ + ε`.

### 6.2 Slicing — TWAP/VWAP are the λ=0 limits of Almgren–Chriss (`execution/slicer.py`)

The optimal execution schedule (Almgren–Chriss 2000) minimizes `E[cost] + λ·Var[cost]` with
temporary impact η and permanent impact γ:
```
κ = √(λ·σ²/η);   x(t) = X·sinh(κ·(T − t))/sinh(κ·T)
```
- λ → 0 (risk-neutral): uniform schedule — **TWAP** (`qty_i = qty//n` + residue, exactly as
  coded at 110–121).
- Volume-weighted: TWAP with volume profile (`DEFAULT_VOLUME_PROFILE`, lines 52–60) — the
  variance-minimizing *implementation shortfall* benchmark, not an optimizer.
- λ → ∞: all at close.
Current slicing is sound as a baseline; the upgrade is to make `n_slices` and the profile
functions of estimated impact (κ from the calibrated cost model) — a closed-form AC trajectory
replaces the urgency→n_slices table (212–223). This is implementable in ~2 days and gives the
RL agent (below) a principled baseline to beat.

### 6.3 Router — point estimates where distributions exist (`execution/router.py`)

`cost(venue) + urgency·2.0`, constants (ALPACA 6.0 / IBKR 5.5 / COINBASE 12.0 bps). Two issues:
(1) constants are guesses, (2) no exploration — a wrong prior is never corrected. Upgrade:
per-venue cost histograms from the TCA feed, and Thompson sampling (or UCB with
exploration bonus `√(2·ln t/n_v)`) on the cost posteriors. Failover order stays.

### 6.4 RL execution — stub today, honest plan (`execution/rl_agent.py`, `lob_simulator.py`)

Current state: no RL (sb3 import only, model stays None, fallback TWAP with hardcoded
8.0 bps shortfall). LOB simulator is a **closed-form slippage formula**
(`slippage = 0.5·spread + η·participation`, lines 38–41) — no queue, no depth dynamics.

2030 math:
- **Baseline (analytic, theory-grounded)**: Almgren–Chriss trajectory from the calibrated κ
  (§6.2) — this is the *benchmark the RL agent must beat*, measured by IS vs arrival.
- **Agent**: PPO with GAE:
```
L^CLIP(θ) = E_t[ min(r_t(θ)·Â_t, clip(r_t(θ), 1−ε, 1+ε)·Â_t) ]
Â_t = Σ_{l≥0} (γλ)^l·δ_{t+l},   δ_t = r_t + γ·V(s_{t+1}) − V(s_t)
reward = −(IS_bps + λ_reg·|inventory|)     (implementation-shortfall shaping)
```
- **Simulator fidelity**: replace the closed form with a queue-based LOB: bid/ask queues with
  exponential arrival (intensity ∝ 1/(spread)), market-order arrivals Poisson, our own orders
  deplete queues; calibrate intensities to intraday data. **Domain randomization** on spread,
  depth, intensity (sim-to-real).
- **Verification**: agent vs AC baseline vs TWAP on out-of-sample days; paired shortfall
  differences with bootstrap CIs; never promote to live before beating AC on 20+ OOS days.

---

## 7. Aggregation & council math

### 7.1 EWM IC-Sharpe (`council/aggregator.py:648–670`)

```
α_ewm = 1 − 2^(−1/h),  h = min(halflife_env=60, max(2, len//2))     (halflife in days)
ICSharpe = EWM_mean(IC) / (EWM_std(IC) + 1e-9) · √252
```
Math notes:
- IC is Spearman cross-sectional, T−1 signal vs T return (line 415) — correct alignment,
  no lookahead. ✓
- The √252 annualization is a constant multiplier: it cancels under the simplex normalization
  of weights, so it is harmless but misleading in logs — consider removing for clarity.
- Noise: with 60 daily ICs and n_t ≈ 53 tickers, SE(IC̄) ≈ σ_IC/√60 ≈ 0.12–0.15 → IC-Sharpe
  values below ~1.5 are statistically indistinguishable from 0. The `max(0.1, ICSharpe)` floor
  (line 507) keeps signals alive regardless — document that floor as an explicit choice.
- **Degeneracy risk in EWM std**: EWM variance with halflife 60 has effective n ≈ 2h ≈ 120
  days but with heavy overlap; the z-scoring at the end (line 363–365) is the real normalizer.

### 7.2 Orthogonality & weight shrinkage (lines 158–185, 316–323)

`w_min ×= 0.5` if |ρ_ij| > 0.65 (90d pairwise); then renormalize **only if** Σw ≥ 0.85 →
`effective_weight_sum < 1` = confidence shrinkage (M6, resolved). This is defensible; note the
pairwise threshold is a crude stand-in for a spectral measure: the *condition number* or the
ratio of the top-2 eigenvalues of the signal correlation matrix would be more informative
(and is one line with numpy). Keep the behavior; upgrade the trigger.

### 7.3 MoE gating — built, never trained (`council/moe_gating.py`)

`gate = softmax(context·W / temperature)`, context = one-hot regime + mean IC; `effective =
gate_i · perf_i` renormalized. The module has **no loss and no training** (checkpoint comes from
a scaffold script). 2030 math for activation (P-1.1):
- Train by **hard-EM**: E-step assigns each day's best-performing expert as label, M-step fits
  W by multinomial logistic on context → softmax gating. (Simple, robust, no autodiff.)
- Or end-to-end: Gumbel-softmax straight-through estimator on the council objective
  (differentiable portfolio already exists in `portfolio_diff.py`).
- Temperature annealing: T: 2 → 1 over training (calibrated gating).

### 7.4 Regime embedding (lines 589–640) — RBF blend

`d_l = Σ(emb − c_l)²`; `softmax(−d)` → convex blend of regime buckets per model. This is a
radial-basis kernel on regime centroids — sound. Note the centroids default axis-aligned;
fine as prior, upgrade with data-fit centroids (k-means on HMM posterior means) once the HMM
has enough history.

---

## 8. Verification matrix — every claim needs its statistic

| Module / feature | Verification statistic | Exists? |
|---|---|---|
| Multivariate VaR (F-0.1) | Kupiec POF, Christoffersen, Acerbi–Szekely ES | partial (MC repro, `test_risk_engine.py:96–163`) |
| Conformal coverage | Empirical coverage + bound 1−2αn/(n+1) | yes (`test_council.py:312–337`, `test_cqr.py:53–62`) |
| Promotion gate | DSR, paired Sharpe (Ledoit–Wolf), Stouffer/Fisher over windows | partial (PBO proxy tests, `test_walkforward_promotion.py:56–64`) |
| IC claims | t-test with Newey–West SE; Spearman SE 1/√(n−1) | yes (`test_models.py:291–318`) |
| Drift detectors | Permutation nulls (block bootstrap); ARL on synthetic drift | no |
| PBO | Full CSCV PBO ∈ [0,1] | proxy only (`test_backtest_validation.py:27–44`) |
| Cost model calibration | IS-bps regression (participation, vol); posterior CI width | no |
| Execution (RL vs AC) | Paired shortfall, bootstrap CI, 20+ OOS days | no |
| Online gate | Power analysis on eval days; Wilcoxon paired IC | no |

---

## 9. Findings digest (the one-page version)

1. **`CVaR = 1.25·VaR₉₅` in "generative" stress is the Gaussian ES/VaR ratio** — the simulation
   is decorative; the tail is parametric. (risk_engine.py:354)
2. **The online-learning IC gate (0.05) is 0.15σ of its own noise** at n=10 — it gates nothing.
   (models/online.py:135)
3. **The promotion tolerance (−0.1 Sharpe) is ~0.7 SE of the estimator** — challengers pass by
   luck; needs DSR with trial multiplicity m. (validation.py:894)
4. **"TDA" is a mean-correlation monitor; "causal drift" is thresholded correlation** — both
   need the actual mathematics (Vietoris–Rips/Betti-1; partial-correlation PC skeleton).
5. **Conformal coverage guarantee is 0.60, not 0.80** (jackknife+ bound) — either document or
   switch to split conformal + weighted conformal under drift.
6. **λ_tc = 2.0** in the optimizer is inconsistent with the objective units; λ_tc = 1 or
   calibrated marginal-TC. (portfolio.py:104)
7. **Kelly docstring vs code mismatch** (rolling vs cross-sectional variance). (fractional_kelly.py:116)
8. **Execution is a stub with constants** (router 6.0/5.5/12.0 bps, RL 8.0 bps fallback) — the
   theory baseline (Almgren–Chriss from the calibrated cost model) is a 2-day implementation.
9. **Ledoit–Wolf exists but VaR uses ridge + eigenvalue clipping** — swap one for the other.
10. **ADWIN's left-only pooled variance** is the one real deviation from the canonical bound.
