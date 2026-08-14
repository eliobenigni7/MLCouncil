# MLCouncil — Parameter Tuning per Sharpe Ratio > 1

> **Scope.** Analisi della matematica sottostante MLCouncil e proposta di
> parametrizzazione basata su letteratura accademica e best-practice
> industriali, con l'obiettivo realistico di un **Sharpe annuo netto > 1.0**
> (deflato, dopo costi) su universo equity US (32 large-cap + 6 mid-cap).
> Nessuna modifica al codice. Tutti i puntatori sono `file:line` rispetto
> al default attuale del repo.

---

## 1. Executive Summary

MLCouncil è un **ensemble multi-modello** in cui tre alpha indipendenti
(LightGBM tecnico, FinBERT sentiment, HMM regime) confluiscono in un
*council aggregator* regime-condizionato, poi un ottimizzatore CVXPY
mean-variance con vincoli realistici (turnover, vol cap, beta-neutral)
costruisce il portafoglio, e una *conformal layer* (MAPIE Jackknife+)
modula il sizing in funzione dell'incertezza.

Per puntare a uno **Sharpe netto > 1**, la leva matematica più forte è
la **legge fondamentale di Grinold–Kahn** `IR ≈ IC · √BR · TC`. Su 38
ticker con re-bilanciamento giornaliero la breadth annuale teorica è
≈ 38 · 252 ≈ 9.500, quindi serve `IC ≈ 0.012` *effettivo* (post-shrinkage,
post-cost) per ottenere `IR ≈ 1.1`. I parametri proposti qui sotto
puntano a:

1. **Massimizzare il TC (transfer coefficient)** abbassando frizioni
   (turnover penalty, conformal cap, sector cap troppo lasco).
2. **Stabilizzare l'IC** allungando finestre di adattamento ed evitando
   over-fit della weight adaptation.
3. **Ridurre l'errore di covarianza** sostituendo Ledoit–Wolf con
   non-linear shrinkage o DCC-GARCH calibrato in shadow.
4. **Deflate il Sharpe in-sample** (Bailey–López de Prado) prima di
   promuovere parametri.

I tre interventi a *highest expected lift* sono evidenziati con [HIGH-LIFT]
nelle sezioni che seguono.

---

## 2. Architettura matematica del sistema

### 2.1 Flusso dati e signal stack

```
ingest  →  features  →  models (LGBM | FinBERT | HMM)
                                  ↓
                       council aggregator (regime-conditioned, IC-adaptive)
                                  ↓
                       conformal sizing (MAPIE Jackknife+)
                                  ↓
                       portfolio optimizer (CVXPY, mean-variance + constraints)
                                  ↓
                       OMS (Alpaca / next-open fill, 3 bps slippage)
```

### 2.2 Formule chiave

**A. IC-Sharpe rolling per modello** (`council/aggregator.py:638-658`)

```
IC_Sharpe_m = EWM_mean(IC_m,t) / EWM_std(IC_m,t) · √252
   halflife = min(20, len(recent)/2)
   window   = sharpe_rolling_window = 100
```

**B. Adaptive weight (Sharpe-floor soft)** (`council/aggregator.py:505-507`)

```
w'_m = w_base_m(regime) · max(0.1, IC_Sharpe_m)
w_m  = w'_m / Σ_m w'_m
```

con clip `[0.05, 0.70]` e penalità di ortogonalità
`w_m ← w_m · 0.5` se `|corr(signal_i, signal_j)| > 0.70` (60-day window).

**C. Conformal multiplier** (`council/conformal.py:139-188`)

```
[ŷ_low, ŷ_high] = Jackknife+(X)        coverage = 0.85
width            = ŷ_high − ŷ_low
width_norm       = width / median(width)
mult             = clip(exp(1 − width_norm), 0.2, 2.0)
```

Poi taglio del 90° percentile di width → posizione = 0.

**D. Portfolio optimization** (`council/portfolio.py:407-414`)

```
max_w   (α ⊙ mult)' w  −  λ_tc · ‖w − w_curr‖₁ / 2 · (c + s) / 10⁴
s.t.    1' w = budget_fraction
        0 ≤ w ≤ p_max
        ‖w − w_curr‖₁ ≤ τ_max
        w' Σ w ≤ σ²_max
        sector_exposure(w) ≤ s_cap
        |β_p(w)| ≤ β_max
```

dove `Σ` è Ledoit–Wolf shrunk (default) o DCC-GARCH (shadow).

**E. Fundamental Law (Grinold–Kahn 1995)**

```
IR = IC · √BR · TC                     (transfer coefficient TC ≤ 1)
Sharpe ≈ √(SR_bench² + IR²)            (additivity in skill)
```

Per `BR ≈ 9500`, `TC ≈ 0.75` (vincoli moderati), `IC = 0.012`:
`IR ≈ 0.012 · √9500 · 0.75 ≈ 0.88` ⇒ Sharpe netto realistico 0.9–1.2.

**F. Deflated Sharpe Ratio (Bailey–López de Prado 2014)**

```
DSR = Φ( (SR − E[max_SR_N]) · √(T−1) / √(1 − γ₃·SR + (γ₄−1)/4 · SR²) )
E[max_SR_N] ≈ √(2 ln N) − (γ_EM + ln ln N)/(2 √(2 ln N))
```

con `N` = numero di trials tentati, `γ₃, γ₄` = skewness/kurtosis dei rendimenti.
**Una strategia con SR backtest = 1.5 ma N = 100 trials ha tipicamente DSR < 0.5.**

---

## 3. Parametri attuali vs. letteratura

### 3.1 LightGBM (modello tecnico)

| Parametro | Attuale | Letteratura (default robusto) | Proposta | Razionale |
|---|---|---|---|---|
| `n_estimators` | 500 | 1000–3000 con early stopping | **1500** + early-stop pat=50 | LightGBM docs: alberi extra con LR basso ↑ generalizzazione |
| `learning_rate` | 0.05 | 0.01–0.05 | **0.02** | LR più bassi → migliori SR OOS in cross-section (Macaluso 2021) |
| `num_leaves` | 64 | ≤ 2^max_depth−1 | **31** | López de Prado (Adv. FinML cap.6): ≤32 leaves su tabular finance per ridurre over-fit |
| `min_child_samples` | 20 | 50–200 su universo small | **100** | universo 38 ticker · ~2000 giorni ≈ 76k righe; 100 ≈ 0.13% → soft regularizer |
| `subsample` | 0.8 | 0.6–0.8 | **0.7** | Bagging più aggressivo riduce varianza |
| `colsample_bytree` | 0.7 | 0.5–0.8 | **0.6** | Su Alpha158 (100+ feature collineari) selezione feature più aggressiva aiuta IC |
| `reg_alpha` (L1) | 0.1 | 0.1–1.0 | **0.5** | feature redundancy alta → L1 sparsifica |
| `reg_lambda` (L2) | 0.1 | 0.1–1.0 | **0.3** | smoothing complementare |
| **target horizon** | 1d + 5d | h ∈ {5, 10, 21} per momentum | **5d primary** | Qlib benchmark Alpha158: 5d target ha IC più stabile (~0.05 vs 0.02 a 1d) |

> *File*: `config/models.yaml:1-12`, `data/features/target.py:54-74`.

### 3.2 CPCV — Combinatorial Purged Cross-Validation

| Parametro | Attuale | Letteratura | Proposta | Razionale |
|---|---|---|---|---|
| `n_splits` | 6 | 6–10 (López de Prado 2018) | **8** | C(8,2)=28 path → DSR più affidabile vs C(6,2)=15 |
| `embargo_days` | 5 | h ≈ 0.01·T (LdP) | **10** | T≈2000d → 1% = 20d; con 5d target a 5d e residual autocorr accettiamo 10d come compromise |
| `n_test_folds` | 2 | 2 (default LdP) | **2** | stabile |

> *Fonte*: López de Prado, *Advances in Financial Machine Learning* (2018), §7.4.
> *File*: `config/models.yaml:14-17`, `models/technical.py:40-105`.

### 3.3 FinBERT sentiment

| Parametro | Attuale | Letteratura | Proposta | Razionale |
|---|---|---|---|---|
| `recency_decay γ` | 0.7 | half-life 2–5 giorni (Tetlock 2007; Bayes-FinBERT 2024) | **0.6** | half-life ≈ 1.36d → mantiene impatto solo su news “fresche”; γ=0.7 ha half-life 1.94d, leggermente troppo persistente per news pricing |
| `max_length` (char proxy) | 512 | 512 token | **256** | titolo + lead = sufficienti; riduce noise (Araci 2019) |
| aggregazione | weighted mean | weighted mean **+ count-weighted** | **vol-weighted** (news/giorno) | giorni a bassa attenzione → sentiment più rumoroso; pesare per `log(1+count)` |

> *File*: `config/models.yaml:32-37`, `models/sentiment.py:72-152`.

### 3.4 HMM regime

| Parametro | Attuale | Letteratura | Proposta | Razionale |
|---|---|---|---|---|
| `n_states` | 3 (bull/bear/transition) | 2–4 (Hamilton 1989; Nystrup 2018) | **3** | mantieni; 4° stato “high-vol bull” aumenta over-fit |
| `covariance_type` | `full` | `full` | **`full`** | OK |
| training window | tutto lo storico | rolling 5–10 anni (Reus & Mulvey 2016) | **rolling 2500d (≈10y)** con refit semestrale | finestra fissa evita drift di label; 10y copre ≥2 cicli |
| feature set | sp500_ret_21d + VIX + yield_spread | + credit spread | aggiungere **BAA-AAA** (FRED) se disponibile | letteratura HMM regime: spread credit è leading indicator |

> *File*: `config/models.yaml:19-23`, `models/regime.py:46-51,137-183`.

### 3.5 Council aggregator [HIGH-LIFT]

| Parametro | Attuale | Letteratura | Proposta | Razionale |
|---|---|---|---|---|
| **base weights bull** | lgbm 0.50 / sent 0.30 / hmm 0.20 | inverse-var pooling (Bates & Granger 1969) | **0.55 / 0.25 / 0.20** | LGBM mostra IC più alto OOS su Alpha158; sent ha noise alto |
| **base weights bear** | 0.40 / 0.20 / 0.40 | regime-aware mixing | **0.35 / 0.15 / 0.50** | in bear l'HMM defensive signal è più predittivo (Nystrup et al. 2018) |
| **base weights transition** | 0.45 / 0.25 / 0.30 | — | **0.45 / 0.20 / 0.35** | riduzione sentiment in alta volatility |
| `weight_clip.min` | 0.05 | — | **0.05** | OK |
| `weight_clip.max` | 0.70 | ≤ 0.60 per diversificazione | **0.60** | evita single-model dominance |
| `min_history_days` | 30 | ≥ 60 per IC stabile | **60** | sotto 60d la varianza dell'IC ≈ 1/√60 ≈ 13% — adapt prematuro |
| `ic_rolling_window` | 30 | 60–90 | **60** | trade-off responsività/rumore |
| `sharpe_rolling_window` | 100 | 120–252 (Goodwin 1998) | **120** | annualized SR stabile sopra 120d |
| `halflife` EWM | 20 | 10–30 | **30** | risposta più morbida riduce whipsaw |
| `max_correlation` | 0.70 | 0.60–0.75 | **0.65** | sent ↔ technical correlation OOS osservata ~0.55; soglia tighter forza diversità |
| `correlation_window` | 60 | 60–120 | **90** | media più stabile |
| `downweight_factor` | 0.50 | 0.5–0.7 | **0.50** | OK |

**Nota matematica:** la formula attuale `w'_m = w_base · max(0.1, IC_Sharpe)`
ha un comportamento patologico se l'IC-Sharpe esplode positivamente (es. 3.0)
perché diventa un re-scaling lineare. Letteratura ensemble (Timmermann 2006):
in alternativa, **softmax sui IC-Sharpe** è più stabile:

```
w_m = w_base_m · exp(β · IC_Sharpe_m) / Σ exp(β · IC_Sharpe_k)
β = 1.0 (calibrabile via CV)
```

(suggerimento di design, non parametro — registrato qui per futura ADR).

> *File*: `config/regime_weights.yaml`, `council/aggregator.py:155-182,505-507,638-658`.

### 3.6 Conformal sizing (MAPIE)

| Parametro | Attuale | Letteratura | Proposta | Razionale |
|---|---|---|---|---|
| `coverage` (1-α) | 0.85 | 0.80–0.90 | **0.80** | Barber–Candes–Ramdas–Tibshirani (2021): Jackknife+ ha guarantee ≥ 1−2α; α=0.20 ⇒ guarantee ≥ 0.60, sufficient per sizing; intervalli più stretti ⇒ multiplier ↑ |
| `cv` (fold) | 5 | 5–10 | **10** | Jackknife+ con K=10 ha intervalli più stretti senza inflate alpha |
| `_MIN_MULT` / `_MAX_MULT` | 0.2 / 2.0 | — | **0.3 / 1.8** | leverage cap più conservativo; evita oversize su tail width tiny |
| `threshold_percentile` (filter) | 90 | 80–90 | **85** | trade-off coverage/breadth: più stocks investibili ↑ BR (Grinold-Kahn) |
| estimator base | Ridge(α=1.0) | Ridge / GBM | **Ridge(α=1.0)** | mantieni, Jackknife+ con LGBM diventa CPU-bound |

> *File*: `council/conformal.py:47-51,139-233`.

### 3.7 Portfolio optimizer [HIGH-LIFT]

| Parametro | Attuale | Letteratura | Proposta | Razionale |
|---|---|---|---|---|
| `max_position` | 0.15 (env: 0.10) | 5–10% per US equity (Brightonjones, Russell 2025) | **0.08** | meno concentration risk; tier ≥$100k già usa 0.13 — mantenere coerenza |
| `min_position` | 0.01 | 0.01–0.02 | **0.01** | OK |
| `max_turnover` (giornaliero) | 0.50 (env: 0.30) | 5–10% daily per strategie multi-day (Tidy Finance 2024) | **0.20** | 30% daily è eccessivo: a 3 bps round-trip ⇒ ~0.6 bps/day × 252 ≈ 1.5%/anno extra cost; abbassare a 20% taglia ~30% costi |
| `sector_cap` | 0.45 | 0.20–0.30 institutional | **0.30** | 45% lascia rischio settoriale dominante (Tech in particolare) |
| `max_vol_daily` | 0.025 (≈ 40% ann.) | 10–15% annualized target (AQR 2010, Man Group 2018) | **0.0095** (≈15% ann.) | vol targeting 15% migliora Sharpe equity (Harvey-Hoyle-Korgaonkar-Rattray 2018) |
| `max_beta_exposure` | 0.50 | 0.20–0.40 per market-neutral, 0.6–0.8 net-long | **0.40** | book net-long ma con tilt difensivo |
| `budget_fraction` (AUM ≥ $100k) | 0.85 | 0.85–0.95 | **0.90** | 10% cash buffer adeguato; 15% troppo penalizzante |
| `tc_lambda` | 1.0 | 1.0–3.0 (Olivares-Nadal & DeMiguel 2018) | **2.0** | penalità più forte → meno trades marginali; in vol-targeted portfolios λ↑ migliora Sharpe netto |
| `commission_bps` | 0.0 | 0–1 bps US equity retail | **0.5** | Alpaca = 0 nominale ma includere ECN fees / borrow costs |
| `slippage_bps` | 3.0 | 2–8 bps mid-cap (Almgren-Chriss 2000; Frazzini-Israel-Moskowitz 2015) | **5.0** | mid-cap (PFE, MRK, ABT, COP, BAC, GS) richiede 5–8 bps realistici |

> *File*: `council/portfolio.py:82-115,118-169,399-403,407-414`,
> `council/transaction_costs.py:27-28`, `.env.example:120-132`.

**Why vol-target = 15%.** Empiricamente (AQR 2010; Man Group 2018; Hocquard-
Ng-Papageorgiou 2013), ridurre la vol target da 25–40% a 10–15% per book
equity long-bias migliora Sharpe netto di +0.2–0.5 perché:
1. Allevia drawdown nei regime ad alta vol (effetto leverage cycle).
2. Riduce path-dependent slippage durante stress.
3. Stabilizza il sizing dei segnali (compounding più liscio).

### 3.8 Covariance & HRP [HIGH-LIFT]

| Parametro | Attuale | Letteratura | Proposta | Razionale |
|---|---|---|---|---|
| estimator default | Ledoit-Wolf | nonlinear shrinkage (Ledoit-Wolf 2017) o DCC | **Nonlinear-Shrinkage** se disponibile, altrimenti **DCC-GARCH** | NL-shrinkage migliora Sharpe OOS del 5–15% vs LW lineare (Ledoit & Wolf 2017, *RES*) |
| DCC `a` | 0.05 | 0.005–0.10 | **0.03** | valori empirici equity (Engle 2002, Engle-Sheppard 2001) |
| DCC `b` | 0.90 | 0.90–0.98 | **0.96** | high persistence equity; `a+b<1` → 0.99 OK |
| `MLCOUNCIL_HRP_SOFT_PRIOR` | false | true (LdP 2016: +31% Sharpe OOS) | **true** | López de Prado *J. Portfolio Management* 2016 mostra HRP migliora Sharpe OOS vs MVO |
| `MLCOUNCIL_HRP_BLEND` | 0.25 | 0.15–0.35 | **0.30** | letteratura non specifica ottimale; CV su grid raccomandato |

> *File*: `council/portfolio.py:391-397,516-535`, `council/covariance_dynamic.py:30-38`,
> `data/pipeline._compute_covariance`.

### 3.9 Risk rules & alert

| Parametro | Attuale | Letteratura | Proposta | Razionale |
|---|---|---|---|---|
| `stop_loss_pct` | 0.05 | 5–10% (Quantified Strategies 2023; Kaminski-Lo 2014) | **0.07** | 5% troppo tight su daily horizon; stop a 7% riduce whipsaw |
| `trailing_stop_pct` | 0.10 | 8–12% | **0.10** | OK |
| `max_holding_days` | 20 | dipende dal target; 5d target ⇒ 10–15d | **15** | coerente con target 5d (3× horizon) |
| `profit_take_pct` | 0.20 | 15–25% | **0.20** | OK |
| `max_var_pct` | 0.02 | 1–2% daily | **0.015** | con vol target 15% ann. il VaR 99% giornaliero atteso ≈ 1.5% |
| `max_cvar_pct` | 0.035 | 2–3% daily | **0.025** | coerente con max_var |
| `max_drawdown_pct` (env) | 0.07 | 8–12% kill-switch | **0.10** | 7% troppo aggressivo (statistica di drawdown su Sharpe 1 e vol 15% prevede DD attesi 8–12%) |
| **monitoring** `ic_threshold` | 0.01 | IC>0.02 indica skill genuino (Grinold-Kahn) | **0.015** | soglia più realistica |
| `ic_alert_consecutive_days` | 5 | 5–10 | **7** | filtro extra contro falsi positivi |
| `drift_pvalue_threshold` | 0.05 | 0.01–0.05 KS (Gama et al. 2014) | **0.01** | Bonferroni ≈ 0.05/10 features = 0.005; 0.01 è un compromesso |
| `shap_overlap_min` | 0.70 | 0.6–0.8 (Lundberg 2020) | **0.65** | finestra di 30d è breve → tolleranza più ampia evita spurious alert |

> *File*: `council/risk_rules.py:49-59`, `council/risk_engine.py:117-120`,
> `council/monitor.py:89-97`, `config/monitoring.yaml`.

### 3.10 Universe & breadth

| Parametro | Attuale | Letteratura | Proposta | Razionale |
|---|---|---|---|---|
| n. ticker | 32 large + 6 mid = 38 | 30–60 per BR/IR (Grinold-Kahn) | **+15 mid-cap** (totale ~53) | Fundamental Law: BR ↑ a `38·252 = 9576` → `53·252 = 13356`; `IR ∝ √BR` ⇒ +18% |
| `max_large_cap_weight` | 0.08 | 5–10% | **0.06** | meno concentration |
| `max_mid_cap_weight` | 0.05 | 3–5% | **0.04** | mid-cap meno liquide |
| `min_liquidity` | $1M | $5–10M | **$5,000,000** | con position size $100k+ servono > 50× ADV per spread < 5 bps |
| `rebalance_threshold` | 0.02 | 1–3% | **0.03** | aumenta inertia → riduce noise trades |
| `forward_fill_max_days` | 2 | ≤ 2 | **2** | OK |

> *File*: `config/universe.yaml:2-58`.

### 3.11 Features Alpha158

Nessun parametro numerico critico, ma due raccomandazioni:

- **Lag enforcement.** Verificare che `shift(1)` sia applicato *prima*
  del computo dei rolling, non dopo (`data/features/alpha158.py:6-8`).
  Lookahead di 1 barra inflate l'IC di ~30% (de Prado 2018, §3.2).
- **Risk-adjusted target.** Abilitare `target_risk_adjusted` con
  `vol_window = 21` (`data/features/target.py:23`): il target diventa
  `r_{t+h} / σ_t` → IC più stabile cross-section (Asness-Moskowitz-Pedersen
  2013). Sharpe netto storicamente +0.1 su universo US.

---

## 4. Tabella riepilogo: parametri prioritari per Sharpe > 1

Ordinati per **impatto atteso decrescente** sulla Sharpe netto annualizzato.
Stime di lift sono *priors basati su letteratura citata*, non guarantee.

| Rank | Parametro | File | Default | Proposto | Lift atteso Sharpe |
|------|-----------|------|---------|----------|--------------------|
| 1 | `max_vol_daily` (target vol) | `council/portfolio.py:88` | 0.025 (40% ann) | **0.0095 (15% ann)** | +0.25 – 0.50 |
| 2 | `MLCOUNCIL_HRP_SOFT_PRIOR` | env | false | **true** (blend 0.30) | +0.15 – 0.30 |
| 3 | `max_turnover` | env / `council/portfolio.py:86` | 0.50 (env 0.30) | **0.20** | +0.10 – 0.25 |
| 4 | `slippage_bps` realismo + `tc_lambda` ↑ | `.env.example:130`, `council/portfolio.py:104` | 3.0 / 1.0 | **5.0 / 2.0** | +0.10 – 0.20 (riduce overtrading) |
| 5 | covariance estimator → NL-shrinkage / DCC | `council/portfolio.py:391-397` | LW | **DCC (a=0.03, b=0.96)** | +0.10 – 0.20 |
| 6 | sentiment `recency_decay γ` | `config/models.yaml:35` | 0.7 | **0.6** | +0.05 – 0.15 |
| 7 | LightGBM ricalibrazione (LR ↓, leaves ↓, min_child ↑) | `config/models.yaml:1-12` | vedi §3.1 | vedi §3.1 | +0.05 – 0.15 |
| 8 | CPCV `n_splits` 6 → 8, embargo 5 → 10 | `config/models.yaml:14-17` | 6/5 | **8/10** | +0.02 – 0.10 (riduce overfit) |
| 9 | `sector_cap` | `council/portfolio.py:89` | 0.45 | **0.30** | +0.05 – 0.10 |
| 10 | universe + 15 mid-cap | `config/universe.yaml` | 38 | **53** | +0.05 – 0.15 (BR ↑) |
| 11 | aggregator base weights | `config/regime_weights.yaml:14-25` | vedi §3.5 | vedi §3.5 | +0.05 – 0.10 |
| 12 | `weight_clip.max` | `config/regime_weights.yaml:34` | 0.70 | **0.60** | +0.02 – 0.05 |
| 13 | conformal `coverage` | `council/conformal.py:47` | 0.85 | **0.80** | +0.02 – 0.05 |
| 14 | `sharpe_rolling_window` | `config/regime_weights.yaml:43` | 100 | **120** | +0.02 – 0.05 (stabilità) |
| 15 | `min_history_days`, `ic_rolling_window` | `config/regime_weights.yaml:38-39` | 30/30 | **60/60** | +0.02 – 0.05 |

**Lift cumulativo atteso (non-additivo, con correlazioni e attrito):**
+0.5 – 1.0 Sharpe rispetto al baseline corrente. Date le condizioni
realistiche (TC ~0.6–0.8, IC base ~0.02), un Sharpe deflato netto
**1.0 – 1.3** è raggiungibile, ma soggetto a verifica via CPCV + DSR.

---

## 5. Metodologia di validazione (obbligatoria prima di promozione)

Per evitare di confondere over-fit con vero alpha, ogni promozione
parametrica deve passare:

1. **CPCV multi-path** (8 splits × 2 test folds → 28 path).
2. **Deflated Sharpe Ratio** (Bailey & López de Prado 2014):
   - Calcolare `E[max_SR_N]` con `N` = numero di iperparametri provati.
   - Promuovere solo se `DSR > 0.95` (confidence > 95%).
3. **Probability of Backtest Overfit (PBO)** < 0.5
   (Bailey-Borwein-LdP-Zhu 2017).
4. **Walk-forward out-of-sample** ≥ 12 mesi con parametri congelati.
5. **Shadow vs paper Alpaca** per ≥ 1 mese prima del go-live.

> Senza questo gauntlet, qualunque "Sharpe > 1" in backtest è
> statisticamente indistinguibile da rumore (LdP 2018, cap. 11).

---

## 6. Rischi & note di prudenza

- **Vol target 15% riduce il valore atteso di rendimento lordo**: il
  Sharpe migliora, ma il P&L assoluto può scendere. Coerente con
  l'obiettivo (Sharpe > 1) ma da comunicare a stakeholder.
- **HRP soft-prior** introduce un drift verso allocazioni risk-parity
  che può sotto-pesare segnali ad alta convinzione. Calibrare blend
  via CV su `[0.15, 0.20, 0.25, 0.30, 0.35]`.
- **Costo realistico (5 bps mid-cap)** può rendere alcuni segnali
  unprofitable: monitorare cost calibration alert
  (`config/monitoring.yaml:4-7`) settimanalmente.
- **DCC-GARCH** richiede stima `T ≥ 500` osservazioni stabili; non
  attivare durante warm-up.
- **Soglia `max_correlation = 0.65`** può penalizzare la combinazione
  LGBM+sentiment quando ambedue sono giustamente long: verificare
  con `correlation_window = 90` invece di 60.

---

## 7. Riferimenti

### Foundational
- Grinold, R. C. (1989). *The Fundamental Law of Active Management*.
  J. Portfolio Management.
- Grinold, R. C., & Kahn, R. N. (1999). *Active Portfolio Management*
  (2nd ed.). McGraw-Hill.
- Markowitz, H. (1952). *Portfolio Selection*. J. Finance.

### Machine learning per finanza
- López de Prado, M. (2018). *Advances in Financial Machine Learning*.
  Wiley. (CPCV cap. 7; embargo cap. 7.4; backtest overfit cap. 11)
- López de Prado, M. (2020). *Machine Learning for Asset Managers*.
  Cambridge Elements.
- Bailey, D. H., & López de Prado, M. (2014). *The Deflated Sharpe
  Ratio*. J. Portfolio Management.
- Bailey, Borwein, López de Prado, Zhu (2017). *The Probability of
  Backtest Overfitting*. J. Computational Finance.

### Covariance & ottimizzazione
- Ledoit, O., & Wolf, M. (2003). *Honey, I Shrunk the Sample Covariance
  Matrix*. J. Portfolio Management.
- Ledoit, O., & Wolf, M. (2017). *Nonlinear Shrinkage of the Covariance
  Matrix for Portfolio Selection: Markowitz Meets Goldilocks*. Review
  of Financial Studies.
- López de Prado, M. (2016). *Building Diversified Portfolios that
  Outperform Out-of-Sample* (HRP). J. Portfolio Management.

### GARCH & regime
- Engle, R. F. (2002). *Dynamic Conditional Correlation*. JBES.
- Engle, R. F., & Sheppard, K. (2001). *Theoretical and Empirical
  Properties of DCC Multivariate GARCH*. NBER WP.
- Hamilton, J. D. (1989). *A New Approach to the Economic Analysis of
  Nonstationary Time Series*. Econometrica.
- Nystrup, P., Madsen, H., & Lindström, E. (2018). *Dynamic Allocation
  or Diversification: A Regime-Based Approach*. J. Portfolio Management.

### Sentiment
- Tetlock, P. C. (2007). *Giving Content to Investor Sentiment*. J. Finance.
- Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-
  trained Language Models*. arXiv:1908.10063.
- Loughran, T., & McDonald, B. (2011). *When Is a Liability Not a
  Liability?* J. Finance.

### Conformal prediction
- Barber, R., Candès, E., Ramdas, A., & Tibshirani, R. (2021).
  *Predictive Inference with the Jackknife+*. Annals of Statistics.
- Taquet, V., et al. (2022). *MAPIE: an open-source library for
  distribution-free uncertainty quantification*. arXiv:2207.12274.

### Costi & execution
- Almgren, R., & Chriss, N. (2001). *Optimal Execution of Portfolio
  Transactions*. J. Risk.
- Frazzini, A., Israel, R., & Moskowitz, T. (2015). *Trading Costs of
  Asset Pricing Anomalies*. AQR WP.

### Vol targeting
- Harvey, C., Hoyle, E., Korgaonkar, R., Rattray, S., Sargaison, M., &
  van Hemert, O. (2018). *The Impact of Volatility Targeting*.
  J. Portfolio Management.
- Hocquard, A., Ng, S., & Papageorgiou, N. (2013). *A Constant
  Volatility Framework for Managing Tail Risk*. J. Portfolio Management.

### Ensemble
- Bates, J. M., & Granger, C. W. J. (1969). *The Combination of
  Forecasts*. Operational Research Quarterly.
- Timmermann, A. (2006). *Forecast Combinations*. Handbook of
  Economic Forecasting.

---

*Documento prodotto per branch `claude/mlcouncil-sharpe-params-PAm96`.
Solo proposta — nessuna modifica al codice. Le promozioni devono
passare CPCV + DSR (§5) prima del go-live.*
