# MLCouncil — Roadmap Disruptive (Wave 1-4)

> **Scope.** Questo piano implementa i cambi *disruptive* della sezione 5 di
> `docs/codebase_analysis.md` (TO-BE), una volta che il primo P2 track
> (self-calibrating cost model) è chiuso. Le 4 wave qui sotto sono progettate
> per essere eseguibili da agenti indipendenti, ciascuno su un branch e ADR
> dedicati, con champion/challenger gating obbligatorio prima di ogni
> promozione in master.

## Stato di partenza (2026-05-21)

**Già consegnato** (Fasi 0-7 di `docs/next-steps-plan-2026-05-21.md`):

| Area | Stato | Riferimento |
|---|---|---|
| Drift register / docs hygiene | ✅ 11/11 | `docs/architecture-as-is-to-be-2026-05-21.md` |
| Fill telemetry (`FillRecord` + OMS hook + backfill) | ✅ | `execution/fill_log.py`, `execution/oms.py` |
| Cost calibration engine + manifest | ✅ | `council/cost_calibration.py` |
| Dagster `cost_calibration_artifact` nightly | ✅ | `data/pipeline.py` |
| `TransactionCostModel` blend lookup+kappa | ✅ | `council/transaction_costs.py` |
| A/B backtest `--cost-mode=both` | ✅ | `scripts/run_strategy_backtest.py` |
| Divergence alert + promotion gate | ✅ | `council/monitor.py`, `backtest/validation.py` |
| Dashboard math-trace + fill quality | ✅ | `dashboard/app.py`, `dashboard/pages/1_Fill_Quality.py` |
| HRP soft prior + spike + ADR | ✅ | `council/hrp.py`, `docs/adr/2026-05-21-hrp-soft-prior.md` |
| Robust opt spike + ADR | ✅ | `scripts/spike_robust_opt.py`, ADR |

**Disruptive ancora da fare** (TO-BE originale, mappato qui sotto):

| Sezione TO-BE | Track | Wave |
|---|---|---|
| 5.1.1 TFT/PatchTST alpha | T2.1 | Wave 2 |
| 5.1.2 FinGPT/RAG sentiment | T2.2 | Wave 2 |
| 5.1.3 Switching state-space regime | T2.3 | Wave 2 |
| 5.1.4 Microstructure / OFI | T2.4 | Wave 2 |
| 5.2.1 MoE gating | T3.1 | Wave 3 |
| 5.2.2 Stacking + CQR | T3.2 | Wave 3 |
| 5.3.3 Differentiable portfolio | T3.3 | Wave 3 |
| 5.3.4 DCC-GARCH / GNN covariance | T3.4 | Wave 3 |
| 5.4.1 RL execution | T4.1 | Wave 4 |
| 5.4.3 Smart order routing | T4.2 | Wave 4 |
| 5.5.1 Generative stress (VAE/Diffusion) | T4.3 | Wave 4 |
| 5.5.2 Causal drift detection (PCMCI) | T4.4 | Wave 4 |
| 5.5.3 TDA crash early warning | T4.5 | Wave 4 |
| 5.6.1 Walk-forward CI | T1.1 | **Wave 1** |
| 5.6.2 Online learning | T1.2 | **Wave 1** |
| 5.6.3 Tri-temporal feature store | T1.3 | **Wave 1** |
| 5.6.4 OpenTelemetry + Grafana | T1.4 | **Wave 1** |

---

## Wave 0 — Pre-requisiti bloccanti (devono chiudere prima di Wave 2-4)

Wave 1 è chiamata "Wave 0" qui solo come reminder che è un **gating wave**:
nessun modello disruptive (T2.x, T3.x, T4.x) può andare in produzione finché
i T1.x non sono solidi.

Sintesi:

- **T1.1 — Walk-forward CI**: senza CI che ritrenni i modelli con purge+embargo
  ogni settimana, ogni challenger che producono i T2.x è ingestibile.
- **T1.2 — Online learning**: il loop daily oggi è batch (re-fit MLflow); per
  reagire a regime shock servono update incrementali.
- **T1.3 — Tri-temporal store**: aggiunge `arrival_time` accanto a
  `transaction_time`. Necessario per replay rigorosi quando si confrontano
  modelli con dipendenze da news/macro che hanno latenza variabile.
- **T1.4 — Observability**: senza tracing OTel + dashboard Grafana, i
  challenger T2-T4 non sono debuggabili in produzione.

Dettaglio dei task in **Wave 1**.

---

## Wave 1 — Foundations (prerequisito per tutto il resto)

Stima totale: 4-6 settimane (di cui ~2 settimane di T1.4 in parallelo a T1.1).

### Track T1.1 — Walk-forward champion/challenger CI

**Obiettivo.** GitHub Action settimanale che (a) ricarica feature store al
giorno X, (b) gira walk-forward purged+embargoed re-fit, (c) confronta
challenger vs champion via `backtest/validation.py`, (d) apre PR
"promote model" se il gate passa.

**File:**
- Nuovo `.github/workflows/walk-forward-ci.yml`
- Nuovo `scripts/run_walkforward_promotion.py` che orchestrora retrain+gate
- Estendere `backtest/validation.py` con `validate_model_promotion()` (esiste
  per costi, replicare per alpha LightGBM, sentiment, HMM)
- Nuovo `docs/adr/YYYY-MM-DD-walkforward-ci.md`

**ADR points:**
- Cadenza: settimanale (lunedì 02:00 UTC) per LightGBM, mensile per FinBERT
- Gating thresholds: `oos_sharpe >= champion - 0.1`, `pbo <= 0.5`,
  `walk_forward_window_count >= 8`
- Auto-promote: solo se challenger > champion per **3 PR consecutive**
- Champion archive: tag git con SHA del checkpoint promosso

**Verifica:**
```bash
gh workflow run walk-forward-ci.yml --ref master
python scripts/run_walkforward_promotion.py --model lightgbm --dry-run
python -m pytest tests/test_walkforward_promotion.py -v
```

**Commit:** `feat(ci): walk-forward champion/challenger CI for alpha models`

---

### Track T1.2 — Online learning scaffolding

**Obiettivo.** Sostituire il pattern "carica checkpoint statico"
(`council/pickle_security.py::trusted_pickle_load`) con un loop incrementale
quotidiano per LightGBM. Drift detection automatica con ADWIN/DDM trigger
del retrain pesante.

**File:**
- Nuovo `models/online.py` con classe `IncrementalLightGBM` wrapper
  (`lightgbm.refit()`)
- Nuovo `council/drift.py` con `ADWINDetector`, `DDMDetector` 
- Modificare `data/pipeline.py::lgbm_signals` per chiamare `model.refit(X_new)`
  prima dell'inference, salvando `lgbm_latest.pkl` + nuovo `.hash`
- Nuovo `docs/adr/YYYY-MM-DD-online-learning.md`

**ADR points:**
- Trade-off: refit incrementale può divergere su corner case → fallback al
  checkpoint precedente se `IC_today < IC_baseline - 0.05`
- Cadenza ADWIN: ogni giorno su rolling 60d returns
- Dipendenza: `pip install river` (CC0)

**Verifica:**
```bash
python -m pytest tests/test_online.py tests/test_drift.py -v
python scripts/run_pipeline.py --partition 2026-05-20 --online
```

**Commit:** `feat(models): incremental online learning + ADWIN drift detector`

---

### Track T1.3 — Tri-temporal feature store

**Obiettivo.** Aggiungere `arrival_time` accanto a `valid_time` e
`transaction_time` per distinguere "quando il dato è disponibile in feed"
vs "quando lo abbiamo ingestito". Permette replay con latenza realistica.

**File:**
- `data/store/arctic_store.py` — estendere `write()` per accettare opzionale
  `arrival_time`; estendere `read()` con `as_of_arrival_time` filter
- Migrazione dati storici: nuovo `scripts/migrate_arrival_time.py` che
  retro-stima `arrival_time` da source metadata (RSS pub_date, FRED
  observation_date)
- `tests/test_arctic_store.py` — nuovi test bi/tri-temporal

**Verifica:**
```bash
python scripts/migrate_arrival_time.py --dry-run
python -m pytest tests/test_arctic_store.py -k "tri_temporal" -v
```

**Commit:** `feat(store): tri-temporal feature store with arrival_time`

---

### Track T1.4 — Observability (OpenTelemetry + Grafana)

**Obiettivo.** Distributed tracing su tutto il flusso ingest → orders.
Latency SLO per layer. Audit trail compliance-ready.

**File:**
- Nuovo `observability/tracing.py` con `init_tracing()` (OTel Python SDK)
- Strumentare i 4 layer Dagster: ingest, features, signals, council
- Nuovo `docker-compose.observability.yml` (Tempo + Grafana + Prometheus)
- Nuovo `dashboards/grafana/mlcouncil.json` (dashboard come codice)
- README sezione "Observability"

**Verifica:**
```bash
docker-compose -f docker-compose.observability.yml up -d
python scripts/run_pipeline.py --partition 2026-05-20
# Apri Grafana :3001, verifica trace di un asset completo
```

**Commit:** `feat(observability): OpenTelemetry tracing + Grafana dashboards`

---

## Wave 2 — Alpha next-gen (4 challenger paralleli)

Stima totale: 8-12 settimane wallclock con 4 agenti in parallelo.
Ogni track sostituisce o affianca **un** modello base; gating obbligatorio
via T1.1 prima della promozione.

### Track T2.1 — Temporal Fusion Transformer (TFT) alpha

**Obiettivo.** Sostituire LightGBM con TFT (Lim et al. 2021) o PatchTST
(Nie et al. 2023) per catturare dipendenze cross-asset e cross-time.

**Matematica chiave:**
- Variable selection network (gate σ(Wx)) + LSTM encoder + multi-head
  self-attention + quantile output (5%/50%/95%)
- Loss: pinball quantile loss (multi-horizon nativo)

**File:**
- Nuovo `models/tft.py` (wrapper su `pytorch-forecasting`)
- Nuovo `docs/adr/YYYY-MM-DD-tft-alpha-challenger.md`
- Nuovo `scripts/train_tft.py`
- `config/models.yaml` — sezione `tft:`
- Test: `tests/test_tft.py`

**Dipendenze:**
- `torch>=2.0`, `pytorch-forecasting>=1.0`, `pytorch-lightning>=2.0`
- GPU consigliata (CPU funziona ma 5-10x più lento)

**Gating (deve girare via T1.1 CI):**
- `oos_sharpe_tft >= oos_sharpe_lgbm + 0.15` su 12 mesi walk-forward
- `oos_max_drawdown_tft <= oos_max_drawdown_lgbm + 2%`
- Interpretability: variable selection top-10 deve essere stabile (Jaccard ≥ 0.6)
- Inference latency: <300ms su CPU per il daily batch

**Verifica:**
```bash
python scripts/train_tft.py --start 2021-01-01 --end 2024-12-31
python -m pytest tests/test_tft.py -v
python scripts/run_walkforward_promotion.py --model tft --challenger lgbm
```

**Commit:** `feat(models): TFT alpha challenger (pytorch-forecasting)`

---

### Track T2.2 — FinGPT/FinMA + RAG sentiment

**Obiettivo.** Sostituire FinBERT (2019, vocabolario limitato) con un LLM
finance-tuned + retrieval da 10-K / 10-Q / earnings transcripts.

**Matematica chiave:**
- Encoder: FinMA-7B (Llama-2 finance-tuned) o FinGPT
- Retrieval: vector store (Chroma/Qdrant) su filings SEC EDGAR
- Aggregation: prompt LLM con top-K passaggi → score in [-1, +1]
- Multimodale: trascrizioni earnings call via Whisper → LLM → tone score

**File:**
- Nuovo `models/sentiment_llm.py` con `LLMSentimentScorer(model="FinMA-7B")`
- Nuovo `data/ingest/sec_filings.py` (EDGAR REST API)
- Nuovo `data/ingest/earnings_transcripts.py` (Whisper-based)
- Nuovo `data/store/vector_store.py` (Chroma wrapper)
- Nuovo `docs/adr/YYYY-MM-DD-finma-rag-sentiment.md`

**Dipendenze:**
- `transformers>=4.40`, `llama-cpp-python` o vLLM per inference locale
- `chromadb` o `qdrant-client` per retrieval
- `whisper` per audio (Optional)
- ~8GB VRAM per FinMA-7B quantized

**Gating:**
- IC delta vs FinBERT: ≥ +0.02 cross-section forward 1d returns
- Event lift study: average return su giorni post-positive-sentiment ≥ +20bps
- Throughput: ≥ 100 headlines/sec con quantization Q4_0
- Hallucination guard: score = 0 se LLM produce risposta non parseable

**Verifica:**
```bash
python scripts/train_finma.py --eval-only --start 2024-01-01
python -m pytest tests/test_sentiment_llm.py tests/test_sec_filings.py -v
```

**Commit:** `feat(models): FinMA/RAG sentiment challenger`

---

### Track T2.3 — Switching State-Space regime

**Obiettivo.** Sostituire HMM gaussiano discreto a 3 stati con un Deep State
Space Model (S4/Mamba) a regime continuo $z_t \in \mathbb{R}^d$.

**Matematica chiave:**
$$
z_t = A_\theta(z_{t-1}) + \epsilon_t, \quad x_t = B_\theta(z_t) + \eta_t
$$
con $A_\theta$ una rete neurale; inference variazionale amortizzato via ELBO.

**File:**
- Nuovo `models/regime_dss.py` (wrapper su `mamba-ssm` o `s4-pytorch`)
- Migrare `council/aggregator.py` per supportare regime continuo (oggi
  hard-code `{bull, bear, transition}`): introdurre `regime_embedding`
  vector in alternativa a `regime_label`
- Nuovo `docs/adr/YYYY-MM-DD-deep-regime.md`

**Dipendenze:**
- `mamba-ssm>=2.0` o `s4`, CUDA per training
- Pyro o NumPyro per VI baseline

**Gating:**
- ELBO migliorato vs baseline HMM: ≥ 5% delta su validation set
- Regime probabilities consistenti: no transition `bull→bear` senza
  decadimento monotono via transition
- Council aggregator non degrada: con regime_embedding al posto di
  regime_label, IC council ≥ baseline ± 0.005

**Verifica:**
```bash
python scripts/train_regime_dss.py
python -m pytest tests/test_regime_dss.py -v
```

**Commit:** `feat(models): deep state-space regime model challenger`

---

### Track T2.4 — Microstructure / Order Flow Imbalance alpha

**Obiettivo.** Aggiungere un quarto modello alpha intraday basato su L2 book
data (Order Flow Imbalance, Lo & MacKinlay).

**Matematica chiave:**
$$
\text{OFI}_t = \Delta \sum_b q^b_t - \Delta \sum_a q^a_t
$$
dove $q^b, q^a$ sono volume cumulato bid/ask a 5 livelli.

**File:**
- Nuovo `models/microstructure.py`
- Nuovo `data/ingest/orderbook.py` (richiede feed L2 — Alpaca data o
  Databento)
- Estendere `intraday/market_data.py` con `compute_ofi(book_snapshot)`
- Aggregator: nuovo entry `microstructure` in `config/regime_weights.yaml`

**Dipendenze:**
- L2 data subscription (Databento Premium o Alpaca elite)
- `polars` per processing tick-level

**Gating:**
- OFI come signal: IC ≥ 0.04 su forward 30-min returns
- Cross-correlation con altri 3 modelli: |ρ| ≤ 0.4 (no overlap)
- Latency intraday: <500ms da tick a signal

**Verifica:**
```bash
python -m pytest tests/test_microstructure.py -v
```

**Commit:** `feat(models): order-flow-imbalance intraday alpha`

---

## Wave 3 — Council & Portfolio evoluti

Stima: 6-8 settimane. **Pre-requisito**: almeno 1 modello di Wave 2 promosso
via T1.1, altrimenti il council ha gli stessi 3 input e MoE/stacking è poco
informativo.

### Track T3.1 — Mixture-of-Experts (MoE) gating

**Obiettivo.** Sostituire la pesatura lineare regime-conditional con un
gating network non lineare $g_\theta(x_t, \text{regime}_t)$.

**Matematica chiave:**
$$
\hat{y} = \sum_k g_k(x_t) \cdot f_k(x_t), \quad \sum_k g_k(x_t) = 1
$$
Gating addestrato su reward = realised IC (REINFORCE / PPO).

**File:**
- Nuovo `council/moe_gating.py`
- Modificare `council/aggregator.py` per supportare modalità `linear` vs `moe`
  via env `MLCOUNCIL_AGGREGATOR_MODE`
- Nuovo `docs/adr/YYYY-MM-DD-moe-gating.md`

**Dipendenze:**
- `torch`, `gymnasium` (per REINFORCE), oppure semplice gradient via
  `pytorch-lightning`

**Gating:**
- IC delta ≥ +0.01 su walk-forward 12 mesi
- Weight stability: cambia smoothly tra regimi (no spike >50% inter-day)

**Verifica:**
```bash
python scripts/train_moe_gating.py
python -m pytest tests/test_moe_gating.py -v
```

**Commit:** `feat(council): MoE non-linear gating aggregator`

---

### Track T3.2 — Stacking + Conditional Quantile Regression (CQR)

**Obiettivo.** Sostituire MAPIE Jackknife+ (coverage marginale) con CQR
(coverage condizionato) e aggiungere un meta-learner sui 3+ modelli base.

**Matematica chiave:**
$$
\hat{C}(x) = [\hat{q}_{\alpha/2}(x) - q^{cal}_{1-\alpha},\; \hat{q}_{1-\alpha/2}(x) + q^{cal}_{1-\alpha}]
$$
Intervalli più stretti in zone "facili", più larghi in code.

**File:**
- Nuovo `council/cqr.py` (sostituisce `council/conformal.py` come secondo
  default; il vecchio resta come fallback)
- Meta-learner XGB su outputs dei 3 modelli base + features hand-picked
- Nuovo `docs/adr/YYYY-MM-DD-stacking-cqr.md`

**Gating:**
- Coverage condizionata: per ogni quintile di volatility forecast, coverage
  empirico nell'intervallo [80%, 90%]
- Width medio non superiore al Jackknife+ di +5%

**Verifica:**
```bash
python -m pytest tests/test_cqr.py -v
```

**Commit:** `feat(council): stacking meta-learner with CQR uncertainty`

---

### Track T3.3 — Differentiable portfolio (cvxpylayers)

**Obiettivo.** Rendere il portfolio optimizer un layer differenziabile (CvxPyLayer)
per addestrare alpha + portfolio end-to-end (decision-focused learning).

**Matematica chiave:**
- Loss = realized Sharpe portfolio
- Gradiente attraverso il QP via implicit function theorem (Agrawal et al. 2019)
- Backprop dentro il solver, aggiornando i pesi del modello alpha
  per **massimizzare il P&L reale** invece di IC.

**File:**
- Nuovo `council/portfolio_diff.py` con `DifferentiablePortfolioConstructor`
- Estendere `models/technical.py` con training loop end-to-end
- Nuovo `docs/adr/YYYY-MM-DD-differentiable-portfolio.md`

**Dipendenze:**
- `cvxpylayers>=0.1.6`, PyTorch

**Gating:**
- Realised Sharpe net costs ≥ baseline + 0.2 su walk-forward 6 mesi
- Training stability: no NaN gradient, convergenza in ≤ 200 epoch
- Constraint feasibility: tutte le solution rispettano sector/turnover/vol

**Verifica:**
```bash
python -m pytest tests/test_portfolio_diff.py -v
python scripts/train_alpha_portfolio_end2end.py
```

**Commit:** `feat(council): differentiable portfolio (cvxpylayers) end-to-end training`

---

### Track T3.4 — DCC-GARCH / GNN conditional covariance

**Obiettivo.** Sostituire la covarianza Ledoit-Wolf statica con DCC-GARCH
dinamica (Engle 2002) o una GNN su grafo di correlazione.

**Matematica DCC:**
$$
\Sigma_t = D_t R_t D_t, \quad D_t = \text{diag}(\sigma_{1,t}, ..., \sigma_{N,t})
$$
con $\sigma_{i,t}$ da GARCH(1,1) univariato e $R_t$ correlation dynamics:
$$
Q_t = (1-a-b)\bar{Q} + a\epsilon_{t-1}\epsilon_{t-1}^\top + bQ_{t-1}
$$

**File:**
- Nuovo `council/covariance_dynamic.py` con `DCCEstimator`
- Alternativa: `council/covariance_gnn.py` con `GNNCovariance` (PyTorch
  Geometric, edges = top-K correlations)
- Modificare `_compute_covariance()` in `data/pipeline.py` per scegliere
  estimator via env
- Nuovo `docs/adr/YYYY-MM-DD-dynamic-covariance.md`

**Dipendenze:**
- `arch>=6.0` per GARCH univariato
- (Opzionale) `torch-geometric` per GNN

**Gating:**
- Portfolio realised vol prediction error: MAPE riduce ≥ 10% vs Ledoit-Wolf
- Drawdown peggiore ≥ -1% vs baseline

**Verifica:**
```bash
python -m pytest tests/test_dcc_garch.py -v
```

**Commit:** `feat(council): DCC-GARCH conditional covariance`

---

## Wave 4 — Execution & Risk avanzati

Stima: 8-12 settimane, parzialmente parallelizzabile.

### Track T4.1 — RL execution agent

**Obiettivo.** Sostituire il TWAP/VWAP rules-based di `execution/slicer.py`
con un agente RL (PPO) su simulatore di order book.

**File:**
- Nuovo `execution/rl_agent.py` con `PPOExecutionAgent`
- Nuovo `execution/lob_simulator.py` (ABIDES o homegrown su book history)
- Modificare `execution/slicer.py` per route a RL agent se
  `MLCOUNCIL_RL_EXECUTION_ENABLED=true`
- Nuovo `docs/adr/YYYY-MM-DD-rl-execution.md`

**Dipendenze:**
- `stable-baselines3>=2.0` o `ray[rllib]`
- 6+ mesi di fill history accumulata via T_phase2 fill_log
- L2 book snapshots (T2.4 prereq oppure stub Bachelier)

**Gating:**
- Implementation Shortfall medio: ≤ TWAP - 1 bps su 10k trade simulati
- Robustezza: no exploitation di edge case del simulator (validation su
  paper trading reale)

**Commit:** `feat(execution): PPO RL execution agent`

---

### Track T4.2 — Smart Order Routing multi-venue

**Obiettivo.** Oggi solo Alpaca. Aggiungere IBKR + Coinbase + (futuro) DEX.

**File:**
- Nuovo `execution/router.py` con `SmartRouter`
- Nuovi adapter: `execution/ibkr_adapter.py`, `execution/coinbase_adapter.py`
- Routing decision: `argmin(expected_cost + λ·urgency)`
- Nuovo `docs/adr/YYYY-MM-DD-smart-order-routing.md`

**Dipendenze:**
- `ib_insync` per IBKR, `coinbase-advanced-trade` per Coinbase
- Sandbox/paper credentials per ciascuna venue

**Gating:**
- Best execution su 100 trade simulati: routing decisione ≥ 90% concordant
  con post-hoc optimum
- Failover: se una venue è down, ordini routati a fallback senza dead-letter

**Commit:** `feat(execution): smart order routing IBKR + Coinbase`

---

### Track T4.3 — Generative stress scenarios (VAE / Diffusion)

**Obiettivo.** Sostituire il VaR storico (sample-limited) con un VaR
generativo: model VAE/Diffusion sui returns multivariati, sample 10k
scenari condizionati su regimi rari.

**Matematica chiave:**
- VAE: encoder $q_\phi(z|x)$, decoder $p_\theta(x|z)$, ELBO con KL su prior
  $\mathcal{N}(0,I)$
- Diffusion: forward $q(x_t|x_{t-1})$ Gaussian noise, reverse $p_\theta(x_{t-1}|x_t)$ NN
- Conditional generation: regime label / VIX level come conditioning vector

**File:**
- Nuovo `council/generative_stress.py`
- Integrare in `council/risk_engine.py::compute_var_montecarlo()` come
  modalità `method="generative"`
- Nuovo `docs/adr/YYYY-MM-DD-generative-stress.md`

**Dipendenze:**
- `torch`, `diffusers` (per Diffusion baseline)

**Gating:**
- Calibration on holdout: 95% VaR empirico in [4.5%, 5.5%]
- Diversity: 10k scenarios coprono ≥ 90% delle tail histories osservate

**Commit:** `feat(risk): generative VAE/Diffusion stress scenarios`

---

### Track T4.4 — Causal inference per drift detection

**Obiettivo.** Sostituire/affiancare il KS test su distribuzioni marginali
con causal discovery (PCMCI, Runge et al. 2019) sulla struttura
feature → return.

**File:**
- Nuovo `council/causal_drift.py` con `PCMCIDriftDetector`
- Aggiungere check a `council/monitor.py`: alert quando il **grafo causale**
  cambia (link aggiunti/rimossi rispetto al baseline)
- Nuovo `docs/adr/YYYY-MM-DD-causal-drift.md`

**Dipendenze:**
- `tigramite>=5.0`

**Gating:**
- False positive rate ≤ 5% su backtest 5 anni
- True positive su 4 eventi noti di regime change (COVID 2020, FOMC pivot
  2022, ecc.)

**Commit:** `feat(monitor): PCMCI causal-graph drift detector`

---

### Track T4.5 — TDA crash early warning

**Obiettivo.** Persistent homology su returns multivariati: lo spike in
$\beta_1$ (loops) precede market stress (Gidea & Katz 2018).

**File:**
- Nuovo `council/tda_warning.py` con `PersistentHomologyAnalyser`
- Nuovo Dagster asset `tda_warning_signal` (settimanale)
- Alert in dashboard quando `beta1 > threshold` su rolling 30d
- Nuovo `docs/adr/YYYY-MM-DD-tda-early-warning.md`

**Dipendenze:**
- `gudhi>=3.8` o `ripser>=0.6`

**Gating:**
- Lead time medio prima di drawdown -5%: ≥ 10 giorni
- False positive rate ≤ 20% (più permissivo, è un early warning)

**Commit:** `feat(risk): TDA persistent homology early warning`

---

## Sequenziamento consigliato

```
Wave 1 — Foundations (settimane 1-6)
  T1.4 Observability        ┐
  T1.1 Walk-forward CI      ├── PARALLELI (3 agenti)
  T1.3 Tri-temporal store   ┘
  T1.2 Online learning  (dopo T1.1)

Wave 2 — Alpha challengers (settimane 5-16, sovrappongono con T1.1 ready)
  T2.1 TFT                  ┐
  T2.2 FinMA RAG            ├── PARALLELI (4 agenti, ciascuno su branch)
  T2.3 Deep regime          │
  T2.4 Microstructure       ┘
  → Promozione via T1.1 CI gating

Wave 3 — Council & Portfolio (settimane 14-22, dopo Wave 2 produce ≥1 champion)
  T3.2 Stacking+CQR         ┐── leggero
  T3.4 DCC-GARCH            ├── PARALLELI
  T3.1 MoE gating           │── medio
  T3.3 Differentiable port. ┘── pesante (dopo T3.1)

Wave 4 — Execution & Risk (settimane 1-20+, alcuni precedono Wave 2/3)
  T4.4 PCMCI                ── settimana 1-4 (utile da subito)
  T4.5 TDA                  ── settimana 1-4 (utile da subito)
  T4.3 Generative stress    ── settimana 5-12
  T4.2 Smart routing        ── settimana 8-16
  T4.1 RL execution         ── settimana 14-22 (dopo accumulo fill history)
```

## Acceptance criteria globale (per chiudere un track)

Per ogni track Tx.y, l'agente deve produrre:

1. ✅ **ADR finalizzato** in `docs/adr/YYYY-MM-DD-<track>.md` con stato
   "Accepted" o "Rejected"
2. ✅ **Tutti i test verdi** su `python -m pytest tests/test_<track>.py -v`
3. ✅ **Walk-forward gate passa** (per T2.x e T3.x) via `T1.1`
4. ✅ **Dashboard panel** dedicato (se UI-relevant)
5. ✅ **Rollback documentato** via env flag — il sistema può tornare
   al campione precedente senza redeploy
6. ✅ **Baseline aggiornato** in `docs/baselines/YYYY-MM-DD-<track>.md`
7. ✅ **PR mergeable** su master con commit message convenzionale

## Risk register

| Rischio | Mitigazione |
|---|---|
| Modello disruptive non passa gating per mesi | Hard timeline: track abbandonato e ADR "Rejected" dopo 12 settimane senza promotion |
| Costo GPU per LLM/TFT eccede budget | Quantization Q4 + caching + dedicated GPU box (no cloud) |
| Dati L2 microstructure costosi | T2.4 deferred finché budget non approva subscription Databento |
| Fill history insufficiente per RL | T4.1 bloccato finché telemetry T_phase2 accumula ≥ 6 mesi di fills |
| Conflitti merge tra agenti paralleli | Ogni track su feature branch dedicato; rebase settimanale su master |

## Hand-off per agenti

Ogni agente che prende un track riceve come prompt minimo:

```
Lavora sul track <Tx.y> del piano docs/disruptive-roadmap-2026-05-21.md.
- Branch: feat/<track-slug>
- ADR template: docs/adr/ADR-template.md
- Walk-forward gate: scripts/run_walkforward_promotion.py
- Verifica finale: pytest + dashboard + ADR Accepted/Rejected
- NON pushare su master direttamente: aprire PR.
```

## Note operative

- **Ogni Wave deve completarsi prima della successiva per i pre-requisiti
  obbligatori** (T1.x → Wave 2, ≥1 champion Wave 2 → Wave 3). Le track
  *all'interno* di una Wave sono indipendenti.
- **No big-bang**: zero modifiche al daily pipeline finché un challenger
  non è promosso. Il modello "challenger" gira in shadow mode (calcola
  signal e logga, ma non entra nel council aggregator) finché il gate
  T1.1 lo promuove.
- **Budget cap**: ogni track ha 12 settimane wallclock. Se non chiude in
  tempo → ADR "Rejected", branch archiviato, prossimo track parte.

---

## Wave 1 — Gap analysis (codebase snapshot 2026-05-21)

Automated inventory against `origin/master` + local untracked work (`data/pipeline.py`,
`council/cost_calibration_gate.py`, etc.). Status: **missing** = track deliverable absent;
**partial** = related primitives exist but track acceptance criteria not met;
**done** = track scope satisfied.

| Track | Status | Key files found | Notes |
|---|---|---|---|
| **T1.1** Walk-forward CI | **partial** | `backtest/validation.py` (`build_purged_walk_forward_splits`, `run_walk_forward_analysis`, `estimate_pbo`, `summarize_walk_forward_metrics`); `council/cost_calibration_gate.py` + `validate_cost_calibration_promotion` (cost-only); `.github/workflows/ci.yml` | Missing: `walk-forward-ci.yml`, `scripts/run_walkforward_promotion.py`, `validate_model_promotion()`, `tests/test_walkforward_promotion.py`, ADR |
| **T1.2** Online learning | **partial** | `data/pipeline.py::lgbm_signals` (static `lgbm_latest.pkl` via `TechnicalModel.predict`); `council/monitor.py` (KS feature drift, not ADWIN/DDM); `council/pickle_security.py` | Missing: `models/online.py`, `council/drift.py`, incremental `refit`, `river` dep, tests |
| **T1.3** Tri-temporal store | **partial** | `data/store/arctic_store.py` (`valid_time` + `transaction_time`, `as_of_transaction_time`); `tests/test_arctic_store.py` (PIT bi-temporal) | Missing: `arrival_time`, `as_of_arrival_time`, `scripts/migrate_arrival_time.py`, tri-temporal tests |
| **T1.4** Observability | **partial** | `dashboard/app.py` + `dashboard/data_loader.py` (council math-trace); `docker-compose.yml` (admin, dashboard, dagster, mlflow) | Missing: `observability/tracing.py`, OTel instrumentation, `docker-compose.observability.yml`, `dashboards/grafana/mlcouncil.json`; no `opentelemetry-*` in requirements |

**Suggested agent branches:** `feat/walkforward-ci` (T1.1), `feat/otel-grafana` (T1.4),
`feat/tri-temporal-store` (T1.3), `feat/online-learning` (T1.2, after T1.1).

**Blockers:** `river` and OpenTelemetry SDKs not in `requirements.txt`; T1.1 must land
before T1.2/T2.x promotion loops; T1.3 migration needs RSS/FRED metadata contracts in ingest.
