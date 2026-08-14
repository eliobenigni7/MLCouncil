# MLCouncil — Piano Operativo Post-AS-IS/TO-BE (2026-05-21)

> **Aggiornamento 2026-05-21:** Fasi 0-7 **completate** su `master` (commit `f95e25e`
> e follow-up). Drift register 11/11 ✅. Cost calibration wired in
> `TransactionCostModel`, promotion gate Dagster (`cost_calibration_gate`),
> dashboard math-trace, HRP soft-prior. Spike artifacts in `artifacts/spikes/`.

## Executive summary

Questo piano nasce **dopo** l'ispezione AS-IS / TO-BE prodotta in `docs/internal/codebase_analysis.md` ed è riconciliato con lo stato attuale di `origin/master` (HEAD `21b63af`). Il team ha già chiuso 8 degli 11 mismatch del drift register (`docs/architecture-as-is-to-be-2026-05-21.md`) e ha selezionato come **prima traccia avanzata** il *Self-Calibrating Cost Model* (ADR `docs/adr/2026-05-21-self-calibrating-cost-model.md`).

Le 7 fasi qui sotto sono progettate per essere eseguite **in sequenza da un agente**: ogni fase ha file target espliciti, comandi di verifica, messaggio di commit e definizione di "fatto". Le fasi 0-2 chiudono il debito residuo; 3-5 implementano il challenger di costo; 6-7 abilitano il prossimo track.

---

## Stato di partenza (riferimento, 2026-05-21)

### Drift register — stato (aggiornato 2026-05-21 dopo Fase 1)

| ID | Item | Stato |
|---|---|---|
| M1 | Alpha158 naming → "Alpha158-inspired" | ✅ Risolto (21b63af) |
| M2 | Portfolio/risk docs vs config | ✅ Risolto (Fase 1.2 — `scripts/generate_risk_doc.py`) |
| M3 | Universe docs (26+6+2 bucket structure) | ✅ Risolto (Fase 1.3) |
| M4 | Parkinson `1/(4 ln 2)` canonical | ✅ Risolto (21b63af) |
| M5 | EWM IC-Sharpe (no "rolling 100-day IR") | ✅ Risolto (Fase 1.1) |
| M6 | Orthogonality confidence shrinkage | ✅ Risolto (21b63af) |
| M7 | Sentiment source weighting nel daily | ✅ Risolto (21b63af) |
| M8 | Target engineering separato da inference | ✅ Risolto (Fase 1.4 — `docs/data-flow-daily-vs-training.md`) |
| M9 | Cost model "heuristic" non Almgren-Chriss | ✅ Risolto (21b63af) |
| M10 | Multivariate MC VaR | ✅ Risolto (12276e8) |
| M11 | Pickle hash sidecar (`trusted_pickle_load`) | ✅ Risolto (21b63af) |

### TO-BE — allineamento con il mio report

| Mia proposta | Stato master |
|---|---|
| 5.4.2 Self-calibrating cost model | ✅ **Scelto come primo P2** (ADR-0003) |
| 5.1 TFT/PatchTST, FinGPT/RAG, switching SSM | 📋 P2 challenger pool |
| 5.2.1 MoE gating | 📋 P2 (post-baseline) |
| 5.3 HRP, robust opt, DCC-GARCH | 📋 P2 (post-cost-track) |
| 5.4.1 RL execution | 📋 P3 (richiede fill history) |
| 5.6.4 Observability (math trace) | 📋 P1 (dashboard redesign) |

---

## Fase 0 — Sync del branch di lavoro con master  ✅ COMPLETATA

**Obiettivo:** allineare il branch `claude/codebase-algorithm-analysis-JmoBp` con `origin/master` per evitare conflitti quando si toccano file modificati di recente (aggregator, alpha158, pipeline, dashboard).

**Esito:** rebase pulito su `origin/master` (HEAD `21b63af`); 64 test target (test_council/test_features/test_artifact_governance) verdi; force-push completato.

---

## Fase 1 — Chiusura drift residui (M2, M3, M5, M8)  ✅ COMPLETATA

**Esito:**
- M5 residue: docstring `council/aggregator.py:7` aggiornato a EWM IC-Sharpe (commit `e130367`).
- M2: nuovo `scripts/generate_risk_doc.py` che rigenera il blocco `<!-- BEGIN risk-table -->` da `PortfolioConstructor` live (commit `dbaf2e5`).
- M3: sezione "Asset Universe" del README rifatta con la struttura 26+6+2 e distinzione research vs trading universe (commit `fb243e3`).
- M8: nuovo `docs/data-flow-daily-vs-training.md` con due diagrammi Mermaid che separano daily inference dal training (commit `70f0f59`).

**Obiettivo:** completare l'allineamento doc/config/code prima di iniziare ogni lavoro avanzato.

### Task 1.1 — M5 (residuo): purgare "rolling 100-day IR" da docstrings interni

**File:**
- `council/aggregator.py` — docstring di modulo (linea 7) ancora dice "rolling 100-day Information Ratio". Sostituire con "EWM IC-Sharpe (halflife ≤ 20)".
- Eventuali commenti residui in `aggregator.py:417,430` ("ic_rolling_30d", "sharpe_rolling_60d"): mantenere come chiavi storiche di telemetria ma aggiungere docstring che chiarisca che il **calcolo** è EWM.

**Verifica:**
```bash
grep -rn "rolling 100-day" council/ docs/ README.md AGENTS.md
# expected: 0 hits
python -m pytest tests/test_council.py -v
```

**Commit:** `docs(council): purge residual rolling-100-day IR wording (M5)`

### Task 1.2 — M2: derivare doc risk da `config/`

**File:**
- `README.md` sezione "Risk constraints"
- `docs/architecture-as-is-to-be-2026-05-21.md` (aggiornare riga M2)
- Nuovo helper `scripts/generate_risk_doc.py` che legge `config/runtime.env` + `council/portfolio.py` e stampa la tabella Markdown.

**Approccio:** invece di hard-codare i valori, usare un blocco markdown auto-generato delimitato da `<!-- BEGIN risk-table -->` / `<!-- END risk-table -->`, rigenerabile via `python scripts/generate_risk_doc.py --inplace`.

**Verifica:**
```bash
python scripts/generate_risk_doc.py --dry-run | diff - <(awk '/BEGIN risk-table/,/END risk-table/' README.md)
# expected: identical (or empty diff)
```

**Commit:** `docs(risk): derive constraint table from config (M2)`

### Task 1.3 — M3: universo reale documentato

**File:** `README.md`, `docs/architecture-as-is-to-be-2026-05-21.md` sezione M3.

Documentare: 54 ticker su 11 settori + BTCUSD/ETHUSD (crypto bucket separato), distinzione tra **research universe** (raw OHLCV) e **trading universe** (post `load_universe_as_of()` + history filter). Includere il diagramma del bucket selection.

**Verifica:** confronto manuale + `python -c "from data.pipeline import load_universe_as_of; print(len(load_universe_as_of()))"` ritorna lo stesso numero documentato.

**Commit:** `docs(universe): document real 54+2 universe and bucket contract (M3)`

### Task 1.4 — M8: separare target engineering da daily inference

**File:**
- Nuovo `docs/data-flow-daily-vs-training.md` con 2 diagrammi Mermaid distinti (daily inference, offline training).
- Aggiornare `docs/architecture-as-is-to-be-2026-05-21.md` M8 con link.

**Verifica:** review manuale; nessun cambio di codice in questo task.

**Commit:** `docs(flow): separate daily-inference vs training-target diagrams (M8)`

**Done della Fase 1:** drift register in `architecture-as-is-to-be-2026-05-21.md` ha tutte le righe M1-M11 marcate ✅; tabelle config rigenerabili da script.

---

## Fase 2 — Telemetria fill: prerequisito per cost calibration  ✅ COMPLETATA

**Esito:**
- `execution/fill_log.py` nuovo: `FillRecord` dataclass + `append_fill/append_fills/read_fills` con parquet mensile atomico (commit `6b3599c`).
- `execution/oms.py` esteso: `Order.decision_price` + hook `_append_fill_record` best-effort in `add_fill`; fix di bug pre-esistente in `_save_fill` (JSON serialization); 4 nuovi test di integrazione (commit `b0ad587`).
- `scripts/backfill_fill_log.py` per importare `data/paper_trades/*.json` storici; idempotente e cross-month-safe; 7 nuovi test (commit `037ff77`).
- Test suite: 24 nuovi test (`test_fill_log.py`, `test_oms_fill_log.py`, `test_backfill_fill_log.py`), tutti verdi.

**Obiettivo:** raccogliere fill normalizzati in formato strutturato per alimentare la calibrazione di Fase 3.

### Task 2.1 — Schema fill record

**File:**
- Nuovo `execution/fill_log.py` con dataclass `FillRecord`:
  ```
  fill_id, order_id, ticker, side, qty, fill_price,
  decision_price, decision_ts, fill_ts, broker, venue,
  pipeline_run_id, config_hash, slippage_bps, commission_bps
  ```
- `data/operations/fills/{YYYY-MM}.parquet` come storage partizionato per mese.

**Verifica:**
```bash
python -m pytest tests/test_fill_log.py -v   # da creare
```

**Commit:** `feat(execution): structured FillRecord + parquet log`

### Task 2.2 — Hook in OMS e Alpaca adapter

**File:**
- `execution/oms.py` linea ~250 (dopo `add_fill`): chiamare `fill_log.append(record)` con `decision_price` letto da `Order.created_at` snapshot del prezzo di riferimento.
- `execution/alpaca_adapter.py:690-703` (paper_trades writer): far passare il record dallo stesso log.

**Critico:** `decision_price` deve essere il prezzo al momento della decisione (last close per market orders, limit price per limit). Aggiungere campo a `Order` dataclass se assente.

**Verifica:**
```bash
python -m pytest tests/test_oms.py tests/test_alpaca_adapter.py -v
# Integration: paper trade end-to-end con assert su esistenza file fill log
```

**Commit:** `feat(oms): persist FillRecord on every fill (calibration input)`

### Task 2.3 — Backfill da paper_trades storici

**File:** `scripts/backfill_fill_log.py` che legge `data/paper_trades/{date}.json` esistenti e li converte in `FillRecord`.

**Verifica:** dopo run, `pl.read_parquet("data/operations/fills/*.parquet").shape[0]` ≥ numero righe in paper_trades.

**Commit:** `chore(execution): backfill historical paper trades into fill log`

**Done della Fase 2:** ogni nuovo fill produce automaticamente un `FillRecord`; storico esistente importato.

---

## Fase 3 — Calibration engine (ADR-0003 Stage B core)  ✅ COMPLETATA

**Esito:**
- `council/cost_calibration.py` nuovo: `CalibrationArtifact` dataclass, `compute_is_bps`, `CostCalibrator` (mediana per ticker e per tier `{mega, large, mid, crypto, default}` con soglia `min_fills=30`), `write_calibration` con manifest SHA-256 sidecar, `load_calibration` fail-closed, `run_calibration_job` end-to-end (commit `3835f32`).
- 13 nuovi test in `tests/test_cost_calibration.py` coprono math, tier mapping, esclusione sotto soglia, round-trip, tamper detection, e job end-to-end.
- Asset Dagster `cost_calibration_artifact` + `cost_calibration_job` + `cost_calibration_schedule` (cron `0 23 * * *`, America/New_York) registrati in `data/pipeline.py` Definitions (commit `9cb7e79`).

**Obiettivo:** modulo che legge `FillRecord` e produce `cost_calibration.json` con `kappa_slippage_bps` per ticker/tier.

### Task 3.1 — Implementation Shortfall computation

**File:** nuovo `council/cost_calibration.py`.

Formula (da ADR):
```
IS_bps = 10_000 * (fill_price - decision_price) / decision_price * sign(side)
```

**API:**
```python
class CostCalibrator:
    def __init__(self, fills_path: Path, min_fills: int = 30): ...
    def compute_kappa(self, tier: str | None = None) -> dict[str, float]: ...
    def write_calibration(self, out_path: Path) -> str: ...  # returns sha256
```

**Sample logic:**
- Per ticker: rolling median IS_bps su ultimi N fills.
- Per tier (mega/large/mid/crypto): aggregate dei ticker del tier.
- Se `n < min_fills` → escludere (fallback al lookup statico).

**Verifica:**
```bash
python -m pytest tests/test_cost_calibration.py -v
# Test: fixtures con fills sintetici → kappa atteso; insufficient sample → key assente
```

**Commit:** `feat(council): cost calibration engine (rolling median IS_bps)`

### Task 3.2 — Manifest + sidecar

**File:** stesso modulo. Output `data/operations/cost_calibration.json` + `.manifest` (JSON con `pipeline_run_id`, `config_hash`, `fill_sample_count`, `calibration_window_end`, `sha256_of_json`).

Riusare `council/pickle_security.py` per pattern di hash sidecar (anche se è JSON, mantenere convenzione SHA-256 sidecar per audit consistency).

**Verifica:** test `test_cost_calibration_manifest.py` che verifica round-trip write→read→hash-check.

**Commit:** `feat(council): cost calibration artifact with hash sidecar`

### Task 3.3 — Job notturno

**File:**
- Nuovo Dagster asset in `data/pipeline.py` (Layer 4 estensione): `cost_calibration_artifact` con `RetryPolicy(max_retries=2, delay=60)`.
- Schedule cron `0 23 * * *` (23:00 UTC, dopo close US).
- Dipendenze: dipende da `daily_orders` (per pipeline_run_id) e da `fill_log` non-Dagster (lettura diretta filesystem).

**Verifica:**
```bash
dagster asset materialize -f data/pipeline.py --select cost_calibration_artifact
# inspect: data/operations/cost_calibration.json esistente, valid JSON
```

**Commit:** `feat(pipeline): nightly cost_calibration_artifact asset`

**Done della Fase 3:** girando il pipeline si produce un file di calibrazione versionato e verificabile.

---

## Fase 4 — Wire calibration in TransactionCostModel  ✅ COMPLETATA

**Obiettivo:** far consumare la calibrazione al modello di costo che entra in CVXPY + backtest, con rollback.

### Task 4.1 — Blend lookup + calibrato

**File:** `council/transaction_costs.py`.

Modifica `TransactionCostModel.from_env()`:
```python
calib_path = os.getenv("MLCOUNCIL_COST_CALIBRATION_PATH",
                      "data/operations/cost_calibration.json")
if exists(calib_path) and manifest_verifies(calib_path):
    kappa_calibrated = load_calibration(calib_path)
    alpha = min(1.0, fill_count / CONFIDENCE_FLOOR)  # 0..1
    slippage_bps[ticker] = (1 - alpha) * lookup[ticker] + alpha * kappa_calibrated[ticker]
else:
    slippage_bps = lookup  # fallback statico
```

**Critico:** comportamento fail-safe — se manifest non verifica, log WARNING e usa lookup statico (NON sollevare eccezione).

**Verifica:**
```bash
python -m pytest tests/test_transaction_costs.py -v
# nuovi test: con calib file, senza calib file, con calib hash mismatch
```

**Commit:** `feat(costs): blend calibrated kappa with static lookup`

### Task 4.2 — Lineage in order/execution log

**File:** `execution/oms.py`, `data/pipeline.py` (daily_orders asset).

Aggiungere campo `cost_calibration_version` (SHA-256 del manifest attivo al momento della decisione) a `Order` dataclass e ai parquet `daily_orders`.

**Verifica:** ispezionare `data/orders/{date}.parquet` e confermare presenza colonna.

**Commit:** `feat(lineage): tag orders with cost_calibration_version`

### Task 4.3 — Backtest costs walk-forward

**File:** `backtest/runner.py`, `backtest/report.py`.

Estendere `_compute_stats` per produrre due metriche separate: `net_sharpe_static_costs` e `net_sharpe_calibrated_costs`. Il delta è l'evidenza promotion per Fase 5.

**Verifica:**
```bash
python scripts/run_strategy_backtest.py --cost-mode=both --output data/results/cost_ab.json
python -m pytest tests/test_backtest_validation.py -v
```

**Commit:** `feat(backtest): A/B static vs calibrated cost reporting`

**Done della Fase 4:** un singolo run di backtest produce sia metriche static che calibrated; live trading usa calibrated con rollback automatico.

---

## Fase 5 — Validation, alerting, promotion  ✅ COMPLETATA

**Obiettivo:** governance per evitare che una calibrazione tossica entri in produzione.

### Task 5.1 — Divergence alert

**File:** `council/monitor.py` (nuovo check `cost_calibration_divergence_check`).

Regola: se `|kappa_calibrated - kappa_lookup| > 5 bps` per ≥5 sessioni consecutive sullo stesso tier → alert WARNING. Soglia: critical se >15 bps.

**File config:** aggiungere chiave in `config/monitoring.yaml` (creare se non esiste).

**Verifica:** `python -m pytest tests/test_monitor.py -k cost_calibration -v`

**Commit:** `feat(monitor): cost calibration divergence alert`

### Task 5.2 — Champion/challenger gate (costo, non modello)

**File:** `backtest/validation.py`.

Nuova funzione `validate_cost_calibration_promotion()` con criteri:
- net Sharpe calibrated >= net Sharpe static − 0.1 (no regressione marcata)
- turnover delta entro ±10%
- implementation shortfall mediano dei fill < lookup
- fill_sample_count >= 30 per tier promosso

Se fallisce → `MLCOUNCIL_COST_CALIBRATION_PATH=""` automatico (revert a statico) via env override scritto in `config/runtime_override.env`.

**Verifica:** `python -m pytest tests/test_validation.py -k cost -v`

**Commit:** `feat(validation): cost calibration promotion gate`

### Task 5.3 — Baseline post-cost-track

**File:** `docs/internal/baselines/YYYY-MM-DD-cost-calibrated-baseline.md` (segue il template di `docs/internal/baselines/2026-05-21-clean-baseline.md`).

Catturare: Sharpe lordo/netto, turnover, IS medio, breach count, runtime. Confronto side-by-side col clean baseline.

**Commit:** `docs(baseline): cost-calibrated baseline measurement`

**Done della Fase 5:** track #1 P2 chiuso con evidenza A/B; sistema può promuovere o rollback.

**Post-f95e25e — promotion gate in produzione:** `cost_calibration_gate` asset Dagster
(chiama `run_cost_calibration_promotion_gate`, scrive `config/runtime_override.env`
se fallisce). `cost_calibration_artifact` dipende da `daily_orders` via
`LastPartitionMapping` per `pipeline_run_id`.

---

## Fase 6 — Dashboard math-trace MVP (P1 observability)  ✅ COMPLETATA (f95e25e)

**Obiettivo:** rendere ogni decisione tracciabile dall'UI senza redesign completo.

### Task 6.1 — Constraint waterfall per il portfolio

**File:** `dashboard/charts.py` nuovo grafico `optimizer_waterfall(date)`.

Mostrare, per una data:
1. Pesi greedy `alpha * multiplier` (pre-CVXPY)
2. Pesi dopo budget constraint
3. Pesi dopo position cap
4. Pesi dopo sector cap
5. Pesi dopo turnover cap
6. Pesi dopo vol cap
7. Pesi finali

Ogni step = barra con delta vs step precedente, evidenziando vincoli binding.

**File:** `council/portfolio.py` deve già esporre `optimization_diagnostics` dict (verificare; se assente, esporlo).

**Verifica:** apri Streamlit, naviga a `/portfolio?date=YYYY-MM-DD`, conferma rendering.

**Commit:** `feat(dashboard): optimizer constraint waterfall chart`

### Task 6.2 — Council attribution trace

**File:** `dashboard/data_loader.py:363` usa già `trusted_pickle_load` sul `aggregator_state`. Estendere la pagina dashboard per mostrare per ogni data:
- `weights_log[date]["weights"]`
- `effective_weight_sum` (M6)
- `ortho_applied` flag
- IC e Sharpe rolling per modello

**Commit:** `feat(dashboard): council attribution math-trace panel`

### Task 6.3 — Fill quality panel

**File:** nuova pagina `dashboard/pages/fill_quality.py`.

Mostrare per ticker: IS mediano, slippage realized vs assunto (lookup), trend kappa_calibrated nel tempo, sample size.

**Commit:** `feat(dashboard): fill quality and cost calibration panel`

**Done della Fase 6:** ogni claim del backtest report è verificabile cliccando una data sul dashboard.

---

## Fase 7 — Selezione del prossimo track avanzato  ✅ SPIKE + HRP MVP

**HRP soft-prior (2026-05-21):** `council/hrp.py`, blend opzionale in `PortfolioConstructor` via `MLCOUNCIL_HRP_SOFT_PRIOR=true` e `MLCOUNCIL_HRP_BLEND=0.25`. ADR Accepted.

## Fase 7 (originale) — Selezione del prossimo track avanzato

**Obiettivo:** decidere se il secondo P2 è HRP-soft-prior, robust optimization, o dashboard product redesign.

### Task 7.1 — Mini-spike HRP

**File:** nuovo `notebooks/2026-XX-XX-hrp-spike.ipynb` (oppure script).

Implementare HRP (López de Prado) standalone su ultimi 90 giorni di returns, confrontare allocazione vs CVXPY MV su 5 date sample. Misura: condition number della matrice di covarianza usata, tail-risk dei pesi.

**Deliverable:** ADR `docs/adr/YYYY-MM-DD-hrp-soft-prior.md` con go/no-go.

### Task 7.2 — Mini-spike Robust Optimization

**File:** nuovo notebook/script con CVXPY robust:
$$
\max_w \hat\alpha^\top w - \kappa \sqrt{w^\top \Sigma w}
$$
calibrare $\kappa$ su uno scan. Confronto rispetto al MV su 12 mesi walk-forward.

**Deliverable:** ADR analogo.

### Task 7.3 — Decision review

Aprire ExitPlanMode con l'utente per scegliere: HRP / Robust / Dashboard redesign / TFT spike.

---

## Riassunto dipendenze tra fasi

```
Fase 0 (sync)                                       ✅
    ├──> Fase 1 (drift residui)                     ✅
    │
    └──> Fase 2 (fill telemetry)                    ✅
              └──> Fase 3 (calibration engine)      ✅
                        └──> Fase 4 (wire in TCM)   ✅  (f95e25e)
                                  └──> Fase 5       ✅  (f95e25e + gate asset)
                                            ├──> Fase 6 (dashboard) ✅
                                            └──> Fase 7 (HRP MVP) ✅
```

Le Fasi 4 e 6 possono partire **in parallelo** appena Fase 3 è ferma (lo è ora): Fase 4 tocca solo `council/transaction_costs.py` + backtest, Fase 6 tocca solo `dashboard/`. Fase 5 (governance) dipende strettamente da Fase 4.

## Acceptance criteria globale

- Tutti i test verdi: `python -m pytest tests/ -v --timeout=120`
- Drift register `architecture-as-is-to-be-2026-05-21.md` 11/11 ✅
- File `data/operations/cost_calibration.json` viene rigenerato ogni notte
- Backtest report mostra `net_sharpe_static` e `net_sharpe_calibrated`
- Dashboard renderizza waterfall + council attribution + fill quality
- Due ADR esistono per selezione del prossimo track

## Comandi rapidi per l'agente

```bash
# Setup
git fetch origin master && git rebase origin/master

# Test runner (rapido)
python -m pytest tests/test_council.py tests/test_features.py tests/test_artifact_governance.py -v

# Test runner (completo)
python -m pytest tests/ -v --timeout=120

# Backtest A/B costi
python scripts/run_strategy_backtest.py --cost-mode=both

# Materializza calibration + promotion gate
dagster asset materialize -f data/pipeline.py --select cost_calibration_artifact,cost_calibration_gate

# Dashboard locale
streamlit run dashboard/app.py
```

## Note operative per l'agente

1. **Ogni fase = 1 PR** con commit atomici (uno per task).
2. **Mai saltare la verifica**: i comandi di test elencati per ogni task sono il gate.
3. **Rollback safe by design**: ogni step di Fase 3-4 deve mantenere il fallback statico se la calibrazione fallisce.
4. **No model retrain in queste fasi**: zero modifiche a LGBM, FinBERT, HMM. Lo scopo è onestà di costo e osservabilità.
5. **Doc-first ai mismatch**: se un task richiede modificare codice e doc, prima il test, poi il codice, poi la doc.
6. **Hash sidecar obbligatorio**: ogni nuovo artifact pickle/JSON in `data/operations/` deve avere `.hash` sibling — riusare `council/pickle_security.py`.
