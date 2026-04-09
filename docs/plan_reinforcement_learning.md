# Piano di Implementazione: Reinforcement Learning in MLCouncil

> Data: 2026-04-08  
> Stato: DRAFT  
> Autore: Claude Code  

---

## Sommario Esecutivo

Questo piano descrive come integrare il Reinforcement Learning (RL) nel sistema MLCouncil in **4 fasi incrementali**, ciascuna indipendentemente utile. L'obiettivo primario e' sostituire le euristiche statiche (IC-Sharpe per i pesi del council, CVXPY per il portafoglio) con agenti RL che apprendono politiche ottimali dai dati storici e si adattano in tempo reale.

**Architettura target**: l'agente RL opera come un **meta-layer** sopra i modelli alpha esistenti. Non sostituisce LightGBM, FinBERT o HMM — impara come combinarli e come tradurre i loro segnali in posizioni.

---

## Fase 1: RL Council Weight Optimizer (Priorita' ALTA)

### Obiettivo
Sostituire l'adattamento euristico dei pesi in `council/aggregator.py` (rolling IC-Sharpe + clip) con un agente RL che impara la mappatura ottimale `(segnali, regime, stato_mercato) → pesi_modelli`.

### Perche' qui prima
- Il sistema di pesi attuale e' una euristica lineare (Sharpe rolling → adjust → clip)
- L'RL puo' catturare relazioni non-lineari tra regime, correlazione dei segnali e peso ottimale
- Impatto immediato sul P&L senza modificare modelli o esecuzione
- Superficie di azione piccola (3 pesi → simplesso), ideale per un primo RL

### Design

#### Ambiente (`council/rl/env_council.py`)

```python
class CouncilWeightEnv(gym.Env):
    """
    Observation space (dim ~25):
      - segnali_correnti: [lgbm_z, sentiment_z, hmm_regime_onehot]  (5)
      - rolling_IC per modello (30d): [ic_lgbm, ic_sent, ic_hmm]    (3)
      - rolling_sharpe per modello (60d): [sh_lgbm, sh_sent, sh_hmm] (3)
      - correlazioni_cross: [corr_lg_se, corr_lg_hm, corr_se_hm]    (3)
      - stato_portafoglio: [cash_pct, gross_exposure, net_beta]       (3)
      - macro_context: [vix_z, yield_spread_z, sp500_ret_21d_z]      (3)
      - meta_features: [days_since_retrain, signal_dispersion,
                         signal_kurtosis, turnover_5d]                (4+)

    Action space: Dirichlet(3) → [w_lgbm, w_sentiment, w_hmm]
      - Bounded: min=0.05, max=0.70 per modello
      - Somma = 1.0

    Reward (giornaliero):
      r_t = portfolio_return_t
            - lambda_risk * max(0, drawdown_t - threshold)
            - lambda_tc  * turnover_cost_t
            - lambda_ic  * (1 - mean_IC_t)  # incentivo a mantenere IC alto
    
    Step = 1 giorno di trading
    Episode = 1 anno (252 giorni) con reset casuale nella storia
    """
```

#### Agente

| Aspetto | Scelta | Motivazione |
|---------|--------|-------------|
| Algoritmo | **PPO** (Proximal Policy Optimization) | Stabile, sample-efficient per azioni continue, robusto a iperparametri |
| Libreria | **Stable-Baselines3** | Matura, ben documentata, facile integrazione con gym |
| Policy network | MLP [64, 64] con LayerNorm | Observation space piccolo, non serve complessita' |
| Action head | Beta distribution (non Gaussian) | Output naturalmente in [0,1], poi normalizzato a simplesso |
| Training | 500k steps su dati 2018-2024, eval su 2024-2025 | Abbastanza storia per convergenza |

#### File da creare/modificare

| File | Azione | Descrizione |
|------|--------|-------------|
| `council/rl/__init__.py` | **NUOVO** | Package init |
| `council/rl/env_council.py` | **NUOVO** | Gym environment per weight optimization |
| `council/rl/agent_council.py` | **NUOVO** | Wrapper PPO con train/predict/save/load |
| `council/rl/reward.py` | **NUOVO** | Funzioni di reward configurabili |
| `council/rl/features.py` | **NUOVO** | Costruzione observation vector da dati pipeline |
| `council/aggregator.py` | **MODIFICA** | Aggiungere modalita' `mode="rl"` a `CouncilAggregator` |
| `config/rl.yaml` | **NUOVO** | Iperparametri RL (lr, gamma, lambda_risk, ecc.) |
| `scripts/train_rl_council.py` | **NUOVO** | Script di training offline |
| `tests/test_rl_council.py` | **NUOVO** | Unit test per env, agent, reward |

#### Integrazione con il sistema esistente

```
                    ┌─────────────────────────────┐
                    │  CouncilAggregator           │
                    │                              │
  signals ────────► │  if mode == "heuristic":     │
  regime  ────────► │    → IC-Sharpe weights       │ ──► council_signal
  macro   ────────► │  elif mode == "rl":           │
  portfolio_state ► │    → PPO agent.predict()     │
                    │    → enforce bounds           │
                    └─────────────────────────────┘
```

L'aggregatore mantiene il fallback euristico. In produzione si puo' passare gradualmente dal 100% euristico al 100% RL tramite un parametro `rl_blend_ratio` in `config/rl.yaml`.

#### Metriche di successo
- Sharpe ratio backtest RL > Sharpe euristico di almeno 0.15
- Max drawdown RL <= Max drawdown euristico
- Turnover medio RL <= 1.2x turnover euristico
- IC medio dei pesi RL >= IC euristico

---

## Fase 2: RL Portfolio Optimizer (Priorita' MEDIA)

### Obiettivo
Aggiungere un agente RL come alternativa/complemento a CVXPY in `council/portfolio.py`. L'RL impara direttamente la mappatura `(alpha_signals, risk_state, current_portfolio) → target_weights` senza bisogno di una matrice di covarianza esplicita.

### Perche'
- CVXPY risolve un QP convesso che assume rendimenti attesi lineari e rischio quadratico
- L'RL puo' catturare costi di transazione non-lineari, impatto di mercato, e vincoli soft
- Puo' imparare implicitamente il trade-off turnover/alpha che oggi e' hardcoded (`tc_lambda=1.0`)

### Design

#### Ambiente (`council/rl/env_portfolio.py`)

```python
class PortfolioEnv(gym.Env):
    """
    Observation space (dim ~60):
      - council_signal per ticker (19 tickers)
      - conformal_multiplier per ticker (19)
      - current_weights (19)
      - portfolio_stats: [total_return_5d, vol_20d, drawdown,
                          turnover_5d, cash_pct, beta]  (6)

    Action space: Box(19) → softmax → target weights
      - Post-processing: clip a max_position=0.10, renormalize

    Reward:
      r_t = portfolio_return_t
            - lambda_vol * max(0, realized_vol - 0.20/sqrt(252))
            - lambda_tc  * transaction_cost_t
            - lambda_conc * concentration_penalty_t

    Constraints (hard, post-action):
      - Long-only: w >= 0
      - Budget: sum(w) = 1
      - Max position: w_i <= 0.10
      - Max sector: sector_sum <= 0.25
    """
```

#### Agente

| Aspetto | Scelta | Motivazione |
|---------|--------|-------------|
| Algoritmo | **SAC** (Soft Actor-Critic) | Migliore per azioni continue ad alta dimensionalita', esplora automaticamente |
| Policy network | MLP [128, 128] | Spazio d'azione piu' grande richiede piu' capacita' |
| Action space | 19-dim continuo → softmax + clip | Garantisce validita' dei pesi |
| Replay buffer | 500k transizioni | SAC e' off-policy, sfrutta replay |

#### File da creare/modificare

| File | Azione |
|------|--------|
| `council/rl/env_portfolio.py` | **NUOVO** |
| `council/rl/agent_portfolio.py` | **NUOVO** |
| `council/portfolio.py` | **MODIFICA** — aggiungere `mode="rl"` |
| `scripts/train_rl_portfolio.py` | **NUOVO** |
| `tests/test_rl_portfolio.py` | **NUOVO** |

#### Integrazione

```
council_signal ──► ┌─────────────────────────┐
conformal_mult ──► │  PortfolioConstructor    │
current_weights ─► │                          │
                   │  if mode == "cvxpy":     │ ──► target_weights ──► orders
                   │    → QP optimization     │
                   │  elif mode == "rl":       │
                   │    → SAC agent.predict() │
                   │    → enforce constraints  │
                   └─────────────────────────┘
```

#### Rischi specifici
- **Overfitting**: Lo spazio d'azione a 19 dimensioni richiede molta piu' storia. Mitigazione: data augmentation (bootstrap dei rendimenti), dropout nella policy, early stopping su validation set.
- **Vincoli violati**: L'RL non garantisce vincoli. Mitigazione: projection layer post-azione che proietta su simplesso + clip.
- **Instabilita' del training**: SAC puo' divergere. Mitigazione: gradient clipping, target network soft update (tau=0.005).

---

## Fase 3: RL Execution Optimizer (Priorita' MEDIA-BASSA)

### Obiettivo
Ottimizzare il timing e lo slicing degli ordini in `execution/slicer.py`. Attualmente TWAP/VWAP sono profili statici. Un agente RL puo' imparare a eseguire minimizzando l'impatto di mercato.

### Design

#### Ambiente (`council/rl/env_execution.py`)

```python
class ExecutionEnv(gym.Env):
    """
    Per ogni ordine parent da eseguire:

    Observation (dim ~15):
      - remaining_qty / total_qty
      - elapsed_time / total_time
      - vwap_so_far vs arrival_price
      - spread_bps, depth_bid, depth_ask
      - volatility_intraday
      - volume_participation_rate
      - market_momentum_5min

    Action: fraction of remaining_qty to execute NOW
      - Continuo in [0, 1]
      - 0 = wait, 1 = execute tutto subito

    Reward (per slice):
      r = -(execution_price - arrival_price) * qty_executed
          - lambda_urgency * remaining_qty * (elapsed / total)^2

    Episode = 1 ordine parent (es. 30 minuti, steps da 1 minuto)
    """
```

#### Considerazioni
- Richiede dati **intraday** (L2 book data o almeno 1-min bars) — attualmente non disponibili nel sistema
- Utile solo quando gli ordini sono abbastanza grandi da avere impatto (>5% ADV)
- Per il volume corrente di MLCouncil (19 tickers, daily rebalance), il beneficio e' marginale
- **Raccomandazione**: implementare solo dopo che Fase 1 e 2 sono in produzione e il capitale gestito giustifica il costo

#### File da creare

| File | Azione |
|------|--------|
| `council/rl/env_execution.py` | **NUOVO** |
| `council/rl/agent_execution.py` | **NUOVO** |
| `execution/slicer.py` | **MODIFICA** — aggiungere `mode="rl"` |
| `data/ingest/intraday.py` | **NUOVO** — ingest dati intraday |

---

## Fase 4: Meta-RL per Regime Adaptation (Priorita' BASSA)

### Obiettivo
Un meta-learner RL che decide *quando* ritrainare i modelli alpha e *come* adattare gli iperparametri del sistema in risposta a cambiamenti di regime.

### Design

#### Ambiente

```python
class MetaRLEnv(gym.Env):
    """
    Observation:
      - regime_history (ultimi 30 giorni)
      - model_staleness (giorni dall'ultimo retrain per modello)
      - performance_decay (IC trend per modello)
      - drift_scores (KS test p-values)

    Action (discreta):
      0: no action
      1: retrain technical model
      2: retrain sentiment model  
      3: retrain regime model
      4: increase risk aversion (lambda_risk *= 1.2)
      5: decrease risk aversion (lambda_risk /= 1.2)
      6: switch to defensive weights
      7: switch to aggressive weights

    Reward: portfolio_return - retrain_cost
    """
```

#### Considerazioni
- Richiede molti episodi di training (ogni episodio = mesi di mercato)
- Il reward signal e' molto ritardato (retrain oggi → effetto in settimane)
- **Raccomandazione**: Fase esplorativa/ricerca. Non implementare in produzione prima di aver validato su almeno 5 anni di backtest.

---

## Infrastruttura Comune (Pre-requisiti per tutte le fasi)

### Dipendenze da aggiungere a `requirements.txt`

```
# Reinforcement Learning
gymnasium>=1.0.0
stable-baselines3>=2.4.0
sb3-contrib>=2.4.0          # per RecurrentPPO se necessario
shimmy>=2.0.0               # compatibilita' gym/gymnasium

# Opzionale per Fase avanzate
tensorboard>=2.18.0         # monitoring del training RL
optuna>=4.0.0               # hyperparameter tuning
```

### Struttura directory

```
council/rl/
├── __init__.py
├── env_council.py          # Fase 1: ambiente pesi council
├── env_portfolio.py        # Fase 2: ambiente portafoglio
├── env_execution.py        # Fase 3: ambiente esecuzione
├── agent_council.py        # Wrapper PPO per council
├── agent_portfolio.py      # Wrapper SAC per portfolio
├── agent_execution.py      # Wrapper per execution
├── reward.py               # Funzioni reward condivise
├── features.py             # Feature engineering per observation space
├── wrappers.py             # Gym wrappers (normalizzazione, logging)
├── callbacks.py            # SB3 callbacks (eval, checkpoint, MLflow)
└── utils.py                # Utility (replay, seeding, ecc.)

config/
├── rl.yaml                 # Iperparametri RL

scripts/
├── train_rl_council.py     # Training Fase 1
├── train_rl_portfolio.py   # Training Fase 2
├── eval_rl.py              # Valutazione comparativa RL vs euristico

models/checkpoints/
├── rl_council_ppo.zip      # Checkpoint PPO
├── rl_portfolio_sac.zip    # Checkpoint SAC

tests/
├── test_rl_council.py
├── test_rl_portfolio.py
├── test_rl_reward.py
```

### Integrazione con MLflow

Ogni training RL logga:
- Iperparametri (lr, gamma, n_steps, ecc.)
- Curve di training (reward medio per episodio)
- Metriche di valutazione (Sharpe, drawdown, turnover su validation set)
- Checkpoint del modello come artifact
- Confronto A/B con baseline euristico

### Integrazione con Dagster

Aggiungere due asset al pipeline (`data/pipeline.py`):

```python
@asset(deps=["council_signal", "current_regime"])
def rl_council_weights(context):
    """Pesi council calcolati dall'agente RL (Fase 1)."""
    agent = RLCouncilAgent.load("models/checkpoints/rl_council_ppo.zip")
    obs = build_council_observation(...)
    weights = agent.predict(obs)
    return weights

@asset(deps=["rl_council_weights", "conformal_multipliers"])
def rl_portfolio_weights(context):
    """Pesi portafoglio calcolati dall'agente RL (Fase 2)."""
    agent = RLPortfolioAgent.load("models/checkpoints/rl_portfolio_sac.zip")
    obs = build_portfolio_observation(...)
    weights = agent.predict(obs)
    return weights
```

---

## Anti-pattern da Evitare

| Anti-pattern | Perche' e' un problema | Soluzione |
|---|---|---|
| Trainare RL su tutto il dataset | Overfitting catastrofico | Train/val/test split temporale rigoroso |
| Reward solo su rendimento | Ignora rischio, l'agente diventa un gambler | Reward multi-obiettivo con penalita' rischio |
| Osservazioni non normalizzate | Gradients esplodono, training instabile | VecNormalize wrapper di SB3 |
| Nessun fallback | Se l'RL crasha, il sistema si ferma | Fallback automatico a euristica |
| Retraining troppo frequente | Instabilita' dei pesi, turnover alto | Retraining settimanale, smoothing esponenziale dei pesi |
| Lookahead bias nell'env | Backtest invalido | Usare solo dati t-1 nell'observation a tempo t |

---

## Piano Temporale Indicativo

| Fase | Effort stimato | Dipendenze |
|------|----------------|------------|
| **Infrastruttura comune** | 2-3 giorni | Nessuna |
| **Fase 1: RL Council Weights** | 5-7 giorni | Infrastruttura |
| **Validazione Fase 1** | 3-4 giorni | Fase 1 completa |
| **Fase 2: RL Portfolio** | 7-10 giorni | Fase 1 validata |
| **Validazione Fase 2** | 4-5 giorni | Fase 2 completa |
| **Fase 3: RL Execution** | 7-10 giorni | Dati intraday disponibili |
| **Fase 4: Meta-RL** | Ricerca | Fase 1+2 in produzione |

---

## Criteri di Go/No-Go per Produzione

Prima di attivare l'RL in produzione (per ogni fase):

1. **Backtest**: Sharpe RL >= Sharpe baseline su test set (2024-2025)
2. **Robustezza**: Performance stabile su 3+ seed di training diversi
3. **Drawdown**: Max drawdown RL <= 1.1x max drawdown baseline
4. **Turnover**: Turnover medio RL <= 1.3x turnover baseline
5. **Paper trading**: Almeno 30 giorni di paper trading senza anomalie
6. **Fallback testato**: Il sistema torna a euristica in <1 secondo se l'RL fallisce
7. **Monitoring**: Alert configurati per reward < soglia e drift delle azioni

---

## Riferimenti Tecnici

- [Stable-Baselines3 docs](https://stable-baselines3.readthedocs.io/)
- [FinRL: Financial RL framework](https://github.com/AI4Finance-Foundation/FinRL)
- Jiang et al., "Deep Reinforcement Learning for Portfolio Management" (2017)
- Fischer & Krauss, "Deep learning with long short-term memory networks for financial market predictions" (2018)
- Yang et al., "Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy" (2020)
