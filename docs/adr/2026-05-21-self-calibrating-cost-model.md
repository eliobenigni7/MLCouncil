# ADR-0003: Self-Calibrating Transaction Cost Model From Realized Fills

- Date: 2026-05-21
- Status: Proposed
- Decision owners: MLCouncil quant platform
- Related PR/Issue: Foundation P0 cost-model feedback (M9)

## Context

Backtests and the portfolio constructor currently use `council/transaction_costs.py`, a configurable heuristic (default commission + slippage bps, optional per-ticker lookup). Documentation previously overstated this as Almgren-Chriss optimal execution, which misrepresents validation credibility.

Paper trading via Alpaca produces realized fills with price, quantity, and timestamps. Without a feedback loop, backtest net metrics can drift from live/paper outcomes as liquidity and participation change.

## Decision

Introduce a **self-calibrating cost layer** that periodically fits slippage/impact coefficients from realized fills and writes versioned parameters consumed by `TransactionCostModel`, while keeping the current heuristic as rollback fallback until calibration quality gates pass.

Calibration target (v1):

- Per-ticker or per-tier `kappa_slippage_bps` adjusted from implementation shortfall:
  `IS_bps = 10_000 * (fill_price - decision_price) / decision_price * sign(side)`
- Use rolling median IS by ticker tier over the last `N` fills (minimum sample threshold).
- Blend calibrated kappa with static lookup: `slippage_bps = (1 - alpha) * lookup + alpha * kappa_calibrated`, with `alpha` capped by fill-count confidence.

Artifacts:

- `data/operations/cost_calibration.json` + `.manifest` sidecar
- Lineage fields: `pipeline_run_id`, `config_hash`, `fill_sample_count`, `calibration_window_end`

## Consequences

- Positive: backtest and paper paths converge when fills accumulate; promotion evidence becomes more credible.
- Trade-off: sparse fill history leaves tiers on lookup fallback; early calibration may be noisy.
- Operations: failed calibration or low sample count must alert but must not block paper trading (fallback to heuristic).

## Alternatives Considered

1. **Keep static lookup only** — simple, but perpetuates gross/net drift (rejected for TO BE).
2. **Full Almgren-Chriss solver online** — theoretically clean, but needs reliable participation forecasts and inventory state not yet modeled (deferred).
3. **Manual quarterly bps table edits** — low engineering cost, no audit trail, poor reproducibility (rejected).

## Rollout Plan

1. Log normalized fill records from `execution/oms.py` / Alpaca adapter into `data/operations/fills/`.
2. Nightly job computes tier-level kappa; writes `cost_calibration.json` when `fill_count >= MIN_FILLS` (proposed: 30 per tier).
3. `TransactionCostModel.from_env()` loads calibration if manifest hash verifies; else static lookup.
4. Dashboard/admin surfaces active calibration version and sample counts.

## Verification

- Unit tests: calibration math on synthetic fills; fallback when `n < MIN_FILLS`.
- Integration: backtest net PnL changes only within tolerance when calibration applied vs baseline lookup.
- Manifest SHA-256 recorded for every calibration artifact.
- Rollback: delete or disable calibration file → model reverts to env/lookup defaults without code deploy.

## Rollback

- Set `MLCOUNCIL_COST_CALIBRATION_PATH` empty or remove calibration JSON.
- Keep prior lookup table in `transaction_costs.py` as authoritative until re-enabled.
