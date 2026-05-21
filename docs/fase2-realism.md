# Phase 2 — Backtest Realism And Transaction Costs

This document describes **implemented** cost and realism behavior in MLCouncil.
It does not describe a full Almgren-Chriss optimal-execution model.

## Transaction Cost Model (AS IS)

Runtime costs are estimated by `council/transaction_costs.py`:

- **Commission:** default 1 bps (`MLCOUNCIL_COMMISSION_BPS`)
- **Slippage:** default 3 bps (`MLCOUNCIL_SLIPPAGE_BPS`), or a per-ticker lookup table when `estimate_slippage_bps()` is used
- **Portfolio/backtest path:** one-way turnover × (commission + slippage) bps × portfolio value

The per-ticker slippage map is a **static liquidity heuristic** (tiered bps by symbol). It is not calibrated from realized fills and does not solve a square-root market-impact optimization problem.

## Gross vs Net Metrics

- Backtests and `backtest/simulator.py` subtract the shared `TransactionCostModel` estimate from gross returns.
- Promotion gates compare gross/net divergence against configured tolerances (see walk-forward validation docs in-repo).

## What Is Not Implemented Yet

- Realized-fill feedback calibration (kappa / slippage drift from OMS fills)
- Full Almgren-Chriss trajectory optimization
- Volume-conditioned dynamic impact beyond the simple optional `dollar_volume` multiplier in `estimate_slippage_bps()`

See [docs/adr/2026-05-21-self-calibrating-cost-model.md](adr/2026-05-21-self-calibrating-cost-model.md) for the proposed self-calibrating design.
