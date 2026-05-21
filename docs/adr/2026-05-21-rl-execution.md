# ADR: RL Execution Agent (T4.1 Shadow)

- Date: 2026-05-21
- Status: Accepted (scaffolding)

## Decision

`execution/rl_agent.py` + `execution/lob_simulator.py`; TWAP/VWAP fallback unless
`MLCOUNCIL_RL_EXECUTION_ENABLED=true` and trained PPO checkpoint exists.

Blocked on ≥6 months fill history for production training.

## Rollback

Unset `MLCOUNCIL_RL_EXECUTION_ENABLED`.
