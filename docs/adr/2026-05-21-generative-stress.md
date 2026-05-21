# ADR: Generative Stress Scenarios (T4.3 Shadow)

- Date: 2026-05-21
- Status: Accepted (scaffolding)

## Decision

`council/generative_stress.py` + `RiskEngine.compute_var(..., method="generative")`.
Gaussian fallback when torch/diffusers unavailable.

## Rollback

Use `method="monte_carlo"` or `historical` in risk calls.
