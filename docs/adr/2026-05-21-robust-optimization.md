# ADR: Robust Portfolio Optimization (Mini-Spike)

- Date: 2026-05-21
- Status: Proposed
- Spike: `scripts/spike_robust_opt.py` → `data/results/spike_robust_opt.json`

## Context

Mean-variance optimisers are sensitive to covariance estimation error. A robust objective

$$\max_w \alpha^\top w - \kappa \sqrt{w^\top \Sigma w}$$

penalises volatility more aggressively than a fixed vol cap alone.

## Spike summary

Synthetic six-asset problem with κ scan `{0, 0.5, 1, 2, 5}`. Records weights, portfolio vol, and condition number.

## Decision

**Conditional go** for a champion/challenger track after cost-calibration baseline is stable: integrate κ as config param, gate promotion on 12-month walk-forward vs MV.

## Consequences

- Positive: may reduce extreme weights under noisy Σ.
- Trade-off: κ tuning and slower solves; requires CVXPY+SCS in production images.

## Rollback

Keep κ=0 (pure MV objective) as default in `council/portfolio.py`.
