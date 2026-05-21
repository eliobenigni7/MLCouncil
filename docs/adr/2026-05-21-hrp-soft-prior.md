# ADR: HRP Soft Prior (Mini-Spike)

- Date: 2026-05-21
- Status: Accepted (soft-prior MVP)
- Spike: `scripts/spike_hrp.py` → `data/results/spike_hrp.json`

## Context

Portfolio construction uses CVXPY mean-variance with sector/vol/turnover caps. HRP (López de Prado) may stabilise weights when covariance is ill-conditioned.

## Spike summary

The spike compares inverse-vol HRP-style weights vs equal weight on ~90 days of returns for six large-cap names. It records `covariance_condition_number` and sample-day PnL deltas.

## Decision

**Conditional go** for a soft-prior track: proceed only if spike `recommendation` is `go` (condition number &lt; 1e4) and a follow-up walk-forward shows tail-risk improvement vs MV.

## Consequences

- Positive: potentially lower concentration in correlated clusters.
- Trade-off: extra optimisation stage; must not break existing CVXPY constraint contract.
- Deferred if spike `no-go` or CVXPY unavailable in target environment.

## Rollback

Set ``MLCOUNCIL_HRP_SOFT_PRIOR=false`` (default). CVXPY solution is unchanged when disabled.

## Implementation (2026-05-21)

- ``council/hrp.py`` — full HRP from covariance via scipy linkage
- ``PortfolioConstructor.optimize()`` — optional blend ``(1-λ)*MV + λ*HRP`` then re-project to capped simplex
- Env: ``MLCOUNCIL_HRP_SOFT_PRIOR``, ``MLCOUNCIL_HRP_BLEND`` (default 0.25)
