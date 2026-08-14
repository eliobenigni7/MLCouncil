# ADR: Differentiable Portfolio Constructor (T3.3 Shadow)

- Date: 2026-05-21
- Status: Accepted (shadow scaffold)
- Related: `docs/internal/disruptive-roadmap-2026-05-21.md` Wave 3 T3.3

## Context

Decision-focused learning requires gradients through the portfolio QP
(Agrawal et al. 2019, cvxpylayers). Production uses CVXPY in
`council/portfolio.py` with hard sector/turnover/vol constraints.

## Decision

1. **`council/portfolio_diff.py`** — `DifferentiablePortfolioConstructor` delegates
   to `PortfolioConstructor` until cvxpylayers E2E training is promoted.
2. **`MLCOUNCIL_PORTFOLIO_MODE`** — `cvxpy` (default) or `diff`.
3. **`get_portfolio_constructor()`** used by `data/pipeline.py` and `scripts/run_pipeline.py`.
4. **`scripts/train_alpha_portfolio_end2end.py`** — synthetic E2E scaffold + results JSON.
5. Optional dependency `cvxpylayers>=0.1.6` not added to requirements until spike
   passes stability gate (no NaN grads, ≤ 200 epochs).

## Rollback

Default `cvxpy` mode.

## Verification

```bash
python scripts/train_alpha_portfolio_end2end.py
python -m pytest tests/test_portfolio_diff.py -v
```
