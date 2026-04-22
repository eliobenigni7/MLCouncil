---
name: Model promotion review
about: Request promotion of a candidate model with required evidence
title: "[PROMOTION] "
labels: model-promotion
assignees: ""
---

## Candidate

- Model name:
- Candidate version:
- Pipeline run id:
- Environment:

## Required Metrics

- `sharpe`:
- `max_drawdown`:
- `turnover`:
- `oos_sharpe`:
- `oos_max_drawdown`:
- `oos_turnover`:
- `walk_forward_window_count`:
- `pbo`:

## Validation Evidence

- [ ] Walk-forward report attached
- [ ] Regime breakdown attached
- [ ] Benchmark comparison attached
- [ ] Gross vs net metrics consistent with transaction cost assumptions

## Governance Evidence

- [ ] MLflow tags complete (`pipeline_run_id`, `data_version`, `feature_version`, `model_version`)
- [ ] Critical artifacts include checksum/manifest sidecars
- [ ] No open high-severity operational blockers

## Decision

- [ ] Approve promotion
- [ ] Reject promotion (include rationale)
