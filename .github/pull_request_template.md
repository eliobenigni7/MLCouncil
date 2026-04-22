## Summary

- What changed
- Why this change is needed
- Scope boundaries (what is intentionally not included)

## Validation

- [ ] `python -m pytest`
- [ ] `python -m pytest --cov=. --cov-report=term --cov-fail-under=68`
- [ ] `python -m ruff check api/main.py api/auth.py api/services/trading_service.py runtime_env.py council/portfolio.py`
- [ ] `python -m mypy --config-file mypy.ini api/main.py api/auth.py api/services/trading_service.py runtime_env.py council/portfolio.py`
- [ ] `python -m pip_audit -r requirements.txt --progress-spinner off`
- [ ] `python -m bandit -q -r api council execution runtime_env.py -lll`

## Risk / Safety

- [ ] Auth and secret handling unchanged or explicitly reviewed
- [ ] Portfolio/risk behavior unchanged or regression-tested
- [ ] Trading-critical paths reviewed (`api/services/trading_service.py`, `execution/`, `council/risk_engine.py`)
- [ ] Data/model lineage impacts documented

## Promotion / Backtest Impact (if applicable)

- [ ] Promotion gate metrics present (`oos_sharpe`, `oos_max_drawdown`, `oos_turnover`, `walk_forward_window_count`, `pbo`)
- [ ] Benchmark and regime impact assessed
- [ ] Artifact manifests/checksums produced for new critical artifacts

## Docs

- [ ] Updated docs/runbook/ADR where behavior or policy changed
