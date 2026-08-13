# Contributing to MLCouncil

Thanks for considering a contribution. MLCouncil is a research-grade paper
trading system: the priority is **mathematical correctness and auditability**
over feature speed.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

Never commit real credentials. Copy `.env.example` to `.env` for local secrets;
`secrets/` and `.env*` are gitignored by design.

## Running tests

```bash
python -m pytest                    # full suite
python -m pytest tests/test_canary.py -v     # canary controller
python -m pytest tests/test_risk_engine.py -v  # VaR / risk engine
python -m pytest tests/test_pipeline.py -v    # Dagster assets
```

Keep tests green for the modules you touch; new features must ship with
statistical verification tests (coverage guarantees, backtests, drift checks),
not just "it runs" tests.

## Code style

- No formatter or linter is configured — follow the style of the surrounding code.
- `data/pipeline.py` uses **Italian comments** — preserve that style.
- `tests/conftest.py` installs a `slowapi` stub; do not add `slowapi` to test
  requirements.

## Repository conventions

- **Wording**: use "EWM IC-Sharpe over recent history (halflife up to 20,
  bounded by the configured history window)", not "rolling 100-day IR".
- **Lookahead safety**: the technical feature set is shifted 1 day — never
  change that without understanding the backtest implications.
- **Flags**: every new `MLCOUNCIL_*` flag must be documented with purpose,
  target phase and expiry in `docs/flag-registry-2026-08-13.md`.
- **Shadow features**: activate through `config/canary.yaml` (gate G1), not by
  default-on flags. Untrained gates stay disabled.
- **Design decisions**: document significant changes as an ADR under
  `docs/adr/` (see `ADR-template.md`).
- **Risk table**: the README risk table is auto-generated —
  run `python scripts/generate_risk_doc.py` after changing risk defaults.

## Commits and branches

- Conventional commits: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`.
- `master` is the only long-lived branch. Open a PR from a short-lived feature
  branch or fork; keep the history linear and meaningful.
- Before pushing, re-check for secrets (`git diff` review) and run the
  affected test suites.

## License

MLCouncil is released under the **GNU Affero General Public License v3.0**
(AGPL-3.0). By contributing, you agree that your contributions are licensed
under the AGPL-3.0 — including the network-service copyleft of section 13
(anyone running a modified version as a service must offer its source).

## Scope notes

The 2026–2030 strategy lives in `docs/roadmap-2026-2030-autonomous-council.md`
and the mathematical foundations in `docs/math-drilldown-2026-2030-autonomous-council.md`.
Check them before starting large quant, risk, or execution work — the roadmap
defines which waves are open for contribution.
