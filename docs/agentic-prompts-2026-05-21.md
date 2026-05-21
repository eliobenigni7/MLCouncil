# Agentic Prompt Pack - MLCouncil AS IS Fixes and TO BE Roadmap

Use these prompts sequentially. Each prompt is designed for a fresh coding agent working in `E:\Github\MLCouncil`.

Global rules for every agent:

- Read `AGENTS.md` first.
- Do not change dashboard UX unless the prompt explicitly asks for dashboard work.
- Preserve Italian comments in `data/pipeline.py`.
- Add or update focused tests before implementation when behavior changes.
- Use current config and code as source of truth, not older README claims.
- Run the exact verification commands listed in the prompt.
- Summarize changed files, test output, and residual risks.

## Prompt 00 - Orientation And Drift Confirmation

```text
You are working in E:\Github\MLCouncil. First read AGENTS.md, README.md, docs/architecture-as-is-to-be-2026-05-21.md, and E:\Github\MLCouncil\tmp\pdfs\MLCouncil_Combined_Analysis.extracted.txt if it exists. If the extracted PDF file is missing, extract E:\Desktop\Sviluppo\MLCouncil_Combined_Analysis.pdf with pdfplumber into tmp/pdfs.

Goal: confirm the current AS IS drift register before any implementation.

Inspect:
- README.md
- AGENTS.md
- config/universe.yaml
- council/portfolio.py
- council/aggregator.py
- data/pipeline.py
- models/sentiment.py
- council/risk_engine.py
- council/transaction_costs.py
- models/regime.py and other model checkpoint save/load paths

Deliver:
- A short table marking each M1-M11 as still open, already fixed, or partly fixed.
- Exact file references for each finding.
- No code edits.
```

## Prompt 01 - Documentation Reconciliation P0

```text
You are working in E:\Github\MLCouncil. Read AGENTS.md and docs/architecture-as-is-to-be-2026-05-21.md.

Goal: reconcile user-facing docs with current code/config for M1, M2, M3, M5, M8, and M9 without changing runtime behavior.

Modify:
- README.md
- AGENTS.md
- docs/fase2-realism.md if cost-model wording is stale
- docs/fase5-governance.md if artifact/risk wording is stale

Required changes:
- Describe the daily ensemble as LightGBM + sentiment, with HMM as regime label/context unless you find current code now passes HMM as an alpha signal.
- Replace "rolling 100-day IR" wording with "EWM IC-Sharpe over recent history, halflife up to 20, bounded by configured history window".
- Replace "19 equities" with the current bucketed universe from config/universe.yaml and mention BTCUSD/ETHUSD crypto support in progress.
- Clarify that daily inference does not compute training targets.
- Clarify that transaction costs are currently a configurable heuristic unless code now implements a calibrated impact model.
- Keep docs concise and operational.

Verify:
- python -m pytest tests/test_council.py -v
- python -m pytest tests/test_dashboard_data_loader.py -v

Deliver:
- Changed files.
- Test output summary.
- Remaining doc drift, if any.
```

## Prompt 02 - Multivariate VaR And Stress Replay P0

```text
You are working in E:\Github\MLCouncil. Read AGENTS.md and docs/architecture-as-is-to-be-2026-05-21.md.

Goal: fix M10 by making Monte Carlo VaR simulate multivariate asset returns, not a univariate Gaussian portfolio return.

Modify:
- council/risk_engine.py
- tests/test_risk_engine.py
- docs/fase3-operational-controls.md or README.md if risk docs describe the old behavior

Implementation requirements:
- Add a test that fails with the current univariate implementation by verifying scenario output changes when cross-asset covariance/correlation changes while marginal vol is held constant.
- In compute_var_monte_carlo(), align returns columns to weights, estimate mean vector and covariance matrix, regularize covariance for numerical stability, and draw multivariate normal scenarios.
- Compute portfolio scenario PnL from scenario matrix @ weight vector * portfolio_value.
- Preserve deterministic seed behavior.
- Add optional historical stress replay helper only if it can be implemented cleanly with existing returns inputs; otherwise document it as a follow-up.

Verify:
- python -m pytest tests/test_risk_engine.py -v
- python -m pytest tests/test_trading_service.py -v

Deliver:
- Changed files.
- Explanation of covariance regularization.
- Test output summary.
```

## Prompt 03 - Pickle Hash Enforcement P0

```text
You are working in E:\Github\MLCouncil. Read AGENTS.md and docs/architecture-as-is-to-be-2026-05-21.md.

Goal: fix M11 by making trusted pickle loading fail closed when hash sidecar policy is violated.

Inspect:
- data/pipeline.py::_safe_pickle_load
- models/regime.py
- models/technical.py
- council/aggregator.py
- any other pickle.load usage under api, data, models, council, execution

Implementation requirements:
- Add tests that cover missing sidecar, mismatched sidecar, and valid sidecar for at least one critical checkpoint loader.
- Make loader behavior explicit: critical model/artifact loads require a .hash sidecar unless a clearly named local/test escape hatch exists.
- Do not break existing tests that create temporary model artifacts; update fixtures to write sidecars where appropriate.
- Avoid broad format migration in this prompt. This is enforcement first, migration later.

Verify:
- python -m pytest tests/test_models.py tests/test_pipeline.py tests/test_artifact_governance.py -v
- python -m bandit -q -r api council execution runtime_env.py -lll

Deliver:
- Changed files.
- Remaining pickle usages and whether each is protected.
```

## Prompt 04 - Sentiment Source Weight Activation P0

```text
You are working in E:\Github\MLCouncil. Read AGENTS.md and docs/architecture-as-is-to-be-2026-05-21.md.

Goal: fix M7 by applying source and/or recency weighting in the operational daily sentiment path.

Inspect:
- data/pipeline.py::sentiment_features
- models/sentiment.py
- data/ingest/news_processor.py
- tests/test_sentiment.py
- tests/test_pipeline.py

Implementation requirements:
- Add a failing test proving two headlines with different source credibility do not contribute equally when source metadata is available.
- Reuse existing source-weighting helpers if present.
- Keep fallback behavior stable when source or timestamp metadata is missing.
- Preserve batching of FinBERT scoring.
- Emit metadata useful for later dashboard display: headline count, weighted average, fallback count if practical.

Verify:
- python -m pytest tests/test_sentiment.py tests/test_pipeline.py -k "sentiment" -v

Deliver:
- Changed files.
- Before/after semantics of sentiment aggregation.
- Test output summary.
```

## Prompt 05 - Council Orthogonality Semantics P0

```text
You are working in E:\Github\MLCouncil. Read AGENTS.md and docs/architecture-as-is-to-be-2026-05-21.md.

Goal: resolve M6 explicitly. Choose one behavior and make code, tests, and docs agree.

Decision options:
1. Confidence shrinkage: keep non-renormalized post-orthogonality weights, expose weight_sum and document that sum < 1 is intentional.
2. Simplex projection: renormalize/project after penalty so weights sum to 1.

Recommended path: choose option 1 unless the user has asked for strict simplex semantics, because the current code comment says non-renormalization is intentional and downstream z-scoring absorbs scale.

Modify:
- council/aggregator.py
- tests/test_council.py
- README.md or docs/architecture-as-is-to-be-2026-05-21.md

Implementation requirements:
- Add/adjust tests for the chosen behavior.
- If keeping confidence shrinkage, log/expose sum of effective weights in attribution.
- If projecting to simplex, ensure the penalty still has an observable effect on relative weights.

Verify:
- python -m pytest tests/test_council.py -v

Deliver:
- Decision taken.
- Changed files.
- Test output summary.
```

## Prompt 06 - Feature Inventory And Parkinson Scaling P0

```text
You are working in E:\Github\MLCouncil. Read AGENTS.md and docs/architecture-as-is-to-be-2026-05-21.md.

Goal: close M1 and M4 by making feature naming and Parkinson volatility mathematically honest.

Modify:
- data/features/alpha158.py
- tests/test_features.py
- README.md
- optional: docs/fase1-foundations.md

Implementation requirements:
- First add a test for the expected Parkinson feature value on a tiny deterministic OHLCV fixture.
- Decide whether to apply the canonical 1/(4 ln 2) scaling or document the existing value as an uncalibrated range-vol feature.
- If applying scaling, update any expected feature values.
- Document actual feature count/inventory from compute_alpha158 output instead of claiming "158+" unless the current output proves that count.

Verify:
- python -m pytest tests/test_features.py -v
- python -m pytest tests/test_pipeline.py -k "alpha158" -v

Deliver:
- Decision taken.
- Changed files.
- Test output summary.
```

## Prompt 07 - Cost Model Feedback Design P0

```text
You are working in E:\Github\MLCouncil. Read AGENTS.md and docs/architecture-as-is-to-be-2026-05-21.md.

Goal: address M9 in two stages: honest naming now, realized-slippage calibration design next.

Stage A implementation:
- Rename misleading docs/comments that imply full Almgren-Chriss if the code is still a heuristic lookup model.
- Ensure README and docs/fase2-realism.md describe actual cost behavior.

Stage B design doc:
- Create docs/adr/YYYY-MM-DD-self-calibrating-cost-model.md using docs/adr/ADR-template.md.
- Define how realized fills from OMS update a kappa/slippage parameter over time.
- Include rollback behavior and minimum fill sample requirements.

Verify:
- python -m pytest tests/test_backtest.py tests/test_trading_service.py -v

Deliver:
- Changed docs.
- ADR path.
- Test output summary.
```

## Prompt 08 - Clean Baseline Measurement After P0

```text
You are working in E:\Github\MLCouncil after P0 fixes are merged. Read AGENTS.md and docs/architecture-as-is-to-be-2026-05-21.md.

Goal: generate a clean baseline that future TO BE challengers can beat.

Run or implement:
- Existing walk-forward validation path.
- Benchmark comparison against equal-weight, momentum, inverse-volatility, and vol-target equal-weight if available.
- Regime breakdown.
- Gross/net cost comparison.
- Runtime duration for major stages.

Expected artifact:
- docs/baselines/YYYY-MM-DD-clean-baseline.md
- Include commands, environment metadata, git sha, config hash, data date range, metrics, and caveats.

Verify:
- python -m pytest tests/test_backtest_validation.py tests/test_retraining.py -v

Deliver:
- Baseline document path.
- Exact commands run.
- Metrics summary and blockers.
```

## Prompt 09 - Dashboard Brainstorming Only

```text
You are working in E:\Github\MLCouncil. Do not implement UI yet.

Goal: prepare a dashboard redesign brainstorming session with the user.

Read:
- dashboard/app.py
- dashboard/charts.py
- dashboard/data_loader.py
- docs/architecture-as-is-to-be-2026-05-21.md
- README.md dashboard/admin sections

Prepare:
- A map of current dashboard pages/widgets/data sources.
- A list of missing diagnostics needed for the desired "agentic quant council" UI: signal lineage, formula trace, weight attribution, constraint waterfall, VaR scenarios, cost attribution, execution feedback.
- 3 dashboard product directions with tradeoffs:
  1. Command center
  2. Quant lab notebook
  3. Agentic council cockpit

Ask the user one question at the end: which direction should be explored first?

No code edits.
```

## Prompt 10 - Advanced TO BE Track Selection

```text
You are working in E:\Github\MLCouncil after P0 cleanup and baseline measurement. Do not implement a disruptive model yet.

Goal: select the first advanced challenger based on expected value and available evidence.

Compare:
- TFT/PatchTST alpha challenger
- MoE council gating
- HRP/robust portfolio construction
- Self-calibrating execution cost model if not already implemented
- Dashboard math-trace implementation

For each option provide:
- Required input data.
- Files likely touched.
- New tests needed.
- Expected metric uplift.
- Failure mode.
- Time estimate.

Recommend one first track and create a one-page ADR draft.
```
