# MLCouncil System Stabilization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the local product actually runnable end-to-end with no application errors in the main surfaces, a working Dagster pipeline path, and a test suite that proves the core contracts stay aligned.

**Architecture:** Stabilize the system from the bottom up. First lock configuration and dependency contracts so runtime code stops disagreeing with YAML, containers, and the local environment. Then harden the Dagster asset path and council aggregation internals, then finish by aligning the FastAPI admin, Streamlit dashboard, and Docker/dev startup behavior with the verified backend contracts.

**Tech Stack:** Python 3.12/3.13, FastAPI, Dagster, Streamlit, Plotly, Polars, Pandas, Docker Compose, pytest, Playwright CLI

---

## Known Baseline Before Starting

- Playwright verified that:
  - Dagster UI loads.
  - FastAPI Swagger loads.
  - Admin UI root and Streamlit dashboard had real runtime bugs, now partially stabilized but still need broader hardening.
- Fresh test evidence showed two pipeline failures:
  - `tests/test_pipeline.py::TestQualityChecks::test_raw_ohlcv_asset_fails_on_empty_download`
  - `tests/test_pipeline.py::TestFullPipelineSynthetic::test_council_signal_aggregation`
- Root causes already visible in code:
  - [config/universe.yaml](E:/Github/MLCouncil/config/universe.yaml) schema no longer matches [data/pipeline.py](E:/Github/MLCouncil/data/pipeline.py).
  - [council/aggregator.py](E:/Github/MLCouncil/council/aggregator.py) assumes a non-empty frame shape when recording first signal history rows.
- The worktree is already dirty. Do not revert unrelated user changes.

## File Map

- Modify: `config/universe.yaml`
  - Restore or normalize a single source of truth for ticker universe shape.
- Modify: `data/pipeline.py`
  - Make config loading resilient, harden runtime fallbacks, and ensure the asset graph can execute with realistic empty-data paths.
- Modify: `council/aggregator.py`
  - Fix first-write behavior for orthogonality history and any assumptions about empty history frames.
- Modify: `api/main.py`
  - Keep admin auth/runtime behavior coherent in local/dev and production.
- Modify: `api/static/js/admin.js`
  - Ensure admin frontend handles auth/config/runtime failures explicitly instead of silently degrading.
- Modify: `dashboard/charts.py`
  - Finish chart hardening so Plotly never raises for supported input shapes.
- Modify: `docker-compose.yml`
  - Align env/config/dev defaults so the documented local stack actually boots into a usable state.
- Modify: `requirements.txt`
  - Keep runtime dependencies aligned with imports used by pipeline and frontends.
- Create/modify tests:
  - `tests/test_pipeline.py`
  - `tests/test_admin_ui.py`
  - `tests/test_dashboard_charts.py`
  - `tests/test_api_health.py`
  - `tests/test_api_pipeline.py`
  - `tests/test_api_config.py`
- Optional docs updates:
  - `README.md`
  - `docs/superpowers/plans/2026-04-03-system-stabilization.md` (checklist updates only during execution)

## Chunk 1: Re-establish Config and Dependency Contracts

### Task 1: Freeze the current failing baseline

**Files:**
- Test: `tests/test_pipeline.py`
- Test: `tests/test_api_config.py`

- [ ] **Step 1: Run the targeted failing baseline**

Run:
```bash
.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -q
```

Expected: existing failures around universe config loading and council aggregation history.

- [ ] **Step 2: Run API-config tests to capture local environment mismatches**

Run:
```bash
.\.venv\Scripts\python.exe -m pytest tests/test_api_config.py -q
```

Expected: if local env/import setup is still inconsistent, record the exact failure before changing code.

- [ ] **Step 3: Commit the baseline note**

```bash
git add docs/superpowers/plans/2026-04-03-system-stabilization.md
git commit -m "docs: capture stabilization plan baseline"
```

### Task 2: Make the universe schema explicit and backward-compatible

**Files:**
- Modify: `config/universe.yaml`
- Modify: `data/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing config-contract tests**

Add tests that assert `_load_universe()` accepts the actual YAML structure in `config/universe.yaml` and returns a flat ticker list.

Suggested test shape:
```python
def test_load_universe_flattens_large_and_mid_cap_buckets():
    tickers = _pipeline._load_universe()
    assert "AAPL" in tickers
    assert "SNOW" in tickers
```

- [ ] **Step 2: Run the new tests to verify they fail for the right reason**

Run:
```bash
.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -k load_universe -q
```

Expected: fail because `cfg["universe"]["tickers"]` does not exist.

- [ ] **Step 3: Implement the smallest contract fix**

Update [data/pipeline.py](E:/Github/MLCouncil/data/pipeline.py) so `_load_universe()`:
- accepts `universe.tickers` if present,
- otherwise flattens named buckets like `large_cap` and `mid_cap`,
- ignores `settings`,
- de-duplicates while preserving order.

Do not redesign the full config system yet.

- [ ] **Step 4: Decide and apply the config normalization**

Choose one of these and implement it consistently:
- keep bucketed YAML and make runtime flatten it, or
- add a canonical flat `tickers` list while preserving buckets for human editing.

The runtime and tests must agree on one contract.

- [ ] **Step 5: Re-run the targeted tests**

Run:
```bash
.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -k "load_universe or raw_ohlcv_asset_fails_on_empty_download" -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add config/universe.yaml data/pipeline.py tests/test_pipeline.py
git commit -m "fix: align universe config with pipeline loader"
```

### Task 3: Verify runtime dependencies match imports

**Files:**
- Modify: `requirements.txt`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write/keep a packaging regression test for imported runtime deps**

If not already covered well enough, add tests asserting core container-installed deps exist for pipeline import, at minimum `polars`.

- [ ] **Step 2: Run the packaging test**

Run:
```bash
.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -k requirements_include -q
```

- [ ] **Step 3: Adjust runtime dependency lists only if still missing**

Keep `requirements.txt` as the single source used by the Dockerfile for services that import pipeline code.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt tests/test_pipeline.py
git commit -m "chore: lock pipeline runtime dependencies"
```

## Chunk 2: Make the Dagster Pipeline Executable, Not Just Importable

### Task 4: Fix orthogonality history initialization

**Files:**
- Modify: `council/aggregator.py`
- Test: `tests/test_pipeline.py`
- Test: `tests/test_council.py`

- [ ] **Step 1: Write the failing regression test for first aggregate call**

Add a focused test covering `CouncilAggregator.aggregate()` with first-time signals and empty history.

Suggested shape:
```python
def test_aggregate_handles_first_signal_history_row():
    agg = CouncilAggregator(...)
    signals = {"lgbm": pd.Series(...), "sentiment": pd.Series(...)}
    result = agg.aggregate(signals, regime="bull", date=date(2024, 1, 15))
    assert not result.empty
```

- [ ] **Step 2: Run it to verify the current `ValueError`**

Run:
```bash
.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -k council_signal_aggregation -q
```

Expected: fail with `cannot set a frame with no defined columns`.

- [ ] **Step 3: Implement the minimal fix in `OrthogonalityMonitor.update_signals()`**

Initialize the per-model history frame with signal index columns on first write, then append the dated row without relying on empty-frame assignment behavior.

- [ ] **Step 4: Add one more guard test**

Cover differing ticker universes across models on first insertion so column alignment is explicit.

- [ ] **Step 5: Re-run targeted tests**

Run:
```bash
.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py tests/test_council.py -k "aggregate or council_signal" -q
```

- [ ] **Step 6: Commit**

```bash
git add council/aggregator.py tests/test_pipeline.py tests/test_council.py
git commit -m "fix: initialize council orthogonality history safely"
```

### Task 5: Make the full synthetic pipeline green

**Files:**
- Modify: `data/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Run the full pipeline test module**

Run:
```bash
.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -q
```

Expected: no failures after Tasks 2 and 4. If more failures appear, use them as the next source of truth.

- [ ] **Step 2: Add or refine failing tests for any newly exposed pipeline issue**

Do not bundle speculative fixes. Add one regression per newly surfaced failure.

- [ ] **Step 3: Fix only the surfaced issue**

Examples that are acceptable:
- empty checkpoint handling,
- empty covariance fallback,
- order serialization assumptions,
- partition-date filtering edge cases.

- [ ] **Step 4: Re-run the module until fully green**

Run:
```bash
.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -q
```

- [ ] **Step 5: Commit**

```bash
git add data/pipeline.py tests/test_pipeline.py
git commit -m "fix: stabilize synthetic dagster pipeline path"
```

### Task 6: Prove the Dagster code location works in the container

**Files:**
- Modify: `docker-compose.yml`
- Modify: `README.md`

- [ ] **Step 1: Run a fresh Dagster container import check**

Run:
```bash
docker compose build dagster
docker compose run --rm dagster python -c "import importlib.util, pathlib; p=pathlib.Path('/app/data/pipeline.py'); spec=importlib.util.spec_from_file_location('pipeline_check', p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('container import ok')"
```

Expected: `container import ok`.

- [ ] **Step 2: Run a real in-process job execution smoke test**

Run:
```bash
docker compose run --rm dagster python -c "from data.pipeline import defs; print(defs.get_job_def('daily_pipeline').name)"
```

Then, if feasible, run one partitioned execution with mocked or fixture-backed data path rather than live network IO.

- [ ] **Step 3: If container execution still needs env/config tweaks, fix them in Compose**

Likely targets:
- mounted config/data paths,
- explicit env vars,
- persistent `DAGSTER_HOME`,
- startup commands.

- [ ] **Step 4: Document the real supported local Dagster run path**

Update [README.md](E:/Github/MLCouncil/README.md) so commands match the working workflow.

- [ ] **Step 5: Commit**

```bash
git add docker-compose.yml README.md
git commit -m "chore: document and harden dagster runtime path"
```

## Chunk 3: Make Admin API and UI Fully Usable

### Task 7: Lock admin root and auth behavior

**Files:**
- Modify: `api/main.py`
- Test: `tests/test_admin_ui.py`
- Test: `tests/test_api_config.py`
- Test: `tests/test_api_pipeline.py`

- [ ] **Step 1: Keep the root-rendering regression tests in place**

Ensure the test suite covers:
- `GET /` returns `200`,
- private API endpoints remain accessible in local/dev when no API key is configured,
- authenticated behavior still works when a key is configured.

- [ ] **Step 2: Add the missing auth-enforcement tests**

Suggested tests:
```python
def test_private_api_requires_key_when_configured(): ...
def test_private_api_accepts_valid_key_when_configured(): ...
```

- [ ] **Step 3: Run those tests red**

Run:
```bash
.\.venv\Scripts\python.exe -m pytest tests/test_admin_ui.py tests/test_api_config.py tests/test_api_pipeline.py -q
```

- [ ] **Step 4: Implement the smallest coherent auth policy**

Policy should be explicit:
- local/dev without `MLCOUNCIL_API_KEY`: UI and API work,
- with `MLCOUNCIL_API_KEY`: private API requires `X-API-Key`,
- docs and health endpoints stay readable.

- [ ] **Step 5: Re-run targeted tests**

Run:
```bash
.\.venv\Scripts\python.exe -m pytest tests/test_admin_ui.py tests/test_api_config.py tests/test_api_pipeline.py -q
```

- [ ] **Step 6: Commit**

```bash
git add api/main.py tests/test_admin_ui.py tests/test_api_config.py tests/test_api_pipeline.py
git commit -m "fix: stabilize admin auth and root rendering"
```

### Task 8: Make admin frontend error handling honest

**Files:**
- Modify: `api/static/js/admin.js`
- Modify: `api/templates/admin.html`
- Modify: `api/static/css/admin.css`
- Test: `tests/test_admin_ui.py`

- [ ] **Step 1: Write a failing UI-contract test or DOM smoke check**

At minimum, add a test that the root HTML includes an area for auth/config/runtime status.

- [ ] **Step 2: Implement explicit frontend states**

The admin JS should:
- render actual API data when available,
- show a visible auth/config warning instead of silent console-only failure,
- avoid crashing the whole page when one section fails,
- optionally support an API key input stored in session/local storage if production-private admin use is required from the browser.

- [ ] **Step 3: Re-run browser verification**

Run:
```bash
docker compose up -d --build admin-api
npx --yes --package @playwright/cli playwright-cli -s=admin-check open http://127.0.0.1:8000/
npx --yes --package @playwright/cli playwright-cli -s=admin-check snapshot
npx --yes --package @playwright/cli playwright-cli -s=admin-check console error
npx --yes --package @playwright/cli playwright-cli -s=admin-check network
```

Expected:
- no application 500s,
- no 401s in local/dev without configured key,
- only ignorable noise like missing favicon if not yet fixed.

- [ ] **Step 4: Commit**

```bash
git add api/static/js/admin.js api/templates/admin.html api/static/css/admin.css tests/test_admin_ui.py
git commit -m "fix: make admin ui resilient to runtime and auth states"
```

## Chunk 4: Make the Streamlit Dashboard Error-Free

### Task 9: Lock chart builders against invalid Plotly input

**Files:**
- Modify: `dashboard/charts.py`
- Test: `tests/test_dashboard_charts.py`

- [ ] **Step 1: Expand the chart regression tests**

Cover:
- valid `fillcolor` generation,
- empty dataframes,
- mixed color formats (`#hex`, `rgb(...)`, `rgba(...)`),
- any chart that currently depends on implicit input shape assumptions.

- [ ] **Step 2: Run the chart tests**

Run:
```bash
.\.venv\Scripts\python.exe -m pytest tests/test_dashboard_charts.py -q
```

- [ ] **Step 3: Harden only failing chart builders**

Likely targets:
- `weight_evolution_chart`,
- any radar/timeline chart using string manipulation for colors,
- date columns with wrong dtypes.

- [ ] **Step 4: Re-run the test file**

Run:
```bash
.\.venv\Scripts\python.exe -m pytest tests/test_dashboard_charts.py -q
```

- [ ] **Step 5: Commit**

```bash
git add dashboard/charts.py tests/test_dashboard_charts.py
git commit -m "fix: harden dashboard chart builders"
```

### Task 10: Verify all Streamlit tabs render without tracebacks

**Files:**
- Modify: `dashboard/app.py`
- Modify: `dashboard/data_loader.py`
- Test: `tests/test_dashboard_charts.py`

- [ ] **Step 1: Run browser verification against the dashboard**

Run:
```bash
docker compose up -d --build dashboard
npx --yes --package @playwright/cli playwright-cli -s=dashboard-check open http://127.0.0.1:8501/
npx --yes --package @playwright/cli playwright-cli -s=dashboard-check snapshot
npx --yes --package @playwright/cli playwright-cli -s=dashboard-check console error
```

- [ ] **Step 2: Click through all tabs and capture each failure**

Run:
```bash
npx --yes --package @playwright/cli playwright-cli -s=dashboard-check click e169
npx --yes --package @playwright/cli playwright-cli -s=dashboard-check snapshot
npx --yes --package @playwright/cli playwright-cli -s=dashboard-check click e172
npx --yes --package @playwright/cli playwright-cli -s=dashboard-check snapshot
```

Use fresh refs from the actual snapshot if they differ.

- [ ] **Step 3: Add regression tests for any newly exposed dashboard failure**

Do not just patch visuals blindly. Reproduce the underlying bad input in tests first.

- [ ] **Step 4: Implement the minimal data-loader or rendering fixes**

Typical fixes:
- normalize date columns,
- protect against empty metrics,
- provide stable fallback frames,
- avoid assumptions about attribution history existence.

- [ ] **Step 5: Re-run Playwright verification**

Expected:
- no traceback blocks,
- no console errors,
- all three tabs render useful content or graceful empty states.

- [ ] **Step 6: Commit**

```bash
git add dashboard/app.py dashboard/data_loader.py tests/test_dashboard_charts.py
git commit -m "fix: stabilize streamlit dashboard rendering"
```

## Chunk 5: End-to-End Hardening and Release Readiness

### Task 11: Make the local Docker stack usable from a clean checkout

**Files:**
- Modify: `docker-compose.yml`
- Modify: `Dockerfile`
- Modify: `README.md`

- [ ] **Step 1: Identify the minimum env vars for local success**

Document which services require:
- none,
- mock/fallback mode,
- real secrets.

- [ ] **Step 2: Change Compose defaults only where it reduces breakage**

Examples:
- remove obsolete `version`,
- add sane dev env defaults,
- persist `DAGSTER_HOME`,
- make admin and dashboard boot consistently without hidden host-only assumptions.

- [ ] **Step 3: Verify from clean rebuild**

Run:
```bash
docker compose down
docker compose up -d --build
docker compose ps
```

Expected: all declared services healthy enough to serve their pages without immediate app errors.

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml Dockerfile README.md
git commit -m "chore: make local docker stack reproducible"
```

### Task 12: Run the full verification matrix

**Files:**
- Modify: `README.md` only if commands differ from reality

- [ ] **Step 1: Run the core pytest matrix**

Run:
```bash
.\.venv\Scripts\python.exe -m pytest tests/test_admin_ui.py tests/test_dashboard_charts.py tests/test_api_*.py tests/test_pipeline.py -q
```

- [ ] **Step 2: Run the broader suite if the core matrix is green**

Run:
```bash
.\.venv\Scripts\python.exe -m pytest -q
```

- [ ] **Step 3: Run browser verification for all main surfaces**

Verify:
- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/api/docs`
- `http://127.0.0.1:8501/`
- `http://127.0.0.1:3000/`

Required evidence:
- snapshots,
- console output,
- network status for critical requests.

- [ ] **Step 4: Run a real Dagster pipeline smoke path**

Minimum requirement:
- import code location,
- resolve `defs`,
- inspect job,
- execute a safe synthetic or fixture-backed partition path without uncaught exceptions.

- [ ] **Step 5: Update README with only verified commands**

Remove or rewrite any command that still relies on broken assumptions.

- [ ] **Step 6: Final commit**

```bash
git add README.md
git commit -m "docs: align runbook with verified system behavior"
```

## Exit Criteria

- `pytest tests/test_pipeline.py -q` is green.
- Admin root `/` returns `200` and its data panels do not generate `401` or `500` in local/dev.
- Streamlit dashboard renders all tabs without traceback or console errors.
- Dagster UI loads and the code location/pipeline job resolve successfully in containerized runtime.
- Docker Compose boots a usable local stack from a clean rebuild.
- README commands match the verified workflow.

Plan complete and saved to `docs/superpowers/plans/2026-04-03-system-stabilization.md`. Ready to execute?
