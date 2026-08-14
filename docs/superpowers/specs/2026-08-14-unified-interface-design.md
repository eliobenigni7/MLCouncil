# Unified Interface Design — one app for experiments and technical operations

- **Date**: 2026-08-14
- **Status**: Draft (in spec review)
- **Scope**: UX/UI consolidation of all MLCouncil user surfaces into a single React SPA served by the existing FastAPI admin API.

## 1. Context

Today there are four separate user surfaces with different frameworks, auth models, and capabilities:

| Surface | Port | Stack | Capabilities | Gaps |
|---|---|---|---|---|
| Streamlit dashboard | 8501 | Streamlit, ~2.7k LOC | Read-only analytics: Performance / Attribution / Regime tabs; Backtest Playground (in-process job runner); Challenger Promotion; Fill Quality | No actions, no auth |
| Admin API | 8000 | FastAPI + vanilla JS SPA, ~4.5k LOC (1.1k HTML/JS/CSS) | Only write surface: Pipeline Control, Config, Monitoring, Trading, Portfolio, Intraday | No experiments, no canary |
| Dagster UI | 3000 | Dagster | Asset orchestration, checks, schedules | Separate surface |
| Grafana + MLflow | observability / 5000 | — | Monitoring dashboards, experiment tracking | Separate surfaces |

Structural problems this design fixes:

1. Experiments (playground, promotion, shadow gates) and technical control (pipeline, config, trading) live in **two different frameworks** and cannot share state or navigation.
2. **Canary has no UI anywhere** — state and flag registry are only inspectable via files/API.
3. Model promotion is only visible in Streamlit; admin has no promotion page.
4. Analytics is unauthenticated by design (public-safe metrics), trading is authenticated — inconsistent trust model.
5. Duplicated charting and data-loading code across two stacks (`dashboard/charts.py` vs `api/static/js/admin.js`).

## 2. Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | **One app for everything**: analytics + experiments + control + trading + canary + promotion. Dagster / Grafana / MLflow reachable via direct links from the sidebar (new tab), not iframes (Grafana would need an auth proxy; iframes are a UX compromise). | Single navigation, single trust model, no duplicated state. |
| D2 | **React 18 + Vite + TypeScript SPA, served by FastAPI on :8000** (same origin). No Next.js: SSR provides nothing for an authenticated internal admin tool; serving from FastAPI keeps one container, kills CORS, and lets session cookies work natively. | Same-origin removes the API-key-in-browser flow and CORS config. |
| D3 | **Single admin login**: `POST /api/auth/login` with credentials from env (`MLCOUNCIL_ADMIN_USERNAME` / `MLCOUNCIL_ADMIN_PASSWORD`), server-side session (HttpOnly cookie) + CSRF token. The existing `X-API-Key` header auth stays for external scripts/automation and for API-only callers; browser sessions use cookies. | Consistent trust model; API key stops circulating in the browser. |
| D4 | **Big-bang migration**: the SPA replaces both Streamlit and the vanilla-JS admin at release. Streamlit service is removed from docker-compose; `dashboard/` code is retired after migration (charts ported to TS, data-loading logic moves into an API service). | User chose big-bang over incremental; no iframe interim. |
| D5 | **Backend extension** (FastAPI stays the single backend): new routers `auth`, `analytics`, `experiments`, `canary`, `promotion`; new `api/services/experiment_service.py` (job queue); analytics read logic moves from `dashboard/data_loader.py` into an API service so the SPA never touches files. | SPA consumes only REST; artifact reads stay server-side where tests exist. |
| D6 | **IA**: five sidebar groups — Overview, Analytics, Experiments, Operations, System — plus a Links group. **UI language: English.** | Grouped by workflow; English matches admin today and codebase conventions. |
| D7 | **Charts**: plotly.js client-side; `dashboard/charts.py` (781 LOC) ported to TypeScript chart components reusing the same layouts/config. | Reuses proven visual design without server-side image generation. |
| D8 | **Error contract**: standardized envelope `{error: {code, message, detail}}` on **all new routers** (auth, analytics, experiments, canary, promotion); 401 → redirect to login; 4xx → toast; 5xx → toast + existing monitoring alert channel. Existing routers (pipeline, portfolio, config, monitoring, trading, intraday) keep their current `HTTPException` contract unchanged — external API-key consumers depend on it. The SPA maps both shapes to the same client error model. | Uniform client handling without breaking the external API contract. |
| D9 | **Testing**: pytest for new routers; Vitest + React Testing Library + MSW for the SPA; Playwright smoke (login → all sections → one backtest job end-to-end); parity checklist vs current UIs; CI runs frontend build + ESLint + pytest. Walk-forward CI unchanged. | Evidence-backed migration; both stacks keep regression coverage. |

## 3. Target architecture

```
Browser (React SPA, served by FastAPI :8000, same origin)
        │  session cookie + CSRF header
        ▼
FastAPI :8000  ── routers ────────────────────────────────────
  auth  analytics  experiments  canary  promotion            │
  pipeline  portfolio  config  monitoring  trading  intraday  │
        │ services: experiment_service (subprocess runner +    │
        │ job registry on data/results/experiments/),          │
        │ analytics_service (artifact reads),                │
        │ trading_service, dagster_client, ...               │
        ▼
Artifacts: data/results/*, data/orders/*, models/checkpoints/*, config/*
Dagster :3000 (GraphQL) · Alpaca (paper) · MLflow :5000
```

Docker: single new build stage `frontend` (Node → `npm run build`) copied into the existing image; `admin-api` serves `/app` static with a catch-all to `index.html` for client-side routing. The `dashboard` service (8501) is deleted from docker-compose at release. Streamlit config `.streamlit/`, `dashboard/` are removed.

### 3.1 Frontend stack

- React 18 + TypeScript + Vite (build output served by FastAPI).
- React Router (client routes mirroring sidebar groups).
- TanStack Query: cache GETs, invalidate on mutations; no retry on 4xx, limited retry on 5xx/network; per-section error boundaries.
- plotly.js via a typed wrapper (ported chart builders).
- No Redux: TanStack Query + an auth context is enough for a single-user tool.

### 3.2 Navigation (IA)

Sidebar, English UI, desktop-first, dark theme (evolved from `api/static/css/admin.css`):

| Group | Sections | Sources |
|---|---|---|
| Overview | KPI, pipeline health, alert banner, canary summary | admin System Overview + Streamlit sidebar + canary (new) |
| Analytics | Performance · Attribution · Regime | Streamlit tabs |
| Experiments | Backtest · Promotion · Canary | Playground + Challenger Promotion + canary (new) |
| Operations | Pipeline · Trading · Intraday · Portfolio | admin pages |
| System | Configuration · Monitoring · Fill Quality | admin Config/Monitoring + Streamlit Fill Quality |
| Links | Dagster · Grafana · MLflow (new tab) | sidebar today |

Shared components: chart components (plotly.ts), DataTable, status badge, alert banner, confirm dialog (inherits admin.js destructive-action confirmations: execute, liquidate, run pipeline), backtest run form.

### 3.3 New/changed backend surface

New routers (all under `/api`):

- `auth.py` — `POST /api/auth/login`, `POST /api/auth/logout`, `GET /api/auth/me`. Session cookie (HttpOnly, SameSite=Lax, Secure in prod) + **double-submit CSRF cookie** (token set in a readable cookie and echoed in a header by the SPA; rotated on login; login rate-limited via existing slowapi limiter). API-key middleware accepts either valid session **or** valid API key on `/api/*`; API-key requests are exempt from CSRF (header auth is not browser-embeddable). Threat model: SameSite=Lax already blocks most CSRF for a single-user tool; the double-submit token is defense in depth, kept minimal.
- `analytics.py` — JSON endpoints exposing exactly the data `dashboard/data_loader.py` reads today: equity curve (`backtest_result.pkl`/parquet/paper snapshots), benchmark (SPY), attribution (`aggregator.pkl`), regime, fills, cost-calibration artifact. Read logic moves into `api/services/analytics_service.py` (port, do not recompute; same numbers from same files). **Keep the equity-to-100 normalization in the port** (`data_loader.py:194` applies it) so chart values are byte-identical to today's Streamlit — parity item 3 depends on it.
- `experiments.py` — `POST /api/experiments/backtest` (enqueue), `GET /api/experiments/jobs`, `GET /api/experiments/jobs/{id}/status`, `GET /api/experiments/jobs/{id}/result`, `POST /api/experiments/jobs/{id}/cancel`, `GET /api/experiments/snapshots`, `GET /api/experiments/snapshots/{id}` (overlay data). Runner ported from `backtest.playground` into `api/services/experiment_service.py`: **subprocess-based runner** (today's `run_playground_backtest` is synchronous with no cancellation hooks — we design fresh): each job spawns `python -m api.services.experiment_worker` (inherits the API process env; per-job args passed on argv), giving real cancel (terminate) and clean restart semantics; `max_workers=1` (single-user tool, no CPU contention with trading/monitoring endpoints). Job registry persisted as `data/results/experiments/{job_id}.json` (state: queued/running/succeeded/cancelled/failed + traceback excerpt); on API boot a **sweep marks stale `running` → `failed`** (orphaned after restart); retention keeps the last 50 registry entries and **prunes on enqueue, not just boot** — snapshot artifacts older than the oldest retained job are deleted so `GET /api/experiments/snapshots` never lists evicted runs. Client polls status every 2 s while running.
- `canary.py` — `GET /api/canary/state` (controller state + `data/results/canary_state.json`), `GET /api/canary/flags` (registry from `docs/flag-registry-2026-08-13.md` / config), `POST /api/canary/apply`. Note: `CanaryController.apply()` mutates `os.environ` of the calling process, which does **not** propagate to Dagster asset processes — so the endpoint must not pretend to toggle live pipeline env. Instead, `apply` **persists the pending flag set to `data/results/canary_state.json`** (the file `apply_canary_features(state_path=...)` already reads at run start) and reports what will be applied on the next run; reverts go through the same file. A read-only preview (`GET /api/canary/apply/preview`) returns what would change without writing. The pending-apply field is **additive** to `CanaryState` (which today holds revert-event data: floor/last_value/date) and is ignored by `check_revert`; `config/canary.yaml` remains the source of truth — the state file is a pending overlay on top of it, and the preview always computes against `canary.yaml` + overlay.
- `promotion.py` — `GET /api/promotion/manifest` (`config/production_manifest.yaml`), `GET /api/promotion/reports` (walk-forward reports for lightgbm/sentiment/hmm/tft), `GET /api/promotion/shadow-artifacts` (existence + freshness of shadow outputs).

Unchanged routers: `pipeline`, `portfolio`, `config`, `monitoring`, `trading`, `intraday` (existing endpoints reused as-is by the SPA).

## 4. Data flow

- GET-heavy analytics: TanStack Query cache, refetch on focus/staleness.
- Mutations (`run pipeline`, `execute orders`, settings PUT, backtest enqueue) invalidate related queries (e.g., after execute → positions/orders/portfolio refetch).
- Backtest: enqueue → poll `status` (2 s) → fetch `result` (snapshot list + overlay). Cancel posts to `jobs/{id}/cancel` (subprocess terminate; state → `cancelled`). Failure surfaces the traceback excerpt in the job card with a retry button.
- Trading: preflight before execute (unchanged semantics).
- Alerts banner: `load_current_alerts` at session start + periodic poll of existing `/api/monitoring/alerts`.

## 5. Security

- Session auth: credentials from env only; constant-time compare (pattern from `api/auth.py`); sessions server-side, HttpOnly cookie; CSRF token required for state-changing requests; rate limiting (existing slowapi limiter) extended to login.
- `MLCOUNCIL_API_KEY` remains for external automation (unchanged middleware behavior when no session present).
- Startup guard (existing `validate_environment`) extended: refuse boot without admin credentials when profile requires auth.

## 6. Migration & parity

Release gate checklist — every item below must exist in the SPA before Streamlit/admin are retired:

1. All Streamlit tabs/pages: Performance, Attribution, Regime, Fill Quality, Challenger Promotion, Backtest Playground (incl. snapshot list + overlay).
2. All admin pages: Overview, Pipeline Control, Portfolio, Configuration, Monitoring, Trading; plus Intraday (API exists, no UI today) and Canary (new UI).
3. Numeric parity, automated: pytest feeds the same artifacts through `analytics_service` and `dashboard/data_loader.py` and asserts numeric equality (incl. the equity-to-100 normalization); plus a manual side-by-side visual spot-check of every chart.
4. Destructive actions (execute, liquidate, run pipeline, settings PUT) keep confirm dialogs and preflight checks.
5. Links: Dagster, Grafana, MLflow.
6. docker-compose: `dashboard` service removed; `admin-api` serves the SPA; `.env` gains `MLCOUNCIL_ADMIN_USERNAME` / `MLCOUNCIL_ADMIN_PASSWORD` (passed through to the `admin-api` service).
7. `dashboard/` directory retired; `.streamlit/` removed; AGENTS.md updated (dashboard command, ports).
8. **Milestone before retirement**: the SPA runs alongside the old UIs during development (no shippable-gap: old UIs stay live until the checklist is green). Coexistence on :8000: the legacy vanilla-JS admin moves to the `/admin` prefix while the SPA's catch-all serves the root and explicitly excludes `/admin` (or, alternatively, the legacy UI is served only when `MLCOUNCIL_LEGACY_UI=true`). Retirement of Streamlit and the legacy admin SPA happens in the same release only after items 1–7 pass.
9. **Rollback path**: pin the previous image tag / `git revert` of the retirement commit + `docker compose` restore of the `dashboard` service; session-auth is additive, so reverting to API-key auth requires no data migration.

## 7. Out of scope (YAGNI)

- Multi-user roles / public read-only view.
- Mobile/responsive beyond graceful desktop scaling.
- Iframe embedding of Dagster/Grafana.
- Rewriting backend services not needed by the SPA.
- Real-time push (SSE/WebSockets); polling is sufficient at this scale.

## 8. Risks

- **Big-bang risk**: long period with nothing shippable → mitigation: keep both old UIs running until the parity checklist passes; ship SPA and retire old surfaces in the same release.
- **Job runner move** (Streamlit process → API service): no cancellation or concurrency semantics exist today to reproduce (playground is synchronous) — designed fresh as subprocess runner with cancel endpoint, boot sweep for stale `running` jobs, and registry persistence; `max_workers=1` avoids CPU contention with trading/monitoring.
- **Chart parity**: plotly.js config differences vs Python plotly must be checked panel-by-panel (parity checklist item 3).
- **Session vs API-key auth coexistence**: middleware must accept either; regression tests for both paths.
