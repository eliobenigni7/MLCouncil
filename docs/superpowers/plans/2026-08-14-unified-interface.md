# Unified Interface Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Streamlit dashboard (:8501) and the vanilla-JS admin SPA with a single React SPA served by FastAPI (:8000), adding session auth and new backend routers (auth, analytics, experiments, canary, promotion).

**Architecture:** FastAPI stays the single backend; the SPA is same-origin on :8000 (no CORS, session cookies). New routers expose analytics artifacts (ported from `dashboard/data_loader.py`), a subprocess-based backtest job runner (ported from `backtest/playground.py`), canary overlay state, and promotion reports. Legacy admin moves to `/admin` behind `MLCOUNCIL_LEGACY_UI` until parity passes, then everything old is retired.

**Tech Stack:** Python 3.10 FastAPI (existing) · React 18 + TypeScript + Vite + TanStack Query + plotly.js · Vitest + MSW (frontend tests) · Playwright (smoke) · pytest (backend) · Docker multi-stage build.

**Spec:** `docs/superpowers/specs/2026-08-14-unified-interface-design.md`

---

## File Structure

**New backend:**
- `api/errors.py` — `ApiError` + envelope handler
- `api/session.py` — in-memory session store, cookie/CSRF helpers
- `api/routers/auth.py` — login/logout/me/csrf
- `api/services/analytics_service.py` — port of `dashboard/data_loader.py` (13 loaders)
- `api/routers/analytics.py`
- `api/services/experiment_service.py` — subprocess job runner + registry
- `api/services/experiment_worker.py` — `python -m` entry point per job
- `api/routers/experiments.py`
- `api/services/canary_service.py`, `api/routers/canary.py`
- `api/services/promotion_service.py`, `api/routers/promotion.py`

**Modified backend:**
- `api/auth.py` — session-aware auth gate
- `api/main.py` — auth router, session/CSRF middleware, SPA static + `/admin` legacy route, startup guard
- `council/canary.py` — additive `pending_apply` overlay on `CanaryState`, honored by `_active_features()`

**Frontend (new `frontend/`):**
- `package.json`, `vite.config.ts`, `tsconfig.json`, `index.html`, `.eslintrc.cjs`, `.gitignore`
- `src/main.tsx`, `src/App.tsx` (router+layout), `src/styles/theme.css`
- `src/api/client.ts`, `src/api/queries.ts`
- `src/auth/AuthContext.tsx`, `src/components/ProtectedRoute.tsx`
- `src/components/layout/Sidebar.tsx`, `src/components/{DataTable,StatusBadge,AlertBanner,KpiCard,ConfirmDialog}.tsx`
- `src/features/analytics/charts.ts` (port of `dashboard/charts.py`)
- `src/features/experiments/jobs.ts`
- `src/pages/{Overview,Performance,Attribution,Regime,Backtest,Promotion,Canary,Pipeline,Trading,Intraday,Portfolio,Config,Monitoring,FillQuality}Page.tsx`, `src/pages/LoginPage.tsx`
- `src/test/setup.ts`, `src/test/server.ts`
- `e2e/smoke.spec.ts`

**Tests:**
- `tests/test_api_auth.py`, `tests/test_api_analytics.py`, `tests/test_analytics_parity.py`, `tests/test_api_experiments.py`, `tests/test_api_canary.py`, `tests/test_api_promotion.py`, `tests/test_canary_pending.py`
- `frontend/src/**/*.test.ts(x)`, `frontend/e2e/smoke.spec.ts`

**Infra:** `Dockerfile` (multi-stage node), `docker-compose.yml`, `.env.example`, `.github/workflows/ci.yml`, `AGENTS.md`

---

## Chunk 1: Backend auth (sessions + CSRF)

### Task 1: `api/errors.py` — error envelope

**Files:**
- Create: `api/errors.py`
- Test: `tests/test_api_auth.py` (uses it)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_auth.py
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.errors import ApiError, api_error_handler


def test_api_error_envelope():
    app = FastAPI()
    app.add_exception_handler(ApiError, api_error_handler)

    @app.get("/boom")
    def boom():
        raise ApiError(404, "artifact_not_found", "Missing artifact", "data/results/equity_curve.parquet")

    client = TestClient(app)
    resp = client.get("/boom")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "artifact_not_found"
    assert body["error"]["message"] == "Missing artifact"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_auth.py::test_api_error_envelope -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.errors'`

- [ ] **Step 3: Write minimal implementation**

```python
# api/errors.py
from __future__ import annotations

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


class ApiError(HTTPException):
    """Errore applicativo con envelope JSON: {error: {code, message, detail}}."""

    def __init__(self, status_code: int, code: str, message: str, detail: str = ""):
        super().__init__(status_code=status_code, detail=detail)
        self.code = code
        self.message = message
        self.error_detail = detail


def api_error_handler(_request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message, "detail": exc.error_detail}},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_api_auth.py::test_api_error_envelope -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api/errors.py tests/test_api_auth.py
git commit -m "feat(api): ApiError envelope for new routers"
```

### Task 2: `api/session.py` — session store + CSRF helpers

**Files:**
- Create: `api/session.py`
- Test: `tests/test_api_auth.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_api_auth.py
from datetime import datetime, timedelta, timezone

from api.session import (SESSION_COOKIE, check_csrf, create_session,
                         destroy_session, get_session, is_session_valid,
                         new_csrf_token)


def test_session_lifecycle():
    token = create_session()
    assert is_session_valid(token)
    assert get_session(token) is not None
    destroy_session(token)
    assert not is_session_valid(token)


def test_session_http_cookie():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient

    from api.session import set_session_cookie

    app = FastAPI()

    @app.get("/login")
    def login():
        resp = JSONResponse({"ok": True})
        set_session_cookie(resp, create_session())
        return resp

    client = TestClient(app)
    resp = client.get("/login")
    assert SESSION_COOKIE in resp.headers["set-cookie"]
    assert "HttpOnly" in resp.headers["set-cookie"]


def test_csrf_double_submit():
    token = new_csrf_token()
    assert check_csrf(token, token)
    assert not check_csrf(token, "wrong")


def test_expired_session_rejected():
    token = create_session()
    import api.session as s
    s._SESSIONS[token] = datetime.now(timezone.utc) - timedelta(hours=25)
    assert not is_session_valid(token)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_auth.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.session'`

- [ ] **Step 3: Write minimal implementation**

```python
# api/session.py
from __future__ import annotations

import os
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import Request
from fastapi.responses import Response

SESSION_COOKIE = "mlcouncil_session"
CSRF_COOKIE = "mlcouncil_csrf"
SESSION_TTL_HOURS = 24

_SESSIONS: dict[str, datetime] = {}
_CSRF_TOKENS: dict[str, datetime] = {}


def _is_prod() -> bool:
    return os.getenv("MLCOUNCIL_ENV_PROFILE", "local").strip().lower() in {"prod", "paper"}


def create_session() -> str:
    token = secrets.token_urlsafe(32)
    _SESSIONS[token] = datetime.now(timezone.utc) + timedelta(hours=SESSION_TTL_HOURS)
    return token


def is_session_valid(token: str | None) -> bool:
    if not token:
        return False
    expiry = _SESSIONS.get(token)
    if expiry is None:
        return False
    if datetime.now(timezone.utc) > expiry:
        _SESSIONS.pop(token, None)
        return False
    return True


def get_session(token: str | None) -> str | None:
    return token if is_session_valid(token) else None


def destroy_session(token: str | None) -> None:
    if token:
        _SESSIONS.pop(token, None)


def set_session_cookie(resp: Response, token: str) -> None:
    resp.set_cookie(
        SESSION_COOKIE, token,
        max_age=SESSION_TTL_HOURS * 3600, httponly=True,
        samesite="lax", secure=_is_prod(),
    )


def clear_session_cookie(resp: Response) -> None:
    resp.delete_cookie(SESSION_COOKIE, samesite="lax", secure=_is_prod())


def new_csrf_token() -> str:
    token = secrets.token_urlsafe(32)
    _CSRF_TOKENS[token] = datetime.now(timezone.utc) + timedelta(hours=12)
    return token


def check_csrf(token: str | None, submitted: str | None) -> bool:
    if not token or not submitted:
        return False
    expiry = _CSRF_TOKENS.get(token)
    if expiry is None or datetime.now(timezone.utc) > expiry:
        _CSRF_TOKENS.pop(token, None)
        return False
    return secrets.compare_digest(token, submitted)


def request_session_token(request: Request) -> str | None:
    return request.cookies.get(SESSION_COOKIE)


def request_csrf_token(request: Request) -> str | None:
    return request.cookies.get(CSRF_COOKIE)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_api_auth.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api/session.py tests/test_api_auth.py
git commit -m "feat(api): in-memory sessions and CSRF double-submit helpers"
```

### Task 3: `api/routers/auth.py` — login/logout/me/csrf

**Files:**
- Create: `api/routers/auth.py`
- Test: `tests/test_api_auth.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_api_auth.py
import os
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.errors import ApiError, api_error_handler
from api.routers import auth


def _auth_app():
    app = FastAPI()
    app.add_exception_handler(ApiError, api_error_handler)
    app.include_router(auth.router, prefix="/api")
    return app


def test_login_success_sets_cookies():
    with patch.dict(os.environ, {"MLCOUNCIL_ADMIN_USERNAME": "admin", "MLCOUNCIL_ADMIN_PASSWORD": "s3cret"}, clear=False):
        client = TestClient(_auth_app())
        resp = client.post("/api/auth/login", json={"username": "admin", "password": "s3cret"})
        assert resp.status_code == 200
        assert "mlcouncil_session" in resp.cookies
        assert "mlcouncil_csrf" in resp.cookies
        assert resp.json()["authenticated"] is True


def test_login_wrong_password_rejected():
    with patch.dict(os.environ, {"MLCOUNCIL_ADMIN_USERNAME": "admin", "MLCOUNCIL_ADMIN_PASSWORD": "s3cret"}, clear=False):
        client = TestClient(_auth_app())
        resp = client.post("/api/auth/login", json={"username": "admin", "password": "nope"})
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "invalid_credentials"


def test_login_unconfigured_returns_503():
    with patch.dict(os.environ, {"MLCOUNCIL_ADMIN_USERNAME": "", "MLCOUNCIL_ADMIN_PASSWORD": ""}, clear=False):
        client = TestClient(_auth_app())
        resp = client.post("/api/auth/login", json={"username": "a", "password": "b"})
        assert resp.status_code == 503


def test_me_requires_session():
    with patch.dict(os.environ, {"MLCOUNCIL_ADMIN_USERNAME": "admin", "MLCOUNCIL_ADMIN_PASSWORD": "s3cret"}, clear=False):
        client = TestClient(_auth_app())
        assert client.get("/api/auth/me").status_code == 401
        login = client.post("/api/auth/login", json={"username": "admin", "password": "s3cret"})
        assert login.status_code == 200
        me = client.get("/api/auth/me")
        assert me.status_code == 200
        assert me.json()["username"] == "admin"


def test_logout_clears_session():
    with patch.dict(os.environ, {"MLCOUNCIL_ADMIN_USERNAME": "admin", "MLCOUNCIL_ADMIN_PASSWORD": "s3cret"}, clear=False):
        client = TestClient(_auth_app())
        client.post("/api/auth/login", json={"username": "admin", "password": "s3cret"})
        assert client.post("/api/auth/logout").status_code == 200
        assert client.get("/api/auth/me").status_code == 401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_auth.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.routers.auth'`

- [ ] **Step 3: Write minimal implementation**

```python
# api/routers/auth.py
from __future__ import annotations

import os
import secrets

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

from api.errors import ApiError
from api.session import (CSRF_COOKIE, clear_session_cookie, create_session,
                         destroy_session, is_session_valid, new_csrf_token,
                         request_session_token, set_session_cookie)

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


def _admin_credentials() -> tuple[str, str]:
    return os.getenv("MLCOUNCIL_ADMIN_USERNAME", ""), os.getenv("MLCOUNCIL_ADMIN_PASSWORD", "")


@router.post("/login")
def login(request: Request, body: LoginRequest, response: Response):
    expected_user, expected_pass = _admin_credentials()
    if not expected_user or not expected_pass:
        raise ApiError(503, "auth_not_configured", "Admin credentials not configured")
    user_ok = secrets.compare_digest(body.username, expected_user)
    pass_ok = secrets.compare_digest(body.password, expected_pass)
    if not (user_ok and pass_ok):
        raise ApiError(401, "invalid_credentials", "Invalid username or password")
    token = create_session()
    set_session_cookie(response, token)
    response.set_cookie(
        CSRF_COOKIE, new_csrf_token(), max_age=12 * 3600,
        httponly=False, samesite="lax", secure=False,
    )
    return {"authenticated": True, "username": body.username}


@router.post("/logout")
def logout(request: Request, response: Response):
    destroy_session(request_session_token(request))
    clear_session_cookie(response)
    response.delete_cookie(CSRF_COOKIE)
    return {"authenticated": False}


@router.get("/me")
def me(request: Request):
    token = request_session_token(request)
    if not token or not is_session_valid(token):
        raise ApiError(401, "not_authenticated", "Not logged in")
    return {"authenticated": True, "username": os.getenv("MLCOUNCIL_ADMIN_USERNAME", "")}
```

Rate limiting: login is a brute-force target; if the existing `api.rate_limit.limiter` decorator works under the test slowapi stub, add `@limiter.limit("5/minute")` to `login` (verify against `tests/conftest.py` stub; skip if the stub does not support it — note it in a comment).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_api_auth.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api/routers/auth.py tests/test_api_auth.py
git commit -m "feat(api): login/logout/me endpoints with session cookies"
```

### Task 4: `api/auth.py` — session-aware auth gate

**Files:**
- Modify: `api/auth.py`
- Test: `tests/test_api_auth.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_api_auth.py
from unittest.mock import patch

from api.auth import is_api_key_required


def test_api_key_required_semantics():
    with patch.dict(os.environ, {"MLCOUNCIL_REQUIRE_API_KEY": "true"}, clear=False):
        assert is_api_key_required()
    with patch.dict(os.environ, {"MLCOUNCIL_ENV_PROFILE": "paper", "MLCOUNCIL_REQUIRE_API_KEY": "false"}, clear=False):
        assert is_api_key_required()
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest tests/test_api_auth.py::test_api_key_required_semantics -v`
Expected: PASS (pins the existing contract; the real change is Task 5's middleware)

- [ ] **Step 3: Append the session-aware gate to `api/auth.py`**

```python
# api/auth.py — append
def request_is_authenticated(request: Request) -> bool:
    """True if a valid session cookie exists OR a valid API key header."""
    from api.session import is_session_valid, request_session_token
    token = request_session_token(request)
    if token and is_session_valid(token):
        return True
    try:
        ensure_request_api_key(request)
        return True
    except HTTPException:
        return False
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_api_auth.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api/auth.py tests/test_api_auth.py
git commit -m "feat(api): session-aware auth gate alongside API key"
```

### Task 5: `api/main.py` — middleware, auth router, startup guard, legacy `/admin`

**Files:**
- Modify: `api/main.py`
- Test: `tests/test_api_auth.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_api_auth.py
from fastapi.testclient import TestClient


def _full_app():
    from api.main import create_app
    return create_app()


def test_session_auth_flows_through_middleware():
    with patch.dict(os.environ, {
        "MLCOUNCIL_ENV_PROFILE": "local",
        "MLCOUNCIL_REQUIRE_API_KEY": "false",
        "MLCOUNCIL_ADMIN_USERNAME": "admin",
        "MLCOUNCIL_ADMIN_PASSWORD": "s3cret",
    }, clear=False):
        app = _full_app()
        client = TestClient(app)
        # unauth: protected endpoint rejected
        assert client.get("/api/pipeline/status").status_code == 401
        # login then access
        login = client.post("/api/auth/login", json={"username": "admin", "password": "s3cret"})
        assert login.status_code == 200
        resp = client.get("/api/pipeline/status")
        # service may 404/500 on missing data, but must NOT be 401
        assert resp.status_code != 401


def test_legacy_admin_at_admin_prefix():
    with patch.dict(os.environ, {
        "MLCOUNCIL_ENV_PROFILE": "local",
        "MLCOUNCIL_REQUIRE_API_KEY": "false",
        "MLCOUNCIL_ADMIN_USERNAME": "admin",
        "MLCOUNCIL_ADMIN_PASSWORD": "s3cret",
    }, clear=False):
        client = TestClient(_full_app())
        resp = client.get("/admin")
        assert resp.status_code == 200
        assert "admin.html" in resp.text or "MLCouncil" in resp.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_auth.py -v`
Expected: FAIL — no `/api/auth` routes and no `/admin` route yet

- [ ] **Step 3: Modify `api/main.py`**

Replace the API-key middleware with session-aware access control (inside `create_app()`):

```python
from api.errors import ApiError, api_error_handler

app.add_exception_handler(ApiError, api_error_handler)

@app.middleware("http")
async def validate_access(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        if is_public_api_path(request.url.path):
            return await call_next(request)
        from api.auth import request_is_authenticated
        from api.session import check_csrf, request_csrf_token, request_session_token
        if not request_is_authenticated(request):
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=401,
                content={"error": {"code": "not_authenticated", "message": "Authentication required", "detail": ""}},
            )
        if request.method in {"POST", "PUT", "DELETE"} and not request.headers.get("X-API-Key"):
            submitted = request.headers.get("X-CSRF-Token")
            if not check_csrf(request_csrf_token(request), submitted):
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=403,
                    content={"error": {"code": "csrf_failed", "message": "CSRF token mismatch", "detail": ""}},
                )
    return await call_next(request)
```

Import and include the new routers; replace the root route with the legacy `/admin` route behind the flag:

```python
    from api.routers import (auth, analytics, canary, config, experiments, health,
                             intraday, monitoring, pipeline, portfolio, promotion, trading)

    app.include_router(auth.router, prefix=API_PREFIX)
    app.include_router(analytics.router, prefix=API_PREFIX)
    app.include_router(experiments.router, prefix=API_PREFIX)
    app.include_router(canary.router, prefix=API_PREFIX)
    app.include_router(promotion.router, prefix=API_PREFIX)

    if os.getenv("MLCOUNCIL_LEGACY_UI", "true").strip().lower() in {"1", "true", "yes", "on"}:
        @app.get("/admin", response_class=HTMLResponse)
        async def legacy_admin(request: Request):
            return templates.TemplateResponse(request=request, name="admin.html")
```

Extend the startup guard in `validate_environment`:

```python
        if is_api_key_required():
            if not os.getenv("MLCOUNCIL_ADMIN_USERNAME") or not os.getenv("MLCOUNCIL_ADMIN_PASSWORD"):
                raise RuntimeError(
                    "MLCOUNCIL_ADMIN_USERNAME and MLCOUNCIL_ADMIN_PASSWORD are required "
                    "for this runtime profile. Refusing startup to avoid unauthenticated admin access."
                )
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_api_auth.py -v`
Expected: PASS. Then run legacy UI/health tests:
Run: `python -m pytest tests/test_admin_ui.py tests/test_api_health.py -v`
Expected: PASS (if `test_admin_ui.py` hits `/`, update those tests to `/admin`)

- [ ] **Step 5: Commit**

```bash
git add api/main.py api/auth.py tests/test_api_auth.py
git commit -m "feat(api): session middleware, CSRF enforcement, legacy /admin route"
```

## Chunk 2: Analytics service + router + parity test

### Task 6: `api/services/analytics_service.py` — port of data_loader

**Files:**
- Create: `api/services/analytics_service.py`
- Test: `tests/test_api_analytics.py`, `tests/test_analytics_parity.py`

- [ ] **Step 1: Write the failing parity test**

```python
# tests/test_analytics_parity.py
"""Parità numerica: analytics_service vs dashboard.data_loader (stessi artifact)."""
from pathlib import Path

import pandas as pd
import pytest

from api.services import analytics_service


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Reindirizza data/ in tmp_path per il test."""
    results = tmp_path / "results"
    results.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(analytics_service, "DATA_DIR", tmp_path)
    return tmp_path


def _write_equity_fixture(tmp_path: Path):
    idx = pd.date_range("2024-01-01", "2024-01-10", freq="B")
    equity = pd.Series([100.0, 101.5, 100.2, 102.0, 101.1, 103.0, 102.4, 104.0], index=idx[:8])
    equity.to_frame(name="equity").to_parquet(tmp_path / "results" / "equity_curve.parquet")
    return equity


def test_equity_curve_matches_data_loader(data_dir, tmp_path):
    import dashboard.data_loader as dl

    _write_equity_fixture(tmp_path)
    service_out = analytics_service.load_equity_curve(mode="Paper Trading")
    loader_out = dl.load_equity_curve(mode="Paper Trading")

    # data_loader returns the normalized pd.Series; service returns {"dates", "values"}
    assert list(service_out["dates"]) == [d.isoformat() for d in loader_out.index]
    assert service_out["values"] == pytest.approx([float(v) for v in loader_out.values])
```

Note: `dashboard.data_loader` uses `trusted_pickle_load` with hash files; the fixture writes only `equity_curve.parquet`, so both sides must take the parquet branch — keep the fixture minimal exactly as above and align `load_equity_curve`'s artifact priority with `data_loader.py:204-249` (pkl first with hash, then parquet, then orders/paper fallbacks). If `trusted_pickle_load` requires a sidecar hash for the pkl branch, the fixture's parquet-only setup avoids it.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_analytics_parity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.services.analytics_service'`

- [ ] **Step 3: Port `dashboard/data_loader.py` into `api/services/analytics_service.py`**

Port exactly these 13 functions (drop `@st.cache_data`; keep `_densify_business_days` (82-101), `_flatten_universe_tickers` (47-57), `trusted_pickle_load` usage, equity-to-100 normalization (192-197), and the `data/results_snapshots/<tag>` branch for `results_tag`; drop the 4 dead `_synthetic_*` functions):

| Function | data_loader ref | JSON shape |
|---|---|---|
| `load_equity_curve(mode="Paper Trading", results_tag=None)` | 181-259 | `{"dates": [...iso], "values": [...]}` |
| `load_benchmark(mode="Paper Trading", results_tag=None)` | 276-331 | series JSON |
| `load_daily_returns(mode="Paper Trading", results_tag=None)` | 333-344 | series JSON |
| `load_model_attribution(start=None, end=None)` | 346-410 | `{"records": [{date, model_name, weight, ic_rolling_30d, sharpe_rolling_60d, pnl_contribution}]}` |
| `load_ic_history()` | 456-479 | records |
| `load_weights_history()` | 481-504 | records |
| `load_current_regime()` | 506-554 | `{"regime", "bull", "bear", "transition"}` |
| `load_regime_history()` | 556-576 | records |
| `load_portfolio_snapshot()` | 619-675 | records |
| `load_sidebar_metrics()` | 703-784 | dict with keys `sharpe_ytd, max_dd, ic_30d, regime, regime_prob, sharpe_delta, dd_delta, ic_delta` |
| `load_optimization_diagnostics(as_of)` | 786-796 | dict (raw JSON) |
| `load_council_weights_log_entry(as_of)` | 798-819 | dict |
| `load_fill_quality_summary()` | 821-863 | records + `is_bps`/kappa columns |

Module skeleton and helpers:

```python
# api/services/analytics_service.py
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from api.errors import ApiError

DATA_DIR = Path(os.getenv("MLCOUNCIL_DATA_DIR", "data"))


def _series_json(s: pd.Series) -> dict:
    s = s.dropna()
    return {"dates": [d.isoformat() for d in s.index], "values": [float(v) for v in s.values]}


def _records(df: pd.DataFrame) -> dict:
    out = []
    for _, row in df.iterrows():
        rec = {}
        for col, val in row.items():
            if pd.isna(val):
                rec[col] = None
            elif hasattr(val, "isoformat"):
                rec[col] = val.isoformat()
            else:
                try:
                    rec[col] = float(val)
                except (TypeError, ValueError):
                    rec[col] = val
        out.append(rec)
    return {"records": out}


def _artifact(path: Path, what: str) -> Path:
    if not path.exists():
        raise ApiError(404, "artifact_not_found", f"{what} not available yet", str(path))
    return path
```

Each loader: read artifacts via `_artifact(...)`, apply the same transformations as the original (`data_loader.py` line refs above), convert with `_series_json`/`_records`. Missing files raise `ApiError(404, "artifact_not_found", ...)` — the SPA renders empty states on 404. Keep the exact same artifact-priority order as the original so numbers match byte-for-byte.

- [ ] **Step 4: Run the parity + analytics tests**

Run: `python -m pytest tests/test_analytics_parity.py -v`
Expected: PASS (numeric equality with `dashboard.data_loader` for the fixture)

- [ ] **Step 5: Commit**

```bash
git add api/services/analytics_service.py tests/test_analytics_parity.py
git commit -m "feat(api): analytics service ported from dashboard data_loader with parity test"
```

### Task 7: `api/routers/analytics.py`

**Files:**
- Create: `api/routers/analytics.py`
- Test: `tests/test_api_analytics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_analytics.py
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.errors import ApiError, api_error_handler
from api.routers import analytics


def _app():
    app = FastAPI()
    app.add_exception_handler(ApiError, api_error_handler)
    app.include_router(analytics.router, prefix="/api")
    return app


def test_equity_endpoint_404_envelope_when_missing(tmp_path, monkeypatch):
    from api.services import analytics_service
    monkeypatch.setattr(analytics_service, "DATA_DIR", tmp_path)
    client = TestClient(_app())
    resp = client.get("/api/analytics/equity")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "artifact_not_found"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_analytics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.routers.analytics'`

- [ ] **Step 3: Write minimal implementation**

```python
# api/routers/analytics.py
from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import APIRouter, Query

from api.services import analytics_service

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/equity")
def equity(mode: str = "Paper Trading", tag: Optional[str] = None):
    return analytics_service.load_equity_curve(mode=mode, results_tag=tag)


@router.get("/benchmark")
def benchmark(mode: str = "Paper Trading", tag: Optional[str] = None):
    return analytics_service.load_benchmark(mode=mode, results_tag=tag)


@router.get("/returns")
def returns(mode: str = "Paper Trading", tag: Optional[str] = None):
    return analytics_service.load_daily_returns(mode=mode, results_tag=tag)


@router.get("/attribution")
def attribution(start: Optional[date] = None, end: Optional[date] = None):
    return analytics_service.load_model_attribution(start=start, end=end)


@router.get("/ic-history")
def ic_history():
    return analytics_service.load_ic_history()


@router.get("/weights-history")
def weights_history():
    return analytics_service.load_weights_history()


@router.get("/regime/current")
def regime_current():
    return analytics_service.load_current_regime()


@router.get("/regime/history")
def regime_history():
    return analytics_service.load_regime_history()


@router.get("/portfolio-snapshot")
def portfolio_snapshot():
    return analytics_service.load_portfolio_snapshot()


@router.get("/sidebar-metrics")
def sidebar_metrics():
    return analytics_service.load_sidebar_metrics()


@router.get("/optimization-diagnostics")
def optimization_diagnostics(as_of: date = Query(...)):
    return analytics_service.load_optimization_diagnostics(as_of)


@router.get("/weights-log")
def weights_log(as_of: date = Query(...)):
    return analytics_service.load_council_weights_log_entry(as_of)


@router.get("/fill-quality")
def fill_quality():
    return analytics_service.load_fill_quality_summary()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_api_analytics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api/routers/analytics.py tests/test_api_analytics.py
git commit -m "feat(api): analytics router exposing artifact JSON"
```

## Chunk 3: Experiments (subprocess job runner)

### Task 8: `api/services/experiment_worker.py` — per-job entry point

**Files:**
- Create: `api/services/experiment_worker.py`
- Test: `tests/test_api_experiments.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_experiments.py
import json
from pathlib import Path


def _write_job(job_dir: Path, job_id: str, params: dict) -> Path:
    job_dir.mkdir(parents=True, exist_ok=True)
    job_file = job_dir / f"{job_id}.json"
    job_file.write_text(json.dumps({
        "id": job_id, "state": "queued", "params": params,
        "created_at": "2026-08-14T00:00:00Z",
    }), encoding="utf-8")
    return job_file


def test_worker_marks_failed_on_bad_params(tmp_path, monkeypatch):
    from api.services import experiment_service, experiment_worker

    job_dir = tmp_path / "experiments"
    job_file = _write_job(job_dir, "job-1", {"universe": [], "start_date": "x"})

    monkeypatch.setattr(experiment_worker, "JOB_DIR", job_dir)
    experiment_worker.run_job("job-1")

    state = json.loads(job_file.read_text(encoding="utf-8"))
    assert state["state"] == "failed"
    assert "error" in state
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_experiments.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.services.experiment_worker'`

- [ ] **Step 3: Write minimal implementation**

```python
# api/services/experiment_worker.py
"""Esegue un singolo job backtest in un subprocess dedicato.

Uso: python -m api.services.experiment_worker <job_id>
Il worker carica il job dal registro, aggiorna lo stato e scrive l'esito.
"""
from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from runtime_env import load_runtime_env

load_runtime_env()

JOB_DIR = Path("data/results/experiments")


def _job_file(job_id: str) -> Path:
    return JOB_DIR / f"{job_id}.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _update(job_id: str, **fields) -> None:
    path = _job_file(job_id)
    if not path.exists():
        return
    entry = json.loads(path.read_text(encoding="utf-8"))
    entry.update(fields)
    path.write_text(json.dumps(entry, indent=2), encoding="utf-8")


def run_job(job_id: str) -> None:
    try:
        path = _job_file(job_id)
        if not path.exists():
            raise FileNotFoundError(f"job {job_id} not in registry")
        entry = json.loads(path.read_text(encoding="utf-8"))
        _update(job_id, state="running", started_at=_now())

        from backtest.playground import PlaygroundParams, run_playground_backtest

        params = PlaygroundParams.from_dict(entry["params"])
        result = run_playground_backtest(params, progress_cb=None)
        _update(
            job_id,
            state="succeeded",
            finished_at=_now(),
            snapshot_path=str(result.snapshot_path) if result.snapshot_path else None,
            elapsed_seconds=result.elapsed_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        excerpt = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        trace = traceback.format_exc()[-2000:]
        _update(job_id, state="failed", finished_at=_now(),
                error=str(exc), traceback_excerpt=excerpt, traceback=trace)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("usage: python -m api.services.experiment_worker <job_id>\n")
        sys.exit(2)
    run_job(sys.argv[1])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_api_experiments.py -v`
Expected: PASS (bad params path → `failed` with error)

- [ ] **Step 5: Commit**

```bash
git add api/services/experiment_worker.py tests/test_api_experiments.py
git commit -m "feat(api): experiment worker entry point"
```

### Task 9: `api/services/experiment_service.py` — registry + subprocess orchestration

**Files:**
- Create: `api/services/experiment_service.py`
- Test: `tests/test_api_experiments.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_api_experiments.py

def test_submit_runs_worker_and_reaches_terminal_state(tmp_path, monkeypatch):
    from api.services import experiment_service, experiment_worker

    job_dir = tmp_path / "experiments"
    monkeypatch.setattr(experiment_service, "JOB_DIR", job_dir)
    monkeypatch.setattr(experiment_worker, "JOB_DIR", job_dir)

    def fake_spawn(worker_args, cwd):
        experiment_worker.run_job(worker_args[-1])
        return None

    monkeypatch.setattr(experiment_service, "_spawn_worker", fake_spawn)

    job_id = experiment_service.submit_backtest({
        "start_date": "2024-01-01", "end_date": "2024-01-10",
        "universe": ["AAPL"], "note": "test",
    })
    assert job_id
    entry = experiment_service.get_job(job_id)
    assert entry["state"] in {"succeeded", "failed"}
    assert entry["id"] == job_id


def test_boot_sweep_marks_stale_running(tmp_path, monkeypatch):
    from api.services import experiment_service

    job_dir = tmp_path / "experiments"
    job_dir.mkdir(parents=True)
    (job_dir / "job-stale.json").write_text(json.dumps({
        "id": "job-stale", "state": "running", "params": {},
        "created_at": "2026-08-01T00:00:00Z",
    }), encoding="utf-8")

    monkeypatch.setattr(experiment_service, "JOB_DIR", job_dir)
    experiment_service.boot_sweep()

    entry = json.loads((job_dir / "job-stale.json").read_text(encoding="utf-8"))
    assert entry["state"] == "failed"
    assert "interrupted by restart" in entry.get("error", "")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_experiments.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.services.experiment_service'`

- [ ] **Step 3: Write minimal implementation**

```python
# api/services/experiment_service.py
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from api.errors import ApiError
from backtest.playground import (PlaygroundParams, list_snapshots,
                                 load_snapshot_equity, load_snapshot_params)

JOB_DIR = Path("data/results/experiments")
SNAPSHOTS_DIR = Path("data/results_playground")
MAX_REGISTRY_ENTRIES = 50

_lock = threading.Lock()
_procs: dict[str, subprocess.Popen] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_file(job_id: str) -> Path:
    return JOB_DIR / f"{job_id}.json"


def _read_job(job_id: str) -> dict:
    path = _job_file(job_id)
    if not path.exists():
        raise ApiError(404, "job_not_found", f"Job {job_id} not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_job(entry: dict) -> None:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    _job_file(entry["id"]).write_text(json.dumps(entry, indent=2), encoding="utf-8")


def _spawn_worker(job_id: str) -> None:
    proc = subprocess.Popen(
        [sys.executable, "-m", "api.services.experiment_worker", job_id],
        cwd=Path(__file__).resolve().parents[2],
        env=dict(os.environ),
    )
    _procs[job_id] = proc


def submit_backtest(params: dict) -> str:
    try:
        PlaygroundParams.from_dict(params)
    except Exception as exc:  # noqa: BLE001
        raise ApiError(400, "invalid_params", f"Invalid backtest params: {exc}") from exc

    job_id = f"job-{uuid.uuid4().hex[:12]}"
    _write_job({"id": job_id, "state": "queued", "params": params, "created_at": _now()})
    with _lock:
        entry = _read_job(job_id)
        entry["state"] = "running"
        entry["started_at"] = _now()
        _write_job(entry)
        _spawn_worker(job_id)
    return job_id


def cancel_job(job_id: str) -> dict:
    entry = _read_job(job_id)
    if entry["state"] in {"running", "queued"}:
        proc = _procs.get(job_id)
        if proc and proc.poll() is None:
            proc.terminate()
        entry["state"] = "cancelled"
        entry["finished_at"] = _now()
        _write_job(entry)
    return entry


def get_job(job_id: str) -> dict:
    return _read_job(job_id)


def list_jobs() -> list[dict]:
    if not JOB_DIR.exists():
        return []
    jobs = []
    for path in sorted(JOB_DIR.glob("job-*.json"), reverse=True):
        jobs.append(json.loads(path.read_text(encoding="utf-8")))
    return jobs


def get_job_result(job_id: str) -> dict:
    entry = _read_job(job_id)
    if entry["state"] != "succeeded" or not entry.get("snapshot_path"):
        raise ApiError(409, "job_not_finished", f"Job {job_id} has no result", entry["state"])
    snap = Path(entry["snapshot_path"])
    if not snap.exists():
        raise ApiError(404, "artifact_not_found", "Snapshot directory missing", str(snap))
    stats_file = snap / "stats.json"
    stats = json.loads(stats_file.read_text(encoding="utf-8")) if stats_file.exists() else {}
    return {"job": entry, "snapshot_dir": str(snap), "stats": stats}


def list_snapshot_records() -> list[dict]:
    df = list_snapshots(base_dir=SNAPSHOTS_DIR)
    return df.to_dict(orient="records")


def get_snapshot(snapshot_dir: str) -> dict:
    snap = Path(snapshot_dir)
    if not snap.exists():
        raise ApiError(404, "artifact_not_found", "Snapshot directory missing", snapshot_dir)
    equity = load_snapshot_equity(snap)
    params = load_snapshot_params(snap)
    return {
        "snapshot_dir": snapshot_dir,
        "equity": {"dates": [d.isoformat() for d in equity.index],
                   "values": [float(v) for v in equity.values]},
        "params": params,
    }


def boot_sweep() -> None:
    """Avvio: running->failed (orfani da restart), prune registro e snapshot."""
    if not JOB_DIR.exists():
        return
    jobs = [json.loads(p.read_text(encoding="utf-8")) for p in JOB_DIR.glob("job-*.json")]
    jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    for entry in jobs:
        if entry["state"] == "running":
            entry["state"] = "failed"
            entry["error"] = "interrupted by restart"
            entry["finished_at"] = _now()
            _write_job(entry)
    keep = jobs[:MAX_REGISTRY_ENTRIES]
    for entry in jobs[MAX_REGISTRY_ENTRIES:]:
        _job_file(entry["id"]).unlink(missing_ok=True)
    oldest_kept = keep[-1]["created_at"] if keep else None
    if oldest_kept and SNAPSHOTS_DIR.exists():
        stamp = oldest_kept[:10].replace("-", "") + "-" + oldest_kept[11:19].replace(":", "")
        for snap in SNAPSHOTS_DIR.iterdir():
            if snap.is_dir() and snap.name < stamp:
                shutil.rmtree(snap, ignore_errors=True)
```

Note: snapshot dirs are named `<YYYYMMDD-HHMMSS>` (`playground.py:507-509`); the prune compares against the oldest kept job's creation stamp. If pruning proves fiddly, keep it conservative (only prune when `len(jobs) > MAX_REGISTRY_ENTRIES`).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_api_experiments.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api/services/experiment_service.py tests/test_api_experiments.py
git commit -m "feat(api): subprocess experiment runner with registry and boot sweep"
```

### Task 10: `api/routers/experiments.py`

**Files:**
- Create: `api/routers/experiments.py`
- Test: `tests/test_api_experiments.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_api_experiments.py
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.errors import ApiError, api_error_handler
from api.routers import experiments


def _app():
    app = FastAPI()
    app.add_exception_handler(ApiError, api_error_handler)
    app.include_router(experiments.router, prefix="/api")
    return app


def test_submit_endpoint_rejects_bad_params():
    client = TestClient(_app())
    resp = client.post("/api/experiments/backtest", json={"params": {"universe": [], "start_date": "x"}})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_params"


def test_job_status_404():
    client = TestClient(_app())
    resp = client.get("/api/experiments/jobs/nope/status")
    assert resp.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_experiments.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.routers.experiments'`

- [ ] **Step 3: Write minimal implementation**

```python
# api/routers/experiments.py
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from api.services import experiment_service

router = APIRouter(prefix="/experiments", tags=["experiments"])


class BacktestRequest(BaseModel):
    params: dict


@router.post("/backtest")
def run_backtest(body: BacktestRequest):
    job_id = experiment_service.submit_backtest(body.params)
    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs")
def jobs():
    return {"jobs": experiment_service.list_jobs()}


@router.get("/jobs/{job_id}/status")
def job_status(job_id: str):
    entry = experiment_service.get_job(job_id)
    return {"id": entry["id"], "state": entry["state"], "error": entry.get("error")}


@router.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    return experiment_service.get_job_result(job_id)


@router.post("/jobs/{job_id}/cancel")
def job_cancel(job_id: str):
    return experiment_service.cancel_job(job_id)


@router.get("/snapshots")
def snapshots():
    return {"snapshots": experiment_service.list_snapshot_records()}


@router.get("/snapshots/{snapshot_dir:path}")
def snapshot(snapshot_dir: str):
    return experiment_service.get_snapshot(snapshot_dir)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_api_experiments.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api/routers/experiments.py tests/test_api_experiments.py
git commit -m "feat(api): experiments router for backtest jobs and snapshots"
```

## Chunk 4: Canary overlay + Promotion routers

### Task 11: `council/canary.py` — additive `pending_apply` overlay

**Files:**
- Modify: `council/canary.py`
- Test: `tests/test_canary_pending.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_canary_pending.py
"""Overlay pending_apply: additivo a CanaryState, ignorato da check_revert."""
from pathlib import Path

import pytest

from council.canary import CanaryFeature, CanaryState, CanaryController


@pytest.fixture
def config() -> list[CanaryFeature]:
    return [
        CanaryFeature(name="online_learning", env="MLCOUNCIL_ONLINE_LEARNING", value="true", enabled=True, floor=0.0, min_days=5),
        CanaryFeature(name="moe_gating", env="MLCOUNCIL_AGGREGATOR_MODE", value="moe", enabled=False, floor=0.0, min_days=5),
    ]


def test_pending_apply_is_persisted(config, tmp_path):
    state_path = tmp_path / "canary_state.json"
    state = CanaryState()
    state.set_pending("moe_gating", True)
    state.save(str(state_path))
    loaded = CanaryState.load(str(state_path), config=config)
    assert loaded.pending_apply["moe_gating"]["enabled"] is True


def test_active_features_honor_pending(config, tmp_path):
    state = CanaryState()
    state.set_pending("moe_gating", True)
    controller = CanaryController(config, state_path=str(tmp_path / "canary_state.json"))
    controller.state = state
    active = [f.name for f in controller._active_features()]
    assert "moe_gating" in active


def test_revert_still_wins_over_pending(config, tmp_path):
    state = CanaryState()
    state.disable("online_learning", reason="test", last_value=0.01, floor=0.05, date="2026-08-14")
    state.set_pending("online_learning", True)
    controller = CanaryController(config, state_path=str(tmp_path / "canary_state.json"))
    controller.state = state
    assert "online_learning" not in [f.name for f in controller._active_features()]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_canary_pending.py -v`
Expected: FAIL — `CanaryState` has no `pending_apply`/`set_pending`

- [ ] **Step 3: Modify `council/canary.py`**

Add field to the `CanaryState` dataclass (`canary.py:124-237`):

```python
    pending_apply: dict[str, dict] = field(default_factory=dict)
```

Add methods:

```python
    def set_pending(self, name: str, enabled: bool) -> None:
        """Overlay in attesa per la prossima run (additivo, ignorato da check_revert)."""
        self.pending_apply[name] = {"enabled": enabled, "at": datetime.now(timezone.utc).isoformat()}

    def clear_pending(self, name: str) -> None:
        self.pending_apply.pop(name, None)

    def pending_enabled(self, name: str, config_enabled: bool) -> bool:
        pending = self.pending_apply.get(name)
        if pending is None:
            return config_enabled
        return bool(pending["enabled"])
```

Wire into `load` (tolerate absence in old files) and `save` (include `pending_apply` in the JSON round-trip), then change `CanaryController._active_features()` (`canary.py:292-294`):

```python
    def _active_features(self) -> list[CanaryFeature]:
        active = []
        for f in self._config:
            if not self.state.is_enabled(f.name) and self.state.pending_enabled(f.name, f.enabled):
                active.append(f)
        return active
```

`check_revert` (332-382) stays untouched — it only reads `state.features` history, so pending is ignored by design.

- [ ] **Step 4: Run the pending + existing canary tests**

Run: `python -m pytest tests/test_canary_pending.py tests/test_canary.py -v`
Expected: PASS (existing suite stays green — files without `pending_apply` behave as before)

- [ ] **Step 5: Commit**

```bash
git add council/canary.py tests/test_canary_pending.py
git commit -m "feat(canary): additive pending_apply overlay honored by active features"
```

### Task 12: `api/services/canary_service.py` + `api/routers/canary.py`

**Files:**
- Create: `api/services/canary_service.py`, `api/routers/canary.py`
- Test: `tests/test_api_canary.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_canary.py
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.errors import ApiError, api_error_handler
from api.routers import canary


def _app():
    app = FastAPI()
    app.add_exception_handler(ApiError, api_error_handler)
    app.include_router(canary.router, prefix="/api")
    return app


def test_flags_endpoint_lists_features(tmp_path, monkeypatch):
    from api.services import canary_service
    canary_yaml = tmp_path / "canary.yaml"
    canary_yaml.write_text(
        "features:\n"
        "  - name: online_learning\n    env: MLCOUNCIL_ONLINE_LEARNING\n    value: 'true'\n    enabled: true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(canary_service, "CANARY_CONFIG_PATH", canary_yaml)
    client = TestClient(_app())
    resp = client.get("/api/canary/flags")
    assert resp.status_code == 200
    assert resp.json()["features"][0]["name"] == "online_learning"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_canary.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.routers.canary'`

- [ ] **Step 3: Write minimal implementation**

```python
# api/services/canary_service.py
from __future__ import annotations

import os
from pathlib import Path

from api.errors import ApiError
from council.canary import CanaryController, CanaryState, load_canary_config

CANARY_CONFIG_PATH = Path(os.getenv("MLCOUNCIL_CANARY_CONFIG", "config/canary.yaml"))
STATE_PATH = Path("data/results/canary_state.json")


def _controller() -> CanaryController:
    config = load_canary_config(str(CANARY_CONFIG_PATH)) if CANARY_CONFIG_PATH.exists() else []
    return CanaryController(config, state_path=str(STATE_PATH))


def get_flags() -> dict:
    if not CANARY_CONFIG_PATH.exists():
        raise ApiError(404, "artifact_not_found", "canary.yaml not found", str(CANARY_CONFIG_PATH))
    controller = _controller()
    state = controller.state
    out = []
    for f in controller._config:
        reverted = not state.is_enabled(f.name)
        pending = state.pending_apply.get(f.name)
        out.append({
            "name": f.name, "env": f.env, "value": f.value,
            "config_enabled": f.enabled, "reverted": reverted,
            "pending_enabled": pending["enabled"] if pending else None,
            "effective_enabled": (pending["enabled"] if pending else f.enabled) and not reverted,
            "floor": f.floor, "min_days": f.min_days,
        })
    return {"features": out}


def get_state() -> dict:
    state = CanaryState.load(str(STATE_PATH), config=_controller()._config)
    return {
        "state_file": str(STATE_PATH),
        "exists": STATE_PATH.exists(),
        "reverted_features": {
            name: {"reverted_at": v.get("reverted_at"), "reason": v.get("revert_reason")}
            for name, v in state.features.items() if not v.get("enabled")
        },
        "pending_apply": state.pending_apply,
        "history": state.history,
    }


def apply_pending(name: str, enabled: bool) -> dict:
    controller = _controller()
    if name not in {f.name for f in controller._config}:
        raise ApiError(404, "unknown_flag", f"Unknown canary flag: {name}")
    controller.state.set_pending(name, enabled)
    controller.state.save(str(STATE_PATH))
    return preview()


def preview() -> dict:
    controller = _controller()
    changes = []
    for f in controller._config:
        pending = controller.state.pending_apply.get(f.name)
        if pending is not None:
            changes.append({
                "name": f.name, "from": f.enabled,
                "to": bool(pending["enabled"]), "at": pending.get("at"),
            })
    return {"pending_changes": changes, "flags": get_flags()["features"]}


def clear_pending(name: str) -> dict:
    controller = _controller()
    controller.state.clear_pending(name)
    controller.state.save(str(STATE_PATH))
    return preview()
```

```python
# api/routers/canary.py
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from api.services import canary_service

router = APIRouter(prefix="/canary", tags=["canary"])


class PendingApply(BaseModel):
    name: str
    enabled: bool


@router.get("/state")
def state():
    return canary_service.get_state()


@router.get("/flags")
def flags():
    return canary_service.get_flags()


@router.get("/apply/preview")
def apply_preview():
    return canary_service.preview()


@router.post("/apply")
def apply(body: PendingApply):
    return canary_service.apply_pending(body.name, body.enabled)


@router.post("/apply/clear")
def apply_clear(body: PendingApply):
    return canary_service.clear_pending(body.name)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_api_canary.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api/services/canary_service.py api/routers/canary.py tests/test_api_canary.py
git commit -m "feat(api): canary flags/state/overlay endpoints"
```

### Task 13: `api/services/promotion_service.py` + `api/routers/promotion.py`

**Files:**
- Create: `api/services/promotion_service.py`, `api/routers/promotion.py`
- Test: `tests/test_api_promotion.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_promotion.py
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.errors import ApiError, api_error_handler
from api.routers import promotion


def _app():
    app = FastAPI()
    app.add_exception_handler(ApiError, api_error_handler)
    app.include_router(promotion.router, prefix="/api")
    return app


def test_manifest_404_when_missing(tmp_path, monkeypatch):
    from api.services import promotion_service
    monkeypatch.setattr(promotion_service, "MANIFEST_PATH", tmp_path / "nope.yaml")
    client = TestClient(_app())
    resp = client.get("/api/promotion/manifest")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "artifact_not_found"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_promotion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.routers.promotion'`

- [ ] **Step 3: Write minimal implementation**

```python
# api/services/promotion_service.py
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from api.errors import ApiError

MANIFEST_PATH = Path(os.getenv("MLCOUNCIL_PRODUCTION_MANIFEST", "config/production_manifest.yaml"))
OPERATIONS_DIR = Path("data/operations")
SHADOW_ARTIFACTS = [
    Path("data/results/tft_shadow_signals.parquet"),
    Path("data/results/shadow_sentiment_llm"),
    Path("data/results/tda_warning_latest.json"),
]


def get_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise ApiError(404, "artifact_not_found", "Production manifest not found", str(MANIFEST_PATH))
    return yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))


def get_reports() -> dict:
    reports = {}
    for path in sorted(OPERATIONS_DIR.glob("walkforward_promotion_*.json")):
        try:
            reports[path.stem.replace("walkforward_promotion_", "")] = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    streaks = {}
    for path in sorted(OPERATIONS_DIR.glob("walkforward_streak_*.json")):
        try:
            streaks[path.stem.replace("walkforward_streak_", "")] = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    return {"reports": reports, "streaks": streaks}


def get_shadow_artifacts() -> dict:
    out = []
    for path in SHADOW_ARTIFACTS:
        try:
            mtime = path.stat().st_mtime if path.exists() else None
        except OSError:
            mtime = None
        out.append({"path": str(path), "exists": path.exists(), "mtime": mtime})
    return {"artifacts": out}
```

```python
# api/routers/promotion.py
from __future__ import annotations

from fastapi import APIRouter

from api.services import promotion_service

router = APIRouter(prefix="/promotion", tags=["promotion"])


@router.get("/manifest")
def manifest():
    return promotion_service.get_manifest()


@router.get("/reports")
def reports():
    return promotion_service.get_reports()


@router.get("/shadow-artifacts")
def shadow_artifacts():
    return promotion_service.get_shadow_artifacts()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_api_promotion.py -v`
Expected: PASS

- [ ] **Step 5: Run full backend suite for this plan**

Run: `python -m pytest tests/test_api_auth.py tests/test_api_analytics.py tests/test_analytics_parity.py tests/test_api_experiments.py tests/test_api_canary.py tests/test_api_promotion.py tests/test_canary_pending.py tests/test_canary.py tests/test_admin_ui.py tests/test_api_health.py -v`
Expected: all PASS (existing tests stay green)

- [ ] **Step 6: Commit**

```bash
git add api/services/promotion_service.py api/routers/promotion.py tests/test_api_promotion.py
git commit -m "feat(api): promotion manifest/reports/shadow endpoints"
```

## Chunk 5: Frontend scaffold + auth UI + SPA serving

### Task 14: Vite + React + TS scaffold

**Files:**
- Create: `frontend/package.json`, `frontend/vite.config.ts`, `frontend/tsconfig.json`, `frontend/index.html`, `frontend/.eslintrc.cjs`, `frontend/.gitignore`
- Create: `frontend/src/main.tsx`, `frontend/src/styles/theme.css`

- [ ] **Step 1: Scaffold with exact files**

```json
// frontend/package.json
{
  "name": "mlcouncil-ui",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview",
    "test": "vitest run",
    "test:watch": "vitest",
    "lint": "eslint src --max-warnings 0",
    "e2e": "playwright test"
  },
  "dependencies": {
    "@tanstack/react-query": "^5.51.0",
    "plotly.js": "^2.34.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-plotly.js": "^2.6.0",
    "react-router-dom": "^6.26.0"
  },
  "devDependencies": {
    "@playwright/test": "^1.45.0",
    "@testing-library/jest-dom": "^6.4.0",
    "@testing-library/react": "^16.0.0",
    "@types/plotly.js": "^2.33.0",
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.0",
    "eslint": "^8.57.0",
    "jsdom": "^24.1.0",
    "msw": "^2.3.0",
    "typescript": "^5.5.0",
    "vite": "^5.4.0",
    "vitest": "^2.0.0"
  }
}
```

```ts
// frontend/vite.config.ts
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://localhost:8000" },
  },
  build: { outDir: "dist", emptyOutDir: true },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    globals: true,
  },
});
```

```json
// frontend/tsconfig.json
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "forceConsistentCasingInFileNames": true,
    "types": ["vitest/globals", "@testing-library/jest-dom"]
  },
  "include": ["src", "vite.config.ts"]
}
```

```html
<!-- frontend/index.html -->
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>MLCouncil</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

```tsx
// frontend/src/main.tsx
import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { AuthProvider } from "./auth/AuthContext";
import { ApiError } from "./api/client";
import "./styles/theme.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: (failureCount, error: unknown) =>
        error instanceof ApiError && error.status >= 500 && failureCount < 2,
    },
  },
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AuthProvider>
          <App />
        </AuthProvider>
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
);
```

`frontend/.gitignore`: `node_modules/`, `dist/`, `playwright-report/`, `test-results/`.
`frontend/.eslintrc.cjs`: flat-style minimal ESLint 8 config with `@typescript-eslint` recommended; keep `--max-warnings 0` buildable (see note in Task 22 CI).
`frontend/src/styles/theme.css`: dark theme (colors from `api/static/css/admin.css`: background `#17181c`, panels `#1f2128`, text `#c9ccd4`, accent `#4e9bde`), `.app-shell` grid (240px sidebar + main), `.sidebar` styles, `.nav-link`/`.nav-link.active`, `.page`, `.kpi-row`, `.form-grid`, `.form-error`, `.login-screen`/`.login-card`, `.boot-screen`, `.alert-banner`/`.alert-item`, `.table` styles, `.status-badge` variants (running/queued/succeeded/failed/cancelled).

- [ ] **Step 2: Build check**

Run: `npm install && npm run build` (in `frontend/`)
Expected: build succeeds; `frontend/dist/index.html` exists

- [ ] **Step 3: Commit**

```bash
git add frontend/
git commit -m "feat(ui): React + Vite + TS scaffold"
```

### Task 15: API client + auth context + login page

**Files:**
- Create: `frontend/src/api/client.ts`, `frontend/src/api/queries.ts`
- Create: `frontend/src/auth/AuthContext.tsx`
- Create: `frontend/src/pages/LoginPage.tsx`
- Create: `frontend/src/components/ProtectedRoute.tsx`
- Create: `frontend/src/test/setup.ts`, `frontend/src/test/server.ts`

- [ ] **Step 1: Write the failing test**

```tsx
// frontend/src/test/setup.ts
import "@testing-library/jest-dom";
import { server } from "./server";

beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());
```

```tsx
// frontend/src/test/server.ts
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";

export const server = setupServer(
  http.get("/api/auth/me", () => HttpResponse.json({ authenticated: true, username: "admin" })),
);
```

```tsx
// frontend/src/auth/AuthContext.test.tsx
import { render, screen, waitFor } from "@testing-library/react";
import { AuthProvider, useAuth } from "./AuthContext";

function Probe() {
  const { status, username } = useAuth();
  return <div data-testid="probe">{status}:{username ?? "none"}</div>;
}

it("loads session on mount", async () => {
  render(
    <AuthProvider>
      <Probe />
    </AuthProvider>,
  );
  await waitFor(() => expect(screen.getByTestId("probe")).toHaveTextContent("authenticated:admin"));
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test` (in `frontend/`)
Expected: FAIL — no AuthContext

- [ ] **Step 3: Write implementation**

```ts
// frontend/src/api/client.ts
export class ApiError extends Error {
  constructor(public status: number, public code: string, public message: string, public detail: string) {
    super(message);
  }
}

function csrfToken(): string | null {
  const match = document.cookie.match(/(?:^|;\s*)mlcouncil_csrf=([^;]+)/);
  return match ? decodeURIComponent(match[1]) : null;
}

export async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { ...(options.headers as Record<string, string>) };
  const method = (options.method ?? "GET").toUpperCase();
  if (method !== "GET" && method !== "HEAD") {
    const token = csrfToken();
    if (token) headers["X-CSRF-Token"] = token;
  }
  if (options.body) headers["Content-Type"] = "application/json";
  const resp = await fetch(path, { ...options, headers, credentials: "same-origin" });
  if (resp.status === 401) {
    window.location.href = "/login";
    throw new ApiError(401, "not_authenticated", "Not logged in", "");
  }
  let body: unknown = null;
  try {
    body = await resp.json();
  } catch {
    /* non-JSON body */
  }
  if (!resp.ok) {
    const err = (body as { error?: { code?: string; message?: string; detail?: string } })?.error;
    throw new ApiError(resp.status, err?.code ?? "http_error", err?.message ?? resp.statusText, err?.detail ?? "");
  }
  return body as T;
}

export const authApi = {
  login: (username: string, password: string) =>
    api<{ authenticated: boolean; username: string }>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),
  logout: () => api<{ authenticated: boolean }>("/api/auth/logout", { method: "POST" }),
  me: () => api<{ authenticated: boolean; username: string }>("/api/auth/me"),
};
```

```ts
// frontend/src/api/queries.ts
import { useQuery } from "@tanstack/react-query";
import { api } from "./client";

export interface Series {
  dates: string[];
  values: number[];
}

export interface AttributionRow {
  date: string;
  model_name: string;
  weight: number | null;
  ic_rolling_30d: number | null;
  sharpe_rolling_60d: number | null;
  pnl_contribution: number | null;
}

export function useEquity(mode = "Paper Trading", tag?: string) {
  const tagPart = tag ? `&tag=${encodeURIComponent(tag)}` : "";
  return useQuery({
    queryKey: ["equity", mode, tag],
    queryFn: () => api<Series>(`/api/analytics/equity?mode=${encodeURIComponent(mode)}${tagPart}`),
    staleTime: 60_000,
  });
}

export function useBenchmark(mode = "Paper Trading") {
  return useQuery({
    queryKey: ["benchmark", mode],
    queryFn: () => api<Series>(`/api/analytics/benchmark?mode=${encodeURIComponent(mode)}`),
    staleTime: 60_000,
  });
}

export function useDailyReturns(mode = "Paper Trading") {
  return useQuery({
    queryKey: ["returns", mode],
    queryFn: () => api<Series>(`/api/analytics/returns?mode=${encodeURIComponent(mode)}`),
    staleTime: 60_000,
  });
}

export function useAttribution(start?: string, end?: string) {
  return useQuery({
    queryKey: ["attribution", start, end],
    queryFn: () =>
      api<{ records: AttributionRow[] }>(
        `/api/analytics/attribution${start || end ? `?start=${start ?? ""}&end=${end ?? ""}` : ""}`,
      ),
    staleTime: 60_000,
  });
}
```

(Add hooks for `ic-history`, `weights-history`, `regime/current`, `regime/history`, `sidebar-metrics`, `portfolio-snapshot`, `optimization-diagnostics`, `weights-log`, `fill-quality`, experiments jobs, canary, promotion following the same pattern.)

```tsx
// frontend/src/auth/AuthContext.tsx
import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { authApi } from "../api/client";

type AuthStatus = "loading" | "authenticated" | "unauthenticated";

interface AuthState {
  status: AuthStatus;
  username: string | null;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [status, setStatus] = useState<AuthStatus>("loading");
  const [username, setUsername] = useState<string | null>(null);

  useEffect(() => {
    authApi
      .me()
      .then((me) => {
        setStatus("authenticated");
        setUsername(me.username);
      })
      .catch(() => setStatus("unauthenticated"));
  }, []);

  const login = useCallback(async (user: string, pass: string) => {
    const me = await authApi.login(user, pass);
    setUsername(me.username);
    setStatus("authenticated");
  }, []);

  const logout = useCallback(async () => {
    try {
      await authApi.logout();
    } finally {
      setStatus("unauthenticated");
      setUsername(null);
    }
  }, []);

  return <AuthContext.Provider value={{ status, username, login, logout }}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth outside AuthProvider");
  return ctx;
}
```

```tsx
// frontend/src/components/ProtectedRoute.tsx
import { Navigate } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";
import { SidebarLayout } from "./layout/Sidebar";

export function ProtectedRoute() {
  const { status } = useAuth();
  if (status === "loading") return <div className="boot-screen">Loading…</div>;
  if (status !== "authenticated") return <Navigate to="/login" replace />;
  return <SidebarLayout />;
}
```

```tsx
// frontend/src/pages/LoginPage.tsx
import { useState } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";

export function LoginPage() {
  const { status, login } = useAuth();
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  if (status === "authenticated") return <Navigate to="/" replace />;

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await login(username, password);
      navigate("/", { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login-screen">
      <form className="login-card" onSubmit={onSubmit}>
        <h1>MLCouncil</h1>
        <label>
          Username
          <input value={username} onChange={(e) => setUsername(e.target.value)} autoFocus />
        </label>
        <label>
          Password
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
        </label>
        {error && <p className="form-error">{error}</p>}
        <button type="submit" disabled={busy}>{busy ? "Signing in…" : "Sign in"}</button>
      </form>
    </div>
  );
}
```

- [ ] **Step 4: Run tests + build**

Run: `npm test && npm run build` (in `frontend/`)
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api frontend/src/auth frontend/src/pages/LoginPage.tsx frontend/src/components/ProtectedRoute.tsx frontend/src/test
git commit -m "feat(ui): API client with CSRF, auth context, login page"
```

### Task 16: Layout + sidebar IA + app routes

**Files:**
- Create: `frontend/src/components/layout/Sidebar.tsx`, `frontend/src/components/AlertBanner.tsx`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Write implementation directly** (layout is structural, covered by the e2e smoke in Chunk 8)

```tsx
// frontend/src/App.tsx
import { Route, Routes } from "react-router-dom";
import { ProtectedRoute } from "./components/ProtectedRoute";
import { LoginPage } from "./pages/LoginPage";
import { OverviewPage } from "./pages/OverviewPage";
import { PerformancePage } from "./pages/PerformancePage";
import { AttributionPage } from "./pages/AttributionPage";
import { RegimePage } from "./pages/RegimePage";
import { BacktestPage } from "./pages/BacktestPage";
import { PromotionPage } from "./pages/PromotionPage";
import { CanaryPage } from "./pages/CanaryPage";
import { PipelinePage } from "./pages/PipelinePage";
import { TradingPage } from "./pages/TradingPage";
import { IntradayPage } from "./pages/IntradayPage";
import { PortfolioPage } from "./pages/PortfolioPage";
import { ConfigPage } from "./pages/ConfigPage";
import { MonitoringPage } from "./pages/MonitoringPage";
import { FillQualityPage } from "./pages/FillQualityPage";

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route element={<ProtectedRoute />}>
        <Route index element={<OverviewPage />} />
        <Route path="/performance" element={<PerformancePage />} />
        <Route path="/attribution" element={<AttributionPage />} />
        <Route path="/regime" element={<RegimePage />} />
        <Route path="/experiments/backtest" element={<BacktestPage />} />
        <Route path="/experiments/promotion" element={<PromotionPage />} />
        <Route path="/experiments/canary" element={<CanaryPage />} />
        <Route path="/operations/pipeline" element={<PipelinePage />} />
        <Route path="/operations/trading" element={<TradingPage />} />
        <Route path="/operations/intraday" element={<IntradayPage />} />
        <Route path="/operations/portfolio" element={<PortfolioPage />} />
        <Route path="/system/config" element={<ConfigPage />} />
        <Route path="/system/monitoring" element={<MonitoringPage />} />
        <Route path="/system/fill-quality" element={<FillQualityPage />} />
      </Route>
    </Routes>
  );
}
```

```tsx
// frontend/src/components/layout/Sidebar.tsx
import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { useAuth } from "../../auth/AuthContext";
import { AlertBanner } from "../AlertBanner";

const NAV_GROUPS: { group: string; links: { to: string; label: string }[] }[] = [
  { group: "", links: [{ to: "/", label: "Overview" }] },
  {
    group: "Analytics",
    links: [
      { to: "/performance", label: "Performance" },
      { to: "/attribution", label: "Attribution" },
      { to: "/regime", label: "Regime" },
    ],
  },
  {
    group: "Experiments",
    links: [
      { to: "/experiments/backtest", label: "Backtest" },
      { to: "/experiments/promotion", label: "Promotion" },
      { to: "/experiments/canary", label: "Canary" },
    ],
  },
  {
    group: "Operations",
    links: [
      { to: "/operations/pipeline", label: "Pipeline" },
      { to: "/operations/trading", label: "Trading" },
      { to: "/operations/intraday", label: "Intraday" },
      { to: "/operations/portfolio", label: "Portfolio" },
    ],
  },
  {
    group: "System",
    links: [
      { to: "/system/config", label: "Configuration" },
      { to: "/system/monitoring", label: "Monitoring" },
      { to: "/system/fill-quality", label: "Fill Quality" },
    ],
  },
];

const EXTERNAL_LINKS = [
  { href: "/mlflow/", label: "MLflow" },
  { href: "https://mlcouncil.duckdns.org:8443/", label: "Dagster" },
  { href: "http://localhost:3001", label: "Grafana" },
];

export function SidebarLayout() {
  const { username, logout } = useAuth();
  const navigate = useNavigate();
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-brand">MLCouncil</div>
        <nav>
          {NAV_GROUPS.map((g) => (
            <div key={g.group || "root"} className="nav-group">
              {g.group && <div className="nav-group-title">{g.group}</div>}
              {g.links.map((l) => (
                <NavLink key={l.to} to={l.to} end={l.to === "/"}
                  className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
                  {l.label}
                </NavLink>
              ))}
            </div>
          ))}
          <div className="nav-group-title">Links</div>
          {EXTERNAL_LINKS.map((l) => (
            <a key={l.href} className="nav-link" href={l.href} target="_blank" rel="noreferrer">
              {l.label} ↗
            </a>
          ))}
        </nav>
        <div className="sidebar-footer">
          <span>{username}</span>
          <button onClick={() => logout().then(() => navigate("/login"))} className="link-button">
            Sign out
          </button>
        </div>
      </aside>
      <main className="main-content">
        <AlertBanner />
        <Outlet />
      </main>
    </div>
  );
}
```

```tsx
// frontend/src/components/AlertBanner.tsx
import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";

interface AlertItem {
  level: string;
  title: string;
  message?: string;
}

export function AlertBanner() {
  const { data } = useQuery({
    queryKey: ["alerts"],
    queryFn: () => api<AlertItem[]>("/api/monitoring/alerts"),
    refetchInterval: 120_000,
  });
  if (!data || data.length === 0) return null;
  return (
    <div className="alert-banner">
      {data.map((a, i) => (
        <div key={i} className={`alert-item alert-${a.level}`}>
          <strong>{a.title}</strong> {a.message}
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 2: Create placeholder pages + build check**

Create one minimal placeholder per page module referenced in `App.tsx` (renders `<h1>Title</h1>` + `.page-empty` note); they are filled in Chunks 6-7.

Run: `npm run build` (in `frontend/`)
Expected: build succeeds

- [ ] **Step 3: Commit**

```bash
git add frontend/src/App.tsx frontend/src/components/layout frontend/src/components/AlertBanner.tsx frontend/src/pages
git commit -m "feat(ui): app shell with sidebar IA and routes"
```

### Task 17: Serve SPA from FastAPI (dev coexistence with legacy `/admin`)

**Files:**
- Modify: `api/main.py`
- Modify: `Dockerfile`
- Modify: `docker-compose.yml` (env pass-through)
- Modify: `.env.example`
- Test: `tests/test_api_auth.py` (append SPA-serving test)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_api_auth.py
def test_spa_served_at_root(tmp_path, monkeypatch):
    from api.main import create_app, SPA_DIST_DIR
    from fastapi.testclient import TestClient

    monkeypatch.setattr("api.main.SPA_DIST_DIR", tmp_path)
    (tmp_path / "index.html").write_text("<html><body>MLCouncil SPA</body></html>", encoding="utf-8")

    with patch.dict(os.environ, {
        "MLCOUNCIL_ENV_PROFILE": "local",
        "MLCOUNCIL_REQUIRE_API_KEY": "false",
    }, clear=False):
        client = TestClient(create_app())
        resp = client.get("/")
        assert resp.status_code == 200
        assert "MLCouncil SPA" in resp.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_auth.py::test_spa_served_at_root -v`
Expected: FAIL — root serves the legacy admin template (no SPA dist mounted)

- [ ] **Step 3: Modify `api/main.py`**

Add SPA constants and routes (keep the `/admin` legacy route from Task 5):

```python
from fastapi.responses import FileResponse, HTMLResponse

SPA_DIST_DIR = Path(__file__).resolve().parents[1] / "api" / "static" / "spa"


def _spa_index_exists() -> bool:
    return (SPA_DIST_DIR / "index.html").exists()
```

In `create_app()`, after mounting `/static` and registering all routers (order matters — the fallback must be LAST):

```python
    if _spa_index_exists():
        app.mount("/app", StaticFiles(directory=str(SPA_DIST_DIR), html=True), name="spa")

        @app.get("/", response_class=HTMLResponse)
        async def spa_root():
            return FileResponse(SPA_DIST_DIR / "index.html")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa_fallback(full_path: str):
            if full_path.startswith(("api/", "static/", "admin", "app/")):
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="Not found")
            candidate = SPA_DIST_DIR / full_path
            if candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(SPA_DIST_DIR / "index.html")
```

- [ ] **Step 4: Modify `Dockerfile`** — multi-stage frontend build

```dockerfile
FROM node:20-alpine AS frontend-build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# existing python stage stays; after `COPY . .` add:
COPY --from=frontend-build /app/dist /app/api/static/spa
```

- [ ] **Step 5: Modify `docker-compose.yml`** — `admin-api` env pass-through

```yaml
      MLCOUNCIL_ADMIN_USERNAME: ${MLCOUNCIL_ADMIN_USERNAME:-}
      MLCOUNCIL_ADMIN_PASSWORD: ${MLCOUNCIL_ADMIN_PASSWORD:-}
      MLCOUNCIL_LEGACY_UI: ${MLCOUNCIL_LEGACY_UI:-true}
```

- [ ] **Step 6: Modify `.env.example`** — new "Security & development" section

```dotenv
# --- Security & development (unified UI) ---
MLCOUNCIL_ADMIN_USERNAME=admin
MLCOUNCIL_ADMIN_PASSWORD=change-me-long-random-password
MLCOUNCIL_LEGACY_UI=true
```

- [ ] **Step 7: Adjust legacy tests** — update `tests/test_admin_ui.py` to hit `/admin` instead of `/` where the legacy route moved.

- [ ] **Step 8: Run tests**

Run: `python -m pytest tests/test_api_auth.py tests/test_admin_ui.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add api/main.py Dockerfile docker-compose.yml .env.example tests/test_admin_ui.py tests/test_api_auth.py
git commit -m "feat(api): serve SPA at root, legacy admin at /admin, docker multi-stage"
```

## Chunk 6: Analytics frontend (charts port + pages)

### Task 18: `frontend/src/features/analytics/charts.ts` — port of `dashboard/charts.py`

**Files:**
- Create: `frontend/src/features/analytics/charts.ts`
- Test: `frontend/src/features/analytics/charts.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
// frontend/src/features/analytics/charts.test.ts
import { equityCurveChart } from "./charts";

it("builds an equity chart with two traces", () => {
  const fig = equityCurveChart(
    { dates: ["2024-01-01", "2024-01-02"], values: [100, 101] },
    { dates: ["2024-01-01", "2024-01-02"], values: [100, 99] },
  );
  expect(fig.data).toHaveLength(2);
  expect(fig.layout.title?.text).toContain("Equity");
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test` (in `frontend/`)
Expected: FAIL — module missing

- [ ] **Step 3: Port `dashboard/charts.py`** — one builder per function (12 total), same layout constants:

| charts.py fn (line) | TS export | Notes |
|---|---|---|
| `equity_curve_chart` (69) | `equityCurveChart(equity, benchmark)` | `connectgaps: false`; title "Equity & benchmark (base 100)" |
| `rolling_sharpe_chart` (161) | `rollingSharpeChart(returns, window=252)` | hline at 0 via `layout.shapes` |
| `monthly_returns_heatmap` (211) | `monthlyReturnsHeatmap(returns)` | Heatmap trace, `hovertemplate` |
| `drawdown_chart` (265) | `drawdownChart(equity)` | fill `tozeroy` |
| `model_contribution_bar` (317) | `modelContributionBar(rows, targetDate?)` | `customdata` + `hovertemplate`, vline via shape |
| `ic_rolling_chart` (389) | `icRollingChart(icHistory)` | hline 0 |
| `weight_evolution_chart` (438) | `weightEvolutionChart(weightsHistory)` | stackgroup `"one"` |
| `regime_timeline` (484) | `regimeTimeline(regimeHistory, equity?)` | 2-row subplot grid; vrect bands via shapes |
| `regime_probability_gauge` (565) | `regimeProbabilityGauge(probs)` | Indicator gauges |
| `current_weights_radar` (620) | `currentWeightsRadar(weights)` | Scatterpolar |
| `optimizer_waterfall` (671) | `optimizerWaterfall(diagnostics, topN=8)` | Waterfall trace |
| `playground_overlay_chart` (736) | `playgroundOverlayChart(curves, benchmark?, title?)` | normalizes each curve to 100 first |

Shared: `DARK_LAYOUT` (from `charts.py:23-29`), `MODEL_COLORS`, `REGIME_COLORS`.

```ts
// frontend/src/features/analytics/charts.ts
import type Plotly from "plotly.js";

export interface Series {
  dates: string[];
  values: number[];
}

export const DARK_LAYOUT = {
  paper_bgcolor: "#17181c",
  plot_bgcolor: "#17181c",
  font: { color: "#c9ccd4", size: 12 },
  margin: { l: 60, r: 20, t: 50, b: 40 },
};

export const MODEL_COLORS: Record<string, string> = { lgbm: "#4e9bde", sentiment: "#e6a23c", hmm: "#9b6bd6" };
export const REGIME_COLORS: Record<string, string> = { bull: "#3fa86d", bear: "#d15b5b", transition: "#d9a441" };

export function equityCurveChart(equity: Series, benchmark: Series) {
  return {
    data: [
      { x: equity.dates, y: equity.values, type: "scatter", mode: "lines", name: "Equity", line: { color: "#4e9bde", width: 2 } },
      { x: benchmark.dates, y: benchmark.values, type: "scatter", mode: "lines", name: "SPY", line: { color: "#8a8f9c", width: 1.5, dash: "dot" } },
    ],
    layout: { ...DARK_LAYOUT, title: { text: "Equity & benchmark (base 100)" } },
  };
}

export function drawdownChart(equity: Series) {
  const dd = equity.values.map((v, i) => {
    const peak = Math.max(...equity.values.slice(0, i + 1));
    return (v / peak - 1) * 100;
  });
  return {
    data: [{ x: equity.dates, y: dd, type: "scatter", mode: "lines", fill: "tozeroy", name: "Drawdown", line: { color: "#d15b5b" } }],
    layout: { ...DARK_LAYOUT, title: { text: "Drawdown (%)" } },
  };
}

export function rollingSharpeChart(returns: Series, window = 252) {
  const y: (number | null)[] = returns.values.map((_, i) => {
    if (i < window) return null;
    const slice = returns.values.slice(i - window, i);
    const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
    const variance = slice.reduce((a, b) => a + (b - mean) ** 2, 0) / (slice.length - 1);
    const std = Math.sqrt(variance);
    return std === 0 ? null : (mean / std) * Math.sqrt(252);
  });
  return {
    data: [{ x: returns.dates, y, type: "scatter", mode: "lines", name: "Rolling Sharpe", line: { color: "#4e9bde" } }],
    layout: { ...DARK_LAYOUT, title: { text: "Rolling Sharpe (252d)" }, shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0, line: { color: "#8a8f9c", dash: "dot" } }] },
  };
}

export function monthlyReturnsHeatmap(returns: Series) {
  const map = new Map<string, number>();
  returns.dates.forEach((d, i) => {
    map.set(d.slice(0, 7), (map.get(d.slice(0, 7)) ?? 0) + (returns.values[i] ?? 0));
  });
  const months = [...map.keys()].sort();
  return {
    data: [{ z: [months.map((m) => map.get(m) ?? 0)], x: months, y: ["Return"], type: "heatmap", colorscale: "RdYlGn" }],
    layout: { ...DARK_LAYOUT, title: { text: "Monthly returns" }, yaxis: { showticklabels: false } },
  };
}
```

Port the remaining 8 builders following the same shape and the mapping table, preserving titles, colors, and hover behavior from the Python originals (reference `dashboard/charts.py` lines in comments).

- [ ] **Step 4: Run tests**

Run: `npm test` (in `frontend/`)
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/features/analytics/charts.ts frontend/src/features/analytics/charts.test.ts
git commit -m "feat(ui): port plotly chart builders to TS"
```

### Task 19: Analytics pages (Overview / Performance / Attribution / Regime)

**Files:**
- Create: `frontend/src/pages/OverviewPage.tsx`, `frontend/src/pages/PerformancePage.tsx`, `frontend/src/pages/AttributionPage.tsx`, `frontend/src/pages/RegimePage.tsx`
- Create: `frontend/src/components/KpiCard.tsx`
- Test: `frontend/src/pages/PerformancePage.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
// frontend/src/pages/PerformancePage.test.tsx
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { server } from "../test/server";
import { PerformancePage } from "./PerformancePage";

it("renders equity chart title after data loads", async () => {
  server.use(
    http.get("/api/analytics/equity", () => HttpResponse.json({ dates: ["2024-01-01"], values: [100] })),
    http.get("/api/analytics/benchmark", () => HttpResponse.json({ dates: ["2024-01-01"], values: [100] })),
    http.get("/api/analytics/returns", () => HttpResponse.json({ dates: ["2024-01-01"], values: [0.01] })),
  );
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={qc}>
      <PerformancePage />
    </QueryClientProvider>,
  );
  expect(await screen.findByText(/Equity & benchmark/)).toBeInTheDocument();
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test` (in `frontend/`)
Expected: FAIL — page missing

- [ ] **Step 3: Write implementation**

```tsx
// frontend/src/pages/PerformancePage.tsx
import Plot from "react-plotly.js";
import { useBenchmark, useDailyReturns, useEquity } from "../api/queries";
import { drawdownChart, equityCurveChart, monthlyReturnsHeatmap, rollingSharpeChart } from "../features/analytics/charts";
import { KpiCard } from "../components/KpiCard";

export function PerformancePage() {
  const equity = useEquity();
  const benchmark = useBenchmark();
  const returns = useDailyReturns();
  const loading = equity.isLoading || benchmark.isLoading || returns.isLoading;
  const missing = [equity.error, benchmark.error, returns.error].some(
    (e) => e && (e as { status?: number }).status === 404,
  );

  if (loading) return <div className="page-empty">Loading…</div>;
  if (missing) return <div className="page-empty">No backtest results yet — run the daily pipeline first.</div>;
  if (!equity.data || !benchmark.data || !returns.data) return null;

  const eq = equity.data;
  const finalEquity = eq.values[eq.values.length - 1];
  const cagr = Math.pow(finalEquity / 100, 365 / Math.max(eq.dates.length, 1)) - 1;
  const dd = drawdownChart(eq).data[0].y as number[];
  const maxDd = Math.min(...dd);

  return (
    <div className="page">
      <h1>Performance</h1>
      <div className="kpi-row">
        <KpiCard label="Final equity (base 100)" value={finalEquity.toFixed(1)} />
        <KpiCard label="Max drawdown" value={`${maxDd.toFixed(1)}%`} />
        <KpiCard label="CAGR" value={`${(cagr * 100).toFixed(1)}%`} />
      </div>
      <Plot {...equityCurveChart(eq, benchmark.data)} style={{ width: "100%", height: 420 }} useResizeHandler />
      <Plot {...drawdownChart(eq)} style={{ width: "100%", height: 300 }} useResizeHandler />
      <Plot {...rollingSharpeChart(returns.data)} style={{ width: "100%", height: 300 }} useResizeHandler />
      <Plot {...monthlyReturnsHeatmap(returns.data)} style={{ width: "100%", height: 360 }} useResizeHandler />
    </div>
  );
}
```

```tsx
// frontend/src/components/KpiCard.tsx
export function KpiCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="kpi-card">
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
    </div>
  );
}
```

Implement `AttributionPage` (model contribution bar + IC rolling + weights evolution via `attribution`, `ic-history`, `weights-history`), `RegimePage` (gauge + timeline + radar via `regime/current`, `regime/history`, `equity`), and `OverviewPage` (KPIs from `sidebar-metrics`, alert banner is global, canary summary via `/api/canary/state`) following the same query+Plot pattern.

- [ ] **Step 4: Run tests**

Run: `npm test` (in `frontend/`)
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages frontend/src/components/KpiCard.tsx
git commit -m "feat(ui): analytics pages (overview, performance, attribution, regime)"
```

## Chunk 7: Experiments + Operations + System pages

### Task 20: Backtest page (job runner UI)

**Files:**
- Create: `frontend/src/pages/BacktestPage.tsx`, `frontend/src/features/experiments/jobs.ts`
- Create: `frontend/src/components/DataTable.tsx`, `frontend/src/components/StatusBadge.tsx`
- Test: `frontend/src/pages/BacktestPage.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
// frontend/src/pages/BacktestPage.test.tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { server } from "../test/server";
import { BacktestPage } from "./BacktestPage";

it("submits a backtest job and shows the job id", async () => {
  server.use(
    http.post("/api/experiments/backtest", () => HttpResponse.json({ job_id: "job-abc", status: "queued" })),
    http.get("/api/experiments/jobs", () =>
      HttpResponse.json({ jobs: [{ id: "job-abc", state: "running", params: { note: "test" }, created_at: "2026-08-14T00:00:00Z" }] }),
    http.get("/api/experiments/snapshots", () => HttpResponse.json({ snapshots: [] })),
  );
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={qc}>
      <BacktestPage />
    </QueryClientProvider>,
  );
  await userEvent.click(await screen.findByRole("button", { name: /run backtest/i }));
  expect(await screen.findByText(/job-abc/)).toBeInTheDocument();
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test` (in `frontend/`)
Expected: FAIL — page missing

- [ ] **Step 3: Write implementation**

```ts
// frontend/src/features/experiments/jobs.ts
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../../api/client";

export interface JobEntry {
  id: string;
  state: "queued" | "running" | "succeeded" | "cancelled" | "failed";
  params: Record<string, unknown>;
  created_at: string;
  snapshot_path?: string | null;
  error?: string;
}

export interface BacktestForm {
  start_date: string;
  end_date: string;
  universe: string[];
  initial_capital: number;
  slippage_bps: number;
  commission_bps: number;
  regime_weights: Record<string, Record<string, number>>;
  weight_clip_min: number;
  weight_clip_max: number;
  ic_rolling_window: number;
  sharpe_rolling_window: number;
  use_orthogonality: boolean;
  max_correlation: number;
  max_position: number;
  max_turnover: number;
  max_vol_ann: number;
  sector_cap: number;
  min_signal_strength: number;
  note: string;
}

export function useSubmitBacktest() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (params: BacktestForm) =>
      api<{ job_id: string; status: string }>("/api/experiments/backtest", {
        method: "POST",
        body: JSON.stringify({ params }),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["jobs"] }),
  });
}

export function useJobs(pollMs = 5000) {
  return useQuery({
    queryKey: ["jobs"],
    queryFn: () => api<{ jobs: JobEntry[] }>("/api/experiments/jobs"),
    refetchInterval: (query) => {
      const jobs = query.state.data?.jobs ?? [];
      return jobs.some((j) => j.state === "running" || j.state === "queued") ? 2000 : pollMs;
    },
  });
}

export function useCancelJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => api<JobEntry>(`/api/experiments/jobs/${jobId}/cancel`, { method: "POST" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["jobs"] }),
  });
}
```

```tsx
// frontend/src/pages/BacktestPage.tsx
import { useState } from "react";
import { useCancelJob, useJobs, useSubmitBacktest, type JobEntry } from "../features/experiments/jobs";
import { StatusBadge } from "../components/StatusBadge";
import { DataTable } from "../components/DataTable";

const DEFAULTS = {
  start_date: "2024-01-01",
  end_date: "2025-01-01",
  universe: ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
  initial_capital: 100000,
  slippage_bps: 3,
  commission_bps: 0.5,
  regime_weights: {
    bull: { lgbm: 0.55, sentiment: 0.25, hmm: 0.2 },
    bear: { lgbm: 0.35, sentiment: 0.15, hmm: 0.5 },
    transition: { lgbm: 0.45, sentiment: 0.2, hmm: 0.35 },
  },
  weight_clip_min: 0.05,
  weight_clip_max: 0.6,
  ic_rolling_window: 60,
  sharpe_rolling_window: 120,
  use_orthogonality: true,
  max_correlation: 0.65,
  max_position: 0.08,
  max_turnover: 0.2,
  max_vol_ann: 0.3,
  sector_cap: 0.45,
  min_signal_strength: 0.2,
  note: "",
};

export function BacktestPage() {
  const jobs = useJobs();
  const submit = useSubmitBacktest();
  const cancel = useCancelJob();
  const [note, setNote] = useState("");

  async function onRun() {
    await submit.mutateAsync({ ...DEFAULTS, note });
    setNote("");
  }

  const columns = ["id", "state", "created_at", "snapshot_path", "error"];
  return (
    <div className="page">
      <h1>Backtest Playground</h1>
      <p className="caption">
        Ad-hoc backtest without Alpaca: deterministic proxy signals, 5-day rebalance, snapshots persisted under
        data/results_playground/.
      </p>
      <div className="form-grid">
        <label>Start <input type="date" defaultValue={DEFAULTS.start_date} /></label>
        <label>End <input type="date" defaultValue={DEFAULTS.end_date} /></label>
        <label>Note <input value={note} onChange={(e) => setNote(e.target.value)} /></label>
        <button onClick={onRun} disabled={submit.isPending || !DEFAULTS.universe.length}>
          {submit.isPending ? "Enqueuing…" : "Run Backtest"}
        </button>
      </div>
      {submit.error && <p className="form-error">{String(submit.error)}</p>}
      <h2>Jobs</h2>
      <DataTable
        rows={jobs.data?.jobs ?? []}
        columns={columns}
        renderCell={(col, row) => {
          if (col === "state") return <StatusBadge state={row.state} />;
          if (col === "id" && (row.state === "running" || row.state === "queued")) {
            return (
              <span>
                {row.id}{" "}
                <button className="link-button" onClick={() => cancel.mutate(row.id)}>cancel</button>
              </span>
            );
          }
          return String(row[col as keyof JobEntry] ?? "");
        }}
      />
      <h2>Snapshots</h2>
      <SnapshotOverlay />
    </div>
  );
}
```

Implement `SnapshotOverlay` (fetch `/api/experiments/snapshots`; multiselect rows; fetch each `/api/experiments/snapshots/{dir}`; render `playgroundOverlayChart`), `DataTable` (generic: `rows`, `columns`, optional `renderCell`), and `StatusBadge` (state → colored `.status-badge`). The full parameter form (all `PlaygroundParams` fields: dates, capital, slippage/commission bps, regime weight sliders normalized per regime, clipping, windows, orthogonality, max_correlation, position/turnover/vol/sector caps, min_signal_strength) uses defaults from `dashboard/pages/3_Backtest_Playground.py:95-205` (English labels).

- [ ] **Step 4: Run tests**

Run: `npm test` (in `frontend/`)
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/BacktestPage.tsx frontend/src/features/experiments frontend/src/components/DataTable.tsx frontend/src/components/StatusBadge.tsx
git commit -m "feat(ui): backtest job runner page with polling and cancel"
```

### Task 21: Promotion + Canary pages

**Files:**
- Create: `frontend/src/pages/PromotionPage.tsx`, `frontend/src/pages/CanaryPage.tsx`
- Test: `frontend/src/pages/CanaryPage.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
// frontend/src/pages/CanaryPage.test.tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { server } from "../test/server";
import { CanaryPage } from "./CanaryPage";

it("shows flags and applies a pending change", async () => {
  server.use(
    http.get("/api/canary/flags", () =>
      HttpResponse.json({
        features: [{
          name: "online_learning", env: "MLCOUNCIL_ONLINE_LEARNING", value: "true",
          config_enabled: true, reverted: false, pending_enabled: null,
          effective_enabled: true, floor: 0, min_days: 5,
        }],
      })),
    http.get("/api/canary/state", () =>
      HttpResponse.json({ state_file: "x", exists: true, reverted_features: {}, pending_apply: {}, history: {} })),
    http.post("/api/canary/apply", () => HttpResponse.json({ pending_changes: [], flags: [] })),
  );
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={qc}>
      <CanaryPage />
    </QueryClientProvider>,
  );
  expect(await screen.findByText(/online_learning/)).toBeInTheDocument();
  await userEvent.click(await screen.findByRole("button", { name: /apply/i }));
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test` (in `frontend/`)
Expected: FAIL — pages missing

- [ ] **Step 3: Write implementation**

`PromotionPage` (no test needed — read-only): fetch `/api/promotion/manifest` (render `models`/`council`/`experts`/`promotion_history` tables — shapes in `config/production_manifest.yaml`), `/api/promotion/reports` (table: model, status, promotion_passed, streak.consecutive_passes, auto_promote_eligible, shadow_mode), `/api/promotion/shadow-artifacts` (existence badges + mtime). Empty states when 404 (`artifact_not_found`).

`CanaryPage`: fetch `/api/canary/flags` + `/api/canary/state`; per flag render name, env, config_enabled, reverted (red badge), pending_enabled, effective_enabled, floor/min_days; an "Apply" button toggles pending (POST `/api/canary/apply` `{name, enabled}`), a "Clear" button POSTs `/api/canary/apply/clear`; show `state.pending_apply` and `reverted_features` panels; mutations invalidate `["canary-flags"]`/`["canary-state"]` queries. Use `ConfirmDialog` (Task 22) for apply actions.

- [ ] **Step 4: Run tests**

Run: `npm test` (in `frontend/`)
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/PromotionPage.tsx frontend/src/pages/CanaryPage.tsx frontend/src/pages/CanaryPage.test.tsx
git commit -m "feat(ui): promotion and canary pages"
```

### Task 22: Operations pages (Pipeline / Trading / Intraday / Portfolio)

**Files:**
- Create: `frontend/src/pages/PipelinePage.tsx`, `frontend/src/pages/TradingPage.tsx`, `frontend/src/pages/IntradayPage.tsx`, `frontend/src/pages/PortfolioPage.tsx`
- Create: `frontend/src/components/ConfirmDialog.tsx`
- Test: `frontend/src/pages/TradingPage.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
// frontend/src/pages/TradingPage.test.tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { server } from "../test/server";
import { TradingPage } from "./TradingPage";

it("requires confirmation before execute", async () => {
  server.use(
    http.get("/api/trading/status", () => HttpResponse.json({ account: { equity: 100000, buying_power: 50000 }, positions: [], pending_orders: [] })),
    http.get("/api/trading/orders/latest", () => HttpResponse.json({ orders: [] })),
    http.post("/api/trading/execute", () => HttpResponse.json({ ok: true })),
  );
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={qc}>
      <TradingPage />
    </QueryClientProvider>,
  );
  await userEvent.click(await screen.findByRole("button", { name: /execute/i }));
  expect(await screen.findByRole("button", { name: /confirm/i })).toBeInTheDocument();
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test` (in `frontend/`)
Expected: FAIL — page missing

- [ ] **Step 3: Write implementation** — port the admin SPA panels, mapping to existing endpoints:

| Page | Endpoints (GET / POST) | Renders (from `admin.js`) |
|---|---|---|
| Pipeline | `GET /api/pipeline/status`, `GET /api/pipeline/latest-partition`, `GET /api/automation/{run_id}`, `POST /api/pipeline/run` | status KPIs, latest partition, run trigger (confirm dialog) |
| Trading | `GET /api/trading/status`, `/orders/latest`, `/orders/pending/{date}`, `/preflight/{date}`, `/reconcile/{date}`, `/history`; `POST /api/trading/execute {date}`, `/liquidate` | account KPIs, positions table, pending orders + Execute (preflight + confirm), trade history, auto-execute toggle |
| Intraday | `GET /api/intraday/status`, `/decisions/latest`, `/decisions/{id}/explain`; `POST /api/intraday/control/start\|pause\|resume\|stop`, `/cycle`, `/decisions/{id}/execute` | supervisor state, latest decision + explain, control buttons |
| Portfolio | `GET /api/portfolio/weights`, `/orders/dates`, `/orders/{date}` | weights table + order history |

`ConfirmDialog` component: `{open, title, body, confirmLabel, onConfirm, onCancel}` — used for execute, liquidate, run pipeline (inherits the confirm semantics of `admin.js`). Trading page test above covers the flow: click Execute → dialog appears → Confirm posts.

- [ ] **Step 4: Run tests**

Run: `npm test` (in `frontend/`)
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages frontend/src/components/ConfirmDialog.tsx
git commit -m "feat(ui): operations pages (pipeline, trading, intraday, portfolio)"
```

### Task 23: System pages (Config / Monitoring / Fill Quality)

**Files:**
- Create: `frontend/src/pages/ConfigPage.tsx`, `frontend/src/pages/MonitoringPage.tsx`, `frontend/src/pages/FillQualityPage.tsx`
- Modify: `api/routers/analytics.py` (+`/calibration` endpoint)
- Modify: `api/services/analytics_service.py` (+`load_cost_calibration`)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_api_analytics.py
def test_calibration_endpoint_404_when_missing(tmp_path, monkeypatch):
    from api.services import analytics_service
    monkeypatch.setattr(analytics_service, "CALIBRATION_PATH", tmp_path / "nope.json")
    client = TestClient(_app())
    resp = client.get("/api/analytics/calibration")
    assert resp.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_analytics.py::test_calibration_endpoint_404_when_missing -v`
Expected: FAIL — endpoint missing

- [ ] **Step 3: Backend additions**

```python
# api/services/analytics_service.py — add
CALIBRATION_PATH = Path(os.getenv("MLCOUNCIL_CALIBRATION_PATH", "data/operations/cost_calibration.json"))


def load_cost_calibration() -> dict:
    path = _artifact(CALIBRATION_PATH, "Cost calibration")
    import json
    return json.loads(path.read_text(encoding="utf-8"))
```

```python
# api/routers/analytics.py — add
@router.get("/calibration")
def calibration():
    return analytics_service.load_cost_calibration()
```

- [ ] **Step 4: Frontend pages**

| Page | Endpoints | Renders |
|---|---|---|
| Config | `GET/PUT /api/config/universe`, `GET /api/config/models`, `GET/PUT /api/config/regime-weights` | universe editors (ticker lists, settings, macro), models table, regime weights form; PUT with confirm |
| Monitoring | `GET /api/monitoring/alerts`, `/alerts/history?limit=30`, `/health`, `/settings`; `PUT /api/monitoring/settings {values}` | alert table, health signals, settings form (18 fields from `SETTINGS_FIELDS`, secrets masked, immutable keys disabled) |
| FillQuality | `GET /api/analytics/fill-quality`, `GET /api/analytics/calibration` | fill-quality table (median IS, lookup slippage, calibrated kappa) + calibration artifact JSON |

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_api_analytics.py -v` and `npm test` (in `frontend/`)
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add api/services/analytics_service.py api/routers/analytics.py tests/test_api_analytics.py frontend/src/pages
git commit -m "feat(ui): system pages (config, monitoring, fill quality) + calibration endpoint"
```

## Chunk 8: E2E smoke, CI, parity gate, retirement

### Task 24: Playwright smoke test

**Files:**
- Create: `frontend/playwright.config.ts`, `frontend/e2e/smoke.spec.ts`

- [ ] **Step 1: Write the smoke spec**

```ts
// frontend/playwright.config.ts
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  use: {
    baseURL: process.env.UI_BASE_URL ?? "http://localhost:8000",
    ...devices["Desktop Chrome"],
  },
  // Il test richiede l'API in esecuzione; in CI viene skippato
  grep: process.env.RUN_E2E ? undefined : /smoke/,
});
```

```ts
// frontend/e2e/smoke.spec.ts
import { expect, test } from "@playwright/test";

const USER = process.env.MLCOUNCIL_ADMIN_USERNAME ?? "admin";
const PASS = process.env.MLCOUNCIL_ADMIN_PASSWORD ?? "change-me";

test.skip(!process.env.RUN_E2E, "E2E requires the API running (RUN_E2E=1)");

test("smoke: login, navigate all sections, run a backtest job", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel("Username").fill(USER);
  await page.getByLabel("Password").fill(PASS);
  await page.getByRole("button", { name: /sign in/i }).click();
  await expect(page).toHaveURL(/\/$/);

  const sections = ["Performance", "Attribution", "Regime", "Backtest", "Promotion", "Canary", "Pipeline", "Trading", "Portfolio", "Configuration", "Monitoring", "Fill Quality"];
  for (const s of sections) {
    await page.getByRole("link", { name: new RegExp(s, "i") }).first().click();
    await expect(page.locator("h1").first()).toContainText(s, { timeout: 10_000 });
  }

  await page.getByRole("link", { name: /backtest/i }).click();
  await page.getByRole("button", { name: /run backtest/i }).click();
  await expect(page.getByText(/job-/)).toBeVisible({ timeout: 15_000 });
});
```

- [ ] **Step 2: Verify locally (manual, API running)**

Run: `RUN_E2E=1 npm run e2e` (in `frontend/`, with `python run_admin.py` running on :8000 and admin creds in env)
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add frontend/playwright.config.ts frontend/e2e/smoke.spec.ts
git commit -m "test(ui): playwright smoke covering login, sections, backtest job"
```

### Task 25: CI — frontend build + lint + tests

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Add a frontend job to `ci.yml`** (before `docker-build`)

```yaml
  frontend:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: npm
          cache-dependency-path: frontend/package-lock.json

      - name: Install
        run: npm ci
        working-directory: frontend

      - name: Lint
        run: npm run lint
        working-directory: frontend

      - name: Test
        run: npm test
        working-directory: frontend

      - name: Build
        run: npm run build
        working-directory: frontend
```

- [ ] **Step 2: Extend the ruff/mypy lint scopes** in the existing `lint` and `typecheck` jobs to cover the new backend modules:

```
api/errors.py api/session.py api/routers/auth.py api/routers/analytics.py \
api/routers/experiments.py api/routers/canary.py api/routers/promotion.py \
api/services/analytics_service.py api/services/experiment_service.py \
api/services/experiment_worker.py api/services/canary_service.py \
api/services/promotion_service.py council/canary.py
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: frontend build/lint/test job and extended backend lint scope"
```

### Task 26: Parity gate + retirement of legacy surfaces

**Files:**
- Delete: `dashboard/`, `.streamlit/`, `api/templates/admin.html`, `api/static/js/admin.js`, `api/static/css/admin.css` (after parity passes)
- Modify: `docker-compose.yml` (remove `dashboard` service), `api/main.py` (remove `/admin` route + `MLCOUNCIL_LEGACY_UI`), `.env.example`, `AGENTS.md`, `.streamlit` removal, `docs/architecture-as-is-to-be-2026-05-21.md` update

- [ ] **Step 1: Run the parity checklist** (manual, both UIs up side by side)

1. All Streamlit tabs/pages reproduced: Performance, Attribution, Regime, Fill Quality, Challenger Promotion, Backtest Playground (snapshot list + overlay) — verify each renders identical numbers.
2. All admin pages reproduced: Overview, Pipeline Control, Portfolio, Configuration, Monitoring, Trading + Intraday + Canary (new).
3. Numeric parity: `python -m pytest tests/test_analytics_parity.py -v` PASS + side-by-side visual spot-check of every chart.
4. Destructive actions keep confirm dialogs and preflight checks (execute, liquidate, run pipeline, settings PUT).
5. Links: MLflow `/mlflow/`, Dagster, Grafana — open in new tab.
6. Legacy admin still reachable at `/admin` (flag on) and the new SPA at `/`.

- [ ] **Step 2: When the checklist passes, remove the legacy surfaces**

```bash
git rm -r dashboard .streamlit api/templates/admin.html api/static/js/admin.js api/static/css/admin.css
```

- [ ] **Step 3: Modify `docker-compose.yml`** — delete the `dashboard` service block (73-96) and its `DASHBOARD_PORT` mention; keep `admin-api` as the single web entry.

- [ ] **Step 4: Modify `api/main.py`** — remove the `/admin` legacy route and the `MLCOUNCIL_LEGACY_UI` check (SPA root stays).

- [ ] **Step 5: Update `.env.example`** — remove `DASHBOARD_PORT=8501`; update the ports comment; keep the Security & development section.

- [ ] **Step 6: Update `AGENTS.md`** — Commands: remove `streamlit run dashboard/app.py` line; note "admin API serves the unified UI at http://localhost:8000 (login: MLCOUNCIL_ADMIN_USERNAME/PASSWORD)". Architecture: replace dashboard mentions with `frontend/`. Remove `requirements_dashboard.txt` note if dashboard-specific.

- [ ] **Step 7: Full verification before final commit**

Run: `python -m pytest -q` (backend) and `npm test` (frontend)
Expected: all PASS

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore: retire streamlit dashboard and legacy admin SPA after parity gate"
```

## Execution order and parallelization notes

- Chunks 1-4 (backend) are **independent file sets** and can run in parallel lanes; each lane writes its own new files and tests. Do NOT edit `api/main.py` in the lanes — router registration is a single integration step after all four chunks land (Task 5's main.py snippet applies at integration; only the `auth` router import/registration must be present for `test_session_auth_flows_through_middleware`, so if lanes are parallel, the auth lane may register only `auth` and integration adds the rest).
- Chunk 5 (frontend scaffold + auth + layout + SPA serving) must land before Chunks 6-7 (pages depend on scaffold) — same lane, sequential.
- Chunk 8 (e2e, CI, parity, retirement) is last; retirement steps only after the parity checklist passes.
- Verification per task: exact commands are listed; the full backend test set from Task 13 Step 5 is the backend gate, `npm test` the frontend gate.




