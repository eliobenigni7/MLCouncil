from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded
from slowapi.extension import _rate_limit_exceeded_handler
from runtime_env import load_runtime_env

from api.auth import is_api_key_required, is_public_api_path
from api.rate_limit import limiter

API_PREFIX = "/api"
STATIC_DIR = Path(__file__).parent / "static"
SPA_DIST_DIR = Path(__file__).resolve().parents[1] / "api" / "static" / "spa"

load_runtime_env()


def get_allowed_origins() -> list[str]:
    origins = os.getenv("MLCOUNCIL_ALLOWED_ORIGINS", "")
    return [o.strip() for o in origins.split(",") if o.strip()]


def create_app() -> FastAPI:
    app = FastAPI(
        title="MLCouncil Admin API",
        version="0.1.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    from api.errors import ApiError, api_error_handler

    app.add_exception_handler(ApiError, api_error_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_allowed_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    )

    @app.middleware("http")
    async def validate_access(request: Request, call_next):
        if request.url.path.startswith("/api/"):
            if is_public_api_path(request.url.path):
                return await call_next(request)
            from api.session import check_csrf, is_session_valid, request_csrf_token, request_session_token
            session_token = request_session_token(request)
            if session_token and is_session_valid(session_token):
                # CSRF vale solo per le sessioni browser (cookie). Le richieste via
                # API key o in modalità permissiva (nessuna sessione) non lo richiedono.
                if request.method in {"POST", "PUT", "DELETE"}:
                    submitted = request.headers.get("X-CSRF-Token")
                    if not check_csrf(request_csrf_token(request), submitted):
                        from fastapi.responses import JSONResponse
                        return JSONResponse(
                            status_code=403,
                            content={"error": {"code": "csrf_failed", "message": "CSRF token mismatch", "detail": ""}},
                        )
            else:
                # Nessuna sessione: la richiesta deve superare il gate API key.
                # Manteniamo la forma legacy {"detail": ...} per non rompere i
                # consumer esterni dell'API key (contratto D8).
                from api.auth import ensure_request_api_key
                try:
                    ensure_request_api_key(request)
                except Exception as exc:  # noqa: BLE001
                    from fastapi.responses import JSONResponse
                    return JSONResponse(
                        status_code=getattr(exc, "status_code", 401),
                        content={"detail": getattr(exc, "detail", str(exc))},
                    )
        return await call_next(request)

    @app.on_event("startup")
    async def validate_environment():
        if is_api_key_required() and not os.getenv("MLCOUNCIL_API_KEY"):
            raise RuntimeError(
                "MLCOUNCIL_API_KEY is required for this runtime profile. "
                "Refusing startup to avoid unauthenticated admin access."
            )
        if is_api_key_required():
            if not os.getenv("MLCOUNCIL_ADMIN_USERNAME") or not os.getenv("MLCOUNCIL_ADMIN_PASSWORD"):
                raise RuntimeError(
                    "MLCOUNCIL_ADMIN_USERNAME and MLCOUNCIL_ADMIN_PASSWORD are required "
                    "for this runtime profile. Refusing startup to avoid unauthenticated admin access."
                )
        # Recupero job esperimenti orfani (running -> failed) e prune registro.
        from api.services import experiment_service
        try:
            experiment_service.boot_sweep()
        except Exception:  # noqa: BLE001 — mai bloccare lo startup per la pulizia
            import traceback
            traceback.print_exc()

    from api.routers import (analytics, auth, canary, config, experiments, health,
                             intraday, monitoring, pipeline, portfolio, promotion,
                             trading)

    app.include_router(auth.router, prefix=API_PREFIX)
    app.include_router(analytics.router, prefix=API_PREFIX)
    app.include_router(experiments.router, prefix=API_PREFIX)
    app.include_router(canary.router, prefix=API_PREFIX)
    app.include_router(promotion.router, prefix=API_PREFIX)
    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(pipeline.router, prefix=API_PREFIX)
    app.include_router(portfolio.router, prefix=API_PREFIX)
    app.include_router(config.router, prefix=API_PREFIX)
    app.include_router(monitoring.router, prefix=API_PREFIX)
    app.include_router(trading.router, prefix=API_PREFIX)
    app.include_router(intraday.router, prefix=API_PREFIX)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # SPA unificata: servita a /app (statici) con fallback client-side routing.
    if (SPA_DIST_DIR / "index.html").exists():
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

    return app


app = create_app()
