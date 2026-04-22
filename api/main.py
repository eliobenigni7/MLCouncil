from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger
from slowapi.errors import RateLimitExceeded
from slowapi.extension import _rate_limit_exceeded_handler
from runtime_env import load_runtime_env

from api.auth import ensure_request_api_key, is_api_key_required, is_public_api_path
from api.rate_limit import limiter

API_PREFIX = "/api"
STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"

load_runtime_env()


def get_allowed_origins() -> list[str]:
    origins = os.getenv("MLCOUNCIL_ALLOWED_ORIGINS", "http://localhost:8501")
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

    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_allowed_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    )

    @app.middleware("http")
    async def validate_api_key(request: Request, call_next):
        if request.url.path.startswith("/api/"):
            if is_public_api_path(request.url.path):
                return await call_next(request)
            try:
                ensure_request_api_key(request)
                logger.info(
                    "Admin API auth success path={} client={}",
                    request.url.path,
                    request.client.host if request.client else "unknown",
                )
            except Exception as exc:  # noqa: BLE001
                from fastapi.responses import JSONResponse

                logger.warning(
                    "Admin API auth failure path={} client={} status={} detail={}",
                    request.url.path,
                    request.client.host if request.client else "unknown",
                    getattr(exc, "status_code", 500),
                    getattr(exc, "detail", str(exc)),
                )
                return JSONResponse(
                    status_code=getattr(exc, "status_code", 500),
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

    from api.routers import health, pipeline, portfolio, config, monitoring, trading, intraday

    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(pipeline.router, prefix=API_PREFIX)
    app.include_router(portfolio.router, prefix=API_PREFIX)
    app.include_router(config.router, prefix=API_PREFIX)
    app.include_router(monitoring.router, prefix=API_PREFIX)
    app.include_router(trading.router, prefix=API_PREFIX)
    app.include_router(intraday.router, prefix=API_PREFIX)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    @app.get("/", response_class=HTMLResponse)
    async def admin_ui(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="admin.html",
        )

    return app


app = create_app()
