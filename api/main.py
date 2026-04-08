from __future__ import annotations

import os
import secrets
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from slowapi import Limiter
from slowapi.util import get_remote_address
from runtime_env import load_runtime_env

API_PREFIX = "/api"
STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"

load_runtime_env()

limiter = Limiter(key_func=get_remote_address)


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
            if request.url.path in ["/api/health", "/api/docs", "/api/openapi.json"]:
                return await call_next(request)
            valid_key = os.getenv("MLCOUNCIL_API_KEY", "")
            if not valid_key:
                return await call_next(request)
            api_key = request.headers.get("X-API-Key")
            if not api_key:
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Missing X-API-Key header"}
                )
            if not secrets.compare_digest(api_key, valid_key):
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Invalid API key"}
                )
        return await call_next(request)

    @app.on_event("startup")
    async def validate_environment():
        required_vars = ["MLCOUNCIL_API_KEY"]
        missing = [v for v in required_vars if not os.getenv(v)]
        if missing:
            import warnings
            warnings.warn(f"Missing environment variables: {missing}. API may not be fully functional.")

    from api.routers import health, pipeline, portfolio, config, monitoring, trading

    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(pipeline.router, prefix=API_PREFIX)
    app.include_router(portfolio.router, prefix=API_PREFIX)
    app.include_router(config.router, prefix=API_PREFIX)
    app.include_router(monitoring.router, prefix=API_PREFIX)
    app.include_router(trading.router, prefix=API_PREFIX)

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

