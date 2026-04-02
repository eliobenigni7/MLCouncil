from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

API_PREFIX = "/api"
STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_app() -> FastAPI:
    app = FastAPI(
        title="MLCouncil Admin API",
        version="0.1.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from api.routers import health, pipeline, portfolio, config, monitoring

    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(pipeline.router, prefix=API_PREFIX)
    app.include_router(portfolio.router, prefix=API_PREFIX)
    app.include_router(config.router, prefix=API_PREFIX)
    app.include_router(monitoring.router, prefix=API_PREFIX)

    return app


app = create_app()
