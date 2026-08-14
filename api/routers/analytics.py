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
