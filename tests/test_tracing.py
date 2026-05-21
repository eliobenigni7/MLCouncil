"""Tests for observability.tracing (no-op path must not require OTel SDK)."""

from __future__ import annotations

import importlib
import os

import pytest


@pytest.fixture(autouse=True)
def _reset_tracing_state(monkeypatch):
    """Isolate env flag and module globals between tests."""
    import observability.tracing as tracing

    monkeypatch.delenv("MLCOUNCIL_OTEL_ENABLED", raising=False)
    tracing._OTEL_ENABLED = None
    tracing._INITIALIZED = False
    tracing._tracer = None
    yield
    tracing._OTEL_ENABLED = None
    tracing._INITIALIZED = False
    tracing._tracer = None


def test_tracing_disabled_by_default():
    import observability.tracing as tracing

    assert tracing.is_tracing_enabled() is False
    assert tracing.init_tracing() is False


def test_trace_span_noop_when_disabled():
    import observability.tracing as tracing

    with tracing.trace_span("test.span", layer="ingest"):
        pass
    tracing.record_span_event("noop")


def test_tracing_enabled_flag(monkeypatch):
    import observability.tracing as tracing

    monkeypatch.setenv("MLCOUNCIL_OTEL_ENABLED", "true")
    assert tracing.is_tracing_enabled() is True


def test_pipeline_imports_without_otel(monkeypatch):
    """Dagster pipeline module loads when OTel is disabled."""
    monkeypatch.setenv("MLCOUNCIL_OTEL_ENABLED", "false")
    import observability.tracing as tracing

    tracing._OTEL_ENABLED = None
    tracing._INITIALIZED = False
    mod = importlib.import_module("data.pipeline")
    assert hasattr(mod, "raw_ohlcv")
    assert hasattr(mod, "daily_orders")
