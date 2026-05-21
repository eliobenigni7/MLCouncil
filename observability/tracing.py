"""OpenTelemetry tracing for MLCouncil (no-op when disabled or SDK missing).

Enable with ``MLCOUNCIL_OTEL_ENABLED=true`` and optional OTLP endpoint vars.
When disabled or packages are absent, :func:`trace_span` is a zero-cost context manager.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

_OTEL_ENABLED: bool | None = None
_INITIALIZED = False
_tracer: Any = None


def is_tracing_enabled() -> bool:
    """Return True when OTel export is requested via env flag."""
    global _OTEL_ENABLED
    if _OTEL_ENABLED is None:
        raw = os.getenv("MLCOUNCIL_OTEL_ENABLED", "").strip().lower()
        _OTEL_ENABLED = raw in {"1", "true", "yes", "on"}
    return _OTEL_ENABLED


def _sdk_available() -> bool:
    try:
        import opentelemetry  # noqa: F401
        import opentelemetry.sdk  # noqa: F401
    except ImportError:
        return False
    return True


def init_tracing(service_name: str | None = None) -> bool:
    """Configure OTLP HTTP exporter and global tracer provider.

    Returns True when a real tracer was configured, False for no-op mode.
    Safe to call multiple times; only the first successful call configures SDK state.
    """
    global _INITIALIZED, _tracer

    if _INITIALIZED:
        return _tracer is not None

    _INITIALIZED = True

    if not is_tracing_enabled() or not _sdk_available():
        _tracer = None
        return False

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resolved_name = (
        service_name
        or os.getenv("OTEL_SERVICE_NAME")
        or os.getenv("MLCOUNCIL_OTEL_SERVICE_NAME")
        or "mlcouncil"
    )
    endpoint = os.getenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318/v1/traces"),
    )

    resource = Resource.create(
        {
            "service.name": resolved_name,
            "service.namespace": os.getenv("MLCOUNCIL_OTEL_NAMESPACE", "mlcouncil"),
            "deployment.environment": os.getenv(
                "MLCOUNCIL_ENV_PROFILE",
                os.getenv("OTEL_DEPLOYMENT_ENVIRONMENT", "local"),
            ),
        }
    )
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("mlcouncil.tracing", "1.0.0")
    return True


def _normalize_attributes(attributes: dict[str, Any] | None) -> dict[str, Any]:
    if not attributes:
        return {}
    out: dict[str, Any] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
        else:
            out[key] = str(value)
    return out


@contextmanager
def trace_span(
    name: str,
    *,
    layer: str | None = None,
    asset: str | None = None,
    partition_date: str | None = None,
    **attributes: Any,
) -> Iterator[None]:
    """Lightweight span wrapper; no-op when tracing is off or SDK unavailable."""
    if not is_tracing_enabled():
        yield
        return

    if _tracer is None and not init_tracing():
        yield
        return

    merged = _normalize_attributes(attributes)
    if layer is not None:
        merged["mlcouncil.layer"] = layer
    if asset is not None:
        merged["dagster.asset"] = asset
    if partition_date is not None:
        merged["dagster.partition"] = partition_date

    with _tracer.start_as_current_span(name, attributes=merged):
        yield


def record_span_event(name: str, **attributes: Any) -> None:
    """Attach an event to the current span when tracing is active."""
    if not is_tracing_enabled() or _tracer is None:
        return
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span is None or not span.is_recording():
            return
        span.add_event(name, attributes=_normalize_attributes(attributes))
    except Exception:
        return
