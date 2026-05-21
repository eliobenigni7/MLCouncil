"""MLCouncil observability helpers (OpenTelemetry tracing)."""

from observability.tracing import (
    init_tracing,
    is_tracing_enabled,
    record_span_event,
    trace_span,
)

__all__ = [
    "init_tracing",
    "is_tracing_enabled",
    "record_span_event",
    "trace_span",
]
