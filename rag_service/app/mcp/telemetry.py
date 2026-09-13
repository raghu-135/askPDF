"""Optional tracing seams for MCP calls."""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

logger = logging.getLogger(__name__)


def otel_enabled() -> bool:
    """Return whether MCP OpenTelemetry instrumentation is enabled.

    Tracing is opt-in for MCP.  Invalid values fail closed so a typo cannot
    unexpectedly enable telemetry in production.
    """
    value = os.getenv("MCP_OTEL_ENABLED", "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    logger.warning("Invalid MCP_OTEL_ENABLED=%r; MCP tracing disabled", value)
    return False


def inject_trace_context(carrier: dict[str, Any]) -> dict[str, Any]:
    """Inject W3C trace context without making telemetry mandatory."""
    if not otel_enabled():
        return carrier
    try:
        from opentelemetry import propagate
        propagate.inject(carrier)
    except Exception:
        logger.debug("Unable to inject MCP trace context", exc_info=True)
    return carrier


@asynccontextmanager
async def extracted_trace_context(carrier: dict[str, Any] | None):
    if not otel_enabled():
        yield
        return
    token = None
    try:
        from opentelemetry import context, propagate
        wire_carrier = dict(carrier or {})
        nested = wire_carrier.get("com.askpdf/runtime-context")
        if isinstance(nested, dict):
            # Runtime metadata is the single carrier location used by both
            # transports.  Keep unrelated MCP metadata intact.
            for key in ("traceparent", "tracestate"):
                if key in nested:
                    wire_carrier[key] = nested[key]
        parent = propagate.extract(wire_carrier)
        token = context.attach(parent)
    except Exception:
        logger.debug("Unable to extract MCP trace context", exc_info=True)
    try:
        yield
    finally:
        if token is not None:
            try:
                from opentelemetry import context
                context.detach(token)
            except Exception:
                logger.debug("Unable to detach MCP trace context", exc_info=True)


@asynccontextmanager
async def tool_span(name: str, **attributes: Any):
    """Start an OTel span when available; never make tracing a hard dependency."""
    if not otel_enabled():
        yield None
        return
    try:
        from opentelemetry import trace
    except ImportError:
        yield None
        return

    # Do not wrap the ``yield`` in the fallback handler. If the tool body
    # raises, asynccontextmanager throws that exception back through the yield;
    # yielding again from the exception handler causes ``generator didn't stop
    # after athrow()`` and masks the real tool failure.
    try:
        tracer = trace.get_tracer("askpdf.mcp")
        span_context = tracer.start_as_current_span(name)
    except Exception:
        logger.debug("MCP tracing unavailable", exc_info=True)
        yield None
        return

    with span_context as span:
        for key, value in attributes.items():
            if value is not None:
                try:
                    span.set_attribute(key, str(value))
                except Exception:
                    logger.debug("MCP trace attribute unavailable", exc_info=True)
        yield span
