"""Distributed tracing and performance instrumentation for the realtime pipeline.

This module provides:
- Async context managers for tracing code spans
- Prometheus histogram metrics for latency tracking
- Structured logging for performance analysis
- Integration with existing metrics infrastructure
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Optional

from prometheus_client import Histogram

logger = logging.getLogger(__name__)

# Context variable for storing the current trace context
_trace_context: ContextVar[Optional["TraceContext"]] = ContextVar(
    "trace_context", default=None
)


@dataclass
class SpanMetrics:
    """Metrics collected during a span."""

    name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    attributes: dict[str, Any] = field(default_factory=dict)
    parent_name: Optional[str] = None

    def finish(self) -> float:
        """Mark span as finished and calculate duration."""
        self.end_time = time.monotonic()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        return self.duration_ms


@dataclass
class TraceContext:
    """Context for a distributed trace across the pipeline."""

    trace_id: str
    spans: list[SpanMetrics] = field(default_factory=list)
    current_span: Optional[SpanMetrics] = None

    def add_span(self, span: SpanMetrics):
        """Add a completed span to the trace."""
        self.spans.append(span)


# Prometheus histograms for different pipeline stages
# Using millisecond buckets appropriate for each stage
SPAN_DURATION = Histogram(
    "unmute_span_duration_ms",
    "Duration of traced spans in milliseconds",
    labelnames=["span_name"],
    buckets=[1, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000],
)

AUDIO_INGESTION_DURATION = Histogram(
    "unmute_audio_ingestion_duration_ms",
    "Audio ingestion and Opus decoding duration",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50],
)

STT_FLUSH_DURATION = Histogram(
    "unmute_stt_flush_duration_ms",
    "STT flush pipeline duration (pause detection to flush complete)",
    buckets=[50, 100, 200, 300, 500, 750, 1000, 1500, 2000],
)

LLM_WORD_GENERATION_DURATION = Histogram(
    "unmute_llm_word_generation_ms",
    "Time between successive LLM words",
    buckets=[1, 5, 10, 20, 50, 100, 200, 500],
)

TTS_WORD_PROCESSING_DURATION = Histogram(
    "unmute_tts_word_processing_ms",
    "TTS processing time per word",
    buckets=[10, 25, 50, 100, 200, 300, 500, 1000],
)

WEBSOCKET_SEND_DURATION = Histogram(
    "unmute_websocket_send_duration_ms",
    "WebSocket send operation duration",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50],
)

TASK_GROUP_OVERHEAD = Histogram(
    "unmute_task_group_overhead_ms",
    "Overhead from asyncio.TaskGroup operations",
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 5, 10],
)

QUEUE_WAIT_DURATION = Histogram(
    "unmute_queue_wait_duration_ms",
    "Time spent waiting on queue operations",
    labelnames=["queue_name", "operation"],
    buckets=[0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500],
)


@asynccontextmanager
async def trace_span(
    name: str,
    *,
    attributes: Optional[dict[str, Any]] = None,
    record_to_prometheus: bool = True,
    histogram: Optional[Histogram] = None,
):
    """Async context manager for tracing a span of execution.

    Usage:
        async with trace_span("stt_flush", attributes={"pause_score": 0.7}):
            await flush_stt()

    Args:
        name: Name of the span (used for metrics and logging)
        attributes: Optional dict of attributes to record with the span
        record_to_prometheus: Whether to record duration to Prometheus
        histogram: Specific histogram to use (defaults to SPAN_DURATION)
    """
    span = SpanMetrics(
        name=name,
        start_time=time.monotonic(),
        attributes=attributes or {},
    )

    # Get current trace context
    ctx = _trace_context.get()
    if ctx:
        span.parent_name = ctx.current_span.name if ctx.current_span else None
        old_span = ctx.current_span
        ctx.current_span = span
    else:
        old_span = None

    try:
        yield span
    finally:
        duration_ms = span.finish()

        # Record to Prometheus
        if record_to_prometheus:
            if histogram:
                histogram.observe(duration_ms)
            else:
                SPAN_DURATION.labels(span_name=name).observe(duration_ms)

        # Log slow spans
        if duration_ms > 100:
            logger.warning(
                f"Slow span: {name} took {duration_ms:.2f}ms",
                extra={"span": span, "duration_ms": duration_ms},
            )
        elif duration_ms > 10:
            logger.debug(
                f"Span: {name} took {duration_ms:.2f}ms",
                extra={"span": span, "duration_ms": duration_ms},
            )

        # Add to trace context
        if ctx:
            ctx.add_span(span)
            ctx.current_span = old_span


@asynccontextmanager
async def trace_queue_operation(queue_name: str, operation: str):
    """Trace time spent in queue operations.

    Args:
        queue_name: Name of the queue (e.g., "output_queue", "emit_queue")
        operation: Operation type (e.g., "put", "get")
    """
    start = time.monotonic()
    try:
        yield
    finally:
        duration_ms = (time.monotonic() - start) * 1000
        QUEUE_WAIT_DURATION.labels(queue_name=queue_name, operation=operation).observe(
            duration_ms
        )
        if duration_ms > 10:
            logger.debug(
                f"Queue operation {queue_name}.{operation} took {duration_ms:.2f}ms"
            )


def start_trace(trace_id: str) -> TraceContext:
    """Start a new trace context for a session.

    Args:
        trace_id: Unique identifier for this trace (e.g., session_id)

    Returns:
        TraceContext that will collect spans
    """
    ctx = TraceContext(trace_id=trace_id)
    _trace_context.set(ctx)
    return ctx


def get_trace_context() -> Optional[TraceContext]:
    """Get the current trace context."""
    return _trace_context.get()


def end_trace() -> Optional[TraceContext]:
    """End the current trace and return collected spans."""
    ctx = _trace_context.get()
    _trace_context.set(None)
    return ctx


def format_trace_summary(ctx: TraceContext) -> str:
    """Format a trace context into a human-readable summary.

    Args:
        ctx: TraceContext to format

    Returns:
        Formatted string with trace statistics
    """
    if not ctx.spans:
        return f"Trace {ctx.trace_id}: No spans recorded"

    lines = [f"Trace {ctx.trace_id}: {len(ctx.spans)} spans"]

    # Group spans by name
    span_groups: dict[str, list[float]] = {}
    for span in ctx.spans:
        if span.duration_ms is not None:
            if span.name not in span_groups:
                span_groups[span.name] = []
            span_groups[span.name].append(span.duration_ms)

    # Calculate statistics for each span type
    for name, durations in sorted(span_groups.items()):
        count = len(durations)
        total = sum(durations)
        avg = total / count
        min_dur = min(durations)
        max_dur = max(durations)
        lines.append(
            f"  {name}: {count} calls, avg={avg:.2f}ms, "
            f"min={min_dur:.2f}ms, max={max_dur:.2f}ms, total={total:.2f}ms"
        )

    return "\n".join(lines)


async def profile_task_group_overhead(num_tasks: int = 100) -> float:
    """Benchmark asyncio.TaskGroup overhead.

    Args:
        num_tasks: Number of minimal tasks to create

    Returns:
        Average overhead per task in milliseconds
    """
    async def minimal_task():
        pass

    start = time.monotonic()
    async with asyncio.TaskGroup() as tg:
        for _ in range(num_tasks):
            tg.create_task(minimal_task())
    duration = time.monotonic() - start

    overhead_per_task = (duration * 1000) / num_tasks
    TASK_GROUP_OVERHEAD.observe(overhead_per_task)

    logger.info(
        f"TaskGroup overhead: {overhead_per_task:.4f}ms per task ({num_tasks} tasks)"
    )
    return overhead_per_task
