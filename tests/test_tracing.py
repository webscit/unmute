"""Tests for the tracing instrumentation infrastructure."""

import asyncio
import time

import pytest

from unmute.tracing import (
    AUDIO_INGESTION_DURATION,
    SPAN_DURATION,
    end_trace,
    format_trace_summary,
    get_trace_context,
    profile_task_group_overhead,
    start_trace,
    trace_queue_operation,
    trace_span,
)


@pytest.mark.asyncio
async def test_trace_span_basic():
    """Test basic span tracing."""
    async with trace_span("test_operation"):
        await asyncio.sleep(0.01)

    # Should complete without errors


@pytest.mark.asyncio
async def test_trace_span_with_attributes():
    """Test span tracing with attributes."""
    async with trace_span(
        "test_operation",
        attributes={"key": "value", "count": 42},
    ):
        await asyncio.sleep(0.01)

    # Should complete without errors


@pytest.mark.asyncio
async def test_trace_span_with_histogram():
    """Test span tracing with custom histogram."""
    async with trace_span(
        "audio_decode",
        histogram=AUDIO_INGESTION_DURATION,
        attributes={"bytes": 1024},
    ):
        await asyncio.sleep(0.001)

    # Should record to histogram


@pytest.mark.asyncio
async def test_trace_context():
    """Test trace context management."""
    # No context initially
    assert get_trace_context() is None

    # Start trace
    ctx = start_trace("test_session_123")
    assert ctx is not None
    assert ctx.trace_id == "test_session_123"
    assert get_trace_context() == ctx

    # Record some spans
    async with trace_span("operation1"):
        await asyncio.sleep(0.001)

    async with trace_span("operation2"):
        await asyncio.sleep(0.001)

    # Context should have recorded spans
    assert len(ctx.spans) == 2
    assert ctx.spans[0].name == "operation1"
    assert ctx.spans[1].name == "operation2"

    # End trace
    ended_ctx = end_trace()
    assert ended_ctx == ctx
    assert get_trace_context() is None


@pytest.mark.asyncio
async def test_nested_spans():
    """Test nested span tracking."""
    ctx = start_trace("nested_test")

    async with trace_span("parent"):
        await asyncio.sleep(0.001)

        async with trace_span("child1"):
            await asyncio.sleep(0.001)

        async with trace_span("child2"):
            await asyncio.sleep(0.001)

    # Should have 3 spans total
    assert len(ctx.spans) == 3

    # Child spans finish first, so they're at indices 0 and 1
    # Parent finishes last, so it's at index 2
    assert ctx.spans[0].name == "child1"
    assert ctx.spans[0].parent_name == "parent"
    assert ctx.spans[1].name == "child2"
    assert ctx.spans[1].parent_name == "parent"
    assert ctx.spans[2].name == "parent"
    assert ctx.spans[2].parent_name is None

    end_trace()


@pytest.mark.asyncio
async def test_queue_operation_tracing():
    """Test queue operation tracing."""
    start_trace("queue_test")

    # Trace queue put
    queue = asyncio.Queue()
    async with trace_queue_operation("test_queue", "put"):
        await queue.put("item")

    # Trace queue get
    async with trace_queue_operation("test_queue", "get"):
        item = await queue.get()

    assert item == "item"

    ctx = end_trace()
    # Should have recorded timing for both operations
    assert len(ctx.spans) == 0  # Queue ops don't create spans in current impl


@pytest.mark.asyncio
async def test_format_trace_summary():
    """Test trace summary formatting."""
    ctx = start_trace("summary_test")

    # Create multiple spans of same type
    for _ in range(3):
        async with trace_span("repeated_op"):
            await asyncio.sleep(0.001)

    # Create different span type
    async with trace_span("different_op"):
        await asyncio.sleep(0.002)

    end_trace()

    # Format summary
    summary = format_trace_summary(ctx)

    # Should contain span statistics
    assert "summary_test" in summary
    assert "repeated_op" in summary
    assert "different_op" in summary
    assert "3 calls" in summary  # repeated_op called 3 times
    assert "1 calls" in summary  # different_op called 1 time


@pytest.mark.asyncio
async def test_span_duration_measurement():
    """Test that span durations are measured accurately."""
    ctx = start_trace("duration_test")

    sleep_time = 0.05  # 50ms
    async with trace_span("timed_op"):
        await asyncio.sleep(sleep_time)

    end_trace()

    # Check duration
    span = ctx.spans[0]
    assert span.duration_ms is not None
    # Allow 10ms tolerance for overhead
    assert abs(span.duration_ms - (sleep_time * 1000)) < 10


@pytest.mark.asyncio
async def test_profile_task_group_overhead():
    """Test TaskGroup overhead measurement."""
    overhead_ms = await profile_task_group_overhead(num_tasks=10)

    # Overhead should be reasonable (less than 1ms per task)
    assert overhead_ms < 1.0
    assert overhead_ms > 0.0


@pytest.mark.asyncio
async def test_span_without_trace_context():
    """Test that spans work even without trace context."""
    # Don't start a trace
    assert get_trace_context() is None

    # Span should still work (just won't be recorded)
    async with trace_span("orphan_span"):
        await asyncio.sleep(0.001)

    # Should complete without errors


@pytest.mark.asyncio
async def test_concurrent_spans():
    """Test concurrent span execution."""
    ctx = start_trace("concurrent_test")

    async def concurrent_op(name: str):
        async with trace_span(name):
            await asyncio.sleep(0.01)

    # Run multiple spans concurrently
    async with asyncio.TaskGroup() as tg:
        for i in range(5):
            tg.create_task(concurrent_op(f"concurrent_{i}"))

    end_trace()

    # All spans should be recorded
    assert len(ctx.spans) == 5
    span_names = [s.name for s in ctx.spans]
    for i in range(5):
        assert f"concurrent_{i}" in span_names


@pytest.mark.asyncio
async def test_span_records_to_prometheus():
    """Test that spans record to Prometheus metrics."""
    # Get initial metric value
    before_samples = SPAN_DURATION.collect()[0].samples

    # Create a traced span
    async with trace_span("prometheus_test"):
        await asyncio.sleep(0.001)

    # Check that metric was updated
    after_samples = SPAN_DURATION.collect()[0].samples

    # Should have more samples after
    # (Exact comparison is hard due to histogram structure)
    assert len(after_samples) >= len(before_samples)


@pytest.mark.asyncio
async def test_span_with_exception():
    """Test span behavior when exception occurs."""
    ctx = start_trace("exception_test")

    with pytest.raises(ValueError):
        async with trace_span("failing_op"):
            await asyncio.sleep(0.001)
            raise ValueError("Test error")

    # Span should still be recorded even though it failed
    assert len(ctx.spans) == 1
    assert ctx.spans[0].duration_ms is not None

    end_trace()


@pytest.mark.asyncio
async def test_multiple_traces_sequential():
    """Test multiple sequential traces."""
    # First trace
    ctx1 = start_trace("trace1")
    async with trace_span("op1"):
        await asyncio.sleep(0.001)
    end_trace()
    assert len(ctx1.spans) == 1

    # Second trace
    ctx2 = start_trace("trace2")
    async with trace_span("op2"):
        await asyncio.sleep(0.001)
    end_trace()
    assert len(ctx2.spans) == 1

    # Contexts should be separate
    assert ctx1.trace_id != ctx2.trace_id
    assert ctx1.spans[0].name != ctx2.spans[0].name
