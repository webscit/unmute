# Realtime Pipeline Profiling Guide

This guide covers the performance instrumentation, profiling, and optimization workflow for the Unmute realtime pipeline.

## Overview

The realtime pipeline has comprehensive performance instrumentation:

1. **Distributed Tracing**: Async context managers for timing code spans
2. **Prometheus Metrics**: Histograms for latency tracking across all pipeline stages
3. **pyinstrument Profiling**: CPU profiling for identifying bottlenecks
4. **Conformance Harness**: Load testing with latency benchmarks

## Quick Start

### Run a performance profile:

```bash
python scripts/profile_pipeline.py
```

This will:
- Enable profiling in the server
- Start the server
- Run conformance harness tests
- Collect metrics
- Generate a performance report

### View results:

- `performance_report.md` - Comprehensive metrics analysis
- `profile.html` - Interactive pyinstrument flamegraph (open in browser)
- `metrics_snapshot.json` - Raw Prometheus metrics for comparison

## Architecture

### Tracing Infrastructure

Location: `unmute/tracing.py`

The tracing module provides async context managers for instrumenting code:

```python
from unmute.tracing import trace_span, AUDIO_INGESTION_DURATION

async def process_audio(opus_bytes: bytes):
    async with trace_span(
        "opus_decode",
        histogram=AUDIO_INGESTION_DURATION,
        attributes={"bytes": len(opus_bytes)}
    ):
        pcm = await decode_opus(opus_bytes)
    return pcm
```

**Features:**
- Zero-overhead when profiling is disabled
- Automatic Prometheus metric recording
- Structured logging for slow spans (>100ms warning, >10ms debug)
- Hierarchical span tracking with parent/child relationships

### Instrumented Pipeline Stages

All critical pipeline stages are instrumented:

#### 1. Audio Ingestion (main_websocket.py, realtime_buffer.py)
- WebSocket receive operations
- Opus decoding (thread pool execution)
- Audio buffer management
- **Metrics**: `unmute_audio_ingestion_duration_ms`

#### 2. STT Flush Pipeline (unmute_handler.py:437-476)
- VAD-based pause detection
- Flush trigger and zero-padding
- STT completion
- **Metrics**: `unmute_stt_flush_duration_ms`, `worker_stt_ttft`, `worker_vad_speech_duration`

#### 3. LLM Streaming (unmute_handler.py:277-302)
- Word generation from LLM
- Inter-word latency tracking
- Interruption handling
- **Metrics**: `unmute_llm_word_generation_ms`, `worker_vllm_ttft`, `worker_vllm_gen_duration`

#### 4. TTS Processing (unmute_handler.py:302)
- Word-to-TTS streaming
- Audio synthesis latency
- **Metrics**: `unmute_tts_word_processing_ms`, `worker_tts_ttft`, `worker_tts_gen_duration`

#### 5. WebSocket Emit Loop (main_websocket.py:827-836)
- Event serialization
- WebSocket send operations
- Network latency
- **Metrics**: `unmute_websocket_send_duration_ms`, `worker_emit_queue_size`

#### 6. Queue Operations (All modules)
- Output queue wait times
- Emit queue wait times
- Queue depth monitoring
- **Metrics**: `unmute_queue_wait_duration_ms`, `worker_output_queue_size`

### Per-Session Tracing

Each WebSocket session gets its own distributed trace context:

```python
# Start trace (main_websocket.py:338)
session_id = ora.random_id("sess")
start_trace(session_id)

# Spans are automatically tracked
async with trace_span("llm_word_stream"):
    await process_word(word)

# End trace and log summary (main_websocket.py:357)
trace_ctx = end_trace()
if trace_ctx:
    logger.info("Session trace summary:\n%s", format_trace_summary(trace_ctx))
```

The trace summary shows:
- Total spans recorded
- Per-span-type statistics (count, avg, min, max, total)

## Prometheus Metrics

### Key Metrics

Access metrics at `http://localhost:8000/metrics`

**Latency Histograms:**
- `unmute_span_duration_ms{span_name="..."}` - Generic span timing
- `unmute_audio_ingestion_duration_ms` - Opus decode latency
- `unmute_stt_flush_duration_ms` - STT flush pipeline latency
- `unmute_llm_word_generation_ms` - Inter-word LLM latency
- `unmute_tts_word_processing_ms` - TTS word processing latency
- `unmute_websocket_send_duration_ms` - WebSocket send latency
- `unmute_queue_wait_duration_ms{queue_name="...", operation="..."}` - Queue wait times

**TTFT (Time To First Token):**
- `worker_stt_ttft` - STT first word latency
- `worker_vllm_ttft` - LLM first token latency
- `worker_tts_ttft` - TTS first audio latency

**Queue Depth Gauges:**
- `worker_output_queue_size` - Handler output queue depth
- `worker_emit_queue_size` - WebSocket emit queue depth

**VAD Events:**
- `worker_vad_speech_started` - Speech start events
- `worker_vad_speech_stopped` - Speech stop events
- `worker_vad_speech_duration` - Speech segment durations

**Error Counters:**
- `worker_buffer_overflow_errors` - Audio buffer overflows
- `worker_backpressure_errors` - Queue backpressure errors
- `worker_silence_timeout_errors` - Silence timeout errors

### Querying Metrics

**Average latency over last 5 minutes:**
```promql
rate(unmute_audio_ingestion_duration_ms_sum[5m]) /
rate(unmute_audio_ingestion_duration_ms_count[5m])
```

**95th percentile TTFT:**
```promql
histogram_quantile(0.95, rate(worker_vllm_ttft_bucket[5m]))
```

**Queue depth over time:**
```promql
worker_output_queue_size
```

## pyinstrument Profiling

### Enable profiling:

Edit `unmute/main_websocket.py`:
```python
PROFILE_ACTIVE = True  # Change from False
```

Or use the profiling script which does this automatically.

### Access profile:

While server is running with profiling enabled:
```
http://localhost:8000/profile
```

This returns an interactive HTML flamegraph showing:
- Function call hierarchy
- Time spent in each function
- Hot paths through the code

### Interpreting profiles:

**Look for:**
- Wide bars = expensive functions
- Tall stacks = deep call chains (may need optimization)
- Unexpected GIL contention (synchronous I/O, CPU-bound work)
- Redundant operations in hot paths

**Common bottlenecks:**
- Synchronous I/O blocking event loop
- Large JSON serialization
- CPU-bound operations on event loop
- Queue contention
- Lock contention

## Load Testing

### Using the conformance harness:

```bash
# Run all fixtures
python -m tests.realtime_harness.runner \
    --url ws://localhost:8000/v1/realtime \
    --benchmark

# Run specific fixture
python -m tests.realtime_harness.runner \
    --url ws://localhost:8000/v1/realtime \
    --fixture tests/realtime_harness/fixtures/audio_streaming.json \
    --benchmark

# Run with concurrency
python -m tests.realtime_harness.runner \
    --url ws://localhost:8000/v1/realtime \
    --concurrent 8 \
    --benchmark
```

### Using the profiling script:

```bash
# Profile with 4 concurrent sessions
python scripts/profile_pipeline.py --concurrent-sessions 4

# Profile specific fixtures
python scripts/profile_pipeline.py \
    --fixtures tests/realtime_harness/fixtures/*.json

# Compare against baseline
python scripts/profile_pipeline.py \
    --baseline-metrics baseline_metrics.json \
    --output comparison_report.md
```

## Optimization Workflow

### 1. Establish Baseline

```bash
# Run baseline profiling
python scripts/profile_pipeline.py --output baseline_report.md

# Save baseline metrics
cp metrics_snapshot.json baseline_metrics.json
```

### 2. Identify Bottlenecks

**Review metrics:**
- Check TTFT percentiles - are they within thresholds?
- Check queue depths - are they building up?
- Check error rates - any backpressure/overflow errors?

**Review profile:**
- Open `profile.html` in browser
- Look for wide bars (expensive operations)
- Check for unexpected synchronous I/O
- Identify hot paths

**Review logs:**
- Check for slow span warnings (>100ms)
- Look for queue operation warnings (>10ms)

### 3. Implement Optimizations

**Common optimizations:**

**Offload CPU-bound work:**
```python
# Before: Blocking event loop
result = expensive_computation(data)

# After: Run in thread pool
result = await asyncio.to_thread(expensive_computation, data)
```

**Reduce queue contention:**
```python
# Before: High contention
async def process():
    for item in items:
        await queue.put(item)

# After: Batch operations
async def process():
    await queue.put_batch(items)
```

**Optimize serialization:**
```python
# Before: Slow JSON serialization
json_str = json.dumps(large_dict)

# After: Use orjson or msgpack
json_str = orjson.dumps(large_dict)
```

**Add bounded queues:**
```python
# Before: Unbounded queue
queue = asyncio.Queue()

# After: Bounded with backpressure
queue = asyncio.Queue(maxsize=100)
```

### 4. Measure Improvements

```bash
# Run profiling with same parameters
python scripts/profile_pipeline.py \
    --baseline-metrics baseline_metrics.json \
    --output optimized_report.md

# Compare reports
diff baseline_report.md optimized_report.md
```

### 5. Document Results

Update this document with:
- What was optimized
- Before/after metrics
- Lessons learned

## Troubleshooting

### High queue depths

**Symptom:** `worker_output_queue_size` or `worker_emit_queue_size` > 50

**Possible causes:**
- Downstream consumer too slow (STT, TTS, LLM, WebSocket)
- Network congestion
- CPU saturation
- GIL contention

**Investigation:**
1. Check which queue is building up
2. Profile to find the consumer bottleneck
3. Check network latency (for WebSocket/service calls)
4. Monitor CPU usage

### Slow TTFT

**Symptom:** `worker_vllm_ttft` or `worker_stt_ttft` p95 > threshold

**Possible causes:**
- Service startup latency
- Network latency to service
- Service overload
- Cold start penalties

**Investigation:**
1. Check service health endpoints
2. Profile service startup path
3. Check service logs for slow requests
4. Monitor service resource usage

### Audio buffer overflows

**Symptom:** `worker_buffer_overflow_errors` > 0

**Possible causes:**
- Client sending too fast
- Server processing too slow
- Pause detection not working

**Investigation:**
1. Check `unmute_audio_ingestion_duration_ms` - is Opus decoding slow?
2. Check if STT is keeping up - look at `worker_stt_sent_frames` vs `worker_stt_recv_frames`
3. Review VAD/pause detection logic

### Backpressure errors

**Symptom:** `worker_backpressure_errors` > 0

**Possible causes:**
- Output queue saturated (>100 items)
- Consumer blocked or too slow
- Upstream producing too fast

**Investigation:**
1. Profile the emit loop - where is time being spent?
2. Check WebSocket send latency
3. Check if any async operations are unexpectedly synchronous

## Performance Targets

Based on conformance harness thresholds:

| Metric | Target | Notes |
|--------|--------|-------|
| TTFT (overall) | < 1000ms | Time from response created to first content |
| STT TTFT | < 100ms | First word from STT after audio starts |
| LLM TTFT | < 300ms | First token from LLM |
| TTS TTFT | < 500ms | First audio from TTS |
| STT flush latency | < 300ms | Buffer commit to transcription complete |
| Tool call RTT | < 2000ms | Function call to result received |
| Opus decode | < 10ms | Per-frame decoding latency |
| WebSocket send | < 5ms | Event serialization and send |
| Queue wait time | < 10ms | Time spent waiting on queue ops |

## Advanced Topics

### Custom Histograms

To add a new performance metric:

```python
from prometheus_client import Histogram

MY_OPERATION_DURATION = Histogram(
    "unmute_my_operation_duration_ms",
    "Duration of my operation in milliseconds",
    buckets=[1, 5, 10, 25, 50, 100, 200, 500],
)

# In your code:
from unmute.tracing import trace_span

async def my_operation():
    async with trace_span("my_operation", histogram=MY_OPERATION_DURATION):
        # ... operation code ...
        pass
```

### Task Group Overhead

To measure asyncio.TaskGroup overhead:

```python
from unmute.tracing import profile_task_group_overhead

overhead_ms = await profile_task_group_overhead(num_tasks=100)
print(f"TaskGroup overhead: {overhead_ms:.4f}ms per task")
```

### GIL Contention Analysis

If profiling shows GIL contention:

1. Identify synchronous operations in hot paths
2. Offload to thread pool with `asyncio.to_thread()`
3. Consider process-based parallelism for CPU-bound work
4. Use `py-spy` for multi-threaded profiling:

```bash
py-spy record -o profile.svg -native -format speedscope -- python -m uvicorn unmute.main_websocket:app
```

### Memory Profiling

For memory leaks or excessive memory usage:

```bash
# Install memory_profiler
pip install memory_profiler

# Profile specific function
python -m memory_profiler unmute/main_websocket.py

# Or use memray
pip install memray
memray run unmute/main_websocket.py
memray flamegraph output.bin
```

## References

- [Prometheus Python Client](https://github.com/prometheus/client_python)
- [pyinstrument](https://github.com/joerick/pyinstrument)
- [asyncio Performance Tips](https://docs.python.org/3/library/asyncio-dev.html#asyncio-debug-mode)
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime)

## Changelog

### 2026-01-26: Initial instrumentation
- Added distributed tracing infrastructure (`unmute/tracing.py`)
- Instrumented audio ingestion, STT flush, LLM streaming, TTS processing, WebSocket emit
- Added Prometheus histograms for all pipeline stages
- Created profiling script (`scripts/profile_pipeline.py`)
- Documented profiling workflow

### Future Work
- Add OpenTelemetry integration for distributed tracing across services
- Implement continuous performance monitoring with alerting
- Add GPU profiling for VLLM/TTS models
- Create performance regression CI checks
