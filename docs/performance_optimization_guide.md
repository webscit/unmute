# Performance Optimization Guide

This document provides a structured approach to identifying and mitigating performance bottlenecks in the Unmute realtime pipeline.

## Common Bottlenecks and Mitigations

### 1. GIL Contention

**Symptoms:**
- pyinstrument profile shows time spent in GIL acquisition
- Poor scaling with concurrent sessions
- CPU-bound operations blocking event loop

**Causes:**
- Synchronous I/O operations (file reads, network calls)
- CPU-intensive operations (serialization, encoding, computation)
- Long-running synchronous functions

**Mitigations:**

#### A. Offload to Thread Pool
```python
# Before: Blocks event loop
pcm = opus_reader.decode(opus_bytes)

# After: Runs in thread pool
pcm = await asyncio.to_thread(opus_reader.decode, opus_bytes)
```

**Status:** ✅ Already implemented in `realtime_buffer.py:353`

#### B. Use Async-Native Libraries
```python
# Before: Synchronous HTTP with requests
response = requests.get(url)

# After: Async HTTP with aiohttp
async with aiohttp.ClientSession() as session:
    async response = await session.get(url)
```

#### C. Process-Based Parallelism
For CPU-bound work that can't be offloaded:
```python
from concurrent.futures import ProcessPoolExecutor

executor = ProcessPoolExecutor(max_workers=4)
result = await loop.run_in_executor(executor, cpu_intensive_func, data)
```

---

### 2. Queue Bottlenecks

**Symptoms:**
- `worker_output_queue_size` or `worker_emit_queue_size` > 50
- Backpressure errors in logs
- Increasing latency over time

**Causes:**
- Consumer slower than producer
- Unbounded queue growth
- No backpressure mechanism

**Mitigations:**

#### A. Bounded Queues with Backpressure
```python
# Before: Unbounded queue
output_queue = asyncio.Queue()

# After: Bounded queue
output_queue = asyncio.Queue(maxsize=100)
```

**Status:** ⚠️  Currently unbounded - consider implementing

#### B. Prioritization
```python
from asyncio import PriorityQueue

class PrioritizedItem:
    def __init__(self, priority: int, item: Any):
        self.priority = priority
        self.item = item

    def __lt__(self, other):
        return self.priority < other.priority

# High-priority items processed first
priority_queue = PriorityQueue()
```

**Use case:** Prioritize user interruptions over bot speech

#### C. Rate Limiting
```python
import asyncio

class RateLimiter:
    def __init__(self, max_rate: int, window: float):
        self.max_rate = max_rate
        self.window = window
        self.tokens = max_rate
        self.last_update = time.monotonic()

    async def acquire(self):
        while self.tokens < 1:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens += elapsed * (self.max_rate / self.window)
            self.tokens = min(self.tokens, self.max_rate)
            self.last_update = now
            await asyncio.sleep(0.01)
        self.tokens -= 1
```

---

### 3. WebSocket Send Latency

**Symptoms:**
- `unmute_websocket_send_duration_ms` > 5ms
- Network congestion
- Emit queue buildup

**Causes:**
- Large payloads (audio, JSON)
- Network latency
- Slow serialization
- Client-side receive buffer full

**Mitigations:**

#### A. Optimize Serialization
```python
# Before: stdlib json
json_str = json.dumps(event.model_dump())

# After: orjson (5-10x faster)
import orjson
json_bytes = orjson.dumps(event.model_dump())
json_str = json_bytes.decode()
```

#### B. Compression
```python
# Enable WebSocket permessage-deflate
import websockets

async with websockets.connect(
    uri,
    compression="deflate",
    compression_level=6
) as ws:
    await ws.send(data)
```

**Note:** Adds CPU overhead - benchmark first

#### C. Batching
```python
# Before: Send each event individually
for event in events:
    await websocket.send_text(event.json())

# After: Batch multiple events
batch = [e.model_dump() for e in events]
await websocket.send_text(json.dumps(batch))
```

---

### 4. Audio Buffer Overflow

**Symptoms:**
- `worker_buffer_overflow_errors` > 0
- Frames discarded
- Audio quality degradation

**Causes:**
- Client sending faster than server can process
- Slow Opus decoding
- Downstream STT/processing lag

**Mitigations:**

#### A. Increase Buffer Size
```python
# In realtime_buffer.py
MAX_BUFFER_SAMPLES = 14_400_000  # 10 minutes at 24kHz instead of 5
```

**Trade-off:** More memory usage

#### B. Adaptive Rate Control
```python
class AdaptiveBufferManager:
    def __init__(self):
        self.target_buffer_ms = 500
        self.max_buffer_ms = 5000

    async def should_accept_frame(self, current_buffer_ms: float) -> bool:
        if current_buffer_ms > self.max_buffer_ms:
            return False  # Drop frame
        return True

    def suggest_client_rate(self, current_buffer_ms: float) -> float:
        """Suggest playback rate adjustment to client."""
        if current_buffer_ms > self.target_buffer_ms * 2:
            return 0.9  # Slow down client
        elif current_buffer_ms < self.target_buffer_ms * 0.5:
            return 1.1  # Speed up client
        return 1.0
```

#### C. Optimize Opus Decoding
```python
# Already using thread pool - check if sphn can be optimized
# Consider native Opus library bindings if sphn is slow
```

---

### 5. STT Flush Latency

**Symptoms:**
- `unmute_stt_flush_duration_ms` > 300ms
- Slow response to user speech
- High `STT_flush` benchmark failures

**Causes:**
- Network latency to STT service
- STT service overload
- Large delay compensation (`stt.delay_sec`)

**Mitigations:**

#### A. Reduce Delay Compensation
```python
# If STT service is fast, reduce safety margin
# In speech_to_text.py or configuration
stt.delay_sec = 0.3  # Instead of 0.5-1.0
```

**Trade-off:** May cut off final words if STT is slow

#### B. Parallel STT Processing
```python
# Instead of sequential flush
# Process multiple audio segments in parallel
async with asyncio.TaskGroup() as tg:
    task1 = tg.create_task(stt.process_segment(segment1))
    task2 = tg.create_task(stt.process_segment(segment2))
```

#### C. STT Service Connection Pooling
```python
class STTConnectionPool:
    def __init__(self, pool_size: int = 3):
        self.pool = asyncio.Queue()
        for _ in range(pool_size):
            self.pool.put_nowait(None)  # Placeholder

    async def get_connection(self) -> SpeechToText:
        await self.pool.get()
        return await find_instance("stt", SpeechToText)

    async def release_connection(self, stt: SpeechToText):
        await self.pool.put(None)
```

---

### 6. LLM Streaming Latency

**Symptoms:**
- `unmute_llm_word_generation_ms` high
- `worker_vllm_ttft` > 500ms
- Slow first token

**Causes:**
- LLM service cold start
- Large context window
- Network latency
- Slow word rechunking

**Mitigations:**

#### A. Reduce Context Size
```python
# Truncate older messages
def preprocess_messages_for_llm(chat_history, max_messages=10):
    # Keep system prompt + last N messages
    return [chat_history[0]] + chat_history[-max_messages:]
```

**Status:** Already implemented with history management

#### B. Stream at Token Level (Not Word Level)
```python
# Before: Rechunk to words (adds latency)
async for delta in rechunk_to_words(llm.chat_completion(messages)):
    await tts.send(delta)

# After: Send tokens directly if TTS supports it
async for token in llm.chat_completion(messages):
    await tts.send(token)
```

**Trade-off:** TTS may need word boundaries for proper prosody

#### C. Speculative Decoding
```python
# Use smaller draft model to speed up first tokens
# Then verify with full model
# This is a model-level optimization - requires VLLM support
```

---

### 7. TTS Processing Latency

**Symptoms:**
- `unmute_tts_word_processing_ms` > 200ms
- `worker_tts_ttft` > 500ms
- Audio output lag

**Causes:**
- TTS model inference time
- Audio encoding overhead
- Network latency to TTS service

**Mitigations:**

#### A. Streaming TTS
```python
# Already implemented - TTS streams audio as it generates
# Monitor `tts.received_samples_yielded` vs `tts.received_samples`
# to ensure streaming is working
```

#### B. Audio Buffer Prefetch
```python
# Prefetch next audio chunk while sending current
class PrefetchingTTSClient:
    async def send_word(self, word: str):
        self.prefetch_task = asyncio.create_task(self._fetch_audio(word))

    async def get_audio(self) -> np.ndarray:
        return await self.prefetch_task
```

#### C. Lower Audio Quality (if acceptable)
```python
# Trade quality for speed
# E.g., use faster TTS model or lower sample rate
SAMPLE_RATE = 16000  # Instead of 24000
```

---

### 8. TaskGroup Overhead

**Symptoms:**
- `unmute_task_group_overhead_ms` > 0.1ms per task
- Many short-lived tasks
- High task creation rate

**Causes:**
- asyncio.TaskGroup has per-task overhead
- Task creation and teardown costs

**Mitigations:**

#### A. Task Reuse
```python
# Before: Create task for each operation
async with asyncio.TaskGroup() as tg:
    for item in items:
        tg.create_task(process(item))

# After: Batch operations in fewer tasks
async with asyncio.TaskGroup() as tg:
    for batch in batched(items, batch_size=10):
        tg.create_task(process_batch(batch))
```

#### B. Long-Lived Workers
```python
# Instead of creating tasks on demand
# Use long-lived worker tasks with queues
async def worker(queue: asyncio.Queue):
    while True:
        item = await queue.get()
        await process(item)
        queue.task_done()

# Create workers once
for _ in range(num_workers):
    asyncio.create_task(worker(work_queue))
```

---

## Optimization Workflow

### Phase 1: Measure

1. Run baseline profiling:
   ```bash
   python scripts/profile_pipeline.py --output baseline.md
   ```

2. Identify top bottlenecks:
   - Review `baseline.md` for high latencies
   - Check `profile.html` for hot functions
   - Review logs for warnings

3. Set optimization targets:
   - Pick 1-2 specific metrics to improve
   - Define success criteria (e.g., "reduce LLM TTFT to <200ms")

### Phase 2: Optimize

1. Implement mitigation (see above)
2. Add tests to verify behavior unchanged
3. Add instrumentation if not already present

### Phase 3: Validate

1. Run profiling again:
   ```bash
   python scripts/profile_pipeline.py \
       --baseline-metrics baseline_metrics.json \
       --output optimized.md
   ```

2. Compare metrics:
   - Is target metric improved?
   - Did any other metrics regress?
   - Are there new bottlenecks?

3. Document results in this file

### Phase 4: Deploy

1. Commit changes with detailed commit message
2. Update this guide with findings
3. Monitor production metrics

---

## Checklist for New Code

When adding new features, ensure performance by design:

- [ ] Use async I/O for network/disk operations
- [ ] Offload CPU-bound work to thread pool
- [ ] Add tracing spans for operations >10ms
- [ ] Use bounded queues where appropriate
- [ ] Add Prometheus metrics for critical paths
- [ ] Test with load testing harness
- [ ] Document expected performance characteristics

---

## Quick Reference

### Performance Targets

| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| LLM TTFT | <300ms | >500ms |
| STT TTFT | <100ms | >200ms |
| TTS TTFT | <500ms | >1000ms |
| Overall TTFT | <1000ms | >2000ms |
| STT Flush | <300ms | >500ms |
| Opus Decode | <10ms | >20ms |
| WebSocket Send | <5ms | >10ms |
| Queue Wait | <10ms | >50ms |
| Output Queue Size | <10 | >50 |

### Instrumentation Examples

**Add tracing span:**
```python
from unmute.tracing import trace_span

async with trace_span("my_operation"):
    result = await my_operation()
```

**Add custom histogram:**
```python
from prometheus_client import Histogram

MY_METRIC = Histogram(
    "unmute_my_metric_ms",
    "Description",
    buckets=[1, 5, 10, 25, 50, 100],
)

async with trace_span("my_op", histogram=MY_METRIC):
    await my_operation()
```

**Track queue operations:**
```python
from unmute.tracing import trace_queue_operation

async with trace_queue_operation("my_queue", "put"):
    await queue.put(item)
```

---

## Resources

- [Python asyncio Performance Tips](https://docs.python.org/3/library/asyncio-dev.html)
- [FastAPI Performance](https://fastapi.tiangolo.com/deployment/concepts/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/instrumentation/)
- [Real-time Systems Design](https://en.wikipedia.org/wiki/Real-time_computing)

---

## Changelog

### 2026-01-26: Initial version
- Documented common bottlenecks and mitigations
- Added optimization workflow
- Created quick reference guide
