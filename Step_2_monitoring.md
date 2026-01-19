# Step 2 – Monitoring, Telemetry & Latency Analytics

## Objectives
- Provide real-time observability suited for responsiveness-critical robotics, leveraging OpenTelemetry to capture every hop in the speech/LLM/tool pipeline.
- Surface actionable latency insights (TTFT, STT flush, actuator command RTT, GPU utilization) so regressions are caught early and tuning on Jetson Thor is data-driven.
- Offer lightweight dashboards and replay tools inspired by OpenOmniFramework while respecting embedded constraints.

## Key Constraints & Assumptions
- OpenTelemetry SDK/exporters available in Python services; minimal overhead required for Jetson deployment.
- Metrics must correlate across services (websocket backend, STT, LLM router, actuator executors) even if they run in separate processes or hosts.
- Existing Prometheus metrics can remain but will be fed via OTel collectors when needed.

## Workstreams

### 1. Telemetry Foundations
- Introduce a telemetry module that initializes OpenTelemetry Tracer/Meter Providers with context propagation over asyncio tasks and websockets.
- Define canonical attributes: `session.id`, `client.id`, `device.type=robot`, `modalities`, `llm.model`, `tool.name`, `gpu.id`, etc.
- Export via OTLP to a local OTel Collector configured for Jetson (low memory footprint) and optionally bridge to Prometheus/OpenOmni dashboards.

### 2. Stage-Level Instrumentation
- Websocket ingress: span for each received event (`input_audio_buffer.append`, `response.create`, tool outputs), recording queue wait time, payload size, validation errors.
- STT pipeline: spans for streaming batches, pause detection, transcription deltas; metrics for RTF (real-time factor), confidence scores, backlog.
- LLM streaming: spans covering request/response, tool-call decision, tokens/sec metrics, TTFT histogram.
- Tool executors: spans for command dispatch, actuator ack, camera capture, failure reasons; metrics for success rate, retries, physical latency budgets.
- TTS/audio egress: spans for synthesis, buffering, and websocket send operations; metrics for jitter and dropout count.

### 3. Correlation & Diagnostics
- Propagate trace/span context through QuestManager / async tasks so STT/LLM/TTS logs include shared IDs.
- Attach telemetry hooks to error-handling paths (e.g., `MissingServiceTimeout`, actuator timeouts) so alerts can distinguish capacity vs. bugs.
- Record derived metrics: session duration percentiles, simultaneous user count, GPU utilization (via NVML bindings) tagged per service.

### 4. Visualization & Alerting
- Deploy an embedded OpenTelemetry Collector + lightweight backend (Tempo/Loki/Grafana or OpenOmni-like stack) optimized for Jetson resources.
- Provide default dashboards: latency waterfall per session, TTFT tracking, STT flush vs. threshold, actuator RTT distribution, tool success/failure timeline.
- Configure alert rules (or simple CLI scripts) for SLA breaches: STT flush > 300 ms, TTFT > 800 ms, actuator RTT > 500 ms, GPU > 95% for sustained periods.

### 5. Trace Replay & Dev Tooling
- Implement CLI/notebook helpers that fetch traces for a given `session.id`, reconstruct the conversation timeline, and highlight latency spikes.
- Offer local development mode with console exporters and sampling controls so engineers can test instrumentation without heavy infra.

## Milestones & Exit Criteria
1. **Telemetry SDK Integrated** – Backend initializes OpenTelemetry with OTLP export; spans/metrics appear in collector during manual tests.
2. **Full-Pipeline Coverage** – Every major stage emits spans/metrics with proper correlation IDs; errors report structured diagnostics.
3. **Dashboards Online** – Default latency/health dashboards render on target hardware; alerts configured for key SLA thresholds.
4. **Trace Replay Tool** – CLI/notebook workflow retrieves and visualizes per-session traces for debugging and benchmarking.
5. **Performance Acceptance** – Telemetry overhead measured on Jetson Thor (<5% CPU/GPU impact) while still capturing required fidelity.
