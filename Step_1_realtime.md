# Step 1 – OpenAI Realtime Compliance & Multimodal Routing

## Objectives
- Reach full parity with OpenAI Realtime websocket protocol (client/server events, session lifecycle, tool calling) so any compliant client can talk to our backend without adapters.
- Preserve Unmute-specific capabilities (voices, debugging, recordings) via negotiated extensions that never break compatibility.
- Guarantee end-to-end responsiveness suited for a Jetson Thor robot: sub-300 ms STT flush time, low jitter audio output, fast actuator tool-call turnaround, and smooth failover when services hiccup.

## Key Constraints & Assumptions
- STT/TTS engines are preselected and already optimized for the target hardware; work focuses on orchestration, not model changes.
- Websocket is the only transport; no HTTP fallback or polling paths.
- Python remains the orchestration language; Rust/CUDA components stay untouched except via documented APIs.
- Robot client requires actuator tool calls (`move_arm`, `set_gaze`, `capture_frame`, etc.) and camera image uploads, both of which must traverse the Realtime channel.

## Workstreams

### 1. Protocol & Session Layer
- Map every OpenAI Realtime client/server event to Pydantic models (see `unmute/openai_realtime_api_events.py`) and close gaps: conversation items, response states, tool-call events, errors, interruptions, and session updates.
- Refactor websocket handshake to enforce `realtime` subprotocol, negotiate capabilities (e.g., `unmute.extensions`), and validate auth/config payloads.
- Implement state reconciliation: allow clients to send `session.update`, `response.create`, `input_audio_buffer.*`, `conversation.item.*`, `response.cancel`, etc. Ensure proper acknowledgements (`*.updated`, `response.completed`, `response.failed`).

### 2. Audio Pipeline Alignment
- Rework audio ingestion so `input_audio_buffer.append` frames are stored per-session with timestamps, sequence counters, and base64/audio codecs matching OpenAI specs (PCM16 or Opus as required).
- Introduce a buffer manager that streams frames into the existing STT service, tracks flush points, and emits `conversation.item.input_audio_transcription.delta/done` events with latency metadata.
- Ensure voice-activity and pause-detection logic trigger canonical events (`input_audio_buffer.speech_started/stopped`) and that backpressure or silence timeouts produce compliant errors.

### 3. Multimodal Inputs (Images & Metadata)
- Extend protocol handling for `input_image_buffer.append` (or equivalent) and wire decoded images straight into the LLM context builder; store references so subsequent `response.create` can cite them.
- Support metadata items (e.g., robot sensor readings) using OpenAI “modalities” fields, guaranteeing the conversation state remains deterministic if clients reconnect.

### 4. Tool Calling for Robot Actuators & Sensors
- Define a tool schema library describing each actuator command (`move_arm`, `rotate_base`, `play_sound`, `capture_frame`) plus expected arguments, validation rules, and timeout budgets.
- When the LLM emits `response.tool_call.delta/done`, route the payload to the appropriate actuator microservice, await execution/ack, and stream back `response.tool_call.output` with success/failure info.
- Provide interrupt semantics: allow clients or STT-triggered VAD interruptions to cancel in-flight tool calls safely.

### 5. Conformance Tester & Regression Harness
- Build a Python-based harness that:
  - Replays canonical OpenAI trace files (audio bursts, tool call sequences, multimodal inputs) and asserts byte-level compliance (event order, payload schemas, timing windows).
  - Injects fuzz cases (duplicate IDs, out-of-order frames, invalid tool responses) and ensures the server returns the correct `error` events.
  - Measures latency budgets (TTFT, STT flush, tool-call RTT) and flags regressions with strict thresholds tailored for the robot use case.
- Integrate harness into CI (headless FastAPI + uvicorn instance) and provide scripts for developers to run locally.

### 6. Jetson Thor Deployment & Performance Tuning
- Document recommended service layout for a Jetson Thor board: process pinning, GPU sharing strategy, CUDA versioning, and how to co-locate websocket + router + telemetry without starving STT/TTS GPUs.
- Profile critical paths (audio receive loop, STT flushing, LLM streaming, actuator responses) using async tracepoints; eliminate Python GIL contention via task groups, bounded queues, and `asyncio.StreamReader` optimizations.
- Add health probes that reflect realtime readiness (STT backlog, LLM queue depth, actuator availability) to avoid accepting sessions when latency SLAs cannot be met.

## Milestones & Exit Criteria
1. **Protocol MVP** – All mandatory OpenAI Realtime events implemented; simple echo client passes handshake and round-trips text/audio.
2. **Audio + Tool Integration** – Audio routed through STT with compliant transcription events; actuator tool calls executed via websocket with confirmation events.
3. **Multimodal Ready** – Image ingestion works end-to-end with LLM context; conversation state survives reconnects.
4. **Conformance Harness Green** – Automated suite passes across success/failure cases; latency metrics logged for every run.
5. **Jetson Validation** – Deployed on target hardware, demonstrating latency targets and stable resource usage under representative robot workloads.

## Detailed Action Plan

1. **Protocol & Session Layer**
   - Audit current Pydantic models in `unmute/openai_realtime_api_events.py`, add missing OpenAI events, enums, and acknowledgement payloads.
   - Refactor websocket handshake to enforce the `realtime` subprotocol, negotiate `unmute.extensions`, and validate auth/config payloads.
   - Implement session state reconciliation covering `session.update`, `response.*`, `input_audio_buffer.*`, `conversation.item.*`, and cancellation paths with the corresponding acknowledgements.
   - Expand regression coverage for protocol sequencing and lifecycle transitions.

2. **Audio Pipeline Alignment**
   - Design a per-session audio buffer manager that tracks timestamps, sequence numbers, codec metadata, and storage quotas per the OpenAI spec.
   - Stream buffered frames into the STT service, emitting compliant `conversation.item.input_audio_transcription.delta/done` events with latency annotations.
   - Implement voice-activity and pause detection emitting `input_audio_buffer.speech_started/stopped`, and surface backpressure or silence violations as compliant errors.

3. **Multimodal Inputs**
   - Extend handlers for image append events, decoding and storing references that downstream `response.create` calls can cite.
   - Support metadata modalities (sensor readings) ensuring deterministic conversation state persistence for reconnect scenarios.
   - Update the LLM context builder to compose modalities and persist replayable state.

4. **Tool Calling Orchestration**
   - Define a tool schema library for actuator commands with validation rules, timeout budgets, and serialization helpers.
   - Implement routing from `response.tool_call.delta/done` events to the actuator microservices, awaiting execution and relaying `response.tool_call.output` events.
   - Add interrupt semantics enabling client-issued or VAD-triggered cancellations with safe rollback and logging.

5. **Conformance Harness**
   - Build a Python harness capable of replaying canonical OpenAI traces and asserting event order, payload schemas, and timing windows.
   - Add fuzzing modules for duplicate IDs, out-of-order frames, and invalid tool responses, ensuring the correct `error` events fire.
   - Integrate latency monitors (TTFT, STT flush, tool-call RTT) with configurable thresholds and CI hooks.

6. **Jetson Thor Deployment & Performance**
   - Document the recommended Jetson layout (process pinning, GPU sharing, CUDA versions) for co-locating websocket/router/telemetry with STT/TTS services.
   - Instrument audio/STT/LLM/tool-call loops using async tracepoints, profiling for GIL contention, queue bottlenecks, and stream reader inefficiencies.
   - Add realtime health probes (STT backlog, LLM queue depth, actuator availability) to gate new sessions when latency SLAs are at risk.

7. **Milestone Tracking**
   - Define validation checklists for each milestone and align dependencies with actuator/sensor teams.
   - Establish reporting cadence for parity, audio/tool integration, multimodal readiness, harness health, and Jetson validation progress.

## Test Plan

1. **Protocol Conformance**
   - Unit tests validating every Pydantic model against OpenAI examples, including optional fields and enum values.
   - Integration tests covering websocket handshake negotiation, capability extensions, and session reconciliation flows across success/failure paths.
   - Negative tests for malformed events, unauthorized sessions, and capability mismatches ensuring correct `error` events.

2. **Audio Pipeline**
   - Simulated audio streams verifying per-session buffering, sequence handling, codec validation, and STT flush latency (<300 ms).
   - VAD event tests for speech start/stop, silence timeouts, and backpressure-induced errors with retransmission behavior.
   - Regression tests comparing emitted transcription deltas/done payloads against STT outputs and metadata expectations.

3. **Multimodal & Metadata**
   - Image ingestion tests ensuring buffers persist, link to conversation state, and can be referenced post-reconnect.
   - Metadata determinism tests replaying sensor readings to confirm identical state reconstruction and modality ordering.

4. **Tool Calling**
   - Schema validation tests for each actuator tool input/output contract.
   - Integration tests covering tool call lifecycle: success, failure, timeout, interruption, and concurrent calls.
   - Stress tests validating interrupt semantics and safe cancellation under load.

5. **Conformance Harness Suite**
   - Replay canonical trace fixtures, comparing event logs byte-for-byte against expected sequences.
   - Fuzz campaigns injecting duplicates, out-of-order frames, and invalid payloads, asserting correct `error` emissions.
   - Latency benchmark tests with automated threshold alerts for TTFT, STT flush, and tool-call round-trip times.

6. **Performance & Deployment**
   - Load tests emulating Jetson Thor workloads to monitor audio latency, actuator throughput, and queue depths.
   - Health probe validation ensuring readiness/ liveness endpoints reflect realtime capacity under contention and during failover drills.

7. **CI Integration**
   - Automated harness execution in CI (headless FastAPI + uvicorn) with artifacts for traces, latency metrics, and fuzz outcomes.
   - Smoke tests on every merge, nightly extended trace/fuzz suites, and dashboards surfacing regressions for rapid triage.
