# Step 3 – Semantic LLM Router Integration

## Objectives
- Replace the single-LLM assumption with a semantic router (vLLM semantic-router) so requests can be dispatched to the best model for the situation while keeping Realtime compliance intact.
- Maintain ultra-low latency for robot interactions by routing based on modality, actuator criticality, and live performance telemetry.
- Ensure tool-calling, multimodal context, and session history remain consistent regardless of which downstream model handles a turn.

## Key Constraints & Assumptions
- Router will run locally (Jetson Thor or adjacent GPU node) and must expose latency/health signals for decisions.
- Backends may vary (local vLLM instances, remote OpenAI endpoints, distilled models); the router chooses per request and can fall back automatically.
- Tool-call schemas must be honored identically regardless of chosen model.

## Workstreams

### 1. Abstraction Layer
- Introduce an LLM service interface (`LLMClient`) with async streaming, tool-call support, and metadata hooks; existing VLLM/OpenAI adapters implement it.
- Modify `UnmuteHandler` to depend on the abstract interface rather than a specific client.
- Define payload converters between OpenAI Realtime messages and router-specific request objects (prompts, images, tool schemas, conversation history).

### 2. Semantic Router Integration
- Embed the vLLM semantic-router as a microservice or in-process module, fed with:
  - Message content (text, images, metadata)
  - Session/robot context (task urgency, actuator state, sensors)
  - Telemetry feedback (latency, error rate, GPU availability)
- Configure routing policies:
  - **Latency-first** for urgent actuator instructions or back-and-forth dialogue.
  - **Quality-first** for open-ended reasoning when latency budget allows.
  - **Modality-aware**: image-heavy prompts route to vision-capable models.
  - **Fallback**: automatic downgrade to resilient model if preferred model overloaded or failing conformance checks.

### 3. Tool-Call Consistency
- Ensure router-selected models receive identical tool schema definitions and return outputs convertible into OpenAI Realtime tool events.
- Build normalization layer that translates model-specific tool-call formats back into canonical responses before emitting over websocket.
- Log router decisions alongside tool execution telemetry for debugging and auditing.

### 4. Observability & Feedback Loop
- Feed monitoring data (from Step 2) into the router to influence decisions dynamically (e.g., degrade to smaller model when TTFT spikes).
- Emit router spans/metrics: decision latency, selected model, confidence score, fallback reason, per-model success rates.
- Add CLI/debug views to inspect routing decisions per session and simulate policy changes.

### 5. Validation & Rollout
- Extend conformance harness to cover multi-model routing scenarios: ensure identical Realtime behavior regardless of backend choice and verify tool-call outputs still meet schema.
- Run load tests on Jetson Thor to confirm router overhead stays within latency targets; adjust batching/prefetch strategies as needed.
- Document deployment recipes: router configuration files, model registry, health probes, scaling guidelines.

## Milestones & Exit Criteria
1. **LLM Interface Refactor** – Backend uses abstract interface; existing single-model path works unchanged through adapter.
2. **Router Operational** – Semantic router selects among at least two models with configurable policies; telemetry recorded for each decision.
3. **Tool-Call Parity** – Tool interactions succeed across all routed models, with identical events observed by clients.
4. **Adaptive Routing** – Router responds to live telemetry (latency/health) and performs fallbacks without manual intervention.
5. **Validated on Target Hardware** – Demonstrated improvements (or maintained latency) on Jetson Thor with robot workloads; conformance harness green across routing scenarios.
