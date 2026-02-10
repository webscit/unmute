# Realtime API Conformance Test Fixtures

This directory contains trace fixtures for testing OpenAI Realtime API protocol compliance. Each fixture defines a sequence of client events, expected server responses, and validation assertions.

## Fixture Format

Fixtures are JSON files following the `TraceFixture` schema defined in `../fixture_schema.py`. Each fixture contains:

### Metadata
```json
{
  "metadata": {
    "name": "unique_fixture_name",
    "description": "Human-readable description",
    "category": "text_only|audio_input|tool_call|multimodal|session_update|error_handling",
    "tags": ["tag1", "tag2"],
    "timeout_seconds": 30
  }
}
```

### Timing Mode
- `relative`: Each event's `delay_ms` is relative to the previous event
- `absolute`: Each event's `timestamp_ms` is absolute from trace start
- `immediate`: Send all events as fast as possible

### Client Events
Sequence of events to send to the server:
```json
{
  "client_events": [
    {
      "event": {
        "type": "session.update",
        "session": { ... }
      },
      "delay_ms": 100,
      "description": "Optional description"
    }
  ]
}
```

### Assertions

#### Event Assertions
Validate that expected server events are received:
```json
{
  "event_assertions": [
    {
      "event_type": "response.created",
      "match_fields": {
        "response.status": "in_progress"
      },
      "min_occurrences": 1,
      "max_occurrences": 1,
      "timeout_ms": 5000,
      "description": "Response should be created"
    }
  ]
}
```

- `match_fields`: Dot-notation field paths that must match exactly
- `exclude_fields`: Dot-notation paths that should NOT be present
- `min_occurrences`/`max_occurrences`: Occurrence constraints
- `timeout_ms`: Max wait time for this event

#### Ordering Assertions
Validate event ordering:
```json
{
  "ordering_assertions": [
    {
      "before": "response.created",
      "after": "response.text.delta",
      "strict": true,
      "description": "Response must be created before deltas"
    }
  ]
}
```

- `strict: true`: No `after` events can occur before the first `before` event
- `strict: false`: Just ensure at least one `before` comes before at least one `after`

#### Timing Assertions
Validate timing between events:
```json
{
  "timing_assertions": [
    {
      "event_a": "response.created",
      "event_b": "response.done",
      "max_latency_ms": 10000,
      "description": "Response should complete within 10s"
    }
  ]
}
```

## Available Fixtures

### text_only_basic.json
Basic text-only conversation testing:
- Session creation and update
- Text message exchange
- Response streaming (text deltas)
- Event ordering (created → delta → done)

### audio_input_basic.json
Audio input testing:
- Audio buffer append/commit
- Transcription events
- Audio → text conversion
- Buffer state transitions

### tool_call_basic.json
Tool/function calling testing:
- Session with tool definitions
- Tool invocation from LLM
- Function call argument streaming
- Tool call completion

## Creating New Fixtures

1. **Define the scenario**: What protocol behavior are you testing?
2. **Capture client events**: What events should the client send?
3. **Define expected responses**: What server events must occur?
4. **Add constraints**: Ordering, timing, field matching
5. **Validate schema**: Run `python -m tests.realtime_harness.validate_fixture <file>`

### Tips

- Start with existing fixtures as templates
- Keep fixtures focused on one scenario
- Use descriptive names and tags
- Test your fixture with the harness: `python -m tests.realtime_harness.runner --fixture <name>`
- For audio fixtures, use real Opus-encoded samples (see `../test_audio_samples/`)

## Audio Samples

Audio fixtures use base64-encoded Opus audio. Placeholder values (`AUDIO_PLACEHOLDER_BASE64_*`) should be replaced with real samples:

1. Generate test audio: `python scripts/generate_test_audio.py`
2. The script creates Opus-encoded samples in `tests/realtime_harness/test_audio_samples/`
3. Replace placeholders in fixtures with actual base64 data

## Schema Validation

All fixtures are validated against the Pydantic schema on load. Common validation errors:

- **Empty client_events**: At least one client event required
- **Negative delays**: `delay_ms` must be >= 0
- **Invalid event types**: Must match OpenAI Realtime API spec
- **Invalid timing**: `max_latency_ms` must be positive

Run validation manually:
```bash
python -c "from tests.realtime_harness.fixture_schema import TraceFixture; import json; TraceFixture(**json.load(open('fixtures/my_fixture.json')))"
```
