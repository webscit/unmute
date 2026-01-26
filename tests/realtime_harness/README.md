# OpenAI Realtime API Conformance Harness

A comprehensive testing harness for validating OpenAI Realtime API protocol compliance in the Unmute WebSocket implementation.

## Overview

The conformance harness:
- Launches a headless FastAPI+uvicorn instance
- Replays canonical OpenAI trace fixtures over the `/v1/realtime` WebSocket
- Validates byte-level protocol compliance (event order, payload schemas, timing)
- Generates structured reports (JSON, terminal output)
- Runs in CI with <5 minute target runtime

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Harness Runner                        │
├─────────────────────────────────────────────────────────┤
│  1. ServerLauncher                                      │
│     └─> uvicorn unmute.main_websocket:app              │
│                                                         │
│  2. FixtureParser                                       │
│     └─> Load & validate JSON fixtures                  │
│                                                         │
│  3. WebSocketReplayer                                   │
│     └─> Connect, send events, collect responses        │
│                                                         │
│  4. ProtocolValidator                                   │
│     └─> Validate events, ordering, timing              │
│                                                         │
│  5. Report Generator                                    │
│     └─> JSON/terminal output                           │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Run All Fixtures

```bash
# From project root
python -m tests.realtime_harness.runner
```

### Run Specific Fixture

```bash
python -m tests.realtime_harness.runner --fixture text_only_basic
```

### Run by Category

```bash
python -m tests.realtime_harness.runner --category audio_input
```

### Run with External Server

```bash
# Start server manually
uvicorn unmute.main_websocket:app --host 127.0.0.1 --port 8000

# Run harness without starting server
python -m tests.realtime_harness.runner --no-server
```

### Generate JSON Report

```bash
python -m tests.realtime_harness.runner --output reports/conformance_report.json
```

## CLI Options

```
Options:
  -f, --fixture FIXTURE     Run specific fixture(s) (can specify multiple)
  -c, --category CATEGORY   Filter by category (text_only, audio_input, tool_call, etc.)
  -t, --tag TAG            Filter by tag(s) (can specify multiple)
  -o, --output FILE        Write JSON report to file
  --no-server              Don't start server (connect to external server)
  --host HOST              Server host (default: 127.0.0.1)
  --port PORT              Server port (default: 8000)
```

## Fixture Categories

- **text_only**: Text-only conversations, no audio
- **audio_input**: Audio buffer operations, transcription
- **tool_call**: Function calling and tool use
- **multimodal**: Image and metadata inputs
- **session_update**: Session configuration changes
- **error_handling**: Error cases and edge conditions

## Creating Fixtures

See [`fixtures/README.md`](fixtures/README.md) for detailed fixture format documentation.

### Minimal Example

```json
{
  "metadata": {
    "name": "my_fixture",
    "description": "Test scenario description",
    "category": "text_only",
    "tags": ["basic"],
    "timeout_seconds": 15
  },
  "timing_mode": "relative",
  "client_events": [
    {
      "event": {
        "type": "session.update",
        "session": {
          "modalities": ["text"]
        }
      },
      "delay_ms": 0
    }
  ],
  "event_assertions": [
    {
      "event_type": "session.updated",
      "min_occurrences": 1,
      "timeout_ms": 1000
    }
  ],
  "ordering_assertions": [],
  "timing_assertions": []
}
```

## Validation Types

### Event Assertions

Validate that expected server events occur with correct payloads:

```json
{
  "event_type": "response.created",
  "match_fields": {
    "response.status": "in_progress"
  },
  "exclude_fields": ["error"],
  "min_occurrences": 1,
  "max_occurrences": 1,
  "timeout_ms": 5000
}
```

### Ordering Assertions

Validate event ordering constraints:

```json
{
  "before": "response.created",
  "after": "response.text.delta",
  "strict": true,
  "description": "Response must be created before text deltas"
}
```

- **strict: true**: No `after` events can occur before first `before`
- **strict: false**: At least one `before` must come before at least one `after`

### Timing Assertions

Validate latency between events:

```json
{
  "event_a": "response.created",
  "event_b": "response.done",
  "max_latency_ms": 10000,
  "description": "Response should complete within 10s"
}
```

## Report Format

### Terminal Output

```
Conformance Harness Report
Timestamp: 2026-01-26T10:30:00
Duration: 12543ms

┏━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric        ┃ Value ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Total Fixtures│ 3     │
│ Passed        │ 2     │
│ Failed        │ 1     │
└───────────────┴───────┘
```

### JSON Report

```json
{
  "timestamp": "2026-01-26T10:30:00",
  "total_fixtures": 3,
  "fixtures_passed": 2,
  "fixtures_failed": 1,
  "total_duration_ms": 12543.5,
  "fixture_results": [
    {
      "fixture_name": "text_only_basic",
      "passed": true,
      "duration_ms": 3421.2,
      "assertions_passed": 11,
      "assertions_failed": 0,
      "failures": [],
      "warnings": [],
      "server_events_received": 15
    }
  ],
  "environment": {
    "server_url": "ws://127.0.0.1:8000",
    "python_version": "3.12.0"
  }
}
```

## Module Reference

### `fixture_schema.py`

Pydantic models for fixture structure:
- `TraceFixture`: Complete fixture definition
- `ClientEvent`: Client event to send
- `EventAssertion`: Server event validation
- `OrderingAssertion`: Event ordering constraint
- `TimingAssertion`: Event timing constraint

### `fixture_parser.py`

Fixture loading and validation:
- `FixtureParser`: Load fixtures from files
- `get_nested_field()`: Access nested fields with dot notation
- `matches_field_constraints()`: Validate field constraints

### `websocket_replayer.py`

WebSocket client for replay:
- `WebSocketReplayer`: Connect and replay fixtures
- `ReplayTrace`: Captured trace of server responses
- `ReceivedEvent`: Individual server event with timestamp

### `validators.py`

Protocol compliance validation:
- `ProtocolValidator`: Validate traces against fixtures
- `ValidationResult`: Validation outcome with failures
- `ValidationFailure`: Individual validation failure

### `runner.py`

Main harness orchestrator:
- `ServerLauncher`: Manage FastAPI server lifecycle
- `HarnessRunner`: Run fixtures and generate reports

## CI Integration

### GitHub Actions Example

```yaml
name: Conformance Tests

on: [push, pull_request]

jobs:
  conformance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run conformance harness
        run: |
          python -m tests.realtime_harness.runner \
            --output conformance_report.json
      - name: Upload report
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: conformance-report
          path: conformance_report.json
```

## Development

### Adding New Assertions

1. Add assertion type to `fixture_schema.py`
2. Implement validator in `validators.py`
3. Update fixture examples
4. Add tests

### Debugging Fixtures

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Inspect trace manually:

```python
from tests.realtime_harness.fixture_parser import FixtureParser
from tests.realtime_harness.websocket_replayer import replay_fixture_standalone

fixture = FixtureParser().load_fixture("text_only_basic")
trace = await replay_fixture_standalone(fixture)

# Inspect events
for event in trace.received_events:
    print(f"{event.timestamp_ms:.2f}ms: {event.event_type}")
```

## Performance

Target metrics:
- Fixture execution: <5s each
- Full harness run: <5min (for CI)
- Server startup: <10s
- WebSocket connection: <1s

## Troubleshooting

### Server fails to start

```
RuntimeError: Server failed to start within 30 seconds
```

**Solutions**:
- Check port 8000 is not in use: `lsof -i :8000`
- Verify dependencies installed: `pip install -r requirements.txt`
- Check server logs in subprocess stderr

### Connection refused

```
websockets.exceptions.InvalidStatusCode: server rejected WebSocket connection
```

**Solutions**:
- Verify server is running: `curl http://localhost:8000/v1/health`
- Check handshake headers (realtime subprotocol)
- Review server logs for errors

### Fixture validation fails

```
FixtureLoadError: Fixture validation failed
```

**Solutions**:
- Validate fixture schema: `python -m tests.realtime_harness.fixture_parser fixtures/my_fixture.json`
- Check JSON syntax: `jq . fixtures/my_fixture.json`
- Review error details in output

### Assertions fail unexpectedly

**Solutions**:
- Inspect trace events: Look at `server_events_received` in report
- Check timing: Events may arrive out of order due to async processing
- Relax constraints: Increase timeouts or use non-strict ordering
- Review server logs for errors

## License

Part of the Unmute project. See main LICENSE file.
