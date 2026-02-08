"""Convert an OpenAI Realtime API JSONL recording to a TraceFixture.

Reads a JSONL file with {"direction": "sent"|"received", "payload": {...}, "timestamp": "..."}
and produces a TraceFixture with non-audio client events and scripted mock data.
"""

import json
from pathlib import Path
from typing import Any

from tests.realtime_harness.fixture_schema import (
    ClientEvent,
    EventAssertion,
    FixtureEventType,
    FixtureMetadata,
    OrderingAssertion,
    TimingMode,
    TraceFixture,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_client_events(records: list[dict[str, Any]]) -> list[ClientEvent]:
    """Extract non-audio client events from the recording.

    Filters out input_audio_buffer.append events (bulk audio data).
    Keeps: session.update, conversation.item.create, response.create
    """
    client_events: list[ClientEvent] = []
    for record in records:
        if record["direction"] != "sent":
            continue
        payload = record["payload"]
        event_type = payload.get("type", "")

        # Skip audio append events (too many and too large)
        if event_type == "input_audio_buffer.append":
            continue

        client_events.append(
            ClientEvent(
                event=payload,
                delay_ms=100,  # Small delay between events
                description=f"Client: {event_type}",
            )
        )
    return client_events


def extract_server_event_types(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count server event types from the recording."""
    counts: dict[str, int] = {}
    for record in records:
        if record["direction"] != "received":
            continue
        event_type = record["payload"].get("type", "unknown")
        counts[event_type] = counts.get(event_type, 0) + 1
    return counts


def extract_function_calls(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract function call details from server events."""
    calls = []
    for record in records:
        if record["direction"] != "received":
            continue
        payload = record["payload"]
        if payload.get("type") == "response.function_call_arguments.done":
            calls.append(
                {
                    "name": payload.get("name", ""),
                    "arguments": payload.get("arguments", ""),
                    "call_id": payload.get("call_id", ""),
                }
            )
    return calls


def extract_transcriptions(records: list[dict[str, Any]]) -> list[str]:
    """Extract transcription texts from server events."""
    transcripts = []
    for record in records:
        if record["direction"] != "received":
            continue
        payload = record["payload"]
        if (
            payload.get("type")
            == "conversation.item.input_audio_transcription.completed"
        ):
            transcripts.append(payload.get("transcript", ""))
    return transcripts


def extract_llm_responses(records: list[dict[str, Any]]) -> list[str]:
    """Extract LLM response texts from server events."""
    responses = []
    for record in records:
        if record["direction"] != "received":
            continue
        payload = record["payload"]
        if payload.get("type") == "response.output_audio_transcript.done":
            responses.append(payload.get("transcript", ""))
    return responses


def jsonl_to_fixture(
    path: Path,
    name: str = "session_replay",
) -> TraceFixture:
    """Convert a JSONL recording to a TraceFixture.

    Args:
        path: Path to the JSONL file
        name: Name for the fixture

    Returns:
        TraceFixture ready for replay
    """
    records = load_jsonl(path)
    client_events = extract_client_events(records)
    server_event_counts = extract_server_event_types(records)

    # Build event assertions based on what the recording shows
    event_assertions = [
        EventAssertion(
            event_type="session.updated",
            min_occurrences=1,
            description="Session should be updated after session.update",
        ),
    ]

    # Add assertions for event types seen in the recording
    for event_type, count in server_event_counts.items():
        if event_type in (
            "response.created",
            "response.done",
            "conversation.item.added",
            "conversation.item.done",
        ):
            event_assertions.append(
                EventAssertion(
                    event_type=event_type,
                    min_occurrences=1,
                    description=f"Expected at least 1 {event_type} (recording had {count})",
                )
            )

    # Add function call assertion if present in recording
    if "response.function_call_arguments.done" in server_event_counts:
        event_assertions.append(
            EventAssertion(
                event_type="response.function_call_arguments.done",
                min_occurrences=1,
                description="Function calls should be completed",
            )
        )

    ordering_assertions = [
        OrderingAssertion(
            before="session.updated",
            after="response.created",
            description="Session must be configured before response generation",
        ),
        OrderingAssertion(
            before="response.created",
            after="response.done",
            description="Response must be created before done",
        ),
    ]

    fixture = TraceFixture(
        metadata=FixtureMetadata(
            name=name,
            description=f"Replay of {path.name}",
            category=FixtureEventType.TOOL_CALL,
            tags=["replay", "function_calling"],
            timeout_seconds=60,
        ),
        timing_mode=TimingMode.RELATIVE,
        client_events=client_events,
        event_assertions=event_assertions,
        ordering_assertions=ordering_assertions,
    )

    return fixture


def extract_mock_data(path: Path) -> dict[str, Any]:
    """Extract data needed for mock services from a JSONL recording.

    Returns dict with:
        - transcriptions: list of STT transcription texts
        - llm_responses: list of LLM response texts
        - function_calls: list of function call details
    """
    records = load_jsonl(path)
    return {
        "transcriptions": extract_transcriptions(records),
        "llm_responses": extract_llm_responses(records),
        "function_calls": extract_function_calls(records),
    }
