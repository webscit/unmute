"""Fuzz generators for conformance harness.

This module provides fuzzing capabilities to test protocol robustness by generating:
- Duplicate event IDs
- Out-of-order frames
- Invalid tool outputs
- Malformed payloads

Fuzzers can mutate existing fixtures or generate standalone malformed events.
"""

import copy
import random
import uuid
from typing import Any, Callable, Optional

from .fixture_schema import ClientEvent, EventAssertion, TraceFixture


class FuzzStrategy:
    """Base class for fuzz mutation strategies."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def mutate_fixture(self, fixture: TraceFixture) -> TraceFixture:
        """Apply mutation to a fixture and return mutated copy."""
        raise NotImplementedError


class DuplicateIDFuzzer(FuzzStrategy):
    """Inject duplicate event IDs to test ID collision handling."""

    def __init__(self, duplicate_probability: float = 0.3):
        super().__init__(
            name="duplicate_ids",
            description="Injects duplicate event_id values to test collision handling",
        )
        self.duplicate_probability = duplicate_probability

    def mutate_fixture(self, fixture: TraceFixture) -> TraceFixture:
        """Duplicate random event IDs in the fixture."""
        mutated = copy.deepcopy(fixture)
        mutated.metadata.name = f"{mutated.metadata.name}_fuzz_duplicate_ids"
        mutated.metadata.tags.append("fuzz")
        mutated.metadata.tags.append("duplicate_ids")

        # Collect event IDs from events that have them
        seen_ids = []
        for client_event in mutated.client_events:
            if "event_id" in client_event.event:
                seen_ids.append(client_event.event["event_id"])

        # Randomly replace some IDs with duplicates
        if seen_ids:
            for client_event in mutated.client_events:
                if "event_id" in client_event.event and random.random() < self.duplicate_probability:
                    client_event.event["event_id"] = random.choice(seen_ids)

        # Add assertion for error event
        mutated.event_assertions.append(
            EventAssertion(
                event_type="error",
                match_fields={"error.type": "invalid_request_error"},
                min_occurrences=0,  # Server may or may not emit error for duplicate IDs
                max_occurrences=None,
                timeout_ms=5000,
                description="Server may emit error for duplicate event IDs",
            )
        )

        return mutated


class OutOfOrderFuzzer(FuzzStrategy):
    """Shuffle event order to test state machine resilience."""

    def __init__(self, shuffle_intensity: float = 0.5):
        super().__init__(
            name="out_of_order",
            description="Shuffles event order to test protocol state machine",
        )
        self.shuffle_intensity = shuffle_intensity  # 0.0 = no shuffle, 1.0 = full shuffle

    def mutate_fixture(self, fixture: TraceFixture) -> TraceFixture:
        """Shuffle client events while preserving some dependencies."""
        mutated = copy.deepcopy(fixture)
        mutated.metadata.name = f"{mutated.metadata.name}_fuzz_out_of_order"
        mutated.metadata.tags.append("fuzz")
        mutated.metadata.tags.append("out_of_order")

        # Group events that should stay together (e.g., response.create after item.create)
        # For simplicity, just shuffle with probability based on intensity
        events = mutated.client_events[:]
        num_to_shuffle = int(len(events) * self.shuffle_intensity)

        if num_to_shuffle > 1:
            # Select random indices to shuffle
            indices = random.sample(range(len(events)), num_to_shuffle)
            values = [events[i] for i in indices]
            random.shuffle(values)
            for i, val in zip(indices, values):
                events[i] = val

        mutated.client_events = events

        # Clear timing mode to IMMEDIATE since order is now scrambled
        mutated.timing_mode = "immediate"
        for event in mutated.client_events:
            event.delay_ms = 0
            event.timestamp_ms = None

        # Relax ordering assertions to non-strict
        for ordering in mutated.ordering_assertions:
            ordering.strict = False

        return mutated


class InvalidToolOutputFuzzer(FuzzStrategy):
    """Inject invalid tool call outputs to test error handling."""

    def __init__(self):
        super().__init__(
            name="invalid_tool_output",
            description="Injects malformed tool call responses",
        )

    def mutate_fixture(self, fixture: TraceFixture) -> TraceFixture:
        """Add invalid tool outputs to response.create events."""
        mutated = copy.deepcopy(fixture)
        mutated.metadata.name = f"{mutated.metadata.name}_fuzz_invalid_tool_output"
        mutated.metadata.tags.append("fuzz")
        mutated.metadata.tags.append("invalid_tool_output")

        # Find conversation.item.create events that might trigger tool calls
        for client_event in mutated.client_events:
            if client_event.event.get("type") == "conversation.item.create":
                item = client_event.event.get("item", {})
                # If this is a function call output, corrupt it
                if item.get("type") == "function_call_output":
                    # Inject invalid outputs
                    corruptions = [
                        {"output": None},  # Null output
                        {"output": 123},  # Wrong type (number instead of string)
                        {"output": {"invalid": "structure"}},  # Dict instead of string
                        {},  # Missing output field entirely
                    ]
                    item.update(random.choice(corruptions))

        # Expect error events
        mutated.event_assertions.append(
            EventAssertion(
                event_type="error",
                match_fields={"error.type": "invalid_request_error"},
                min_occurrences=1,
                max_occurrences=None,
                timeout_ms=5000,
                description="Server should emit error for invalid tool output",
            )
        )

        return mutated


class MalformedPayloadFuzzer(FuzzStrategy):
    """Generate malformed JSON payloads to test parser robustness."""

    def __init__(self):
        super().__init__(
            name="malformed_payload",
            description="Injects malformed event structures",
        )

    def mutate_fixture(self, fixture: TraceFixture) -> TraceFixture:
        """Corrupt event structures."""
        mutated = copy.deepcopy(fixture)
        mutated.metadata.name = f"{mutated.metadata.name}_fuzz_malformed"
        mutated.metadata.tags.append("fuzz")
        mutated.metadata.tags.append("malformed")

        # Apply random corruptions to events
        corruptions: list[Callable[[dict[str, Any]], dict[str, Any]]] = [
            lambda e: {**e, "type": None},  # Null type
            lambda e: {**e, "type": 123},  # Wrong type for type field
            lambda e: {k: v for k, v in e.items() if k != "type"},  # Missing type
            lambda e: {**e, "unknown_field": "x" * 10000},  # Huge unexpected field
            lambda e: {**e, "event_id": ""},  # Empty ID
            lambda e: {**e, "event_id": "x" * 1000},  # Extremely long ID
        ]

        for client_event in mutated.client_events[:len(mutated.client_events) // 2]:
            client_event.event = random.choice(corruptions)(client_event.event)

        # Expect errors
        mutated.event_assertions.append(
            EventAssertion(
                event_type="error",
                match_fields={"error.type": "invalid_request_error"},
                min_occurrences=1,
                max_occurrences=None,
                timeout_ms=5000,
                description="Server should emit errors for malformed payloads",
            )
        )

        # Clear other assertions since protocol may not proceed normally
        mutated.ordering_assertions = []
        mutated.timing_assertions = []

        return mutated


class FuzzCampaign:
    """Orchestrates multiple fuzz strategies against fixtures."""

    def __init__(self):
        self.strategies: list[FuzzStrategy] = [
            DuplicateIDFuzzer(),
            OutOfOrderFuzzer(),
            InvalidToolOutputFuzzer(),
            MalformedPayloadFuzzer(),
        ]

    def generate_fuzzed_fixtures(
        self, base_fixtures: list[TraceFixture], strategies: Optional[list[str]] = None
    ) -> list[TraceFixture]:
        """Generate fuzzed variants of base fixtures.

        Args:
            base_fixtures: Original fixtures to mutate
            strategies: List of strategy names to apply (None = all)

        Returns:
            List of fuzzed fixtures
        """
        fuzzed = []
        active_strategies = self.strategies

        if strategies:
            active_strategies = [s for s in self.strategies if s.name in strategies]

        for base_fixture in base_fixtures:
            for strategy in active_strategies:
                try:
                    fuzzed_fixture = strategy.mutate_fixture(base_fixture)
                    fuzzed.append(fuzzed_fixture)
                except Exception as e:
                    # Skip fixtures that fail to mutate
                    print(f"Warning: Failed to apply {strategy.name} to {base_fixture.metadata.name}: {e}")

        return fuzzed

    def list_strategies(self) -> list[dict[str, str]]:
        """List available fuzz strategies."""
        return [{"name": s.name, "description": s.description} for s in self.strategies]


def create_edge_case_fixture(
    name: str,
    description: str,
    client_events: list[dict[str, Any]],
    expect_error: bool = True,
) -> TraceFixture:
    """Helper to create standalone edge case fixtures.

    Args:
        name: Fixture name
        description: Human-readable description
        client_events: List of raw client event dicts
        expect_error: Whether to expect error events

    Returns:
        TraceFixture configured for edge case testing
    """
    fixture_events = [
        ClientEvent(event=event, delay_ms=100) for event in client_events
    ]

    assertions = []
    if expect_error:
        assertions.append(
            EventAssertion(
                event_type="error",
                match_fields={"error.type": "invalid_request_error"},
                min_occurrences=1,
                timeout_ms=5000,
                description="Server should emit error for invalid input",
            )
        )

    from .fixture_schema import FixtureEventType, FixtureMetadata

    return TraceFixture(
        metadata=FixtureMetadata(
            name=name,
            description=description,
            category=FixtureEventType.ERROR_HANDLING,
            tags=["edge_case", "fuzz"],
        ),
        timing_mode="relative",
        client_events=fixture_events,
        event_assertions=assertions,
    )


# Predefined edge case fixtures
def get_predefined_edge_cases() -> list[TraceFixture]:
    """Get collection of predefined edge case fixtures."""
    return [
        create_edge_case_fixture(
            name="empty_event_id",
            description="Event with empty event_id",
            client_events=[
                {
                    "type": "session.update",
                    "event_id": "",
                    "session": {"modalities": ["text"]},
                }
            ],
        ),
        create_edge_case_fixture(
            name="missing_event_type",
            description="Event missing type field",
            client_events=[
                {"event_id": str(uuid.uuid4()), "session": {"modalities": ["text"]}}
            ],
        ),
        create_edge_case_fixture(
            name="null_event_type",
            description="Event with null type",
            client_events=[
                {
                    "type": None,
                    "event_id": str(uuid.uuid4()),
                    "session": {"modalities": ["text"]},
                }
            ],
        ),
        create_edge_case_fixture(
            name="unknown_event_type",
            description="Event with unknown type",
            client_events=[
                {
                    "type": "unknown.event.type",
                    "event_id": str(uuid.uuid4()),
                }
            ],
        ),
    ]
