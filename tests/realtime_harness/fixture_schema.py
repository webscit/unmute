"""Pydantic schemas for conformance harness trace fixtures.

This module defines the structure of trace fixtures used to test OpenAI Realtime API
protocol compliance. Fixtures capture sequences of client events, expected server
responses, and timing/ordering constraints.
"""

from datetime import timedelta
from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


class TimingMode(str, Enum):
    """How to interpret timing values in fixtures."""

    ABSOLUTE = "absolute"  # Exact timestamps from trace start
    RELATIVE = "relative"  # Delays from previous event
    IMMEDIATE = "immediate"  # Send as fast as possible


class FixtureEventType(str, Enum):
    """Categories of fixture scenarios."""

    TEXT_ONLY = "text_only"
    AUDIO_INPUT = "audio_input"
    TOOL_CALL = "tool_call"
    MULTIMODAL = "multimodal"
    SESSION_UPDATE = "session_update"
    ERROR_HANDLING = "error_handling"


class ClientEvent(BaseModel):
    """A client event to send during fixture replay."""

    event: dict[str, Any] = Field(
        ..., description="Raw event payload (will be validated against OpenAI schema)"
    )
    delay_ms: int = Field(
        0, description="Milliseconds to wait before sending (in RELATIVE timing mode)"
    )
    timestamp_ms: Optional[int] = Field(
        None,
        description="Absolute timestamp from trace start (in ABSOLUTE timing mode)",
    )
    description: Optional[str] = Field(
        None, description="Human-readable description of this event"
    )

    @field_validator("delay_ms")
    @classmethod
    def validate_delay(cls, v: int) -> int:
        if v < 0:
            raise ValueError("delay_ms must be non-negative")
        return v


class EventAssertion(BaseModel):
    """Assertion about a server event that should be received."""

    event_type: str = Field(..., description="Expected event.type field value")
    match_fields: Optional[dict[str, Any]] = Field(
        None,
        description="Fields that must match exactly (dot notation supported, e.g., 'session.voice')",
    )
    exclude_fields: Optional[list[str]] = Field(
        None,
        description="Fields that should NOT be present (dot notation supported)",
    )
    min_occurrences: int = Field(1, description="Minimum times this event must occur")
    max_occurrences: Optional[int] = Field(
        None, description="Maximum times this event can occur (None = unlimited)"
    )
    timeout_ms: int = Field(
        5000, description="Max milliseconds to wait for this event"
    )
    description: Optional[str] = Field(
        None, description="Human-readable description of this assertion"
    )

    @field_validator("min_occurrences")
    @classmethod
    def validate_min_occurrences(cls, v: int) -> int:
        if v < 0:
            raise ValueError("min_occurrences must be non-negative")
        return v


class OrderingAssertion(BaseModel):
    """Assertion about the ordering of events."""

    before: str = Field(..., description="Event type that must come before")
    after: str = Field(..., description="Event type that must come after")
    strict: bool = Field(
        False,
        description="If True, no events of 'after' type can occur before first 'before'",
    )
    description: Optional[str] = Field(
        None, description="Human-readable description of this ordering constraint"
    )


class TimingAssertion(BaseModel):
    """Assertion about timing between events."""

    event_a: str = Field(..., description="First event type")
    event_b: str = Field(..., description="Second event type")
    max_latency_ms: int = Field(
        ..., description="Max milliseconds allowed between event_a and event_b"
    )
    description: Optional[str] = Field(
        None, description="Human-readable description of this timing constraint"
    )

    @field_validator("max_latency_ms")
    @classmethod
    def validate_latency(cls, v: int) -> int:
        if v < 0:
            raise ValueError("max_latency_ms must be non-negative")
        return v


class FixtureMetadata(BaseModel):
    """Metadata describing a trace fixture."""

    name: str = Field(..., description="Unique fixture name")
    description: str = Field(..., description="Human-readable description")
    category: FixtureEventType = Field(..., description="Fixture category")
    tags: list[str] = Field(default_factory=list, description="Tags for filtering")
    timeout_seconds: int = Field(
        30, description="Maximum time for entire fixture to complete"
    )


class TraceFixture(BaseModel):
    """Complete trace fixture for protocol conformance testing."""

    metadata: FixtureMetadata = Field(..., description="Fixture metadata")
    timing_mode: TimingMode = Field(
        TimingMode.RELATIVE, description="How to interpret timing values"
    )
    client_events: list[ClientEvent] = Field(
        ..., description="Sequence of client events to send"
    )
    event_assertions: list[EventAssertion] = Field(
        default_factory=list,
        description="Assertions about server events that should be received",
    )
    ordering_assertions: list[OrderingAssertion] = Field(
        default_factory=list, description="Assertions about event ordering"
    )
    timing_assertions: list[TimingAssertion] = Field(
        default_factory=list, description="Assertions about event timing"
    )

    @field_validator("client_events")
    @classmethod
    def validate_client_events(cls, v: list[ClientEvent]) -> list[ClientEvent]:
        if not v:
            raise ValueError("client_events cannot be empty")
        return v


class FixtureResult(BaseModel):
    """Result of running a single fixture."""

    fixture_name: str
    passed: bool
    duration_ms: float
    assertions_passed: int
    assertions_failed: int
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    server_events_received: int = 0


class HarnessReport(BaseModel):
    """Overall report from harness run."""

    timestamp: str
    total_fixtures: int
    fixtures_passed: int
    fixtures_failed: int
    total_duration_ms: float
    fixture_results: list[FixtureResult]
    environment: dict[str, str] = Field(default_factory=dict)
