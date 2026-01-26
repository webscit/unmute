"""Unit tests for the conformance harness components.

Tests fixture parsing, validation, field matching, and protocol compliance checking.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.realtime_harness.fixture_parser import (
    FixtureParser,
    get_nested_field,
    matches_field_constraints,
)
from tests.realtime_harness.fixture_schema import (
    ClientEvent,
    EventAssertion,
    FixtureEventType,
    FixtureMetadata,
    OrderingAssertion,
    TimingAssertion,
    TimingMode,
    TraceFixture,
)
from tests.realtime_harness.validators import (
    ProtocolValidator,
    ValidationFailure,
)
from tests.realtime_harness.websocket_replayer import ReceivedEvent, ReplayTrace


class TestFixtureSchema:
    """Test fixture schema validation."""

    def test_minimal_fixture(self):
        """Test minimal valid fixture."""
        fixture_data = {
            "metadata": {
                "name": "test_minimal",
                "description": "Minimal test fixture",
                "category": "text_only",
                "timeout_seconds": 10,
            },
            "timing_mode": "relative",
            "client_events": [
                {"event": {"type": "session.update", "session": {}}, "delay_ms": 0}
            ],
        }

        fixture = TraceFixture(**fixture_data)
        assert fixture.metadata.name == "test_minimal"
        assert len(fixture.client_events) == 1
        assert len(fixture.event_assertions) == 0

    def test_empty_client_events_fails(self):
        """Test that fixtures must have at least one client event."""
        fixture_data = {
            "metadata": {
                "name": "test_empty",
                "description": "Empty events",
                "category": "text_only",
            },
            "client_events": [],
        }

        with pytest.raises(ValidationError) as exc_info:
            TraceFixture(**fixture_data)

        assert "client_events cannot be empty" in str(exc_info.value)

    def test_negative_delay_fails(self):
        """Test that negative delays are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ClientEvent(event={"type": "test"}, delay_ms=-100)

        assert "delay_ms must be non-negative" in str(exc_info.value)

    def test_negative_latency_fails(self):
        """Test that negative latency constraints are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TimingAssertion(
                event_a="a",
                event_b="b",
                max_latency_ms=-1000,
            )

        assert "max_latency_ms must be non-negative" in str(exc_info.value)

    def test_event_assertion_defaults(self):
        """Test event assertion default values."""
        assertion = EventAssertion(event_type="test.event")

        assert assertion.min_occurrences == 1
        assert assertion.max_occurrences is None
        assert assertion.timeout_ms == 5000
        assert assertion.match_fields is None

    def test_timing_modes(self):
        """Test timing mode enum values."""
        assert TimingMode.RELATIVE == "relative"
        assert TimingMode.ABSOLUTE == "absolute"
        assert TimingMode.IMMEDIATE == "immediate"

    def test_fixture_categories(self):
        """Test fixture category enum values."""
        assert FixtureEventType.TEXT_ONLY == "text_only"
        assert FixtureEventType.AUDIO_INPUT == "audio_input"
        assert FixtureEventType.TOOL_CALL == "tool_call"


class TestFieldMatching:
    """Test nested field access and matching utilities."""

    def test_get_nested_field_dict(self):
        """Test accessing nested fields in dictionaries."""
        obj = {"a": {"b": {"c": "value"}}}

        assert get_nested_field(obj, "a.b.c") == "value"
        assert get_nested_field(obj, "a.b") == {"c": "value"}
        assert get_nested_field(obj, "a") == {"b": {"c": "value"}}

    def test_get_nested_field_not_found(self):
        """Test that missing fields return None."""
        obj = {"a": {"b": "value"}}

        assert get_nested_field(obj, "a.x") is None
        assert get_nested_field(obj, "x.y.z") is None

    def test_matches_field_constraints_success(self):
        """Test successful field matching."""
        event = {
            "type": "test.event",
            "data": {"status": "success", "count": 42},
        }

        matches, reason = matches_field_constraints(
            event, match_fields={"data.status": "success", "data.count": 42}
        )

        assert matches is True
        assert reason is None

    def test_matches_field_constraints_mismatch(self):
        """Test field matching with mismatched values."""
        event = {"type": "test.event", "data": {"status": "failed"}}

        matches, reason = matches_field_constraints(
            event, match_fields={"data.status": "success"}
        )

        assert matches is False
        assert "expected success, got failed" in reason

    def test_exclude_fields_present(self):
        """Test exclude_fields when field is present."""
        event = {"type": "test.event", "error": "some error"}

        matches, reason = matches_field_constraints(
            event, exclude_fields=["error"]
        )

        assert matches is False
        assert "should not be present" in reason

    def test_exclude_fields_absent(self):
        """Test exclude_fields when field is absent."""
        event = {"type": "test.event", "data": "value"}

        matches, reason = matches_field_constraints(
            event, exclude_fields=["error"]
        )

        assert matches is True
        assert reason is None


class TestFixtureParser:
    """Test fixture parser and loader."""

    @pytest.fixture
    def temp_fixtures_dir(self, tmp_path):
        """Create temporary fixtures directory."""
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        return fixtures_dir

    @pytest.fixture
    def sample_fixture_file(self, temp_fixtures_dir):
        """Create sample fixture file."""
        fixture_data = {
            "metadata": {
                "name": "sample",
                "description": "Sample fixture",
                "category": "text_only",
                "tags": ["test", "basic"],
            },
            "client_events": [
                {"event": {"type": "test"}, "delay_ms": 0}
            ],
        }

        fixture_file = temp_fixtures_dir / "sample.json"
        fixture_file.write_text(json.dumps(fixture_data))
        return fixture_file

    def test_load_fixture_success(self, temp_fixtures_dir, sample_fixture_file):
        """Test loading a valid fixture."""
        parser = FixtureParser(fixtures_dir=temp_fixtures_dir)
        fixture = parser.load_fixture("sample")

        assert fixture.metadata.name == "sample"
        assert len(fixture.client_events) == 1

    def test_load_fixture_with_extension(self, temp_fixtures_dir, sample_fixture_file):
        """Test loading fixture with .json extension."""
        parser = FixtureParser(fixtures_dir=temp_fixtures_dir)
        fixture = parser.load_fixture("sample.json")

        assert fixture.metadata.name == "sample"

    def test_load_fixture_not_found(self, temp_fixtures_dir):
        """Test loading non-existent fixture."""
        parser = FixtureParser(fixtures_dir=temp_fixtures_dir)

        with pytest.raises(Exception) as exc_info:
            parser.load_fixture("nonexistent")

        assert "not found" in str(exc_info.value).lower()

    def test_load_all_fixtures(self, temp_fixtures_dir):
        """Test loading all fixtures from directory."""
        # Create multiple fixtures
        for i in range(3):
            fixture_data = {
                "metadata": {
                    "name": f"fixture_{i}",
                    "description": f"Fixture {i}",
                    "category": "text_only",
                },
                "client_events": [{"event": {"type": "test"}, "delay_ms": 0}],
            }
            (temp_fixtures_dir / f"fixture_{i}.json").write_text(
                json.dumps(fixture_data)
            )

        parser = FixtureParser(fixtures_dir=temp_fixtures_dir)
        fixtures = parser.load_all_fixtures()

        assert len(fixtures) == 3

    def test_load_fixtures_with_category_filter(self, temp_fixtures_dir):
        """Test loading fixtures filtered by category."""
        # Create fixtures with different categories
        for category in ["text_only", "audio_input"]:
            fixture_data = {
                "metadata": {
                    "name": f"{category}_test",
                    "description": f"Test {category}",
                    "category": category,
                },
                "client_events": [{"event": {"type": "test"}, "delay_ms": 0}],
            }
            (temp_fixtures_dir / f"{category}.json").write_text(
                json.dumps(fixture_data)
            )

        parser = FixtureParser(fixtures_dir=temp_fixtures_dir)
        fixtures = parser.load_all_fixtures(category=FixtureEventType.TEXT_ONLY)

        assert len(fixtures) == 1
        assert fixtures[0].metadata.category == FixtureEventType.TEXT_ONLY

    def test_validate_fixture_file_success(self, temp_fixtures_dir, sample_fixture_file):
        """Test validating a valid fixture file."""
        parser = FixtureParser(fixtures_dir=temp_fixtures_dir)
        is_valid, error = parser.validate_fixture_file(sample_fixture_file)

        assert is_valid is True
        assert error is None

    def test_validate_fixture_file_invalid_json(self, temp_fixtures_dir):
        """Test validating file with invalid JSON."""
        invalid_file = temp_fixtures_dir / "invalid.json"
        invalid_file.write_text("{invalid json")

        parser = FixtureParser(fixtures_dir=temp_fixtures_dir)
        is_valid, error = parser.validate_fixture_file(invalid_file)

        assert is_valid is False
        assert "JSON" in error


class TestReplayTrace:
    """Test replay trace data structures."""

    def test_trace_duration(self):
        """Test trace duration calculation."""
        trace = ReplayTrace(fixture_name="test", start_time=100.0, end_time=105.0)

        assert trace.duration_ms == 5000.0

    def test_get_events_by_type(self):
        """Test filtering events by type."""
        trace = ReplayTrace(fixture_name="test", start_time=0.0)
        trace.received_events = [
            ReceivedEvent(event={"type": "a"}, timestamp_ms=100, event_type="a"),
            ReceivedEvent(event={"type": "b"}, timestamp_ms=200, event_type="b"),
            ReceivedEvent(event={"type": "a"}, timestamp_ms=300, event_type="a"),
        ]

        events_a = trace.get_events_by_type("a")
        assert len(events_a) == 2

        events_b = trace.get_events_by_type("b")
        assert len(events_b) == 1

    def test_get_first_event(self):
        """Test getting first event of a type."""
        trace = ReplayTrace(fixture_name="test", start_time=0.0)
        trace.received_events = [
            ReceivedEvent(event={"type": "a"}, timestamp_ms=100, event_type="a"),
            ReceivedEvent(event={"type": "a"}, timestamp_ms=200, event_type="a"),
        ]

        first_a = trace.get_first_event("a")
        assert first_a.timestamp_ms == 100

        first_b = trace.get_first_event("b")
        assert first_b is None

    def test_get_latency_ms(self):
        """Test calculating latency between events."""
        trace = ReplayTrace(fixture_name="test", start_time=0.0)
        trace.received_events = [
            ReceivedEvent(event={"type": "a"}, timestamp_ms=100, event_type="a"),
            ReceivedEvent(event={"type": "b"}, timestamp_ms=250, event_type="b"),
        ]

        latency = trace.get_latency_ms("a", "b")
        assert latency == 150.0

    def test_get_latency_ms_missing_events(self):
        """Test latency calculation with missing events."""
        trace = ReplayTrace(fixture_name="test", start_time=0.0)
        trace.received_events = [
            ReceivedEvent(event={"type": "a"}, timestamp_ms=100, event_type="a"),
        ]

        latency = trace.get_latency_ms("a", "b")
        assert latency is None


class TestProtocolValidator:
    """Test protocol compliance validators."""

    def test_validate_event_assertion_success(self):
        """Test successful event assertion."""
        fixture = TraceFixture(
            metadata=FixtureMetadata(
                name="test",
                description="Test",
                category=FixtureEventType.TEXT_ONLY,
            ),
            client_events=[ClientEvent(event={"type": "test"})],
            event_assertions=[
                EventAssertion(
                    event_type="response.created",
                    min_occurrences=1,
                    max_occurrences=1,
                )
            ],
        )

        trace = ReplayTrace(fixture_name="test", start_time=0.0, end_time=1.0)
        trace.received_events = [
            ReceivedEvent(
                event={"type": "response.created"},
                timestamp_ms=100,
                event_type="response.created",
            )
        ]

        validator = ProtocolValidator(fixture, trace)
        result = validator.validate_event_assertions()

        assert result.passed is True
        assert result.passed_assertions == 1
        assert result.failed_assertions == 0

    def test_validate_event_assertion_min_occurrences_fail(self):
        """Test event assertion failing on min_occurrences."""
        fixture = TraceFixture(
            metadata=FixtureMetadata(
                name="test",
                description="Test",
                category=FixtureEventType.TEXT_ONLY,
            ),
            client_events=[ClientEvent(event={"type": "test"})],
            event_assertions=[
                EventAssertion(event_type="response.created", min_occurrences=2)
            ],
        )

        trace = ReplayTrace(fixture_name="test", start_time=0.0, end_time=1.0)
        trace.received_events = [
            ReceivedEvent(
                event={"type": "response.created"},
                timestamp_ms=100,
                event_type="response.created",
            )
        ]

        validator = ProtocolValidator(fixture, trace)
        result = validator.validate_event_assertions()

        assert result.passed is False
        assert result.failed_assertions == 1
        assert "at least 2" in result.failures[0].details

    def test_validate_ordering_assertion_success(self):
        """Test successful ordering assertion."""
        fixture = TraceFixture(
            metadata=FixtureMetadata(
                name="test",
                description="Test",
                category=FixtureEventType.TEXT_ONLY,
            ),
            client_events=[ClientEvent(event={"type": "test"})],
            ordering_assertions=[
                OrderingAssertion(before="event_a", after="event_b", strict=False)
            ],
        )

        trace = ReplayTrace(fixture_name="test", start_time=0.0, end_time=1.0)
        trace.received_events = [
            ReceivedEvent(event={"type": "a"}, timestamp_ms=100, event_type="event_a"),
            ReceivedEvent(event={"type": "b"}, timestamp_ms=200, event_type="event_b"),
        ]

        validator = ProtocolValidator(fixture, trace)
        result = validator.validate_ordering_assertions()

        assert result.passed is True

    def test_validate_ordering_assertion_strict_fail(self):
        """Test strict ordering assertion failure."""
        fixture = TraceFixture(
            metadata=FixtureMetadata(
                name="test",
                description="Test",
                category=FixtureEventType.TEXT_ONLY,
            ),
            client_events=[ClientEvent(event={"type": "test"})],
            ordering_assertions=[
                OrderingAssertion(before="event_a", after="event_b", strict=True)
            ],
        )

        trace = ReplayTrace(fixture_name="test", start_time=0.0, end_time=1.0)
        trace.received_events = [
            ReceivedEvent(event={"type": "b"}, timestamp_ms=100, event_type="event_b"),
            ReceivedEvent(event={"type": "a"}, timestamp_ms=200, event_type="event_a"),
        ]

        validator = ProtocolValidator(fixture, trace)
        result = validator.validate_ordering_assertions()

        assert result.passed is False
        assert "strict mode" in result.failures[0].details

    def test_validate_timing_assertion_success(self):
        """Test successful timing assertion."""
        fixture = TraceFixture(
            metadata=FixtureMetadata(
                name="test",
                description="Test",
                category=FixtureEventType.TEXT_ONLY,
            ),
            client_events=[ClientEvent(event={"type": "test"})],
            timing_assertions=[
                TimingAssertion(
                    event_a="event_a", event_b="event_b", max_latency_ms=1000
                )
            ],
        )

        trace = ReplayTrace(fixture_name="test", start_time=0.0, end_time=1.0)
        trace.received_events = [
            ReceivedEvent(event={"type": "a"}, timestamp_ms=100, event_type="event_a"),
            ReceivedEvent(event={"type": "b"}, timestamp_ms=500, event_type="event_b"),
        ]

        validator = ProtocolValidator(fixture, trace)
        result = validator.validate_timing_assertions()

        assert result.passed is True

    def test_validate_timing_assertion_exceeded(self):
        """Test timing assertion exceeding max latency."""
        fixture = TraceFixture(
            metadata=FixtureMetadata(
                name="test",
                description="Test",
                category=FixtureEventType.TEXT_ONLY,
            ),
            client_events=[ClientEvent(event={"type": "test"})],
            timing_assertions=[
                TimingAssertion(
                    event_a="event_a", event_b="event_b", max_latency_ms=100
                )
            ],
        )

        trace = ReplayTrace(fixture_name="test", start_time=0.0, end_time=1.0)
        trace.received_events = [
            ReceivedEvent(event={"type": "a"}, timestamp_ms=100, event_type="event_a"),
            ReceivedEvent(event={"type": "b"}, timestamp_ms=500, event_type="event_b"),
        ]

        validator = ProtocolValidator(fixture, trace)
        result = validator.validate_timing_assertions()

        assert result.passed is False
        assert "exceeded maximum" in result.failures[0].details

    def test_validate_all_combined(self):
        """Test combined validation of all assertion types."""
        fixture = TraceFixture(
            metadata=FixtureMetadata(
                name="test",
                description="Test",
                category=FixtureEventType.TEXT_ONLY,
            ),
            client_events=[ClientEvent(event={"type": "test"})],
            event_assertions=[
                EventAssertion(event_type="event_a", min_occurrences=1)
            ],
            ordering_assertions=[
                OrderingAssertion(before="event_a", after="event_b", strict=False)
            ],
            timing_assertions=[
                TimingAssertion(
                    event_a="event_a", event_b="event_b", max_latency_ms=1000
                )
            ],
        )

        trace = ReplayTrace(fixture_name="test", start_time=0.0, end_time=1.0)
        trace.received_events = [
            ReceivedEvent(event={"type": "a"}, timestamp_ms=100, event_type="event_a"),
            ReceivedEvent(event={"type": "b"}, timestamp_ms=200, event_type="event_b"),
        ]

        validator = ProtocolValidator(fixture, trace)
        result = validator.validate_all()

        assert result.passed is True
        assert result.total_assertions == 3
        assert result.passed_assertions == 3
