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


class TestFuzzGenerators:
    """Test fuzz generator functionality."""

    def test_duplicate_id_fuzzer(self):
        """Test duplicate ID fuzzer mutates event IDs."""
        from tests.realtime_harness.fuzz_generators import DuplicateIDFuzzer

        fixture = self._create_base_fixture()
        fuzzer = DuplicateIDFuzzer(duplicate_probability=1.0)
        mutated = fuzzer.mutate_fixture(fixture)

        # Check mutation applied
        assert mutated.metadata.name != fixture.metadata.name
        assert "fuzz" in mutated.metadata.tags
        assert "duplicate_ids" in mutated.metadata.tags

        # Check error assertion added
        error_assertions = [
            a for a in mutated.event_assertions if a.event_type == "error"
        ]
        assert len(error_assertions) > 0

    def test_out_of_order_fuzzer(self):
        """Test out-of-order fuzzer shuffles events."""
        from tests.realtime_harness.fuzz_generators import OutOfOrderFuzzer

        fixture = self._create_base_fixture()
        original_order = [e.event.get("type") for e in fixture.client_events]

        fuzzer = OutOfOrderFuzzer(shuffle_intensity=1.0)
        mutated = fuzzer.mutate_fixture(fixture)

        # Check mutation applied
        assert "out_of_order" in mutated.metadata.tags
        assert mutated.timing_mode == "immediate"

        # Check ordering assertions relaxed
        for ordering in mutated.ordering_assertions:
            assert ordering.strict is False

    def test_invalid_tool_output_fuzzer(self):
        """Test invalid tool output fuzzer corrupts tool responses."""
        from tests.realtime_harness.fuzz_generators import InvalidToolOutputFuzzer

        fixture = self._create_fixture_with_tool_call()
        fuzzer = InvalidToolOutputFuzzer()
        mutated = fuzzer.mutate_fixture(fixture)

        # Check mutation applied
        assert "invalid_tool_output" in mutated.metadata.tags

        # Check error assertion added
        error_assertions = [
            a for a in mutated.event_assertions if a.event_type == "error"
        ]
        assert len(error_assertions) > 0

    def test_malformed_payload_fuzzer(self):
        """Test malformed payload fuzzer corrupts event structures."""
        from tests.realtime_harness.fuzz_generators import MalformedPayloadFuzzer

        fixture = self._create_base_fixture()
        fuzzer = MalformedPayloadFuzzer()
        mutated = fuzzer.mutate_fixture(fixture)

        # Check mutation applied
        assert "malformed" in mutated.metadata.tags

        # Check error assertion added
        error_assertions = [
            a for a in mutated.event_assertions if a.event_type == "error"
        ]
        assert len(error_assertions) > 0

        # Check other assertions cleared
        assert len(mutated.ordering_assertions) == 0
        assert len(mutated.timing_assertions) == 0

    def test_fuzz_campaign(self):
        """Test fuzz campaign generates multiple variants."""
        from tests.realtime_harness.fuzz_generators import FuzzCampaign

        base_fixtures = [self._create_base_fixture()]
        campaign = FuzzCampaign()

        fuzzed = campaign.generate_fuzzed_fixtures(base_fixtures)

        # Should generate one variant per strategy
        assert len(fuzzed) >= 4  # At least 4 strategies

        # Each should have unique name
        names = [f.metadata.name for f in fuzzed]
        assert len(names) == len(set(names))

    def test_fuzz_campaign_with_strategy_filter(self):
        """Test fuzz campaign with specific strategies."""
        from tests.realtime_harness.fuzz_generators import FuzzCampaign

        base_fixtures = [self._create_base_fixture()]
        campaign = FuzzCampaign()

        fuzzed = campaign.generate_fuzzed_fixtures(
            base_fixtures, strategies=["duplicate_ids"]
        )

        # Should only have duplicate_ids variants
        assert len(fuzzed) == 1
        assert "duplicate_ids" in fuzzed[0].metadata.tags

    def test_predefined_edge_cases(self):
        """Test predefined edge case fixtures."""
        from tests.realtime_harness.fuzz_generators import get_predefined_edge_cases

        edge_cases = get_predefined_edge_cases()

        assert len(edge_cases) > 0
        for fixture in edge_cases:
            assert fixture.metadata.category == FixtureEventType.ERROR_HANDLING
            assert "edge_case" in fixture.metadata.tags

    def test_create_edge_case_fixture(self):
        """Test edge case fixture creation helper."""
        from tests.realtime_harness.fuzz_generators import create_edge_case_fixture

        fixture = create_edge_case_fixture(
            name="test_edge",
            description="Test edge case",
            client_events=[{"type": "test", "event_id": ""}],
            expect_error=True,
        )

        assert fixture.metadata.name == "test_edge"
        assert len(fixture.client_events) == 1
        assert len(fixture.event_assertions) > 0

    def _create_base_fixture(self):
        """Helper to create a base fixture for testing."""
        return TraceFixture(
            metadata=FixtureMetadata(
                name="test_base",
                description="Base fixture",
                category=FixtureEventType.TEXT_ONLY,
                tags=["test"],
            ),
            timing_mode=TimingMode.RELATIVE,
            client_events=[
                ClientEvent(
                    event={"type": "session.update", "event_id": "evt1"},
                    delay_ms=0,
                ),
                ClientEvent(
                    event={"type": "conversation.item.create", "event_id": "evt2"},
                    delay_ms=100,
                ),
            ],
            event_assertions=[
                EventAssertion(event_type="session.created"),
            ],
            ordering_assertions=[
                OrderingAssertion(
                    before="session.created",
                    after="conversation.item.created",
                    strict=True,
                )
            ],
        )

    def _create_fixture_with_tool_call(self):
        """Helper to create fixture with tool call for testing."""
        return TraceFixture(
            metadata=FixtureMetadata(
                name="test_tool_call",
                description="Tool call fixture",
                category=FixtureEventType.TOOL_CALL,
                tags=["test"],
            ),
            timing_mode=TimingMode.RELATIVE,
            client_events=[
                ClientEvent(
                    event={
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "output": "result",
                        },
                    },
                    delay_ms=0,
                ),
            ],
        )


class TestBenchmarks:
    """Test latency benchmarking functionality."""

    def test_benchmark_thresholds_defaults(self):
        """Test benchmark thresholds have sensible defaults."""
        from tests.realtime_harness.benchmarks import BenchmarkThresholds

        thresholds = BenchmarkThresholds()

        assert thresholds.ttft_ms == 1000
        assert thresholds.stt_flush_ms == 300
        assert thresholds.tool_call_rtt_ms == 2000
        assert thresholds.actuator_rtt_ms == 500

    def test_latency_measurement(self):
        """Test single latency measurement."""
        from tests.realtime_harness.benchmarks import LatencyBenchmarker

        trace = self._create_trace_with_events([
            ("response.created", 0.0),
            ("response.text.delta", 0.5),
        ])

        benchmarker = LatencyBenchmarker()
        measurement = benchmarker.measure_latency(
            trace, "response.created", "response.text.delta", "TTFT", 1000
        )

        assert measurement is not None
        assert measurement.name == "TTFT"
        assert measurement.latency_ms == 500.0
        assert measurement.passed is True

    def test_latency_measurement_exceeds_threshold(self):
        """Test latency measurement fails when exceeding threshold."""
        from tests.realtime_harness.benchmarks import LatencyBenchmarker

        trace = self._create_trace_with_events([
            ("response.created", 0.0),
            ("response.text.delta", 1.5),
        ])

        benchmarker = LatencyBenchmarker()
        measurement = benchmarker.measure_latency(
            trace, "response.created", "response.text.delta", "TTFT", 1000
        )

        assert measurement is not None
        assert measurement.latency_ms == 1500.0
        assert measurement.passed is False

    def test_latency_measurement_missing_events(self):
        """Test latency measurement returns None for missing events."""
        from tests.realtime_harness.benchmarks import LatencyBenchmarker

        trace = self._create_trace_with_events([
            ("response.created", 0.0),
        ])

        benchmarker = LatencyBenchmarker()
        measurement = benchmarker.measure_latency(
            trace, "response.created", "missing.event", "test", 1000
        )

        assert measurement is None

    def test_measure_ttft(self):
        """Test TTFT measurement."""
        from tests.realtime_harness.benchmarks import LatencyBenchmarker

        trace = self._create_trace_with_events([
            ("response.created", 0.0),
            ("response.text.delta", 0.3),
        ])

        benchmarker = LatencyBenchmarker()
        measurement = benchmarker.measure_ttft(trace)

        assert measurement is not None
        assert measurement.name == "TTFT_text"
        assert measurement.latency_ms == 300.0

    def test_measure_stt_flush(self):
        """Test STT flush latency measurement."""
        from tests.realtime_harness.benchmarks import LatencyBenchmarker

        trace = self._create_trace_with_events([
            ("input_audio_buffer.committed", 0.0),
            ("conversation.item.input_audio_transcription.completed", 0.25),
        ])

        benchmarker = LatencyBenchmarker()
        measurement = benchmarker.measure_stt_flush(trace)

        assert measurement is not None
        assert measurement.name == "STT_flush"
        assert measurement.latency_ms == 250.0

    def test_benchmark_trace(self):
        """Test benchmarking complete trace."""
        from tests.realtime_harness.benchmarks import LatencyBenchmarker

        trace = self._create_trace_with_events([
            ("response.created", 0.0),
            ("response.text.delta", 0.3),
            ("input_audio_buffer.committed", 1.0),
            ("conversation.item.input_audio_transcription.completed", 1.2),
        ])

        benchmarker = LatencyBenchmarker()
        result = benchmarker.benchmark_trace(trace, "test_fixture")

        assert result.fixture_name == "test_fixture"
        assert result.total_measurements >= 2  # TTFT and STT
        assert result.passed_measurements >= 0
        assert len(result.measurements) >= 2

    def test_benchmark_runner(self):
        """Test benchmark runner with multiple traces."""
        from tests.realtime_harness.benchmarks import BenchmarkRunner

        traces = [
            (
                self._create_trace_with_events([
                    ("response.created", 0.0),
                    ("response.text.delta", 0.3),
                ]),
                "fixture1",
            ),
            (
                self._create_trace_with_events([
                    ("response.created", 0.0),
                    ("response.text.delta", 0.5),
                ]),
                "fixture2",
            ),
        ]

        runner = BenchmarkRunner()
        report = runner.run_benchmarks(traces)

        assert report.total_fixtures == 2
        assert len(report.benchmark_results) == 2
        assert report.timestamp is not None

    def test_benchmark_report_formatting(self):
        """Test benchmark report formatting."""
        from tests.realtime_harness.benchmarks import BenchmarkRunner

        traces = [
            (
                self._create_trace_with_events([
                    ("response.created", 0.0),
                    ("response.text.delta", 0.3),
                ]),
                "test_fixture",
            ),
        ]

        runner = BenchmarkRunner()
        report = runner.run_benchmarks(traces)
        formatted = runner.format_report(report)

        assert "LATENCY BENCHMARK REPORT" in formatted
        assert "test_fixture" in formatted
        assert "Thresholds:" in formatted

    def _create_trace_with_events(self, events):
        """Helper to create trace with specific events.

        Args:
            events: List of (event_type, timestamp) tuples
        """
        trace = ReplayTrace(
            fixture_name="test",
            start_time=0.0,
            end_time=max(t for _, t in events) if events else 1.0,
        )

        trace.received_events = [
            ReceivedEvent(
                event={"type": event_type},
                timestamp_ms=timestamp * 1000,
                event_type=event_type,
            )
            for event_type, timestamp in events
        ]

        return trace
