"""Protocol compliance validators for conformance harness.

This module provides validators for checking that server behavior matches
fixture assertions for events, ordering, and timing.
"""

from dataclasses import dataclass, field
from typing import Optional

from tests.realtime_harness.fixture_parser import matches_field_constraints
from tests.realtime_harness.fixture_schema import (
    EventAssertion,
    OrderingAssertion,
    TimingAssertion,
    TraceFixture,
)
from tests.realtime_harness.websocket_replayer import ReceivedEvent, ReplayTrace


@dataclass
class ValidationFailure:
    """A validation failure with details."""

    assertion_type: str  # "event" | "ordering" | "timing"
    description: str
    details: str
    severity: str = "error"  # "error" | "warning"


@dataclass
class ValidationResult:
    """Result of validating a replay trace against fixture assertions."""

    passed: bool
    total_assertions: int
    passed_assertions: int
    failed_assertions: int
    failures: list[ValidationFailure] = field(default_factory=list)
    warnings: list[ValidationFailure] = field(default_factory=list)

    @property
    def failure_messages(self) -> list[str]:
        """Get all failure messages."""
        return [f"{f.assertion_type}: {f.description} - {f.details}" for f in self.failures]

    @property
    def warning_messages(self) -> list[str]:
        """Get all warning messages."""
        return [f"{w.assertion_type}: {w.description} - {w.details}" for w in self.warnings]


class ProtocolValidator:
    """Validates replay traces against fixture assertions."""

    def __init__(self, fixture: TraceFixture, trace: ReplayTrace):
        """Initialize validator with fixture and trace.

        Args:
            fixture: Fixture with assertions
            trace: Replay trace to validate
        """
        self.fixture = fixture
        self.trace = trace

    def validate_all(self) -> ValidationResult:
        """Run all validators and return combined result.

        Returns:
            Validation result with all failures and warnings
        """
        result = ValidationResult(
            passed=True,
            total_assertions=0,
            passed_assertions=0,
            failed_assertions=0,
        )

        # Validate event assertions
        if self.fixture.event_assertions:
            event_result = self.validate_event_assertions()
            self._merge_result(result, event_result)

        # Validate ordering assertions
        if self.fixture.ordering_assertions:
            ordering_result = self.validate_ordering_assertions()
            self._merge_result(result, ordering_result)

        # Validate timing assertions
        if self.fixture.timing_assertions:
            timing_result = self.validate_timing_assertions()
            self._merge_result(result, timing_result)

        return result

    def validate_event_assertions(self) -> ValidationResult:
        """Validate event assertions from fixture.

        Returns:
            Validation result for event assertions
        """
        result = ValidationResult(
            passed=True,
            total_assertions=len(self.fixture.event_assertions),
            passed_assertions=0,
            failed_assertions=0,
        )

        for assertion in self.fixture.event_assertions:
            failure = self._validate_single_event_assertion(assertion)

            if failure:
                result.failed_assertions += 1
                result.failures.append(failure)
                result.passed = False
            else:
                result.passed_assertions += 1

        return result

    def _validate_single_event_assertion(
        self, assertion: EventAssertion
    ) -> Optional[ValidationFailure]:
        """Validate a single event assertion.

        Args:
            assertion: Event assertion to validate

        Returns:
            ValidationFailure if assertion failed, None otherwise
        """
        # Find matching events
        matching_events = self.trace.get_events_by_type(assertion.event_type)

        # Apply field constraints if specified
        if assertion.match_fields or assertion.exclude_fields:
            filtered_events = []
            for event in matching_events:
                matches, _ = matches_field_constraints(
                    event.event, assertion.match_fields, assertion.exclude_fields
                )
                if matches:
                    filtered_events.append(event)
            matching_events = filtered_events

        occurrences = len(matching_events)

        # Check min_occurrences
        if occurrences < assertion.min_occurrences:
            return ValidationFailure(
                assertion_type="event",
                description=assertion.description or f"Event {assertion.event_type}",
                details=f"Expected at least {assertion.min_occurrences} occurrence(s), got {occurrences}",
            )

        # Check max_occurrences
        if assertion.max_occurrences is not None and occurrences > assertion.max_occurrences:
            return ValidationFailure(
                assertion_type="event",
                description=assertion.description or f"Event {assertion.event_type}",
                details=f"Expected at most {assertion.max_occurrences} occurrence(s), got {occurrences}",
            )

        # Check field constraints for first matching event
        if matching_events and assertion.match_fields:
            first_event = matching_events[0]
            matches, reason = matches_field_constraints(
                first_event.event, assertion.match_fields, assertion.exclude_fields
            )

            if not matches:
                return ValidationFailure(
                    assertion_type="event",
                    description=assertion.description or f"Event {assertion.event_type}",
                    details=f"Field constraint failed: {reason}",
                )

        return None

    def validate_ordering_assertions(self) -> ValidationResult:
        """Validate ordering assertions from fixture.

        Returns:
            Validation result for ordering assertions
        """
        result = ValidationResult(
            passed=True,
            total_assertions=len(self.fixture.ordering_assertions),
            passed_assertions=0,
            failed_assertions=0,
        )

        for assertion in self.fixture.ordering_assertions:
            failure = self._validate_single_ordering_assertion(assertion)

            if failure:
                result.failed_assertions += 1
                result.failures.append(failure)
                result.passed = False
            else:
                result.passed_assertions += 1

        return result

    def _validate_single_ordering_assertion(
        self, assertion: OrderingAssertion
    ) -> Optional[ValidationFailure]:
        """Validate a single ordering assertion.

        Args:
            assertion: Ordering assertion to validate

        Returns:
            ValidationFailure if assertion failed, None otherwise
        """
        before_events = self.trace.get_events_by_type(assertion.before)
        after_events = self.trace.get_events_by_type(assertion.after)

        # Check that both event types occurred
        if not before_events:
            return ValidationFailure(
                assertion_type="ordering",
                description=assertion.description
                or f"{assertion.before} before {assertion.after}",
                details=f"Event type '{assertion.before}' never occurred",
            )

        if not after_events:
            return ValidationFailure(
                assertion_type="ordering",
                description=assertion.description
                or f"{assertion.before} before {assertion.after}",
                details=f"Event type '{assertion.after}' never occurred",
            )

        first_before = before_events[0]
        first_after = after_events[0]

        if assertion.strict:
            # Strict: NO 'after' events before first 'before'
            for after_event in after_events:
                if after_event.timestamp_ms < first_before.timestamp_ms:
                    return ValidationFailure(
                        assertion_type="ordering",
                        description=assertion.description
                        or f"{assertion.before} before {assertion.after}",
                        details=f"'{assertion.after}' occurred at {after_event.timestamp_ms:.2f}ms "
                        f"before first '{assertion.before}' at {first_before.timestamp_ms:.2f}ms (strict mode)",
                    )
        else:
            # Non-strict: Just ensure at least one 'before' comes before at least one 'after'
            if first_after.timestamp_ms <= first_before.timestamp_ms:
                return ValidationFailure(
                    assertion_type="ordering",
                    description=assertion.description
                    or f"{assertion.before} before {assertion.after}",
                    details=f"First '{assertion.after}' at {first_after.timestamp_ms:.2f}ms "
                    f"did not come after first '{assertion.before}' at {first_before.timestamp_ms:.2f}ms",
                )

        return None

    def validate_timing_assertions(self) -> ValidationResult:
        """Validate timing assertions from fixture.

        Returns:
            Validation result for timing assertions
        """
        result = ValidationResult(
            passed=True,
            total_assertions=len(self.fixture.timing_assertions),
            passed_assertions=0,
            failed_assertions=0,
        )

        for assertion in self.fixture.timing_assertions:
            failure = self._validate_single_timing_assertion(assertion)

            if failure:
                result.failed_assertions += 1
                result.failures.append(failure)
                result.passed = False
            else:
                result.passed_assertions += 1

        return result

    def _validate_single_timing_assertion(
        self, assertion: TimingAssertion
    ) -> Optional[ValidationFailure]:
        """Validate a single timing assertion.

        Args:
            assertion: Timing assertion to validate

        Returns:
            ValidationFailure if assertion failed, None otherwise
        """
        event_a = self.trace.get_first_event(assertion.event_a)
        event_b = self.trace.get_first_event(assertion.event_b)

        # Check that both events occurred
        if event_a is None:
            return ValidationFailure(
                assertion_type="timing",
                description=assertion.description
                or f"{assertion.event_a} -> {assertion.event_b}",
                details=f"Event type '{assertion.event_a}' never occurred",
            )

        if event_b is None:
            return ValidationFailure(
                assertion_type="timing",
                description=assertion.description
                or f"{assertion.event_a} -> {assertion.event_b}",
                details=f"Event type '{assertion.event_b}' never occurred",
            )

        # Calculate latency
        latency_ms = event_b.timestamp_ms - event_a.timestamp_ms

        if latency_ms < 0:
            return ValidationFailure(
                assertion_type="timing",
                description=assertion.description
                or f"{assertion.event_a} -> {assertion.event_b}",
                details=f"'{assertion.event_b}' occurred before '{assertion.event_a}' "
                f"(negative latency: {latency_ms:.2f}ms)",
            )

        if latency_ms > assertion.max_latency_ms:
            return ValidationFailure(
                assertion_type="timing",
                description=assertion.description
                or f"{assertion.event_a} -> {assertion.event_b}",
                details=f"Latency {latency_ms:.2f}ms exceeded maximum {assertion.max_latency_ms}ms",
            )

        return None

    def _merge_result(
        self, target: ValidationResult, source: ValidationResult
    ) -> None:
        """Merge source validation result into target.

        Args:
            target: Target result to merge into
            source: Source result to merge from
        """
        target.total_assertions += source.total_assertions
        target.passed_assertions += source.passed_assertions
        target.failed_assertions += source.failed_assertions
        target.failures.extend(source.failures)
        target.warnings.extend(source.warnings)

        if not source.passed:
            target.passed = False
