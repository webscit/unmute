"""Fixture parser and validator for conformance harness.

This module provides utilities for loading, validating, and accessing trace fixtures
used in protocol conformance testing.
"""

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from tests.realtime_harness.fixture_schema import FixtureEventType, TraceFixture


class FixtureLoadError(Exception):
    """Raised when a fixture cannot be loaded or is invalid."""

    pass


class FixtureParser:
    """Parser for trace fixture files."""

    def __init__(self, fixtures_dir: Optional[Path] = None):
        """Initialize parser with fixtures directory.

        Args:
            fixtures_dir: Path to directory containing fixture files.
                         Defaults to fixtures/ relative to this module.
        """
        if fixtures_dir is None:
            fixtures_dir = Path(__file__).parent / "fixtures"
        self.fixtures_dir = Path(fixtures_dir)

        if not self.fixtures_dir.exists():
            raise FixtureLoadError(
                f"Fixtures directory not found: {self.fixtures_dir}"
            )

    def load_fixture(self, fixture_name: str) -> TraceFixture:
        """Load a single fixture by name.

        Args:
            fixture_name: Name of fixture file (with or without .json extension)

        Returns:
            Validated TraceFixture instance

        Raises:
            FixtureLoadError: If fixture file not found or invalid
        """
        # Add .json extension if not present
        if not fixture_name.endswith(".json"):
            fixture_name = f"{fixture_name}.json"

        fixture_path = self.fixtures_dir / fixture_name

        if not fixture_path.exists():
            raise FixtureLoadError(f"Fixture not found: {fixture_path}")

        try:
            with open(fixture_path, "r") as f:
                fixture_data = json.load(f)
        except json.JSONDecodeError as e:
            raise FixtureLoadError(f"Invalid JSON in {fixture_path}: {e}")

        try:
            fixture = TraceFixture(**fixture_data)
        except ValidationError as e:
            raise FixtureLoadError(f"Fixture validation failed for {fixture_path}:\n{e}")

        return fixture

    def load_all_fixtures(
        self,
        category: Optional[FixtureEventType] = None,
        tags: Optional[list[str]] = None,
    ) -> list[TraceFixture]:
        """Load all fixtures from the fixtures directory.

        Args:
            category: Optional filter by fixture category
            tags: Optional filter by tags (fixture must have ALL specified tags)

        Returns:
            List of validated TraceFixture instances
        """
        fixtures = []

        for fixture_file in self.fixtures_dir.glob("*.json"):
            try:
                fixture = self.load_fixture(fixture_file.name)

                # Apply filters
                if category and fixture.metadata.category != category:
                    continue

                if tags and not all(tag in fixture.metadata.tags for tag in tags):
                    continue

                fixtures.append(fixture)

            except FixtureLoadError as e:
                # Log warning but continue loading other fixtures
                print(f"Warning: Skipping {fixture_file.name}: {e}")
                continue

        return fixtures

    def list_fixtures(self) -> list[str]:
        """List all available fixture names.

        Returns:
            List of fixture names (without .json extension)
        """
        return [f.stem for f in self.fixtures_dir.glob("*.json")]

    def validate_fixture_file(self, fixture_path: Path) -> tuple[bool, Optional[str]]:
        """Validate a fixture file without loading it fully.

        Args:
            fixture_path: Path to fixture file

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            with open(fixture_path, "r") as f:
                fixture_data = json.load(f)
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"
        except Exception as e:
            return False, f"Error reading file: {e}"

        try:
            TraceFixture(**fixture_data)
            return True, None
        except ValidationError as e:
            return False, str(e)


def get_nested_field(obj: Any, field_path: str) -> Any:
    """Get a nested field from an object using dot notation.

    Args:
        obj: Object (dict or Pydantic model) to access
        field_path: Dot-separated path (e.g., 'session.voice')

    Returns:
        Field value or None if not found

    Examples:
        >>> get_nested_field({'a': {'b': 'c'}}, 'a.b')
        'c'
        >>> get_nested_field({'a': {'b': 'c'}}, 'a.x')
        None
    """
    parts = field_path.split(".")
    current = obj

    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            # Pydantic model or object
            current = getattr(current, part, None)

        if current is None:
            return None

    return current


def matches_field_constraints(
    event: dict[str, Any],
    match_fields: Optional[dict[str, Any]] = None,
    exclude_fields: Optional[list[str]] = None,
) -> tuple[bool, Optional[str]]:
    """Check if an event matches field constraints.

    Args:
        event: Event dictionary to check
        match_fields: Fields that must match exactly (dot notation)
        exclude_fields: Fields that must NOT be present (dot notation)

    Returns:
        Tuple of (matches, failure_reason)
    """
    # Check match_fields
    if match_fields:
        for field_path, expected_value in match_fields.items():
            actual_value = get_nested_field(event, field_path)

            if actual_value != expected_value:
                return (
                    False,
                    f"Field {field_path}: expected {expected_value}, got {actual_value}",
                )

    # Check exclude_fields
    if exclude_fields:
        for field_path in exclude_fields:
            actual_value = get_nested_field(event, field_path)

            if actual_value is not None:
                return False, f"Field {field_path} should not be present"

    return True, None


# CLI utility for validating fixtures
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m tests.realtime_harness.fixture_parser <fixture_file>")
        sys.exit(1)

    fixture_path = Path(sys.argv[1])

    if not fixture_path.exists():
        print(f"Error: File not found: {fixture_path}")
        sys.exit(1)

    parser = FixtureParser()
    is_valid, error = parser.validate_fixture_file(fixture_path)

    if is_valid:
        print(f"✓ Fixture is valid: {fixture_path}")
        # Load and print summary
        fixture = parser.load_fixture(fixture_path.name)
        print(f"\nFixture: {fixture.metadata.name}")
        print(f"Category: {fixture.metadata.category}")
        print(f"Description: {fixture.metadata.description}")
        print(f"Client events: {len(fixture.client_events)}")
        print(f"Event assertions: {len(fixture.event_assertions)}")
        print(f"Ordering assertions: {len(fixture.ordering_assertions)}")
        print(f"Timing assertions: {len(fixture.timing_assertions)}")
    else:
        print(f"✗ Fixture is invalid: {fixture_path}")
        print(f"\nError:\n{error}")
        sys.exit(1)
