"""Tests for exception utilities and error event creation.

This test module ensures that:
1. make_ora_error creates valid error events with correct parameters
2. The function signature is used correctly throughout the codebase
"""

import pytest

import unmute.openai_realtime_api_events as ora
from unmute.exceptions import make_ora_error


class TestMakeOraError:
    """Tests for make_ora_error function."""

    def test_creates_error_with_type_and_message(self):
        """Test that make_ora_error creates a valid Error event."""
        error = make_ora_error(type="test_error", message="Test message")

        assert isinstance(error, ora.Error)
        assert error.error.type == "test_error"
        assert error.error.message == "Test message"

    def test_accepts_various_error_types(self):
        """Test that make_ora_error accepts different error type strings."""
        test_types = [
            "server_overloaded",
            "buffer_overflow",
            "backpressure_error",
            "silence_timeout",
            "warning",
            "fatal",
        ]

        for error_type in test_types:
            error = make_ora_error(type=error_type, message=f"Testing {error_type}")
            assert error.error.type == error_type
            assert error_type in error.error.message

    def test_error_is_json_serializable(self):
        """Test that error can be serialized to JSON."""
        error = make_ora_error(type="test", message="Test message")
        json_str = error.model_dump_json()

        assert isinstance(json_str, str)
        assert '"type":"test"' in json_str or '"type": "test"' in json_str
        assert "Test message" in json_str

    def test_parameter_name_is_type_not_code(self):
        """Test that the function uses 'type' parameter, not 'code'.

        This test documents the correct parameter name to prevent regressions
        where callers might incorrectly use 'code' instead of 'type'.
        """
        # This should work
        error = make_ora_error(type="test_error", message="Test")
        assert error.error.type == "test_error"

        # This should raise TypeError (documenting the incorrect usage)
        with pytest.raises(TypeError, match="unexpected keyword argument 'code'"):
            make_ora_error(code="test_error", message="Test")

    def test_both_parameters_required(self):
        """Test that both type and message are required."""
        with pytest.raises(TypeError):
            make_ora_error(type="test")

        with pytest.raises(TypeError):
            make_ora_error(message="test")

        with pytest.raises(TypeError):
            make_ora_error()

    def test_empty_strings_allowed(self):
        """Test that empty strings are allowed for type and message."""
        error = make_ora_error(type="", message="")
        assert error.error.type == ""
        assert error.error.message == ""

    def test_long_messages_handled(self):
        """Test that long error messages are handled correctly."""
        long_message = "Error: " + "x" * 1000
        error = make_ora_error(type="test", message=long_message)
        assert error.error.message == long_message
