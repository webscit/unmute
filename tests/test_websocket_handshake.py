"""Tests for WebSocket handshake, subprotocol negotiation, and extension handling.

Tests cover:
- Successful handshake with proper subprotocol/headers
- Failure cases (missing subprotocol, unsupported extensions, malformed config)
- Extension negotiation
- Auth error responses
"""

import json

import pytest

import unmute.openai_realtime_api_events as ora
from unmute.websocket_auth import (
    OPENAI_BETA_REALTIME,
    SUPPORTED_EXTENSIONS,
    NegotiatedSession,
    UnmuteExtension,
    create_extension_error,
    create_handshake_error_event,
    extract_authorization,
    perform_handshake_validation,
    validate_extensions,
    validate_openai_beta_header,
    validate_subprotocol,
)


class MockWebSocket:
    """Mock WebSocket for testing validation functions."""

    def __init__(self, headers: dict[str, str] | None = None):
        self.headers = headers or {}
        self.accepted = False
        self.closed = False
        self.sent_messages: list[str] = []
        self.close_code: int | None = None
        self.close_reason: str | None = None
        self.accepted_subprotocol: str | None = None

    async def accept(self, subprotocol: str | None = None) -> None:
        self.accepted = True
        self.accepted_subprotocol = subprotocol

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = True
        self.close_code = code
        self.close_reason = reason

    async def send_text(self, data: str) -> None:
        self.sent_messages.append(data)


class TestSubprotocolValidation:
    """Tests for subprotocol validation."""

    def test_valid_subprotocol(self):
        """Test that 'realtime' subprotocol is accepted."""
        ws = MockWebSocket(headers={"sec-websocket-protocol": "realtime"})
        assert validate_subprotocol(ws) is True

    def test_valid_subprotocol_with_multiple(self):
        """Test that 'realtime' is accepted when multiple subprotocols are listed."""
        ws = MockWebSocket(
            headers={"sec-websocket-protocol": "graphql, realtime, chat"}
        )
        assert validate_subprotocol(ws) is True

    def test_valid_subprotocol_with_spaces(self):
        """Test that spaces around subprotocol names are handled."""
        ws = MockWebSocket(headers={"sec-websocket-protocol": " realtime , other "})
        assert validate_subprotocol(ws) is True

    def test_missing_subprotocol_header(self):
        """Test that missing subprotocol header is rejected."""
        ws = MockWebSocket(headers={})
        assert validate_subprotocol(ws) is False

    def test_empty_subprotocol_header(self):
        """Test that empty subprotocol header is rejected."""
        ws = MockWebSocket(headers={"sec-websocket-protocol": ""})
        assert validate_subprotocol(ws) is False

    def test_wrong_subprotocol(self):
        """Test that wrong subprotocol is rejected."""
        ws = MockWebSocket(headers={"sec-websocket-protocol": "graphql, chat"})
        assert validate_subprotocol(ws) is False


class TestOpenAIBetaHeaderValidation:
    """Tests for OpenAI-Beta header validation."""

    def test_valid_header(self):
        """Test that correct OpenAI-Beta header is accepted."""
        ws = MockWebSocket(headers={"openai-beta": OPENAI_BETA_REALTIME})
        assert validate_openai_beta_header(ws) is True

    def test_missing_header_is_allowed(self):
        """Test that missing header is allowed (for compatibility)."""
        ws = MockWebSocket(headers={})
        assert validate_openai_beta_header(ws) is True

    def test_empty_header_is_allowed(self):
        """Test that empty header is allowed."""
        ws = MockWebSocket(headers={"openai-beta": ""})
        assert validate_openai_beta_header(ws) is True

    def test_header_with_additional_values(self):
        """Test that header with additional values is accepted if realtime=v1 is present."""
        ws = MockWebSocket(headers={"openai-beta": "realtime=v1, other=v2"})
        assert validate_openai_beta_header(ws) is True

    def test_wrong_version(self):
        """Test that wrong version is rejected."""
        ws = MockWebSocket(headers={"openai-beta": "realtime=v2"})
        assert validate_openai_beta_header(ws) is False


class TestAuthorizationExtraction:
    """Tests for authorization token extraction."""

    def test_bearer_token(self):
        """Test extraction of bearer token."""
        ws = MockWebSocket(headers={"authorization": "Bearer test-token-123"})
        assert extract_authorization(ws) == "test-token-123"

    def test_bearer_case_insensitive(self):
        """Test that 'bearer' is case-insensitive."""
        ws = MockWebSocket(headers={"authorization": "BEARER test-token"})
        assert extract_authorization(ws) == "test-token"

    def test_missing_authorization(self):
        """Test that missing authorization returns None."""
        ws = MockWebSocket(headers={})
        assert extract_authorization(ws) is None

    def test_non_bearer_auth(self):
        """Test that non-bearer auth returns None."""
        ws = MockWebSocket(headers={"authorization": "Basic abc123"})
        assert extract_authorization(ws) is None

    def test_token_with_spaces_stripped(self):
        """Test that token whitespace is stripped."""
        ws = MockWebSocket(headers={"authorization": "Bearer   test-token  "})
        assert extract_authorization(ws) == "test-token"


class TestExtensionValidation:
    """Tests for extension validation."""

    def test_all_supported_extensions(self):
        """Test that all supported extensions are accepted."""
        requested = list(SUPPORTED_EXTENSIONS)
        accepted, rejected = validate_extensions(requested)
        assert accepted == SUPPORTED_EXTENSIONS
        assert rejected == []

    def test_single_valid_extension(self):
        """Test that a single valid extension is accepted."""
        requested = [UnmuteExtension.RECORDING.value]
        accepted, rejected = validate_extensions(requested)
        assert accepted == {UnmuteExtension.RECORDING.value}
        assert rejected == []

    def test_unsupported_extension_rejected(self):
        """Test that unsupported extensions are rejected."""
        requested = ["unmute.unsupported", "unmute.another_unsupported"]
        accepted, rejected = validate_extensions(requested)
        assert accepted == set()
        assert set(rejected) == {"unmute.unsupported", "unmute.another_unsupported"}

    def test_mixed_extensions(self):
        """Test mix of supported and unsupported extensions."""
        requested = [
            UnmuteExtension.RECORDING.value,
            "unmute.unsupported",
            UnmuteExtension.DEBUG_OUTPUTS.value,
        ]
        accepted, rejected = validate_extensions(requested)
        assert accepted == {
            UnmuteExtension.RECORDING.value,
            UnmuteExtension.DEBUG_OUTPUTS.value,
        }
        assert rejected == ["unmute.unsupported"]

    def test_none_extensions(self):
        """Test that None extensions list is handled."""
        accepted, rejected = validate_extensions(None)
        assert accepted == set()
        assert rejected == []

    def test_empty_extensions(self):
        """Test that empty extensions list is handled."""
        accepted, rejected = validate_extensions([])
        assert accepted == set()
        assert rejected == []


class TestNegotiatedSession:
    """Tests for NegotiatedSession configuration."""

    def test_initial_state(self):
        """Test initial state of negotiated session."""
        session = NegotiatedSession()
        assert session.voice is None
        assert session.instructions is None
        assert session.extensions == set()
        assert session.allow_recording is True
        assert session.model is None

    def test_update_from_dict(self):
        """Test updating session from dict."""
        session = NegotiatedSession()
        session.update_from_session(
            {
                "voice": "alloy",
                "instructions": "Be helpful",
                "allow_recording": False,
                "model": "gpt-4o-realtime",
            }
        )
        assert session.voice == "alloy"
        assert session.instructions == "Be helpful"
        assert session.allow_recording is False
        assert session.model == "gpt-4o-realtime"

    def test_update_from_session_object(self):
        """Test updating from Session object."""
        session = NegotiatedSession()
        ora_session = ora.Session(
            voice="echo",
            instructions="Be concise",
            allow_recording=True,
        )
        session.update_from_session(ora_session)
        assert session.voice == "echo"
        assert session.instructions == "Be concise"
        assert session.allow_recording is True

    def test_update_preserves_unset_values(self):
        """Test that update preserves existing values when not provided."""
        session = NegotiatedSession(voice="alloy", instructions="Original")
        session.update_from_session({"voice": "echo"})
        assert session.voice == "echo"
        assert session.instructions == "Original"

    def test_update_ignores_none_values(self):
        """Test that None values don't overwrite existing values."""
        session = NegotiatedSession(voice="alloy")
        session.update_from_session({"voice": None})
        assert session.voice == "alloy"


class TestErrorEventCreation:
    """Tests for error event creation."""

    def test_create_handshake_error(self):
        """Test creation of handshake error event."""
        error = create_handshake_error_event(
            error_type="invalid_request_error",
            message="Test error message",
            code="test_code",
        )
        assert isinstance(error, ora.Error)
        assert error.error.type == "invalid_request_error"
        assert error.error.message == "Test error message"
        assert error.error.code == "test_code"

    def test_create_extension_error(self):
        """Test creation of extension error event."""
        error = create_extension_error(["ext1", "ext2"])
        assert isinstance(error, ora.Error)
        assert error.error.type == "invalid_request_error"
        assert error.error.code == "unsupported_extension"
        assert "ext1" in error.error.message
        assert "ext2" in error.error.message

    def test_extension_error_includes_supported(self):
        """Test that extension error includes list of supported extensions."""
        error = create_extension_error(["unknown"])
        for ext in SUPPORTED_EXTENSIONS:
            assert ext in error.error.message


class TestPerformHandshakeValidation:
    """Tests for full handshake validation."""

    @pytest.mark.asyncio
    async def test_valid_handshake(self):
        """Test successful handshake with valid headers."""
        ws = MockWebSocket(
            headers={
                "sec-websocket-protocol": "realtime",
                "openai-beta": OPENAI_BETA_REALTIME,
            }
        )
        valid, error = await perform_handshake_validation(ws)
        assert valid is True
        assert error is None

    @pytest.mark.asyncio
    async def test_valid_handshake_minimal(self):
        """Test successful handshake with minimal valid headers."""
        ws = MockWebSocket(headers={"sec-websocket-protocol": "realtime"})
        valid, error = await perform_handshake_validation(ws)
        assert valid is True
        assert error is None

    @pytest.mark.asyncio
    async def test_invalid_subprotocol(self):
        """Test handshake failure with invalid subprotocol."""
        ws = MockWebSocket(headers={"sec-websocket-protocol": "graphql"})
        valid, error = await perform_handshake_validation(ws)
        assert valid is False
        assert error is not None
        assert error.error.code == "invalid_subprotocol"

    @pytest.mark.asyncio
    async def test_missing_subprotocol(self):
        """Test handshake failure with missing subprotocol."""
        ws = MockWebSocket(headers={})
        valid, error = await perform_handshake_validation(ws)
        assert valid is False
        assert error is not None
        assert "subprotocol" in error.error.message.lower()

    @pytest.mark.asyncio
    async def test_invalid_openai_beta_header(self):
        """Test handshake failure with invalid OpenAI-Beta header."""
        ws = MockWebSocket(
            headers={
                "sec-websocket-protocol": "realtime",
                "openai-beta": "realtime=v99",  # Wrong version
            }
        )
        valid, error = await perform_handshake_validation(ws)
        assert valid is False
        assert error is not None
        assert error.error.code == "invalid_header"


class TestSupportedExtensions:
    """Tests for the supported extensions constant."""

    def test_recording_extension_supported(self):
        """Test that recording extension is in supported list."""
        assert "unmute.recording" in SUPPORTED_EXTENSIONS

    def test_voice_cloning_extension_supported(self):
        """Test that voice cloning extension is in supported list."""
        assert "unmute.voice_cloning" in SUPPORTED_EXTENSIONS

    def test_instructions_object_extension_supported(self):
        """Test that instructions object extension is in supported list."""
        assert "unmute.instructions_object" in SUPPORTED_EXTENSIONS

    def test_debug_outputs_extension_supported(self):
        """Test that debug outputs extension is in supported list."""
        assert "unmute.debug_outputs" in SUPPORTED_EXTENSIONS


class TestErrorEventSerialization:
    """Tests for error event JSON serialization."""

    def test_error_event_json_round_trip(self):
        """Test that error events can be serialized and deserialized."""
        error = create_handshake_error_event(
            error_type="test_error",
            message="Test message",
            code="test_code",
        )
        json_str = error.model_dump_json()
        data = json.loads(json_str)

        assert data["type"] == "error"
        assert data["error"]["type"] == "test_error"
        assert data["error"]["message"] == "Test message"
        assert data["error"]["code"] == "test_code"

    def test_extension_error_json_format(self):
        """Test that extension error JSON is properly formatted."""
        error = create_extension_error(["bad_ext"])
        json_str = error.model_dump_json()
        data = json.loads(json_str)

        assert data["type"] == "error"
        assert data["error"]["code"] == "unsupported_extension"
        assert "bad_ext" in data["error"]["message"]
