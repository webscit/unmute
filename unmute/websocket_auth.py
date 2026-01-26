"""WebSocket authentication and configuration validation helpers.

This module provides utilities for validating WebSocket connections,
handling subprotocol negotiation, and managing Unmute extension capabilities.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from fastapi import WebSocket, status
from pydantic import BaseModel

import unmute.openai_realtime_api_events as ora

logger = logging.getLogger(__name__)

# OpenAI Realtime API subprotocol
REALTIME_SUBPROTOCOL = "realtime"

# OpenAI-Beta header value for realtime API
OPENAI_BETA_REALTIME = "realtime=v1"


class UnmuteExtension(str, Enum):
    """Supported Unmute extensions."""

    RECORDING = "unmute.recording"
    VOICE_CLONING = "unmute.voice_cloning"
    INSTRUCTIONS_OBJECT = "unmute.instructions_object"
    DEBUG_OUTPUTS = "unmute.debug_outputs"


# All supported extensions
SUPPORTED_EXTENSIONS: set[str] = {ext.value for ext in UnmuteExtension}


class HandshakeError(Exception):
    """Error during WebSocket handshake validation."""

    def __init__(self, message: str, error_type: str = "invalid_request_error"):
        self.message = message
        self.error_type = error_type
        super().__init__(message)


class NegotiatedConfig(BaseModel):
    """Configuration negotiated during session setup."""

    voice: str | None = None
    instructions: Any = None
    extensions: set[str] = field(default_factory=set)
    allow_recording: bool = True
    model: str | None = None

    model_config = {"arbitrary_types_allowed": True}


@dataclass
class NegotiatedSession:
    """Session configuration negotiated during WebSocket handshake."""

    voice: str | None = None
    instructions: Any = None
    extensions: set[str] = field(default_factory=set)
    allow_recording: bool = True
    model: str | None = None

    def update_from_session(self, session: ora.Session | dict[str, Any]) -> None:
        """Update configuration from a session.update payload."""
        if isinstance(session, dict):
            if "voice" in session and session["voice"] is not None:
                self.voice = session["voice"]
            if "instructions" in session and session["instructions"] is not None:
                self.instructions = session["instructions"]
            if "allow_recording" in session:
                self.allow_recording = session.get("allow_recording", True)
            if "model" in session and session["model"] is not None:
                self.model = session["model"]
        else:
            if session.voice is not None:
                self.voice = session.voice
            if session.instructions is not None:
                self.instructions = session.instructions
            if session.allow_recording is not None:
                self.allow_recording = session.allow_recording
            if session.model is not None:
                self.model = session.model


def validate_subprotocol(websocket: WebSocket) -> bool:
    """Validate that the client requested the 'realtime' subprotocol.

    Args:
        websocket: The WebSocket connection to validate.

    Returns:
        True if the subprotocol is valid, False otherwise.
    """
    # Get requested subprotocols from the WebSocket headers
    # The Sec-WebSocket-Protocol header contains comma-separated subprotocols
    subprotocols = websocket.headers.get("sec-websocket-protocol", "")
    requested_subprotocols = [s.strip() for s in subprotocols.split(",") if s.strip()]

    return REALTIME_SUBPROTOCOL in requested_subprotocols


def validate_openai_beta_header(websocket: WebSocket) -> bool:
    """Validate the OpenAI-Beta header for realtime API compatibility.

    Args:
        websocket: The WebSocket connection to validate.

    Returns:
        True if the header is valid or not present (optional), False if invalid.
    """
    openai_beta = websocket.headers.get("openai-beta", "")
    if not openai_beta:
        # Header is optional for compatibility
        return True

    # Check if realtime=v1 is present in the header
    return OPENAI_BETA_REALTIME in openai_beta


def extract_authorization(websocket: WebSocket) -> str | None:
    """Extract the authorization token from the WebSocket headers.

    Args:
        websocket: The WebSocket connection.

    Returns:
        The bearer token if present, None otherwise.
    """
    auth_header = websocket.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return None


def validate_extensions(
    requested_extensions: list[str] | None,
) -> tuple[set[str], list[str]]:
    """Validate and filter requested extensions.

    Args:
        requested_extensions: List of extension names requested by client.

    Returns:
        Tuple of (accepted_extensions, rejected_extensions).
    """
    if not requested_extensions:
        return set(), []

    accepted = set()
    rejected = []

    for ext in requested_extensions:
        if ext in SUPPORTED_EXTENSIONS:
            accepted.add(ext)
        else:
            rejected.append(ext)

    return accepted, rejected


def create_handshake_error_event(
    error_type: str, message: str, code: str | None = None
) -> ora.Error:
    """Create a spec-compliant error event for handshake failures.

    Args:
        error_type: The error type (e.g., "invalid_request_error").
        message: Human-readable error message.
        code: Optional error code.

    Returns:
        An Error event that can be sent to the client.
    """
    return ora.Error(
        error=ora.ErrorDetails(
            type=error_type,
            message=message,
            code=code,
        )
    )


async def perform_handshake_validation(
    websocket: WebSocket,
) -> tuple[bool, ora.Error | None]:
    """Perform full handshake validation on a WebSocket connection.

    Args:
        websocket: The WebSocket connection to validate.

    Returns:
        Tuple of (success, error_event). If success is True, error_event is None.
        If success is False, error_event contains the error to send to client.
    """
    # Validate subprotocol
    if not validate_subprotocol(websocket):
        error = create_handshake_error_event(
            error_type="invalid_request_error",
            message=(
                "Missing or invalid Sec-WebSocket-Protocol header. "
                f"Expected subprotocol: '{REALTIME_SUBPROTOCOL}'"
            ),
            code="invalid_subprotocol",
        )
        return False, error

    # Validate OpenAI-Beta header if present
    if not validate_openai_beta_header(websocket):
        error = create_handshake_error_event(
            error_type="invalid_request_error",
            message=(f"Invalid OpenAI-Beta header. Expected: '{OPENAI_BETA_REALTIME}'"),
            code="invalid_header",
        )
        return False, error

    return True, None


async def reject_connection_with_error(
    websocket: WebSocket,
    error: ora.Error,
    close_code: int = status.WS_1008_POLICY_VIOLATION,
) -> None:
    """Reject a WebSocket connection by sending an error and closing.

    This accepts the connection first to be able to send the error message,
    then closes it.

    Args:
        websocket: The WebSocket connection.
        error: The error event to send.
        close_code: The WebSocket close code.
    """
    # We need to accept to send a message, even if we're going to close
    try:
        await websocket.accept(subprotocol=REALTIME_SUBPROTOCOL)
        await websocket.send_text(error.model_dump_json())
        await websocket.close(code=close_code, reason=error.error.message)
    except Exception as e:
        logger.warning(f"Error while rejecting connection: {e}")


def create_extension_error(unsupported_extensions: list[str]) -> ora.Error:
    """Create an error event for unsupported extensions.

    Args:
        unsupported_extensions: List of extension names that are not supported.

    Returns:
        An Error event for the unsupported extensions.
    """
    return create_handshake_error_event(
        error_type="invalid_request_error",
        message=f"Unsupported extensions: {', '.join(unsupported_extensions)}. "
        f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        code="unsupported_extension",
    )
