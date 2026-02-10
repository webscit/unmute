#!/usr/bin/env python3
"""Example WebSocket client for the Unmute Realtime API.

This script demonstrates how to properly connect to the Unmute WebSocket endpoint
with the required headers and subprotocol, and how to negotiate extensions.

Usage:
    python scripts/example_websocket_client.py [--url URL] [--api-key KEY]

Requirements:
    - websockets library: pip install websockets

Example:
    # Connect to local server
    python scripts/example_websocket_client.py

    # Connect with API key
    python scripts/example_websocket_client.py --api-key your-api-key

    # Connect to custom URL
    python scripts/example_websocket_client.py --url ws://your-server:8000/v1/realtime
"""

import argparse
import asyncio
import json
import logging
import sys

try:
    import websockets
    from websockets.asyncio.client import ClientConnection
except ImportError:
    try:
        import websockets
        from websockets.legacy.client import WebSocketClientProtocol as ClientConnection  # type: ignore[assignment]
    except ImportError:
        print("Error: websockets library required. Install with: pip install websockets")
        sys.exit(1)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default WebSocket URL
DEFAULT_URL = "ws://localhost:8000/v1/realtime"

# Required subprotocol for the OpenAI Realtime API
SUBPROTOCOL = "realtime"

# OpenAI-Beta header value
OPENAI_BETA_HEADER = "realtime=v1"


def build_headers(api_key: str | None = None) -> dict[str, str]:
    """Build the required headers for WebSocket connection.

    Args:
        api_key: Optional API key for authorization.

    Returns:
        Dictionary of headers to send with the WebSocket connection.
    """
    headers = {
        "OpenAI-Beta": OPENAI_BETA_HEADER,
    }

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    return headers


def create_session_update(
    voice: str | None = None,
    instructions: str | None = None,
    extensions: list[str] | None = None,
    allow_recording: bool = True,
) -> dict:
    """Create a session.update event payload.

    Args:
        voice: Voice ID for TTS.
        instructions: System instructions.
        extensions: List of Unmute extensions to request.
        allow_recording: Whether to allow session recording.

    Returns:
        Session update event payload.
    """
    session: dict = {}

    if voice:
        session["voice"] = voice

    if instructions:
        session["instructions"] = instructions

    if extensions:
        session["unmute_extensions"] = extensions

    session["allow_recording"] = allow_recording

    return {
        "type": "session.update",
        "session": session,
    }


async def handle_server_messages(websocket: ClientConnection) -> None:
    """Handle incoming messages from the server.

    Args:
        websocket: The WebSocket connection.
    """
    async for message in websocket:
        try:
            data = json.loads(message)
            event_type = data.get("type", "unknown")

            if event_type == "error":
                error = data.get("error", {})
                logger.error(
                    f"Server error: {error.get('type')} - {error.get('message')}"
                )
            elif event_type == "session.updated":
                logger.info("Session updated successfully")
                session = data.get("session", {})
                logger.info(f"  Voice: {session.get('voice', 'not set')}")
            elif event_type == "response.text.delta":
                # Print text deltas without newline
                print(data.get("delta", ""), end="", flush=True)
            elif event_type == "response.text.done":
                print()  # Newline after complete text
                logger.info(f"Response complete: {data.get('text', '')[:100]}...")
            elif event_type == "response.audio.delta":
                # Just log that we received audio
                logger.debug("Received audio delta")
            elif event_type == "response.created":
                logger.info("Response generation started")
            elif event_type == "response.done":
                logger.info("Response generation completed")
            elif event_type.startswith("unmute."):
                logger.debug(f"Unmute extension event: {event_type}")
            else:
                logger.debug(f"Received event: {event_type}")

        except json.JSONDecodeError:
            logger.error(f"Failed to parse server message: {message[:100]}")


async def send_test_messages(websocket: ClientConnection) -> None:
    """Send test messages to demonstrate the API.

    Args:
        websocket: The WebSocket connection.
    """
    # Wait a moment for the connection to be established
    await asyncio.sleep(0.5)

    # Send session.update to configure the session
    logger.info("Sending session.update...")
    session_update = create_session_update(
        voice="alloy",
        instructions="You are a helpful AI assistant. Keep responses brief.",
        extensions=[
            "unmute.recording",
            "unmute.debug_outputs",
        ],
        allow_recording=True,
    )
    await websocket.send(json.dumps(session_update))

    # Wait for the session to be updated
    await asyncio.sleep(1)

    logger.info("Session configured. Listening for events...")
    logger.info("Press Ctrl+C to disconnect")


async def connect_and_run(url: str, api_key: str | None = None) -> None:
    """Connect to the WebSocket and run the example.

    Args:
        url: WebSocket URL to connect to.
        api_key: Optional API key.
    """
    headers = build_headers(api_key)

    logger.info(f"Connecting to {url}...")
    logger.info(f"  Subprotocol: {SUBPROTOCOL}")
    logger.info(f"  Headers: OpenAI-Beta={OPENAI_BETA_HEADER}")
    if api_key:
        logger.info("  Authorization: Bearer ***")

    try:
        async with websockets.connect(
            url,
            subprotocols=[SUBPROTOCOL],  # type: ignore[arg-type]
            additional_headers=headers,
        ) as websocket:
            logger.info("Connected successfully!")
            logger.info(f"  Negotiated subprotocol: {websocket.subprotocol}")

            # Run message handler and sender concurrently
            receive_task = asyncio.create_task(handle_server_messages(websocket))
            send_task = asyncio.create_task(send_test_messages(websocket))

            # Wait for either task to complete or be cancelled
            try:
                done, _pending = await asyncio.wait(
                    [receive_task, send_task],
                    return_when=asyncio.FIRST_EXCEPTION,
                )

                # Check for exceptions
                for task in done:
                    if task.exception():
                        raise task.exception()  # type: ignore

                # Keep receiving messages until cancelled
                await receive_task

            except asyncio.CancelledError:
                logger.info("Connection cancelled")
            finally:
                for task in [receive_task, send_task]:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

    except websockets.exceptions.InvalidStatusCode as e:  # type: ignore[attr-defined]
        logger.error(f"Connection rejected with status {e.status_code}")
        sys.exit(1)
    except websockets.exceptions.InvalidHandshake as e:  # type: ignore[attr-defined]
        logger.error(f"WebSocket handshake failed: {e}")
        sys.exit(1)
    except ConnectionRefusedError:
        logger.error(f"Connection refused. Is the server running at {url}?")
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Example WebSocket client for Unmute Realtime API"
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"WebSocket URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for authorization (optional)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        asyncio.run(connect_and_run(args.url, args.api_key))
    except KeyboardInterrupt:
        logger.info("Disconnected by user")


if __name__ == "__main__":
    main()
