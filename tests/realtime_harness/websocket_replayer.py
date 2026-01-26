"""WebSocket client for replaying trace fixtures against the Realtime API.

This module provides a WebSocket client that connects to the /v1/realtime endpoint,
replays client events from fixtures, and collects server responses for validation.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import websockets
from websockets.asyncio.client import ClientConnection

from tests.realtime_harness.fixture_schema import ClientEvent, TimingMode, TraceFixture


@dataclass
class ReceivedEvent:
    """A server event received during fixture replay."""

    event: dict[str, Any]
    timestamp_ms: float  # Milliseconds since trace start
    event_type: str


@dataclass
class ReplayTrace:
    """Complete trace of a fixture replay session."""

    fixture_name: str
    start_time: float  # Monotonic time
    end_time: Optional[float] = None
    received_events: list[ReceivedEvent] = field(default_factory=list)
    sent_events: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    disconnected: bool = False
    disconnect_reason: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        """Total duration of the trace in milliseconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    def get_events_by_type(self, event_type: str) -> list[ReceivedEvent]:
        """Get all received events of a specific type."""
        return [e for e in self.received_events if e.event_type == event_type]

    def get_first_event(self, event_type: str) -> Optional[ReceivedEvent]:
        """Get the first received event of a specific type."""
        events = self.get_events_by_type(event_type)
        return events[0] if events else None

    def get_latency_ms(
        self, event_type_a: str, event_type_b: str
    ) -> Optional[float]:
        """Calculate latency between first occurrence of two event types.

        Args:
            event_type_a: First event type
            event_type_b: Second event type

        Returns:
            Latency in milliseconds, or None if either event not found
        """
        event_a = self.get_first_event(event_type_a)
        event_b = self.get_first_event(event_type_b)

        if event_a is None or event_b is None:
            return None

        return event_b.timestamp_ms - event_a.timestamp_ms


class WebSocketReplayer:
    """WebSocket client for replaying fixtures."""

    def __init__(
        self,
        base_url: str = "ws://localhost:8000",
        api_key: Optional[str] = None,
        timeout_seconds: int = 30,
    ):
        """Initialize the replayer.

        Args:
            base_url: Base WebSocket URL (e.g., ws://localhost:8000)
            api_key: Optional API key for authorization
            timeout_seconds: Default timeout for operations
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.websocket: Optional[ClientConnection] = None

    async def connect(self) -> None:
        """Establish WebSocket connection to /v1/realtime.

        Raises:
            websockets.exceptions.WebSocketException: If connection fails
        """
        url = f"{self.base_url}/v1/realtime"

        headers = {"Sec-WebSocket-Protocol": "realtime"}

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Add OpenAI Beta header for compatibility
        headers["OpenAI-Beta"] = "realtime=v1"

        self.websocket = await websockets.connect(
            url,
            extra_headers=headers,
            subprotocols=["realtime"],
        )

    async def disconnect(self) -> None:
        """Close the WebSocket connection gracefully."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

    async def send_event(self, event: dict[str, Any]) -> None:
        """Send a client event to the server.

        Args:
            event: Event dictionary to send

        Raises:
            RuntimeError: If not connected
        """
        if not self.websocket:
            raise RuntimeError("Not connected")

        await self.websocket.send(json.dumps(event))

    async def receive_event(self, timeout: Optional[float] = None) -> Optional[dict[str, Any]]:
        """Receive a single server event.

        Args:
            timeout: Optional timeout in seconds (uses default if None)

        Returns:
            Event dictionary or None if timeout/disconnected
        """
        if not self.websocket:
            return None

        timeout = timeout or self.timeout_seconds

        try:
            message = await asyncio.wait_for(self.websocket.recv(), timeout=timeout)
            return json.loads(message)
        except asyncio.TimeoutError:
            return None
        except websockets.exceptions.ConnectionClosed:
            return None

    async def replay_fixture(
        self, fixture: TraceFixture, collect_timeout_ms: int = 2000
    ) -> ReplayTrace:
        """Replay a complete fixture and collect server responses.

        Args:
            fixture: Fixture to replay
            collect_timeout_ms: Milliseconds to wait after last client event
                               to collect remaining server events

        Returns:
            Complete replay trace with all events and timing
        """
        trace = ReplayTrace(
            fixture_name=fixture.metadata.name, start_time=time.monotonic()
        )

        try:
            # Connect
            await self.connect()

            # Start event collection task
            collection_task = asyncio.create_task(
                self._collect_server_events(trace)
            )

            # Send client events with timing
            await self._send_client_events(fixture, trace)

            # Wait for remaining server events
            await asyncio.sleep(collect_timeout_ms / 1000)

            # Stop collection
            collection_task.cancel()
            try:
                await collection_task
            except asyncio.CancelledError:
                pass

        except websockets.exceptions.ConnectionClosed as e:
            trace.disconnected = True
            trace.disconnect_reason = str(e)
            trace.errors.append(f"Connection closed: {e}")

        except Exception as e:
            trace.errors.append(f"Replay error: {e}")

        finally:
            trace.end_time = time.monotonic()
            await self.disconnect()

        return trace

    async def _send_client_events(
        self, fixture: TraceFixture, trace: ReplayTrace
    ) -> None:
        """Send all client events from a fixture.

        Args:
            fixture: Fixture to replay
            trace: Trace to record sent events
        """
        trace_start = time.monotonic()

        if fixture.timing_mode == TimingMode.IMMEDIATE:
            # Send all events as fast as possible
            for client_event in fixture.client_events:
                await self.send_event(client_event.event)
                trace.sent_events.append(client_event.event)

        elif fixture.timing_mode == TimingMode.RELATIVE:
            # Send events with delays relative to previous event
            for client_event in fixture.client_events:
                if client_event.delay_ms > 0:
                    await asyncio.sleep(client_event.delay_ms / 1000)

                await self.send_event(client_event.event)
                trace.sent_events.append(client_event.event)

        elif fixture.timing_mode == TimingMode.ABSOLUTE:
            # Send events at absolute timestamps
            for client_event in fixture.client_events:
                if client_event.timestamp_ms is not None:
                    # Calculate how long to wait
                    elapsed_ms = (time.monotonic() - trace_start) * 1000
                    wait_ms = client_event.timestamp_ms - elapsed_ms

                    if wait_ms > 0:
                        await asyncio.sleep(wait_ms / 1000)

                await self.send_event(client_event.event)
                trace.sent_events.append(client_event.event)

    async def _collect_server_events(self, trace: ReplayTrace) -> None:
        """Continuously collect server events until cancelled.

        Args:
            trace: Trace to record received events
        """
        trace_start = trace.start_time

        while True:
            try:
                event = await self.receive_event(timeout=0.1)

                if event is None:
                    # Small delay to avoid busy-waiting
                    await asyncio.sleep(0.01)
                    continue

                # Record event with timestamp
                timestamp_ms = (time.monotonic() - trace_start) * 1000
                event_type = event.get("type", "unknown")

                received_event = ReceivedEvent(
                    event=event, timestamp_ms=timestamp_ms, event_type=event_type
                )

                trace.received_events.append(received_event)

            except asyncio.CancelledError:
                # Collection stopped
                break
            except Exception as e:
                trace.errors.append(f"Error collecting event: {e}")
                break


async def replay_fixture_standalone(
    fixture: TraceFixture,
    base_url: str = "ws://localhost:8000",
    api_key: Optional[str] = None,
) -> ReplayTrace:
    """Standalone function to replay a fixture.

    Args:
        fixture: Fixture to replay
        base_url: WebSocket base URL
        api_key: Optional API key

    Returns:
        Complete replay trace
    """
    replayer = WebSocketReplayer(base_url=base_url, api_key=api_key)
    return await replayer.replay_fixture(fixture)
