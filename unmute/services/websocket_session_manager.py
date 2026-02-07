"""WebSocket session lifecycle management."""

import asyncio
import base64
import json
import logging
from typing import Annotated

import sphn
from fastapi import WebSocket, WebSocketDisconnect, status
from fastapi.websockets import WebSocketState
from fastrtc import AdditionalOutputs, CloseStream, audio_to_float32
from pydantic import Field, TypeAdapter, ValidationError

import unmute.openai_realtime_api_events as ora
from unmute import metrics as mt
from unmute.exceptions import WebSocketClosedError
from unmute.kyutai_constants import SAMPLE_RATE
from unmute.timer import Stopwatch
from unmute.tracing import (
    WEBSOCKET_SEND_DURATION,
    end_trace,
    start_trace,
    trace_queue_operation,
    trace_span,
)
from unmute.unmute_handler import UnmuteHandler
from unmute.websocket_auth import NegotiatedSession

from .event_dispatcher import EventDispatcher
from .health_service import get_health

logger = logging.getLogger(__name__)

ClientEventAdapter = TypeAdapter(
    Annotated[ora.ClientEvent, Field(discriminator="type")]
)


async def debug_running_tasks():
    while True:
        logger.debug(f"Running tasks: {len(asyncio.all_tasks())}")
        for task in asyncio.all_tasks():
            logger.debug(f"  Task: {task.get_name()} - {task.get_coro()}")
        await asyncio.sleep(5)


class EmitDebugLogger:
    def __init__(self):
        self.last_emitted_n = 0
        self.last_emitted_type = ""

    def on_emit(self, to_emit: ora.ServerEvent):
        if self.last_emitted_type == to_emit.type:
            self.last_emitted_n += 1
        else:
            self.last_emitted_n = 1
            self.last_emitted_type = to_emit.type

        if self.last_emitted_n == 1:
            logger.debug(f"Emitting: {to_emit.type}")
        else:
            logger.debug(f"Emitting ({self.last_emitted_n}): {self.last_emitted_type}")


class WebSocketSessionManager:
    """Manages WebSocket session lifecycle including receive and emit loops."""

    def __init__(self, websocket: WebSocket, handler: UnmuteHandler):
        self.websocket = websocket
        self.handler = handler
        self.session_watch = Stopwatch()
        self.negotiated_session = NegotiatedSession()

    async def run(self):
        """Run the WebSocket session handling receive and emit loops."""
        # Check health before starting
        health = await get_health()
        if not health.ok:
            logger.info("Health check failed, closing WebSocket connection.")
            await self.websocket.close(
                code=status.WS_1011_INTERNAL_ERROR,
                reason=f"Server is not healthy: {health}",
            )
            return

        # Start distributed tracing for this session
        session_id = ora.random_id("sess")
        start_trace(session_id)

        # Store negotiated session on handler for access during message handling
        self.handler.negotiated_session = self.negotiated_session  # type: ignore[attr-defined]

        emit_queue: asyncio.Queue[ora.ServerEvent] = asyncio.Queue()
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(
                    self._receive_loop(emit_queue),
                    name="receive_loop()",
                )
                tg.create_task(
                    self._emit_loop(emit_queue), name="emit_loop()"
                )
                tg.create_task(self.handler.quest_manager.wait(), name="quest_manager.wait()")
                tg.create_task(debug_running_tasks(), name="debug_running_tasks()")
        finally:
            await self.handler.cleanup()
            logger.info("websocket_route() finished")

            # End tracing and log summary
            trace_ctx = end_trace()
            if trace_ctx:
                from unmute.tracing import format_trace_summary

                logger.info("Session trace summary:\n%s", format_trace_summary(trace_ctx))

    async def _receive_loop(self, emit_queue: asyncio.Queue[ora.ServerEvent]):
        """Receive messages from the WebSocket.

        Can decide to send messages via `emit_queue`.
        """
        dispatcher = EventDispatcher(self.handler, emit_queue, self.negotiated_session)

        while True:
            try:
                message_raw = await self.websocket.receive_text()
            except WebSocketDisconnect as e:
                logger.info(
                    "receive_loop() stopped because WebSocket disconnected: "
                    f"{e.code=} {e.reason=}"
                )
                raise WebSocketClosedError() from e
            except RuntimeError as e:
                # This is expected when the client disconnects
                if "WebSocket is not connected" not in str(e):
                    raise  # re-raise unexpected errors

                logger.info("receive_loop() stopped because WebSocket disconnected.")
                raise WebSocketClosedError() from e

            try:
                message: ora.ClientEvent = ClientEventAdapter.validate_json(message_raw)
            except json.JSONDecodeError as e:
                await emit_queue.put(
                    ora.Error(
                        error=ora.ErrorDetails(
                            type="invalid_request_error",
                            message=f"Invalid JSON: {e}",
                        )
                    )
                )
                continue
            except ValidationError as e:
                await emit_queue.put(
                    ora.Error(
                        error=ora.ErrorDetails(
                            type="invalid_request_error",
                            message="Invalid message",
                            details=json.loads(e.json()),
                        )
                    )
                )
                continue

            # Dispatch the message and get what to record
            message_to_record = await dispatcher.dispatch(message)

            if message_to_record is not None and self.handler.recorder is not None:
                await self.handler.recorder.add_event("client", message_to_record)

    async def _emit_loop(self, emit_queue: asyncio.Queue[ora.ServerEvent]):
        """Send messages to the WebSocket."""
        emit_debug_logger = EmitDebugLogger()
        opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE)

        while True:
            if (
                self.websocket.application_state == WebSocketState.DISCONNECTED
                or self.websocket.client_state == WebSocketState.DISCONNECTED
            ):
                logger.info("emit_loop() stopped because WebSocket disconnected")
                raise WebSocketClosedError()

            # Monitor emit_queue size for backpressure
            emit_queue_size = emit_queue.qsize()
            mt.EMIT_QUEUE_SIZE.set(emit_queue_size)

            try:
                # Try to get from buffered queue first (non-blocking)
                async with trace_queue_operation("emit_queue", "get_nowait"):
                    to_emit = emit_queue.get_nowait()
            except asyncio.QueueEmpty:
                # Fall back to handler emit (blocking)
                async with trace_queue_operation("handler", "emit"):
                    emitted_by_handler = await self.handler.emit()

                if emitted_by_handler is None:
                    continue
                elif isinstance(emitted_by_handler, AdditionalOutputs):
                    assert len(emitted_by_handler.args) == 1
                    to_emit = ora.UnmuteAdditionalOutputs(
                        args=emitted_by_handler.args[0],
                    )
                elif isinstance(emitted_by_handler, CloseStream):
                    # Close here explicitly so that the receive loop stops too
                    await self.websocket.close()
                    break
                elif isinstance(emitted_by_handler, ora.ServerEvent):
                    to_emit = emitted_by_handler
                else:
                    _sr, audio = emitted_by_handler
                    audio = audio_to_float32(audio)
                    opus_bytes = await asyncio.to_thread(opus_writer.append_pcm, audio)
                    # Due to buffering/chunking, Opus doesn't necessarily output something on every PCM added
                    if opus_bytes:
                        to_emit = ora.ResponseAudioDelta(
                            delta=base64.b64encode(opus_bytes).decode("utf-8"),
                        )
                    else:
                        continue

            emit_debug_logger.on_emit(to_emit)

            if self.handler.recorder is not None:
                await self.handler.recorder.add_event("server", to_emit)

            try:
                # Trace websocket send operations to detect network/serialization bottlenecks
                async with trace_span(
                    "websocket_send",
                    histogram=WEBSOCKET_SEND_DURATION,
                    attributes={"event_type": to_emit.type},
                ):
                    await self.websocket.send_text(to_emit.model_dump_json())
            except (WebSocketDisconnect, RuntimeError) as e:
                if isinstance(e, RuntimeError):
                    if "Unexpected ASGI message 'websocket.send'" in str(e):
                        # This is expected when the client disconnects
                        message = f"emit_loop() stopped because WebSocket disconnected: {e}"
                    else:
                        raise
                else:
                    message = (
                        "emit_loop() stopped because WebSocket disconnected: "
                        f"{e.code=} {e.reason=}"
                    )

                logger.info(message)
                raise WebSocketClosedError() from e
