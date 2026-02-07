"""Event dispatcher for routing WebSocket messages to appropriate handlers."""

import asyncio
import base64
import logging

import numpy as np

import unmute.openai_realtime_api_events as ora
from unmute import metrics as mt
from unmute.exceptions import make_ora_error
from unmute.kyutai_constants import SAMPLE_RATE
from unmute.unmute_handler import UnmuteHandler
from unmute.websocket_auth import (
    NegotiatedSession,
    create_extension_error,
    validate_extensions,
)

logger = logging.getLogger(__name__)


class EventDispatcher:
    """Handles routing of client events to appropriate handler methods."""

    def __init__(
        self,
        handler: UnmuteHandler,
        emit_queue: asyncio.Queue[ora.ServerEvent],
        negotiated_session: NegotiatedSession,
    ):
        self.handler = handler
        self.emit_queue = emit_queue
        self.negotiated_session = negotiated_session
        self.audio_buffer = handler.audio_buffer_handler.audio_buffer

    async def dispatch(self, message: ora.ClientEvent) -> ora.ClientEvent | None:
        """Dispatch a client event to the appropriate handler.

        Returns the message to record, or None if the message should not be recorded.
        """
        message_to_record: ora.ClientEvent | None = message

        if isinstance(message, ora.InputAudioBufferAppend):
            message_to_record = await self._handle_audio_append(message)

        elif isinstance(message, ora.SessionUpdate):
            await self._handle_session_update(message)

        elif isinstance(message, ora.UnmuteAdditionalOutputs):
            # Don't record this: it's a debugging message and can be verbose. Anything
            # important to store should be in the other event types.
            message_to_record = None

        # Response control events
        elif isinstance(message, ora.ResponseCreate):
            await self.handler.handle_response_create(message)

        elif isinstance(message, ora.ResponseCancel):
            ack = await self.handler.handle_response_cancel()
            if ack is not None:
                await self.emit_queue.put(ack)

        # Conversation item management events
        elif isinstance(message, ora.ConversationItemCreate):
            item, ack = await self.handler.handle_item_create(message)
            await self.emit_queue.put(ack)
            # Record state snapshot after item creation
            if self.handler.recorder is not None:
                await self.handler.recorder.add_state_snapshot(
                    self.handler.session_state
                )

        elif isinstance(message, ora.ConversationItemDelete):
            ack = await self.handler.handle_item_delete(message.item_id)
            await self.emit_queue.put(ack)
            # Record state snapshot after item deletion
            if self.handler.recorder is not None:
                await self.handler.recorder.add_state_snapshot(
                    self.handler.session_state
                )

        elif isinstance(message, ora.ConversationItemRetrieve):
            ack = self.handler.handle_item_retrieve(message.item_id)
            if ack is not None:
                await self.emit_queue.put(ack)
            else:
                await self.emit_queue.put(
                    ora.Error(
                        error=ora.ErrorDetails(
                            type="invalid_request_error",
                            code="item_not_found",
                            message=f"Item '{message.item_id}' not found",
                        )
                    )
                )

        elif isinstance(message, ora.ConversationItemTruncate):
            ack = await self.handler.handle_item_truncate(message)
            if ack is not None:
                await self.emit_queue.put(ack)
            else:
                await self.emit_queue.put(
                    ora.Error(
                        error=ora.ErrorDetails(
                            type="invalid_request_error",
                            code="item_not_found",
                            message=f"Item '{message.item_id}' not found for truncation",
                        )
                    )
                )

        # Input audio buffer control events
        elif isinstance(message, ora.InputAudioBufferCommit):
            item_id, prev_id = await self.handler.commit_audio_buffer()
            await self.emit_queue.put(
                ora.InputAudioBufferCommitted(item_id=item_id, previous_item_id=prev_id)
            )

        elif isinstance(message, ora.InputAudioBufferClear):
            await self.handler.clear_audio_buffer()
            await self.emit_queue.put(ora.InputAudioBufferCleared())

        else:
            logger.info("Ignoring message:", str(message)[:100])

        return message_to_record

    async def _handle_audio_append(
        self, message: ora.InputAudioBufferAppend
    ) -> ora.UnmuteInputAudioBufferAppendAnonymized:
        """Handle audio buffer append event."""
        opus_bytes = base64.b64decode(message.audio)
        # Use audio buffer for Opus decoding with first-packet sync and
        # frame metadata tracking (timestamps, sequences, latency)

        # Check for overflow before appending
        buffer_was_full = (
            self.audio_buffer.total_samples >= self.audio_buffer._max_buffer_samples
        )

        pcm = await self.audio_buffer.append_opus_async(opus_bytes)

        # If buffer overflow detected, emit error event
        if buffer_was_full and pcm is None:
            error = make_ora_error(
                type="buffer_overflow",
                message=f"Audio buffer overflow: {self.audio_buffer.total_samples} samples exceeds maximum {self.audio_buffer._max_buffer_samples}. Frame discarded.",
            )
            await self.emit_queue.put(error)
            mt.BUFFER_OVERFLOW_ERRORS.inc()
            logger.warning("Buffer overflow detected, frame discarded")

        if pcm is not None and pcm.size:
            await self.handler.receive((SAMPLE_RATE, pcm[np.newaxis, :]))

        return ora.UnmuteInputAudioBufferAppendAnonymized(
            number_of_samples=pcm.size if pcm is not None else 0,
        )

    async def _handle_session_update(self, message: ora.SessionUpdate):
        """Handle session update event including extension negotiation."""
        # Handle extension negotiation from session.update payload
        session_dict = (
            message.session
            if isinstance(message.session, dict)
            else message.session.model_dump()
        )

        # Check for requested extensions in the session config
        requested_extensions = session_dict.get("unmute_extensions")
        if requested_extensions:
            accepted, rejected = validate_extensions(requested_extensions)
            if rejected:
                # Send error for unsupported extensions
                error_event = create_extension_error(rejected)
                await self.emit_queue.put(error_event)
                logger.warning(f"Client requested unsupported extensions: {rejected}")
            # Update negotiated session with accepted extensions
            self.negotiated_session.extensions = accepted
            logger.info(f"Negotiated extensions: {accepted}")

        # Update negotiated session with other config
        self.negotiated_session.update_from_session(message.session)

        await self.handler.update_session(message.session)

        # Convert session to Session object if it's a dict for the response
        session_for_response = (
            ora.Session(**message.session)
            if isinstance(message.session, dict)
            else message.session
        )
        await self.emit_queue.put(ora.SessionUpdated(session=session_for_response))

        # Record state snapshot after session update
        if self.handler.recorder is not None:
            await self.handler.recorder.add_state_snapshot(self.handler.session_state)
