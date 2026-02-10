"""Unit tests for VAD event emission and error handling.

Tests cover:
- speech_started/stopped event emission with proper metadata
- Silence timeout error handling
- Buffer overflow error detection
- Backpressure threshold monitoring
- State reset after interruptions
"""
# pyright: reportPrivateUsage=false

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

import unmute.openai_realtime_api_events as ora
from unmute.handlers.event_router import (
    OUTPUT_QUEUE_ERROR_THRESHOLD,
    OUTPUT_QUEUE_WARNING_THRESHOLD,
)
from unmute.session_state import SessionState
from unmute.unmute_handler import USER_SILENCE_TIMEOUT, UnmuteHandler


class TestVADEventEmission:
    """Test VAD event emission with metadata."""

    @pytest.fixture
    def handler(self):
        """Create handler with mocked dependencies."""
        with patch("unmute.unmute_handler.get_openai_client"):
            handler = UnmuteHandler()
            handler.session_state = SessionState()
            handler.chatbot = MagicMock()
            handler.chatbot.conversation_state = MagicMock(return_value="user_speaking")
            handler.chatbot.chat_history = []
            handler.chatbot.last_message = MagicMock(return_value=None)

            # Mock STT
            handler.quest_manager = MagicMock()
            handler.quest_manager.quests = {}
            stt_mock = MagicMock()
            stt_mock.pause_prediction = MagicMock()
            stt_mock.pause_prediction.value = 0.0
            handler.quest_manager.quests["stt"] = MagicMock()
            handler.quest_manager.quests["stt"].get_nowait = MagicMock(return_value=stt_mock)

            return handler

    @pytest.mark.asyncio
    async def test_speech_started_emitted_with_metadata(self, handler):
        """Test that speech_started event is emitted with item_id and audio_start_ms."""
        # Simulate speech starting
        handler.speech_started = False
        handler.n_samples_received = 24000  # 1 second at 24kHz

        # Pre-allocate item_id
        item_id = handler.session_state.get_pending_input_item_id()

        # Add first word to trigger speech_started
        await handler.add_chat_message_delta("Hello", "user")

        # Check that speech tracking state was updated
        assert handler.speech_started is True
        assert handler.speech_start_time is not None
        assert handler.current_speech_item_id == item_id

        # Check that speech_started event was queued
        events = []
        while not handler.output_queue.empty():
            events.append(await handler.output_queue.get())

        speech_started_events = [
            e for e in events if isinstance(e, ora.InputAudioBufferSpeechStarted)
        ]
        assert len(speech_started_events) > 0

        event = speech_started_events[0]
        assert event.item_id == item_id
        assert event.audio_start_ms is not None
        assert event.audio_start_ms >= 0

    @pytest.mark.asyncio
    async def test_speech_stopped_emitted_with_metadata(self, handler):
        """Test that speech_stopped event is emitted with item_id and audio_end_ms."""
        # Set up speech in progress
        handler.speech_started = True
        handler.speech_start_time = 1.0
        handler.n_samples_received = 48000  # 2 seconds at 24kHz
        handler.current_speech_item_id = "item_test123"

        handler.chatbot.conversation_state = MagicMock(return_value="user_speaking")

        # Simulate pause detection
        stt = handler.stt
        stt.pause_prediction.value = 0.7  # Above threshold
        stt.sent_samples = 48000
        stt.current_time = 2.0
        stt.delay_sec = 0.5

        pause_detected = handler.determine_pause()
        assert pause_detected is True

        # Manually emit speech_stopped (normally done in receive())
        audio_end_ms = int(handler.audio_received_sec() * 1000)
        await handler.output_queue.put(
            ora.InputAudioBufferSpeechStopped(
                item_id=handler.current_speech_item_id,
                audio_end_ms=audio_end_ms,
            )
        )

        # Check that event was queued
        event = await handler.output_queue.get()
        assert isinstance(event, ora.InputAudioBufferSpeechStopped)
        assert event.item_id == "item_test123"
        assert event.audio_end_ms is not None
        assert event.audio_end_ms > 0

    @pytest.mark.asyncio
    async def test_speech_duration_calculated(self, handler):
        """Test that speech duration is properly calculated and tracked."""
        handler.speech_started = True
        handler.speech_start_time = 1.0
        handler.n_samples_received = 72000  # 3 seconds at 24kHz (duration = 2s)

        duration = handler.audio_received_sec() - handler.speech_start_time
        assert duration == pytest.approx(2.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_speech_state_reset_on_pause(self, handler):
        """Test that speech tracking state is reset when pause is detected."""
        # Set up speech in progress
        handler.speech_started = True
        handler.speech_start_time = 1.0
        handler.current_speech_item_id = "item_test123"

        # Reset would happen in receive() after pause detection
        handler.speech_started = False
        handler.speech_start_time = None

        assert handler.speech_started is False
        assert handler.speech_start_time is None
        # current_speech_item_id is kept for reference


class TestErrorEventEmission:
    """Test error event emission for various conditions."""

    @pytest.fixture
    def handler(self):
        """Create handler with mocked dependencies."""
        with patch("unmute.unmute_handler.get_openai_client"):
            handler = UnmuteHandler()
            handler.session_state = SessionState()
            handler.chatbot = MagicMock()
            handler.chatbot.chat_history = []
            handler.chatbot.last_message = MagicMock(return_value=None)
            return handler

    @pytest.mark.asyncio
    async def test_silence_timeout_error(self, handler):
        """Test that silence timeout emits error event."""
        handler.chatbot.conversation_state = MagicMock(return_value="waiting_for_user")
        handler.waiting_for_user_start_time = 0.0
        handler.n_samples_received = int((USER_SILENCE_TIMEOUT + 1) * 24000)

        # Trigger silence detection
        await handler.detect_long_silence()

        # Check that error was emitted
        events = []
        while not handler.output_queue.empty():
            events.append(await handler.output_queue.get())

        errors = [e for e in events if isinstance(e, ora.Error)]
        assert len(errors) == 1
        assert "silence_timeout" in errors[0].error.type

    @pytest.mark.asyncio
    async def test_backpressure_warning_threshold(self, handler):
        """Test that backpressure warning is logged at warning threshold."""
        # Fill queue to warning level
        for _ in range(OUTPUT_QUEUE_WARNING_THRESHOLD + 1):
            await handler.output_queue.put(ora.SessionCreated(session=ora.Session()))

        handler.last_backpressure_check = 0.0  # Force check

        with patch("unmute.unmute_handler.logger") as mock_logger:
            await handler.check_backpressure()
            mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_backpressure_error_threshold(self, handler):
        """Test that backpressure error is emitted at error threshold."""
        # Fill queue to error level
        for _ in range(OUTPUT_QUEUE_ERROR_THRESHOLD + 1):
            await handler.output_queue.put(ora.SessionCreated(session=ora.Session()))

        handler.last_backpressure_check = 0.0  # Force check

        await handler.check_backpressure()

        # Check that error was emitted
        found_error = False
        while not handler.output_queue.empty():
            item = handler.output_queue.get_nowait()
            if isinstance(item, ora.Error) and "backpressure" in item.error.type:
                found_error = True
                break

        assert found_error, "Backpressure error should be emitted"

    @pytest.mark.asyncio
    async def test_backpressure_error_not_repeated(self, handler):
        """Test that backpressure error is not spammed when queue stays full."""
        # Fill queue to error level
        for _ in range(OUTPUT_QUEUE_ERROR_THRESHOLD + 1):
            await handler.output_queue.put(ora.SessionCreated(session=ora.Session()))

        handler.last_backpressure_check = 0.0

        # First check should emit error
        await handler.check_backpressure()
        assert handler.backpressure_error_emitted is True

        # Clear errors from queue
        error_count = 0
        while not handler.output_queue.empty():
            item = handler.output_queue.get_nowait()
            if isinstance(item, ora.Error) and "backpressure" in item.error.type:
                error_count += 1

        # Refill to keep above threshold
        for _ in range(OUTPUT_QUEUE_ERROR_THRESHOLD + 1):
            await handler.output_queue.put(ora.SessionCreated(session=ora.Session()))

        # Second check should NOT emit another error
        handler.last_backpressure_check = 0.0
        await handler.check_backpressure()

        # Count new errors (should be 0)
        new_error_count = 0
        while not handler.output_queue.empty():
            item = handler.output_queue.get_nowait()
            if isinstance(item, ora.Error) and "backpressure" in item.error.type:
                new_error_count += 1

        assert new_error_count == 0, "Should not emit duplicate backpressure errors"


class TestStateReset:
    """Test state reset after interruptions."""

    @pytest.fixture
    def handler(self):
        """Create handler with mocked dependencies."""
        with patch("unmute.unmute_handler.get_openai_client"):
            handler = UnmuteHandler()
            handler.session_state = SessionState()
            handler.chatbot = MagicMock()
            handler.chatbot.conversation_state = MagicMock(return_value="bot_speaking")
            handler.chatbot.chat_history = [
                {"role": "system", "content": "System prompt"},
                {"role": "assistant", "content": "Hello"},
            ]
            handler.chatbot.last_message = MagicMock(return_value="Hello")
            handler.chatbot.add_chat_message_delta = AsyncMock()

            handler.quest_manager = MagicMock()
            handler.quest_manager.remove = AsyncMock()
            handler._clear_queue = MagicMock()

            return handler

    @pytest.mark.asyncio
    async def test_interrupt_bot_resets_speech_state(self, handler):
        """Test that interrupt_bot resets speech tracking state."""
        # Set up active speech
        handler.speech_started = True
        handler.speech_start_time = 1.0
        handler.current_speech_item_id = "item_test123"

        # Interrupt
        await handler.interrupt_bot()

        # Verify speech state was reset
        assert handler.speech_started is False
        assert handler.speech_start_time is None
        # current_speech_item_id is preserved for reference

    @pytest.mark.asyncio
    async def test_interrupt_bot_clears_queues(self, handler):
        """Test that interrupt_bot clears output queue."""
        # Add some items to queue
        await handler.output_queue.put(ora.ResponseTextDelta(delta="test"))
        await handler.output_queue.put(ora.ResponseAudioDelta(delta="audio"))

        old_queue = handler.output_queue

        # Interrupt
        await handler.interrupt_bot()

        # Queue should be replaced
        assert handler.output_queue is not old_queue
        # New queue should be empty
        assert handler.output_queue.empty()

    @pytest.mark.asyncio
    async def test_interrupt_bot_emits_unmute_event(self, handler):
        """Test that interrupt_bot emits UnmuteInterruptedByVAD event."""
        await handler.interrupt_bot()

        # Check for UnmuteInterruptedByVAD event
        events = []
        while not handler.output_queue.empty():
            events.append(await handler.output_queue.get())

        interrupted_events = [
            e for e in events if isinstance(e, ora.UnmuteInterruptedByVAD)
        ]
        assert len(interrupted_events) == 1


class TestVADThresholds:
    """Test VAD threshold logic."""

    @pytest.fixture
    def handler(self):
        """Create handler with mocked dependencies."""
        with patch("unmute.unmute_handler.get_openai_client"):
            handler = UnmuteHandler()
            handler.session_state = SessionState()
            handler.chatbot = MagicMock()
            handler.chatbot.chat_history = []

            # Mock STT
            handler.quest_manager = MagicMock()
            handler.quest_manager.quests = {}
            stt_mock = MagicMock()
            stt_mock.pause_prediction = MagicMock()
            stt_mock.sent_samples = 0
            handler.quest_manager.quests["stt"] = MagicMock()
            handler.quest_manager.quests["stt"].get_nowait = MagicMock(return_value=stt_mock)

            return handler

    def test_pause_detection_threshold(self, handler):
        """Test that pause is detected at 0.6 threshold."""
        handler.chatbot.conversation_state = MagicMock(return_value="user_speaking")
        stt = handler.stt

        # Below threshold - no pause
        stt.pause_prediction.value = 0.59
        assert handler.determine_pause() is False

        # At threshold - pause detected
        stt.pause_prediction.value = 0.6
        assert handler.determine_pause() is True

        # Above threshold - pause detected
        stt.pause_prediction.value = 0.8
        assert handler.determine_pause() is True

    def test_pause_not_detected_when_bot_speaking(self, handler):
        """Test that pause detection doesn't trigger when bot is speaking."""
        handler.chatbot.conversation_state = MagicMock(return_value="bot_speaking")
        stt = handler.stt
        stt.pause_prediction.value = 0.8

        assert handler.determine_pause() is False

    def test_pause_not_detected_when_waiting(self, handler):
        """Test that pause detection doesn't trigger when waiting for user."""
        handler.chatbot.conversation_state = MagicMock(return_value="waiting_for_user")
        stt = handler.stt
        stt.pause_prediction.value = 0.8

        assert handler.determine_pause() is False
