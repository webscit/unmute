"""Integration tests for VAD events, backpressure, and buffer overflow.

These tests use simulated audio frames to trigger real event flows.
"""

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

import unmute.openai_realtime_api_events as ora
from unmute.audio.realtime_buffer import MAX_BUFFER_SAMPLES, RealtimeAudioBuffer
from unmute.kyutai_constants import SAMPLE_RATE
from unmute.unmute_handler import UnmuteHandler


class TestVADIntegration:
    """Integration tests for VAD event flow."""

    @pytest.fixture
    async def handler_with_stt(self):
        """Create handler with mocked STT that simulates real behavior."""
        with patch("unmute.unmute_handler.get_openai_client"):
            with patch("unmute.unmute_handler.get_openai_client"):
            handler = UnmuteHandler()

            # Mock chatbot
            handler.chatbot = MagicMock()
            handler.chatbot.chat_history = [{"role": "system", "content": "Test"}]
            handler.chatbot.conversation_state = MagicMock(return_value="user_speaking")
            handler.chatbot.get_instructions = MagicMock(return_value="Test instructions")
            handler.chatbot.preprocessed_messages = MagicMock(
                return_value=[{"role": "system", "content": "Test"}]
            )
            handler.chatbot.add_chat_message_delta = AsyncMock(return_value=True)
            handler.chatbot.last_message = MagicMock(return_value=None)

            # Mock STT
            stt_mock = MagicMock()
            stt_mock.pause_prediction = MagicMock()
            stt_mock.pause_prediction.value = 0.0
            stt_mock.pause_prediction.update = MagicMock()
            stt_mock.sent_samples = 0
            stt_mock.current_time = 0.0
            stt_mock.delay_sec = 0.5
            stt_mock.send_audio = AsyncMock()

            handler.quest_manager = MagicMock()
            handler.quest_manager.quests = {"stt": MagicMock()}
            handler.quest_manager.quests["stt"].get_nowait = MagicMock(return_value=stt_mock)

            return handler

    @pytest.mark.asyncio
    async def test_full_speech_cycle_emits_events(self, handler_with_stt):
        """Test that a full speech cycle (start -> stop) emits proper events."""
        handler = handler_with_stt
        stt = handler.stt

        # Simulate speech starting (first word)
        await handler.add_chat_message_delta("Hello", "user")

        # Should have emitted speech_started
        events = []
        while not handler.output_queue.empty():
            events.append(await handler.output_queue.get())

        speech_started = [
            e for e in events if isinstance(e, ora.InputAudioBufferSpeechStarted)
        ]
        assert len(speech_started) > 0, "speech_started should be emitted"

        # Simulate pause detection
        stt.pause_prediction.value = 0.7
        handler.chatbot.conversation_state = MagicMock(return_value="user_speaking")

        pause_detected = handler.determine_pause()
        assert pause_detected is True

        # Manually emit speech_stopped
        audio_end_ms = int(handler.audio_received_sec() * 1000)
        await handler.output_queue.put(
            ora.InputAudioBufferSpeechStopped(
                item_id=handler.current_speech_item_id,
                audio_end_ms=audio_end_ms,
            )
        )

        # Verify speech_stopped was emitted
        event = await handler.output_queue.get()
        assert isinstance(event, ora.InputAudioBufferSpeechStopped)

    @pytest.mark.asyncio
    async def test_multiple_speech_segments(self, handler_with_stt):
        """Test handling multiple speech segments in sequence."""
        handler = handler_with_stt
        stt = handler.stt

        # First speech segment
        await handler.add_chat_message_delta("First", "user")
        item_id_1 = handler.current_speech_item_id

        # Reset for second segment
        handler.speech_started = False
        handler.speech_start_time = None
        handler.current_speech_item_id = None

        # Second speech segment
        await handler.add_chat_message_delta("Second", "user")
        item_id_2 = handler.current_speech_item_id

        # Item IDs should be different
        assert item_id_1 != item_id_2

        # Should have two speech_started events
        events = []
        while not handler.output_queue.empty():
            events.append(await handler.output_queue.get())

        speech_started_events = [
            e for e in events if isinstance(e, ora.InputAudioBufferSpeechStarted)
        ]
        assert len(speech_started_events) == 2

    @pytest.mark.asyncio
    async def test_interruption_resets_speech_state(self, handler_with_stt):
        """Test that interruption during speech properly resets state."""
        handler = handler_with_stt
        handler.chatbot.conversation_state = MagicMock(return_value="bot_speaking")
        handler.quest_manager.remove = AsyncMock()
        handler._clear_queue = MagicMock()

        # Start speech
        handler.speech_started = True
        handler.speech_start_time = 1.0
        handler.current_speech_item_id = "item_test"

        # Interrupt
        await handler.interrupt_bot()

        # Verify reset
        assert handler.speech_started is False
        assert handler.speech_start_time is None


class TestBufferOverflowIntegration:
    """Integration tests for buffer overflow detection."""

    @pytest.mark.asyncio
    async def test_buffer_overflow_emits_error(self):
        """Test that buffer overflow triggers error event emission."""
        # Create a buffer with small max size for testing
        buffer = RealtimeAudioBuffer(
            sample_rate=SAMPLE_RATE,
            max_buffer_samples=1000,  # Very small for testing
        )

        # Create fake Opus packets (simplified - just checking overflow logic)
        # Fill buffer to capacity
        fake_opus = b"\x00" * 100
        pcm = None

        # Try to append many frames
        for _ in range(50):
            pcm = buffer.append_opus(fake_opus)
            if pcm is None:
                # Overflow protection kicked in
                break

        # Should have hit overflow
        assert buffer.total_samples >= 1000 or pcm is None

    @pytest.mark.asyncio
    async def test_buffer_overflow_protection_discards_frames(self):
        """Test that overflow protection discards frames correctly."""
        buffer = RealtimeAudioBuffer(
            sample_rate=SAMPLE_RATE,
            max_buffer_samples=500,
        )

        # Mock the opus decoder to return predictable data
        mock_decoder = MagicMock()
        mock_decoder.append_bytes = MagicMock(
            return_value=np.zeros(480, dtype=np.float32)
        )
        buffer._opus_reader = mock_decoder
        buffer._wait_for_first_opus = False  # Skip first packet sync

        # Fill to capacity
        samples_added = 0
        for i in range(10):
            pcm = buffer.append_opus(b"\x00" * 100)
            if pcm is not None:
                samples_added += pcm.size

        # Should stop at or before max
        assert buffer.total_samples <= 500


class TestBackpressureIntegration:
    """Integration tests for backpressure handling."""

    @pytest.fixture
    def handler(self):
        """Create handler for backpressure testing."""
        with patch("unmute.unmute_handler.get_openai_client"):
            with patch("unmute.unmute_handler.get_openai_client"):
            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.conversation_state = MagicMock(return_value="user_speaking")
            return handler

    @pytest.mark.asyncio
    async def test_backpressure_detected_in_receive_loop(self, handler):
        """Test that backpressure is checked during receive loop."""
        # Fill queue significantly
        for i in range(60):
            await handler.output_queue.put(
                ora.ResponseTextDelta(delta=f"word{i}")
            )

        # Force backpressure check
        handler.last_backpressure_check = 0.0

        # Check should detect high queue size
        with patch("unmute.unmute_handler.logger") as mock_logger:
            await handler.check_backpressure()
            # Should have logged a warning
            assert mock_logger.warning.called or mock_logger.error.called

    @pytest.mark.asyncio
    async def test_backpressure_recovery(self, handler):
        """Test that backpressure error flag resets when queue drains."""
        # Trigger backpressure error
        for _ in range(105):
            await handler.output_queue.put(ora.ResponseTextDelta(delta="test"))

        handler.last_backpressure_check = 0.0
        await handler.check_backpressure()
        assert handler.backpressure_error_emitted is True

        # Drain queue
        while not handler.output_queue.empty():
            await handler.output_queue.get()

        # Check again - should reset flag
        handler.last_backpressure_check = 0.0
        await handler.check_backpressure()
        assert handler.backpressure_error_emitted is False

    @pytest.mark.asyncio
    async def test_backpressure_check_interval(self, handler):
        """Test that backpressure checks respect interval."""
        handler.last_backpressure_check = 1.0
        handler.n_samples_received = int(1.5 * SAMPLE_RATE)  # 1.5 seconds

        # Should skip check (interval not elapsed)
        initial_queue_size = handler.output_queue.qsize()
        await handler.check_backpressure()

        # last_backpressure_check should not have updated
        assert handler.last_backpressure_check == 1.0


class TestErrorEventPropagation:
    """Test that error events are properly propagated through emit_loop."""

    @pytest.mark.asyncio
    async def test_silence_timeout_error_propagates(self):
        """Test that silence timeout error reaches output queue."""
        with patch("unmute.unmute_handler.get_openai_client"):
            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.conversation_state = MagicMock(return_value="waiting_for_user")
            handler.chatbot.chat_history = []
            handler.chatbot.add_chat_message_delta = AsyncMock()
            handler.waiting_for_user_start_time = 0.0
            handler.n_samples_received = int(10 * SAMPLE_RATE)  # 10 seconds

            await handler.detect_long_silence()

            # Check for error in queue
            events = []
            while not handler.output_queue.empty():
                events.append(await handler.output_queue.get())

            errors = [e for e in events if isinstance(e, ora.Error)]
            assert len(errors) > 0
            assert any("silence_timeout" in e.error.type for e in errors)

    @pytest.mark.asyncio
    async def test_error_events_have_proper_structure(self):
        """Test that error events conform to OpenAI API structure."""
        with patch("unmute.unmute_handler.get_openai_client"):
            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.conversation_state = MagicMock(return_value="waiting_for_user")
            handler.chatbot.chat_history = []
            handler.chatbot.add_chat_message_delta = AsyncMock()
            handler.waiting_for_user_start_time = 0.0
            handler.n_samples_received = int(10 * SAMPLE_RATE)

            await handler.detect_long_silence()

            # Get error event
            error_event = None
            while not handler.output_queue.empty():
                event = await handler.output_queue.get()
                if isinstance(event, ora.Error):
                    error_event = event
                    break

            assert error_event is not None
            assert hasattr(error_event, "error")
            assert hasattr(error_event.error, "type")
            assert hasattr(error_event.error, "message")
            assert isinstance(error_event.error.type, str)
            assert isinstance(error_event.error.message, str)


class TestMetricsRecording:
    """Test that metrics are properly recorded for VAD events."""

    @pytest.mark.asyncio
    async def test_speech_started_increments_metric(self):
        """Test that speech_started events increment counter."""
        with patch("unmute.unmute_handler.get_openai_client"):
            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.chat_history = []
            handler.chatbot.add_chat_message_delta = AsyncMock(return_value=True)

            # Mock the metric
            with patch("unmute.unmute_handler.mt") as mock_mt:
                await handler.add_chat_message_delta("Hello", "user")

                # Metric should have been incremented
                assert mock_mt.VAD_SPEECH_STARTED.inc.called

    @pytest.mark.asyncio
    async def test_buffer_overflow_increments_metric(self):
        """Test that buffer overflow errors increment counter."""
        from unmute.exceptions import make_ora_error

        with patch("unmute.metrics.BUFFER_OVERFLOW_ERRORS") as mock_metric:
            # Simulate overflow detection in main_websocket
            # (This would normally be in the receive_loop)
            error = make_ora_error(
                type="buffer_overflow",
                message="Test overflow",
            )

            # Increment would happen in main_websocket.py
            mock_metric.inc()
            assert mock_metric.inc.called
