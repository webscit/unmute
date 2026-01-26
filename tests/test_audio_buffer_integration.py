"""Integration tests for RealtimeAudioBuffer with handler and websocket flow.

Tests the complete event flow:
- InputAudioBufferAppend -> decode -> buffer -> STT
- InputAudioBufferCommit -> commit buffer -> create item
- InputAudioBufferClear -> clear buffer

Also tests latency tracking and transcription event emission.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from unmute.audio.realtime_buffer import (
    BufferState,
    RealtimeAudioBuffer,
)
from unmute.kyutai_constants import SAMPLE_RATE


class MockOpusDecoder:
    """Mock Opus decoder for integration testing."""

    def __init__(self):
        self.pcm_queue: list[np.ndarray] = []
        self.default_pcm = np.zeros(960, dtype=np.float32)

    def append_bytes(self, data: bytes) -> np.ndarray:
        if self.pcm_queue:
            return self.pcm_queue.pop(0)
        return self.default_pcm.copy()

    def queue_pcm(self, pcm: np.ndarray) -> None:
        """Queue PCM data to return on next decode call."""
        self.pcm_queue.append(pcm)


class TestAppendFlushEventFlow:
    """Test the complete append/flush event flow."""

    def test_append_commit_clear_cycle(self):
        """Test a complete cycle of append -> commit -> reset -> append."""
        mock_decoder = MockOpusDecoder()
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False

        # Phase 1: Append audio frames
        for i in range(5):
            pcm = np.full(480, float(i), dtype=np.float32)
            mock_decoder.queue_pcm(pcm)
            result = buffer.append_opus(f"frame{i}".encode())
            assert result is not None

        assert buffer.total_samples == 2400
        assert buffer.frame_count == 5
        assert buffer.state == BufferState.ACCUMULATING

        # Phase 2: Commit buffer
        total, metrics = buffer.commit("item_test_123")
        assert total == 2400
        assert buffer.state == BufferState.COMMITTED
        assert buffer.pending_item_id == "item_test_123"
        assert metrics.first_frame_arrival is not None

        # Phase 3: Reset for next segment
        buffer.reset()
        assert buffer.state == BufferState.ACCUMULATING
        assert buffer.total_samples == 0
        assert buffer.frame_count == 0

        # Phase 4: New audio segment
        mock_decoder.queue_pcm(np.zeros(960, dtype=np.float32))
        result = buffer.append_opus(b"new_frame")
        assert result is not None
        assert buffer.total_samples == 960

    def test_append_clear_cycle(self):
        """Test append -> clear cycle without commit."""
        mock_decoder = MockOpusDecoder()
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False

        # Append some audio
        for _ in range(3):
            mock_decoder.queue_pcm(np.zeros(960, dtype=np.float32))
            buffer.append_opus(b"frame")

        assert buffer.total_samples == 2880

        # Clear without commit
        cleared = buffer.clear()
        assert cleared == 2880
        assert buffer.state == BufferState.CLEARED
        assert buffer.total_samples == 0

        # Reset and continue
        buffer.reset()
        assert buffer.state == BufferState.ACCUMULATING


class TestLatencyTrackingFlow:
    """Test latency tracking through the event flow."""

    def test_latency_tracked_through_append_commit(self):
        """Test that latency is tracked from first append to commit."""
        mock_decoder = MockOpusDecoder()
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False

        # Append frames
        mock_decoder.queue_pcm(np.zeros(960, dtype=np.float32))
        buffer.append_opus(b"first")

        first_arrival = buffer.latency_metrics.first_frame_arrival
        assert first_arrival is not None

        # More frames
        for _ in range(2):
            mock_decoder.queue_pcm(np.zeros(960, dtype=np.float32))
            buffer.append_opus(b"more")

        # Last frame should update last_frame_arrival
        last_arrival = buffer.latency_metrics.last_frame_arrival
        assert last_arrival is not None
        assert last_arrival >= first_arrival

        # Mark STT flush
        buffer.mark_stt_flush()
        assert buffer.latency_metrics.stt_flush_time is not None
        assert buffer.latency_metrics.arrival_to_flush_ms is not None

        # Commit
        _, metrics = buffer.commit("item_123")
        assert metrics.commit_time is not None
        assert metrics.buffer_duration_ms is not None

    def test_latency_reset_on_clear(self):
        """Test that latency tracking resets on clear."""
        mock_decoder = MockOpusDecoder()
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False

        mock_decoder.queue_pcm(np.zeros(960, dtype=np.float32))
        buffer.append_opus(b"frame")
        assert buffer.latency_metrics.first_frame_arrival is not None

        buffer.clear()
        assert buffer.latency_metrics.first_frame_arrival is None


class TestSTTStreamingFlow:
    """Test STT streaming through the buffer."""

    @pytest.mark.asyncio
    async def test_stream_all_frames_to_stt(self):
        """Test streaming all buffered frames to STT."""
        mock_decoder = MockOpusDecoder()
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False

        # Buffer some frames
        for i in range(4):
            pcm = np.full(480, float(i), dtype=np.float32)
            mock_decoder.queue_pcm(pcm)
            buffer.append_opus(f"frame{i}".encode())

        # Create mock STT
        mock_stt = AsyncMock()

        # Stream to STT
        frames_sent = await buffer.stream_to_stt(mock_stt)

        assert frames_sent == 4
        assert mock_stt.send_audio.call_count == 4

        # Verify correct PCM values were sent
        calls = mock_stt.send_audio.call_args_list
        for i, call in enumerate(calls):
            pcm_arg = call[0][0]
            assert np.all(pcm_arg == float(i))

    @pytest.mark.asyncio
    async def test_stream_partial_to_stt(self):
        """Test streaming partial buffer to STT."""
        mock_decoder = MockOpusDecoder()
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False

        for _ in range(5):
            mock_decoder.queue_pcm(np.zeros(480, dtype=np.float32))
            buffer.append_opus(b"frame")

        mock_stt = AsyncMock()

        # Stream only from frame 3 onwards
        frames_sent = await buffer.stream_to_stt(mock_stt, start_index=3)

        assert frames_sent == 2  # Frames 3 and 4
        assert mock_stt.send_audio.call_count == 2


class TestSequenceValidationFlow:
    """Test sequence validation in event flows."""

    def test_sequence_integrity_maintained(self):
        """Test that sequence numbers remain consistent through flow."""
        mock_decoder = MockOpusDecoder()
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False

        for i in range(10):
            mock_decoder.queue_pcm(np.zeros(480, dtype=np.float32))
            buffer.append_opus(f"frame{i}".encode())

        # Validate sequences
        issues = buffer.validate_sequence()
        assert len(issues) == 0

        # Verify actual sequence values
        for i, frame in enumerate(buffer._frames):
            assert frame.sequence == i


class TestOverflowProtectionFlow:
    """Test overflow protection in event flows."""

    def test_overflow_protection_during_append(self):
        """Test that overflow protection works during append flow."""
        mock_decoder = MockOpusDecoder()
        buffer = RealtimeAudioBuffer(
            opus_decoder=mock_decoder,
            max_buffer_samples=5000,
        )
        buffer._wait_for_first_opus = False

        # Fill up to limit
        mock_decoder.queue_pcm(np.zeros(5000, dtype=np.float32))
        result = buffer.append_opus(b"big_frame")
        assert result is not None
        assert buffer.total_samples == 5000

        # Next append should be rejected
        mock_decoder.queue_pcm(np.zeros(100, dtype=np.float32))
        result = buffer.append_opus(b"overflow")
        assert result is None
        assert buffer.total_samples == 5000  # Unchanged

    def test_overflow_clears_on_reset(self):
        """Test that overflow state clears on reset."""
        mock_decoder = MockOpusDecoder()
        buffer = RealtimeAudioBuffer(
            opus_decoder=mock_decoder,
            max_buffer_samples=1000,
        )
        buffer._wait_for_first_opus = False

        # Fill to limit
        mock_decoder.queue_pcm(np.zeros(1000, dtype=np.float32))
        buffer.append_opus(b"fill")

        # Commit and reset
        buffer.commit("item_123")
        buffer.reset()

        # Should accept new data
        mock_decoder.queue_pcm(np.zeros(500, dtype=np.float32))
        result = buffer.append_opus(b"new")
        assert result is not None
        assert buffer.total_samples == 500


class TestAnonymizedRecordingFlow:
    """Test anonymized recording hooks in event flow."""

    def test_recording_hook_captures_all_frames(self):
        """Test that recording hook is called for every frame."""
        records = []

        def on_recorded(record):
            records.append(record)

        mock_decoder = MockOpusDecoder()
        buffer = RealtimeAudioBuffer(
            opus_decoder=mock_decoder,
            on_frame_recorded=on_recorded,
        )
        buffer._wait_for_first_opus = False

        # Append several frames
        for i in range(5):
            pcm_size = 480 + i * 100
            mock_decoder.queue_pcm(np.zeros(pcm_size, dtype=np.float32))
            buffer.append_opus(f"frame{i}".encode())

        # Verify records
        assert len(records) == 5
        for i, record in enumerate(records):
            assert record.sequence == i
            assert record.num_samples == 480 + i * 100
            assert record.sample_rate == SAMPLE_RATE

    def test_recording_hook_not_called_for_skipped_frames(self):
        """Test that recording hook is not called for skipped frames."""
        records = []

        def on_recorded(record):
            records.append(record)

        mock_decoder = MockOpusDecoder()
        buffer = RealtimeAudioBuffer(
            opus_decoder=mock_decoder,
            on_frame_recorded=on_recorded,
        )
        # Still waiting for first opus packet
        assert buffer._wait_for_first_opus

        # This should be skipped (no first-packet bit)
        mock_decoder.queue_pcm(np.zeros(960, dtype=np.float32))
        result = buffer.append_opus(bytes([0] * 10))
        assert result is None
        assert len(records) == 0

    def test_recording_hook_serialization(self):
        """Test that recording hook data serializes correctly."""
        import json

        records = []

        def on_recorded(record):
            # Verify it can be serialized
            json_str = json.dumps(record.to_dict())
            records.append(json.loads(json_str))

        mock_decoder = MockOpusDecoder()
        buffer = RealtimeAudioBuffer(
            opus_decoder=mock_decoder,
            on_frame_recorded=on_recorded,
        )
        buffer._wait_for_first_opus = False

        mock_decoder.queue_pcm(np.zeros(960, dtype=np.float32))
        buffer.append_opus(b"frame")

        assert len(records) == 1
        assert "sequence" in records[0]
        assert "num_samples" in records[0]
        assert "arrival_time" in records[0]


class TestRewindFlow:
    """Test rewind functionality in event flows."""

    def test_rewind_after_partial_processing(self):
        """Test rewinding buffer after partial processing."""
        mock_decoder = MockOpusDecoder()
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False

        # Add frames
        for i in range(5):
            mock_decoder.queue_pcm(np.full(200, float(i), dtype=np.float32))
            buffer.append_opus(f"frame{i}".encode())

        assert buffer.total_samples == 1000

        # Rewind last 400 samples (2 frames)
        removed = buffer.rewind(400)
        assert removed is not None
        assert removed.size == 400
        assert buffer.total_samples == 600
        assert buffer.frame_count == 3

        # Verify remaining frames
        remaining = buffer.get_all_pcm()
        assert remaining.size == 600

    def test_rewind_and_continue_appending(self):
        """Test that appending continues correctly after rewind."""
        mock_decoder = MockOpusDecoder()
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False

        # Initial frames
        for _ in range(3):
            mock_decoder.queue_pcm(np.zeros(300, dtype=np.float32))
            buffer.append_opus(b"frame")

        # Rewind
        buffer.rewind(300)
        assert buffer.total_samples == 600

        # Continue appending
        mock_decoder.queue_pcm(np.ones(400, dtype=np.float32))
        buffer.append_opus(b"new")
        assert buffer.total_samples == 1000

        # Verify new frame was appended
        all_pcm = buffer.get_all_pcm()
        assert np.all(all_pcm[-400:] == 1.0)


class TestSnapshotRestoreFlow:
    """Test snapshot and state preservation."""

    def test_snapshot_preserves_state_for_recording(self):
        """Test that snapshot captures state for session recording."""
        mock_decoder = MockOpusDecoder()
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False

        # Build up some state
        for _ in range(3):
            mock_decoder.queue_pcm(np.zeros(960, dtype=np.float32))
            buffer.append_opus(b"frame")

        buffer.commit("item_xyz")

        # Capture snapshot
        snapshot = buffer.snapshot()

        assert snapshot["state"] == "COMMITTED"
        assert snapshot["total_samples"] == 2880
        assert snapshot["frame_count"] == 3
        assert snapshot["pending_item_id"] == "item_xyz"
        assert snapshot["duration_sec"] == 2880 / SAMPLE_RATE

    def test_snapshot_json_serializable(self):
        """Test that snapshot is fully JSON serializable."""
        import json

        mock_decoder = MockOpusDecoder()
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False

        mock_decoder.queue_pcm(np.zeros(960, dtype=np.float32))
        buffer.append_opus(b"frame")
        buffer.mark_stt_flush()
        buffer.commit("item_123")

        snapshot = buffer.snapshot()
        json_str = json.dumps(snapshot)
        restored = json.loads(json_str)

        assert restored["state"] == "COMMITTED"
        assert restored["pending_item_id"] == "item_123"
