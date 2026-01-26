"""Tests for RealtimeAudioBuffer class.

Tests the per-session audio buffer manager for:
- Buffer ordering and frame storage
- Sequence validation
- Codec guardrails (Opus decoding)
- Commit/reset semantics
- Latency tracking
- Overflow protection
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from unmute.audio.realtime_buffer import (
    AnonymizedFrameRecord,
    AudioFrame,
    BufferState,
    CodecMetadata,
    LatencyMetrics,
    OpusDecoder,
    RealtimeAudioBuffer,
)
from unmute.kyutai_constants import SAMPLE_RATE


class MockOpusDecoder:
    """Mock Opus decoder for testing."""

    def __init__(self, pcm_to_return: np.ndarray | None = None):
        self.pcm_to_return = pcm_to_return
        self.calls: list[bytes] = []

    def append_bytes(self, data: bytes) -> np.ndarray:
        self.calls.append(data)
        if self.pcm_to_return is not None:
            return self.pcm_to_return
        return np.zeros(960, dtype=np.float32)

    def set_pcm(self, pcm: np.ndarray) -> None:
        """Set the PCM data to return on next call."""
        self.pcm_to_return = pcm


class TestAudioFrame:
    """Test AudioFrame dataclass."""

    def test_frame_properties(self):
        """Test frame property calculations."""
        pcm = np.zeros(960, dtype=np.float32)
        frame = AudioFrame(
            pcm=pcm,
            arrival_time=1000.0,
            sequence=0,
            sample_rate=SAMPLE_RATE,
        )

        assert frame.num_samples == 960
        assert frame.duration_sec == 960 / SAMPLE_RATE

    def test_frame_with_different_sample_rates(self):
        """Test frame duration with different sample rates."""
        pcm = np.zeros(480, dtype=np.float32)
        frame = AudioFrame(
            pcm=pcm,
            arrival_time=1000.0,
            sequence=0,
            sample_rate=16000,
        )

        assert frame.duration_sec == 480 / 16000


class TestCodecMetadata:
    """Test CodecMetadata dataclass."""

    def test_defaults(self):
        """Test default codec metadata."""
        meta = CodecMetadata()
        assert meta.sample_rate == SAMPLE_RATE
        assert meta.channels == 1
        assert meta.codec == "opus"

    def test_to_dict(self):
        """Test serialization to dict."""
        meta = CodecMetadata(sample_rate=16000, channels=2, codec="pcm")
        d = meta.to_dict()

        assert d["sample_rate"] == 16000
        assert d["channels"] == 2
        assert d["codec"] == "pcm"


class TestLatencyMetrics:
    """Test LatencyMetrics dataclass."""

    def test_initial_state(self):
        """Test initial metrics are None."""
        metrics = LatencyMetrics()
        assert metrics.first_frame_arrival is None
        assert metrics.arrival_to_flush_ms is None
        assert metrics.buffer_duration_ms is None

    def test_arrival_to_flush_calculation(self):
        """Test arrival to flush latency calculation."""
        metrics = LatencyMetrics(
            first_frame_arrival=1000.0,
            stt_flush_time=1000.050,
        )

        assert metrics.arrival_to_flush_ms is not None
        assert abs(metrics.arrival_to_flush_ms - 50.0) < 0.001

    def test_buffer_duration_calculation(self):
        """Test buffer duration calculation."""
        metrics = LatencyMetrics(
            first_frame_arrival=1000.0,
            last_frame_arrival=1000.100,
        )

        assert metrics.buffer_duration_ms is not None
        assert abs(metrics.buffer_duration_ms - 100.0) < 0.001

    def test_to_dict(self):
        """Test metrics serialization."""
        metrics = LatencyMetrics(
            first_frame_arrival=1000.0,
            last_frame_arrival=1000.100,
            stt_flush_time=1000.150,
        )
        d = metrics.to_dict()

        assert "arrival_to_flush_ms" in d
        assert "buffer_duration_ms" in d


class TestAnonymizedFrameRecord:
    """Test AnonymizedFrameRecord for recording hooks."""

    def test_to_dict(self):
        """Test serialization without raw audio."""
        record = AnonymizedFrameRecord(
            sequence=5,
            num_samples=960,
            arrival_time=1234.567,
            sample_rate=SAMPLE_RATE,
        )
        d = record.to_dict()

        assert d["sequence"] == 5
        assert d["num_samples"] == 960
        assert d["arrival_time"] == 1234.567
        assert d["sample_rate"] == SAMPLE_RATE
        # Verify no raw audio in serialization
        assert "pcm" not in d
        assert "audio" not in d


class TestRealtimeAudioBufferInitialization:
    """Test buffer initialization."""

    def test_initial_state(self):
        """Test buffer starts in correct state."""
        buffer = RealtimeAudioBuffer()

        assert buffer.state == BufferState.ACCUMULATING
        assert buffer.total_samples == 0
        assert buffer.frame_count == 0
        assert buffer.duration_sec == 0.0
        assert buffer.pending_item_id is None

    def test_custom_sample_rate(self):
        """Test buffer with custom sample rate."""
        buffer = RealtimeAudioBuffer(sample_rate=16000)
        assert buffer.codec_metadata.sample_rate == 16000

    def test_custom_max_buffer(self):
        """Test buffer with custom max size."""
        buffer = RealtimeAudioBuffer(max_buffer_samples=1000)
        # Buffer should accept this configuration
        assert buffer._max_buffer_samples == 1000


class TestRealtimeAudioBufferAppend:
    """Test audio append functionality."""

    def test_append_opus_waits_for_first_packet(self):
        """Test that buffer waits for first valid Opus packet."""
        buffer = RealtimeAudioBuffer()

        # Create a fake Opus packet without the first-packet bit
        fake_opus_no_start = bytes([0] * 10)
        result = buffer.append_opus(fake_opus_no_start)

        assert result is None
        assert buffer.total_samples == 0

    def test_append_opus_accepts_first_packet(self):
        """Test that buffer accepts packet with first-packet bit."""
        mock_decoder = MockOpusDecoder(np.zeros(960, dtype=np.float32))
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)

        # Byte 5 with bit 1 set = 0x02
        fake_opus_with_start = bytes([0, 0, 0, 0, 0, 2]) + bytes([0] * 10)
        result = buffer.append_opus(fake_opus_with_start)

        assert result is not None
        assert buffer.total_samples == 960
        assert buffer.frame_count == 1

    def test_append_increments_sequence(self):
        """Test that sequence counter increments with each frame."""
        mock_decoder = MockOpusDecoder(np.zeros(480, dtype=np.float32))
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False  # Skip first packet check

        buffer.append_opus(b"fake1")
        buffer.append_opus(b"fake2")
        buffer.append_opus(b"fake3")

        assert buffer._sequence_counter == 3
        assert buffer._frames[0].sequence == 0
        assert buffer._frames[1].sequence == 1
        assert buffer._frames[2].sequence == 2

    def test_append_updates_latency_tracking(self):
        """Test that latency metrics are updated on append."""
        mock_decoder = MockOpusDecoder(np.zeros(960, dtype=np.float32))
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False

        buffer.append_opus(b"fake")

        assert buffer.latency_metrics.first_frame_arrival is not None
        assert buffer.latency_metrics.last_frame_arrival is not None

    def test_append_empty_decode_returns_none(self):
        """Test that empty decode result returns None."""
        mock_decoder = MockOpusDecoder(np.array([], dtype=np.float32))
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False

        result = buffer.append_opus(b"fake")

        assert result is None
        assert buffer.total_samples == 0

    def test_append_in_wrong_state_returns_none(self):
        """Test that append fails when buffer not in ACCUMULATING state."""
        mock_decoder = MockOpusDecoder(np.zeros(960, dtype=np.float32))
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False
        buffer._state = BufferState.COMMITTED

        result = buffer.append_opus(b"fake")

        assert result is None

    def test_overflow_protection(self):
        """Test that overflow protection prevents excessive buffering."""
        mock_decoder = MockOpusDecoder()
        buffer = RealtimeAudioBuffer(max_buffer_samples=1000, opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False

        # First append fills buffer
        mock_decoder.set_pcm(np.zeros(1000, dtype=np.float32))
        buffer.append_opus(b"fake1")
        assert buffer.total_samples == 1000

        # Second append should be rejected
        mock_decoder.set_pcm(np.zeros(100, dtype=np.float32))
        result = buffer.append_opus(b"fake2")

        assert result is None
        assert buffer.total_samples == 1000  # Still at limit

    def test_recording_hook_called(self):
        """Test that anonymized recording hook is called on append."""
        records = []

        def on_recorded(record):
            records.append(record)

        mock_decoder = MockOpusDecoder(np.zeros(960, dtype=np.float32))
        buffer = RealtimeAudioBuffer(
            on_frame_recorded=on_recorded, opus_decoder=mock_decoder
        )
        buffer._wait_for_first_opus = False

        buffer.append_opus(b"fake")

        assert len(records) == 1
        assert records[0].num_samples == 960
        assert records[0].sequence == 0


class TestRealtimeAudioBufferCommit:
    """Test buffer commit functionality."""

    def test_commit_changes_state(self):
        """Test that commit changes buffer state."""
        mock_decoder = MockOpusDecoder(np.zeros(960, dtype=np.float32))
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False

        buffer.append_opus(b"fake")

        total, metrics = buffer.commit("item_123")

        assert buffer.state == BufferState.COMMITTED
        assert buffer.pending_item_id == "item_123"
        assert total == 960
        assert metrics.commit_time is not None

    def test_commit_returns_correct_samples(self):
        """Test that commit returns correct sample count."""
        mock_decoder = MockOpusDecoder()
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False

        mock_decoder.set_pcm(np.zeros(480, dtype=np.float32))
        buffer.append_opus(b"fake1")
        mock_decoder.set_pcm(np.zeros(960, dtype=np.float32))
        buffer.append_opus(b"fake2")

        total, _ = buffer.commit("item_123")
        assert total == 1440


class TestRealtimeAudioBufferClear:
    """Test buffer clear functionality."""

    def test_clear_changes_state(self):
        """Test that clear changes buffer state."""
        mock_decoder = MockOpusDecoder(np.zeros(960, dtype=np.float32))
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False

        buffer.append_opus(b"fake")

        cleared = buffer.clear()

        assert buffer.state == BufferState.CLEARED
        assert buffer.total_samples == 0
        assert buffer.frame_count == 0
        assert cleared == 960

    def test_clear_resets_latency_metrics(self):
        """Test that clear resets latency tracking."""
        mock_decoder = MockOpusDecoder(np.zeros(960, dtype=np.float32))
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False

        buffer.append_opus(b"fake")

        buffer.clear()

        assert buffer.latency_metrics.first_frame_arrival is None


class TestRealtimeAudioBufferReset:
    """Test buffer reset functionality."""

    def test_reset_returns_to_accumulating(self):
        """Test that reset returns buffer to ACCUMULATING state."""
        buffer = RealtimeAudioBuffer()
        buffer._state = BufferState.COMMITTED

        buffer.reset()

        assert buffer.state == BufferState.ACCUMULATING
        assert buffer.total_samples == 0
        assert buffer.frame_count == 0
        assert buffer._sequence_counter == 0

    def test_reset_preserves_opus_stream_state(self):
        """Test that reset doesn't reset Opus stream state."""
        buffer = RealtimeAudioBuffer()
        buffer._wait_for_first_opus = False  # Simulate having found first packet

        buffer.reset()

        # Opus stream state should be preserved for continued streaming
        assert not buffer._wait_for_first_opus


class TestRealtimeAudioBufferRewind:
    """Test buffer rewind functionality."""

    def test_rewind_removes_samples(self):
        """Test that rewind removes samples from end."""
        mock_decoder = MockOpusDecoder()
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False

        # Add 3 frames of 480 samples each
        for i in range(3):
            pcm = np.full(480, float(i), dtype=np.float32)
            mock_decoder.set_pcm(pcm)
            buffer.append_opus(f"fake{i}".encode())

        assert buffer.total_samples == 1440

        removed = buffer.rewind(480)

        assert removed is not None
        assert removed.size == 480
        assert buffer.total_samples == 960
        assert buffer.frame_count == 2

    def test_rewind_partial_frame(self):
        """Test that rewind can partially remove a frame."""
        mock_decoder = MockOpusDecoder(np.zeros(960, dtype=np.float32))
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False

        buffer.append_opus(b"fake")

        removed = buffer.rewind(100)

        assert removed is not None
        assert removed.size == 100
        assert buffer.total_samples == 860
        assert buffer.frame_count == 1

    def test_rewind_insufficient_data(self):
        """Test that rewind returns None if insufficient data."""
        mock_decoder = MockOpusDecoder(np.zeros(480, dtype=np.float32))
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False

        buffer.append_opus(b"fake")

        result = buffer.rewind(1000)
        assert result is None
        assert buffer.total_samples == 480  # Unchanged

    def test_rewind_zero_samples(self):
        """Test that rewind with zero samples returns None."""
        buffer = RealtimeAudioBuffer()
        result = buffer.rewind(0)
        assert result is None


class TestRealtimeAudioBufferGetAllPcm:
    """Test getting all PCM samples."""

    def test_get_all_pcm_empty(self):
        """Test getting PCM from empty buffer."""
        buffer = RealtimeAudioBuffer()
        pcm = buffer.get_all_pcm()
        assert pcm.size == 0

    def test_get_all_pcm_concatenates(self):
        """Test that get_all_pcm concatenates all frames."""
        mock_decoder = MockOpusDecoder()
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False

        # Add frames with distinct values
        for i in range(3):
            pcm = np.full(100, float(i), dtype=np.float32)
            mock_decoder.set_pcm(pcm)
            buffer.append_opus(f"fake{i}".encode())

        all_pcm = buffer.get_all_pcm()

        assert all_pcm.size == 300
        # Verify correct ordering
        assert np.all(all_pcm[:100] == 0.0)
        assert np.all(all_pcm[100:200] == 1.0)
        assert np.all(all_pcm[200:300] == 2.0)


class TestRealtimeAudioBufferSequenceValidation:
    """Test sequence validation functionality."""

    def test_validate_sequence_no_issues(self):
        """Test validation with correct sequences."""
        mock_decoder = MockOpusDecoder(np.zeros(480, dtype=np.float32))
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False

        for _ in range(5):
            buffer.append_opus(b"fake")

        issues = buffer.validate_sequence()
        assert len(issues) == 0

    def test_validate_sequence_detects_gap(self):
        """Test that validation detects sequence gaps."""
        buffer = RealtimeAudioBuffer()

        # Manually create frames with a gap
        buffer._frames = [
            AudioFrame(np.zeros(100, dtype=np.float32), 0.0, 0),
            AudioFrame(np.zeros(100, dtype=np.float32), 0.0, 1),
            AudioFrame(np.zeros(100, dtype=np.float32), 0.0, 5),  # Gap!
        ]

        issues = buffer.validate_sequence()

        assert len(issues) == 1
        assert issues[0] == (2, 5)  # Expected 2, got 5


class TestRealtimeAudioBufferLatencyTracking:
    """Test latency tracking functionality."""

    def test_mark_stt_flush(self):
        """Test marking STT flush completion."""
        mock_decoder = MockOpusDecoder(np.zeros(960, dtype=np.float32))
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False

        buffer.append_opus(b"fake")

        buffer.mark_stt_flush()

        assert buffer.latency_metrics.stt_flush_time is not None
        assert buffer.latency_metrics.arrival_to_flush_ms is not None


class TestRealtimeAudioBufferSnapshot:
    """Test snapshot functionality."""

    def test_snapshot_captures_state(self):
        """Test that snapshot captures buffer state."""
        mock_decoder = MockOpusDecoder(np.zeros(960, dtype=np.float32))
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False

        buffer.append_opus(b"fake")

        buffer.commit("item_xyz")
        snapshot = buffer.snapshot()

        assert snapshot["state"] == "COMMITTED"
        assert snapshot["total_samples"] == 960
        assert snapshot["frame_count"] == 1
        assert snapshot["pending_item_id"] == "item_xyz"

    def test_snapshot_serializable(self):
        """Test that snapshot is JSON-serializable."""
        import json

        buffer = RealtimeAudioBuffer()
        snapshot = buffer.snapshot()

        # Should not raise
        json_str = json.dumps(snapshot)
        assert isinstance(json_str, str)


class TestRealtimeAudioBufferRepr:
    """Test string representation."""

    def test_repr_empty(self):
        """Test repr of empty buffer."""
        buffer = RealtimeAudioBuffer()
        repr_str = repr(buffer)

        assert "ACCUMULATING" in repr_str
        assert "samples=0" in repr_str
        assert "frames=0" in repr_str

    def test_repr_with_data(self):
        """Test repr with data."""
        mock_decoder = MockOpusDecoder(np.zeros(960, dtype=np.float32))
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False

        buffer.append_opus(b"fake")

        repr_str = repr(buffer)

        assert "samples=960" in repr_str
        assert "frames=1" in repr_str


@pytest.mark.asyncio
class TestRealtimeAudioBufferAsync:
    """Test async functionality."""

    async def test_append_opus_async(self):
        """Test async append functionality."""
        mock_decoder = MockOpusDecoder(np.zeros(960, dtype=np.float32))
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False

        result = await buffer.append_opus_async(b"fake")

        assert result is not None
        assert buffer.total_samples == 960

    async def test_stream_to_stt(self):
        """Test streaming to STT."""
        mock_decoder = MockOpusDecoder(np.zeros(480, dtype=np.float32))
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False

        for _ in range(3):
            buffer.append_opus(b"fake")

        # Create mock STT
        mock_stt = AsyncMock()

        frames_sent = await buffer.stream_to_stt(mock_stt)

        assert frames_sent == 3
        assert mock_stt.send_audio.call_count == 3

    async def test_stream_to_stt_with_offset(self):
        """Test streaming to STT with offset."""
        mock_decoder = MockOpusDecoder(np.zeros(480, dtype=np.float32))
        buffer = RealtimeAudioBuffer(opus_decoder=mock_decoder)
        buffer._wait_for_first_opus = False

        for _ in range(5):
            buffer.append_opus(b"fake")

        mock_stt = AsyncMock()

        # Stream from index 2 onwards
        frames_sent = await buffer.stream_to_stt(mock_stt, start_index=2)

        assert frames_sent == 3
        assert mock_stt.send_audio.call_count == 3
