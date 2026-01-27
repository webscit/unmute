"""Per-session audio buffer manager for OpenAI Realtime API.

This module provides a buffer manager that:
- Stores input_audio_buffer.append frames with timestamps, sequence counters, codec metadata
- Handles Opus -> PCM decoding with overflow protection
- Exposes APIs for append, commit/flush, rewind, and STT streaming
- Tracks latency metadata (arrival -> STT flush) for transcription events
- Provides hooks for anonymized recording without storing raw audio

Frame Buffer Design:
====================
The buffer stores AudioFrame objects containing:
- PCM samples (decoded from Opus)
- Arrival timestamp (for latency tracking)
- Sequence number (for ordering and gap detection)
- Codec metadata (sample rate, channels, etc.)

State Machine:
==============
- accumulating: Frames are being appended
- committed: Buffer has been committed as a conversation item
- cleared: Buffer was cleared without committing

Latency Tracking:
=================
We track the time from frame arrival to STT flush to enable
emission of proper transcription event metadata.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum, auto
from logging import getLogger
from typing import TYPE_CHECKING, Callable, Protocol

import numpy as np
import sphn

from unmute.kyutai_constants import SAMPLE_RATE
from unmute.tracing import AUDIO_INGESTION_DURATION, trace_span

if TYPE_CHECKING:
    from unmute.stt.speech_to_text import SpeechToText


class OpusDecoder(Protocol):
    """Protocol for Opus decoders to enable testing."""

    def append_bytes(self, data: bytes) -> np.ndarray:
        """Decode Opus bytes to PCM samples."""
        ...

logger = getLogger(__name__)

# Maximum buffer size in samples before overflow protection kicks in
# At 24kHz, 5 minutes = 7,200,000 samples
MAX_BUFFER_SAMPLES = 7_200_000

# Maximum expected gap in sequence numbers before warning
MAX_SEQUENCE_GAP = 5


class BufferState(Enum):
    """State of the audio buffer."""
    ACCUMULATING = auto()
    COMMITTED = auto()
    CLEARED = auto()


@dataclass
class AudioFrame:
    """A single audio frame with metadata."""

    pcm: np.ndarray
    """PCM samples as float32 array."""

    arrival_time: float
    """Wall-clock time when frame arrived (time.monotonic())."""

    sequence: int
    """Sequence number for ordering and gap detection."""

    sample_rate: int = SAMPLE_RATE
    """Sample rate of the PCM data."""

    @property
    def num_samples(self) -> int:
        """Number of PCM samples in this frame."""
        return self.pcm.size

    @property
    def duration_sec(self) -> float:
        """Duration of this frame in seconds."""
        return self.num_samples / self.sample_rate


@dataclass
class CodecMetadata:
    """Metadata about the audio codec configuration."""

    sample_rate: int = SAMPLE_RATE
    channels: int = 1
    codec: str = "opus"

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "codec": self.codec,
        }


@dataclass
class LatencyMetrics:
    """Latency tracking for a buffer segment."""

    first_frame_arrival: float | None = None
    """Monotonic time when first frame arrived."""

    last_frame_arrival: float | None = None
    """Monotonic time when last frame arrived."""

    stt_flush_time: float | None = None
    """Monotonic time when STT flush completed."""

    commit_time: float | None = None
    """Monotonic time when buffer was committed."""

    @property
    def arrival_to_flush_ms(self) -> float | None:
        """Time from first frame arrival to STT flush in milliseconds."""
        if self.first_frame_arrival is None or self.stt_flush_time is None:
            return None
        return (self.stt_flush_time - self.first_frame_arrival) * 1000

    @property
    def buffer_duration_ms(self) -> float | None:
        """Duration of buffering (first to last frame) in milliseconds."""
        if self.first_frame_arrival is None or self.last_frame_arrival is None:
            return None
        return (self.last_frame_arrival - self.first_frame_arrival) * 1000

    def to_dict(self) -> dict:
        """Serialize metrics for event emission."""
        return {
            "arrival_to_flush_ms": self.arrival_to_flush_ms,
            "buffer_duration_ms": self.buffer_duration_ms,
        }


@dataclass
class AnonymizedFrameRecord:
    """Anonymized record of a frame for recording without raw audio."""

    sequence: int
    num_samples: int
    arrival_time: float
    sample_rate: int = SAMPLE_RATE

    def to_dict(self) -> dict:
        """Serialize for recording."""
        return {
            "sequence": self.sequence,
            "num_samples": self.num_samples,
            "arrival_time": self.arrival_time,
            "sample_rate": self.sample_rate,
        }


class RealtimeAudioBuffer:
    """Per-session audio buffer manager for the Realtime API.

    This class manages the lifecycle of audio input from WebSocket clients,
    handling Opus decoding, buffering, and integration with STT.

    Usage:
        buffer = RealtimeAudioBuffer()

        # Append Opus frames
        pcm = buffer.append_opus(opus_bytes)
        if pcm is not None:
            await stt.send_audio(pcm)

        # Commit when ready
        item_id = buffer.commit()

        # Or clear without committing
        buffer.clear()
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        max_buffer_samples: int = MAX_BUFFER_SAMPLES,
        on_frame_recorded: Callable[[AnonymizedFrameRecord], None] | None = None,
        opus_decoder: OpusDecoder | None = None,
    ):
        """Initialize the audio buffer.

        Args:
            sample_rate: Sample rate for audio processing.
            max_buffer_samples: Maximum samples before overflow protection.
            on_frame_recorded: Callback for anonymized recording hook.
            opus_decoder: Optional custom Opus decoder (for testing).
        """
        self._sample_rate = sample_rate
        self._max_buffer_samples = max_buffer_samples
        self._on_frame_recorded = on_frame_recorded

        # Opus decoder - use provided or create default
        self._opus_reader: OpusDecoder = opus_decoder or sphn.OpusStreamReader(sample_rate)
        self._wait_for_first_opus = True

        # Frame storage
        self._frames: list[AudioFrame] = []
        self._sequence_counter = 0
        self._total_samples = 0

        # State tracking
        self._state = BufferState.ACCUMULATING
        self._codec_metadata = CodecMetadata(sample_rate=sample_rate)

        # Latency tracking
        self._latency_metrics = LatencyMetrics()

        # Committed item tracking
        self._pending_item_id: str | None = None

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    @property
    def state(self) -> BufferState:
        """Current buffer state."""
        return self._state

    @property
    def total_samples(self) -> int:
        """Total samples currently in buffer."""
        return self._total_samples

    @property
    def duration_sec(self) -> float:
        """Total duration of buffered audio in seconds."""
        return self._total_samples / self._sample_rate

    @property
    def codec_metadata(self) -> CodecMetadata:
        """Codec configuration metadata."""
        return self._codec_metadata

    @property
    def latency_metrics(self) -> LatencyMetrics:
        """Current latency tracking metrics."""
        return self._latency_metrics

    @property
    def pending_item_id(self) -> str | None:
        """Item ID if buffer has been committed."""
        return self._pending_item_id

    @property
    def frame_count(self) -> int:
        """Number of frames in buffer."""
        return len(self._frames)

    def append_opus(self, opus_bytes: bytes) -> np.ndarray | None:
        """Append Opus-encoded audio to the buffer.

        Decodes Opus to PCM and stores the frame with metadata.
        Handles first-packet synchronization for stream starts.

        Args:
            opus_bytes: Raw Opus packet bytes.

        Returns:
            PCM samples as float32 array, or None if packet was skipped/empty.
            Returns None for:
            - Packets before first valid Opus packet (sync)
            - Empty decode results
            - Overflow protection triggered
        """
        if self._state != BufferState.ACCUMULATING:
            logger.warning(
                f"Attempted to append to buffer in state {self._state.name}"
            )
            return None

        # Wait for first valid Opus packet
        if self._wait_for_first_opus:
            # Check for the first-packet bit in Opus header
            # Byte 5, bit 1 indicates start of stream
            if len(opus_bytes) > 5 and opus_bytes[5] & 2:
                self._wait_for_first_opus = False
            else:
                return None

        # Overflow protection
        if self._total_samples >= self._max_buffer_samples:
            logger.warning(
                f"Buffer overflow protection: {self._total_samples} samples, "
                f"max {self._max_buffer_samples}. Discarding frame."
            )
            return None

        # Decode Opus to PCM
        pcm = self._opus_reader.append_bytes(opus_bytes)

        if pcm.size == 0:
            return None

        # Create frame with metadata
        arrival_time = time.monotonic()
        frame = AudioFrame(
            pcm=pcm,
            arrival_time=arrival_time,
            sequence=self._sequence_counter,
            sample_rate=self._sample_rate,
        )

        # Update latency tracking
        if self._latency_metrics.first_frame_arrival is None:
            self._latency_metrics.first_frame_arrival = arrival_time
        self._latency_metrics.last_frame_arrival = arrival_time

        # Store frame
        self._frames.append(frame)
        self._total_samples += frame.num_samples
        self._sequence_counter += 1

        # Anonymized recording hook
        if self._on_frame_recorded is not None:
            record = AnonymizedFrameRecord(
                sequence=frame.sequence,
                num_samples=frame.num_samples,
                arrival_time=arrival_time,
                sample_rate=self._sample_rate,
            )
            self._on_frame_recorded(record)

        return pcm

    async def append_opus_async(self, opus_bytes: bytes) -> np.ndarray | None:
        """Thread-safe async version of append_opus.

        Runs Opus decoding in a thread pool for better async performance.
        """
        # Run decode in thread to avoid blocking event loop with instrumentation
        async with trace_span(
            "opus_decode",
            histogram=AUDIO_INGESTION_DURATION,
            attributes={"opus_bytes": len(opus_bytes)},
        ):
            pcm = await asyncio.to_thread(self._opus_reader.append_bytes, opus_bytes)

        if self._state != BufferState.ACCUMULATING:
            logger.warning(
                f"Attempted to append to buffer in state {self._state.name}"
            )
            return None

        # Wait for first valid Opus packet
        if self._wait_for_first_opus:
            if len(opus_bytes) > 5 and opus_bytes[5] & 2:
                self._wait_for_first_opus = False
            else:
                return None

        if self._total_samples >= self._max_buffer_samples:
            logger.warning(
                f"Buffer overflow protection: {self._total_samples} samples"
            )
            return None

        if pcm.size == 0:
            return None

        async with self._lock:
            arrival_time = time.monotonic()
            frame = AudioFrame(
                pcm=pcm,
                arrival_time=arrival_time,
                sequence=self._sequence_counter,
                sample_rate=self._sample_rate,
            )

            if self._latency_metrics.first_frame_arrival is None:
                self._latency_metrics.first_frame_arrival = arrival_time
            self._latency_metrics.last_frame_arrival = arrival_time

            self._frames.append(frame)
            self._total_samples += frame.num_samples
            self._sequence_counter += 1

            if self._on_frame_recorded is not None:
                record = AnonymizedFrameRecord(
                    sequence=frame.sequence,
                    num_samples=frame.num_samples,
                    arrival_time=arrival_time,
                    sample_rate=self._sample_rate,
                )
                self._on_frame_recorded(record)

        return pcm

    def commit(self, item_id: str) -> tuple[int, LatencyMetrics]:
        """Commit the buffer as a conversation item.

        Marks the buffer as committed and records latency metrics.

        Args:
            item_id: The conversation item ID for this committed audio.

        Returns:
            Tuple of (total_samples, latency_metrics).
        """
        self._latency_metrics.commit_time = time.monotonic()
        self._state = BufferState.COMMITTED
        self._pending_item_id = item_id

        total_samples = self._total_samples
        metrics = self._latency_metrics

        logger.info(
            f"Buffer committed: {total_samples} samples, "
            f"duration={self.duration_sec:.2f}s, item_id={item_id}"
        )

        return total_samples, metrics

    def mark_stt_flush(self) -> None:
        """Mark that STT flush has completed.

        Called when STT processing of buffered audio is complete,
        used for latency tracking.
        """
        self._latency_metrics.stt_flush_time = time.monotonic()

        if self._latency_metrics.arrival_to_flush_ms is not None:
            logger.debug(
                f"STT flush latency: {self._latency_metrics.arrival_to_flush_ms:.1f}ms"
            )

    def clear(self) -> int:
        """Clear the buffer without committing.

        Returns:
            Number of samples that were cleared.
        """
        cleared_samples = self._total_samples

        self._frames.clear()
        self._total_samples = 0
        self._state = BufferState.CLEARED
        self._latency_metrics = LatencyMetrics()
        self._pending_item_id = None

        logger.info(f"Buffer cleared: {cleared_samples} samples discarded")

        return cleared_samples

    def reset(self) -> None:
        """Reset buffer to initial accumulating state.

        Called after commit or clear to prepare for new audio.
        """
        self._frames.clear()
        self._total_samples = 0
        self._sequence_counter = 0
        self._state = BufferState.ACCUMULATING
        self._latency_metrics = LatencyMetrics()
        self._pending_item_id = None
        # Note: We don't reset _wait_for_first_opus here as the stream continues

    def get_all_pcm(self) -> np.ndarray:
        """Get all buffered PCM samples concatenated.

        Useful for full buffer retrieval (e.g., for rewind scenarios).

        Returns:
            Concatenated PCM samples as float32 array.
        """
        if not self._frames:
            return np.array([], dtype=np.float32)

        return np.concatenate([f.pcm for f in self._frames])

    def rewind(self, num_samples: int) -> np.ndarray | None:
        """Rewind buffer by removing recent samples.

        Useful for replay or adjustment scenarios.

        Args:
            num_samples: Number of samples to rewind.

        Returns:
            The removed samples, or None if insufficient data.
        """
        if num_samples <= 0:
            return None

        if num_samples > self._total_samples:
            logger.warning(
                f"Rewind requested {num_samples} samples but only "
                f"{self._total_samples} available"
            )
            return None

        removed_samples = []
        samples_to_remove = num_samples

        while samples_to_remove > 0 and self._frames:
            last_frame = self._frames[-1]

            if last_frame.num_samples <= samples_to_remove:
                # Remove entire frame
                self._frames.pop()
                self._total_samples -= last_frame.num_samples
                samples_to_remove -= last_frame.num_samples
                removed_samples.insert(0, last_frame.pcm)
            else:
                # Partial frame removal (split)
                split_point = last_frame.num_samples - samples_to_remove
                removed_samples.insert(0, last_frame.pcm[split_point:])

                # Update frame in place
                last_frame.pcm = last_frame.pcm[:split_point]
                self._total_samples -= samples_to_remove
                samples_to_remove = 0

        if removed_samples:
            return np.concatenate(removed_samples)
        return None

    def validate_sequence(self) -> list[tuple[int, int]]:
        """Validate frame sequence for gaps or reordering.

        Returns:
            List of (expected, actual) tuples for any sequence issues found.
        """
        issues = []
        expected_seq = 0

        for frame in self._frames:
            if frame.sequence != expected_seq:
                issues.append((expected_seq, frame.sequence))
                if frame.sequence - expected_seq > MAX_SEQUENCE_GAP:
                    logger.warning(
                        f"Large sequence gap: expected {expected_seq}, "
                        f"got {frame.sequence}"
                    )
            expected_seq = frame.sequence + 1

        return issues

    async def stream_to_stt(
        self,
        stt: SpeechToText,
        start_index: int = 0,
    ) -> int:
        """Stream buffered frames to STT processor.

        Args:
            stt: The STT instance to send audio to.
            start_index: Frame index to start streaming from.

        Returns:
            Number of frames streamed.
        """
        frames_sent = 0

        for frame in self._frames[start_index:]:
            await stt.send_audio(frame.pcm)
            frames_sent += 1

        return frames_sent

    def snapshot(self) -> dict:
        """Create a serializable snapshot of buffer state.

        For session recording and replay support.
        """
        return {
            "state": self._state.name,
            "total_samples": self._total_samples,
            "frame_count": len(self._frames),
            "sequence_counter": self._sequence_counter,
            "pending_item_id": self._pending_item_id,
            "codec_metadata": self._codec_metadata.to_dict(),
            "latency_metrics": self._latency_metrics.to_dict(),
            "duration_sec": self.duration_sec,
        }

    def __repr__(self) -> str:
        return (
            f"RealtimeAudioBuffer("
            f"state={self._state.name}, "
            f"samples={self._total_samples}, "
            f"frames={len(self._frames)}, "
            f"duration={self.duration_sec:.2f}s)"
        )
