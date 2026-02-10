"""Integration test for Opus error handling in audio buffer.

This test reproduces the bug reported where invalid Opus data
(e.g., [0, 0, 0, 0]) caused a ValueError crash in the async path.
"""
# pyright: reportPrivateUsage=false

import asyncio

import numpy as np
import pytest

from unmute.audio.realtime_buffer import RealtimeAudioBuffer


class RealOpusDecoderSimulator:
    """Simulates the real sphn.OpusStreamReader behavior with invalid data."""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate

    def append_bytes(self, data: bytes) -> np.ndarray:
        """Simulate Opus decoder that throws on invalid data."""
        # Check for invalid Ogg capture pattern (common error)
        if len(data) >= 4 and data[:4] == bytes([0, 0, 0, 0]):
            raise ValueError("unexpected ogg capture pattern [0, 0, 0, 0]")

        # For valid-looking data, return some PCM
        return np.zeros(960, dtype=np.float32)


@pytest.mark.asyncio
class TestOpusErrorHandling:
    """Test that Opus decoder errors are handled gracefully."""

    async def test_invalid_opus_data_before_first_packet(self):
        """Test the exact scenario from the bug report.

        When invalid data (like all zeros) is sent before the first valid
        Opus packet, the async path should skip it without crashing.
        """
        buffer = RealtimeAudioBuffer(opus_decoder=RealOpusDecoderSimulator())

        # Send invalid data that would trigger "unexpected ogg capture pattern"
        invalid_data = bytes([0, 0, 0, 0, 0, 0, 0, 0])

        # This should NOT crash, should return None
        result = await buffer.append_opus_async(invalid_data)

        assert result is None
        assert buffer.total_samples == 0
        assert buffer.frame_count == 0

    async def test_invalid_opus_data_after_first_packet(self):
        """Test invalid data handling after stream has started."""
        buffer = RealtimeAudioBuffer(opus_decoder=RealOpusDecoderSimulator())

        # Skip the first packet check for this test
        buffer._wait_for_first_opus = False

        # Send invalid data that would trigger decoder error
        invalid_data = bytes([0, 0, 0, 0])

        # Should handle gracefully, return None
        result = await buffer.append_opus_async(invalid_data)

        assert result is None
        assert buffer.total_samples == 0

    async def test_valid_data_after_invalid_data(self):
        """Test that buffer recovers after receiving invalid data."""
        buffer = RealtimeAudioBuffer(opus_decoder=RealOpusDecoderSimulator())

        # Skip first packet check
        buffer._wait_for_first_opus = False

        # Send invalid data first
        invalid_data = bytes([0, 0, 0, 0])
        result1 = await buffer.append_opus_async(invalid_data)
        assert result1 is None

        # Send valid data
        valid_data = bytes([1, 2, 3, 4, 5, 6, 7, 8])
        result2 = await buffer.append_opus_async(valid_data)

        # Should work and add samples
        assert result2 is not None
        assert buffer.total_samples == 960

    async def test_concurrent_invalid_data(self):
        """Test handling multiple invalid packets concurrently."""
        buffer = RealtimeAudioBuffer(opus_decoder=RealOpusDecoderSimulator())
        buffer._wait_for_first_opus = False

        # Send multiple invalid packets concurrently
        invalid_data = bytes([0, 0, 0, 0])

        tasks = [
            buffer.append_opus_async(invalid_data)
            for _ in range(10)
        ]

        results = await asyncio.gather(*tasks)

        # All should return None without crashing
        assert all(r is None for r in results)
        assert buffer.total_samples == 0

    async def test_error_logged_on_invalid_data(self, caplog: pytest.LogCaptureFixture):
        """Test that invalid data errors are logged appropriately."""
        import logging

        buffer = RealtimeAudioBuffer(opus_decoder=RealOpusDecoderSimulator())
        buffer._wait_for_first_opus = False

        # Capture log messages at WARNING level
        with caplog.at_level(logging.WARNING):
            invalid_data = bytes([0, 0, 0, 0])
            result = await buffer.append_opus_async(invalid_data)

        # Should handle gracefully
        assert result is None
        assert buffer.total_samples == 0

        # Verify warning was logged
        assert any("Invalid Opus data" in record.message for record in caplog.records)
