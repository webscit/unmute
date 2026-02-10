"""Integration tests for health probes with simulated degraded services.

Tests that health probes correctly detect degraded services and block session admission.
"""

import pytest
from unittest.mock import patch

from unmute.services.health_service import (
    HealthStatus,
    RealtimeCapacity,
    check_session_admission,
    MAX_OUTPUT_QUEUE_SIZE,
    MAX_EMIT_QUEUE_SIZE,
    MAX_ACTIVE_SESSIONS,
)


class TestDegradedServiceDetection:
    """Test health probe behavior when services are degraded."""

    @pytest.mark.asyncio
    async def test_degraded_stt_service_blocks_admission(self):
        """Test that degraded STT service blocks new session admission."""
        # Mock health status with STT down
        mock_health = HealthStatus(
            tts_up=True,
            stt_up=False,
            llm_up=True,
            voice_cloning_up=True,
            capacity=RealtimeCapacity(
                output_queue_size=10,
                emit_queue_size=10,
                active_sessions=5,
                stt_active_sessions=0,
                vllm_active_sessions=2,
                tts_active_sessions=3,
            ),
        )

        with patch("unmute.services.health_service.get_health", return_value=mock_health):
            can_admit, reason = await check_session_admission()

        assert can_admit is False
        assert "STT service unavailable" in reason

    @pytest.mark.asyncio
    async def test_degraded_llm_service_blocks_admission(self):
        """Test that degraded LLM service blocks new session admission."""
        mock_health = HealthStatus(
            tts_up=True,
            stt_up=True,
            llm_up=False,
            voice_cloning_up=True,
            capacity=RealtimeCapacity(
                output_queue_size=10,
                emit_queue_size=10,
                active_sessions=5,
                stt_active_sessions=3,
                vllm_active_sessions=0,
                tts_active_sessions=3,
            ),
        )

        with patch("unmute.services.health_service.get_health", return_value=mock_health):
            can_admit, reason = await check_session_admission()

        assert can_admit is False
        assert "LLM service unavailable" in reason

    @pytest.mark.asyncio
    async def test_degraded_tts_service_blocks_admission(self):
        """Test that degraded TTS service blocks new session admission."""
        mock_health = HealthStatus(
            tts_up=False,
            stt_up=True,
            llm_up=True,
            voice_cloning_up=True,
            capacity=RealtimeCapacity(
                output_queue_size=10,
                emit_queue_size=10,
                active_sessions=5,
                stt_active_sessions=3,
                vllm_active_sessions=2,
                tts_active_sessions=0,
            ),
        )

        with patch("unmute.services.health_service.get_health", return_value=mock_health):
            can_admit, reason = await check_session_admission()

        assert can_admit is False
        assert "TTS service unavailable" in reason


class TestBackpressureDetection:
    """Test health probe behavior under backpressure conditions."""

    @pytest.mark.asyncio
    async def test_high_output_queue_blocks_admission(self):
        """Test that high output queue depth blocks new session admission."""
        mock_health = HealthStatus(
            tts_up=True,
            stt_up=True,
            llm_up=True,
            voice_cloning_up=True,
            capacity=RealtimeCapacity(
                output_queue_size=MAX_OUTPUT_QUEUE_SIZE,
                emit_queue_size=10,
                active_sessions=5,
                stt_active_sessions=3,
                vllm_active_sessions=2,
                tts_active_sessions=3,
            ),
        )

        with patch("unmute.services.health_service.get_health", return_value=mock_health):
            can_admit, reason = await check_session_admission()

        assert can_admit is False
        assert "Output queue size" in reason
        assert str(MAX_OUTPUT_QUEUE_SIZE) in reason

    @pytest.mark.asyncio
    async def test_high_emit_queue_blocks_admission(self):
        """Test that high emit queue depth blocks new session admission."""
        mock_health = HealthStatus(
            tts_up=True,
            stt_up=True,
            llm_up=True,
            voice_cloning_up=True,
            capacity=RealtimeCapacity(
                output_queue_size=10,
                emit_queue_size=MAX_EMIT_QUEUE_SIZE,
                active_sessions=5,
                stt_active_sessions=3,
                vllm_active_sessions=2,
                tts_active_sessions=3,
            ),
        )

        with patch("unmute.services.health_service.get_health", return_value=mock_health):
            can_admit, reason = await check_session_admission()

        assert can_admit is False
        assert "Emit queue size" in reason
        assert str(MAX_EMIT_QUEUE_SIZE) in reason

    @pytest.mark.asyncio
    async def test_max_sessions_blocks_admission(self):
        """Test that reaching max sessions blocks new session admission."""
        mock_health = HealthStatus(
            tts_up=True,
            stt_up=True,
            llm_up=True,
            voice_cloning_up=True,
            capacity=RealtimeCapacity(
                output_queue_size=10,
                emit_queue_size=10,
                active_sessions=MAX_ACTIVE_SESSIONS,
                stt_active_sessions=5,
                vllm_active_sessions=5,
                tts_active_sessions=5,
            ),
        )

        with patch("unmute.services.health_service.get_health", return_value=mock_health):
            can_admit, reason = await check_session_admission()

        assert can_admit is False
        assert "Active sessions" in reason
        assert str(MAX_ACTIVE_SESSIONS) in reason


class TestHealthRecovery:
    """Test health probe behavior when services recover."""

    @pytest.mark.asyncio
    async def test_service_recovery_allows_admission(self):
        """Test that recovered services allow new session admission."""
        mock_health = HealthStatus(
            tts_up=True,
            stt_up=True,
            llm_up=True,
            voice_cloning_up=True,
            capacity=RealtimeCapacity(
                output_queue_size=20,
                emit_queue_size=15,
                active_sessions=5,
                stt_active_sessions=3,
                vllm_active_sessions=2,
                tts_active_sessions=3,
            ),
        )

        with patch("unmute.services.health_service.get_health", return_value=mock_health):
            can_admit, reason = await check_session_admission()

        assert can_admit is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_queue_draining_allows_admission(self):
        """Test that queue draining below threshold allows admission."""
        mock_health = HealthStatus(
            tts_up=True,
            stt_up=True,
            llm_up=True,
            voice_cloning_up=True,
            capacity=RealtimeCapacity(
                output_queue_size=MAX_OUTPUT_QUEUE_SIZE - 1,
                emit_queue_size=MAX_EMIT_QUEUE_SIZE - 1,
                active_sessions=MAX_ACTIVE_SESSIONS - 1,
                stt_active_sessions=5,
                vllm_active_sessions=5,
                tts_active_sessions=5,
            ),
        )

        with patch("unmute.services.health_service.get_health", return_value=mock_health):
            can_admit, reason = await check_session_admission()

        assert can_admit is True
        assert reason == ""
