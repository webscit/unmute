"""Tests for health service and realtime capacity probes.

Tests health checking, capacity metrics gathering, and session admission gating.
"""
# pyright: reportPrivateUsage=false

import pytest
from unittest.mock import Mock, patch, AsyncMock

from unmute.services.health_service import (
    HealthStatus,
    RealtimeCapacity,
    _get_realtime_capacity,
    check_session_admission,
    get_health,
    MAX_OUTPUT_QUEUE_SIZE,
    MAX_EMIT_QUEUE_SIZE,
    MAX_ACTIVE_SESSIONS,
)


class TestRealtimeCapacity:
    """Test RealtimeCapacity model and capacity checking."""

    def test_has_capacity_when_all_metrics_ok(self):
        """Test that has_capacity returns True when all metrics are below thresholds."""
        capacity = RealtimeCapacity(
            output_queue_size=10,
            emit_queue_size=10,
            active_sessions=5,
            stt_active_sessions=3,
            vllm_active_sessions=2,
            tts_active_sessions=3,
        )
        assert capacity.has_capacity is True

    def test_no_capacity_when_output_queue_full(self):
        """Test that has_capacity returns False when output queue is at threshold."""
        capacity = RealtimeCapacity(
            output_queue_size=MAX_OUTPUT_QUEUE_SIZE,
            emit_queue_size=10,
            active_sessions=5,
            stt_active_sessions=3,
            vllm_active_sessions=2,
            tts_active_sessions=3,
        )
        assert capacity.has_capacity is False

    def test_no_capacity_when_emit_queue_full(self):
        """Test that has_capacity returns False when emit queue is at threshold."""
        capacity = RealtimeCapacity(
            output_queue_size=10,
            emit_queue_size=MAX_EMIT_QUEUE_SIZE,
            active_sessions=5,
            stt_active_sessions=3,
            vllm_active_sessions=2,
            tts_active_sessions=3,
        )
        assert capacity.has_capacity is False

    def test_no_capacity_when_sessions_at_max(self):
        """Test that has_capacity returns False when active sessions at max."""
        capacity = RealtimeCapacity(
            output_queue_size=10,
            emit_queue_size=10,
            active_sessions=MAX_ACTIVE_SESSIONS,
            stt_active_sessions=5,
            vllm_active_sessions=5,
            tts_active_sessions=5,
        )
        assert capacity.has_capacity is False


class TestGetRealtimeCapacity:
    """Test gathering realtime capacity metrics from Prometheus gauges."""

    @patch("unmute.services.health_service.mt.OUTPUT_QUEUE_SIZE")
    @patch("unmute.services.health_service.mt.EMIT_QUEUE_SIZE")
    @patch("unmute.services.health_service.mt.ACTIVE_SESSIONS")
    @patch("unmute.services.health_service.mt.STT_ACTIVE_SESSIONS")
    @patch("unmute.services.health_service.mt.VLLM_ACTIVE_SESSIONS")
    @patch("unmute.services.health_service.mt.TTS_ACTIVE_SESSIONS")
    def test_get_realtime_capacity(
        self,
        mock_tts_active,
        mock_vllm_active,
        mock_stt_active,
        mock_active,
        mock_emit_queue,
        mock_output_queue,
    ):
        """Test that _get_realtime_capacity reads from Prometheus gauges correctly."""
        # Mock the gauge values
        mock_output_queue._value.get.return_value = 25
        mock_emit_queue._value.get.return_value = 15
        mock_active._value.get.return_value = 7
        mock_stt_active._value.get.return_value = 4
        mock_vllm_active._value.get.return_value = 3
        mock_tts_active._value.get.return_value = 5

        capacity = _get_realtime_capacity()

        assert capacity.output_queue_size == 25
        assert capacity.emit_queue_size == 15
        assert capacity.active_sessions == 7
        assert capacity.stt_active_sessions == 4
        assert capacity.vllm_active_sessions == 3
        assert capacity.tts_active_sessions == 5


class TestHealthStatus:
    """Test HealthStatus model with capacity integration."""

    def test_ok_when_all_services_up_and_has_capacity(self):
        """Test that ok returns True when all services up and has capacity."""
        capacity = RealtimeCapacity(
            output_queue_size=10,
            emit_queue_size=10,
            active_sessions=5,
            stt_active_sessions=3,
            vllm_active_sessions=2,
            tts_active_sessions=3,
        )
        health = HealthStatus(
            tts_up=True,
            stt_up=True,
            llm_up=True,
            voice_cloning_up=True,
            capacity=capacity,
        )
        assert health.ok is True

    def test_not_ok_when_service_down(self):
        """Test that ok returns False when a service is down."""
        capacity = RealtimeCapacity(
            output_queue_size=10,
            emit_queue_size=10,
            active_sessions=5,
            stt_active_sessions=3,
            vllm_active_sessions=2,
            tts_active_sessions=3,
        )
        health = HealthStatus(
            tts_up=False,
            stt_up=True,
            llm_up=True,
            voice_cloning_up=True,
            capacity=capacity,
        )
        assert health.ok is False

    def test_not_ok_when_no_capacity(self):
        """Test that ok returns False when system has no capacity."""
        capacity = RealtimeCapacity(
            output_queue_size=MAX_OUTPUT_QUEUE_SIZE,
            emit_queue_size=10,
            active_sessions=5,
            stt_active_sessions=3,
            vllm_active_sessions=2,
            tts_active_sessions=3,
        )
        health = HealthStatus(
            tts_up=True,
            stt_up=True,
            llm_up=True,
            voice_cloning_up=True,
            capacity=capacity,
        )
        assert health.ok is False

    def test_ok_when_capacity_none(self):
        """Test that ok returns True based on services when capacity is None."""
        health = HealthStatus(
            tts_up=True,
            stt_up=True,
            llm_up=True,
            voice_cloning_up=True,
            capacity=None,
        )
        assert health.ok is True


class TestCheckSessionAdmission:
    """Test session admission gating based on health probes."""

    @pytest.mark.asyncio
    async def test_admission_allowed_when_healthy(self):
        """Test that admission is allowed when system is healthy."""
        mock_health = HealthStatus(
            tts_up=True,
            stt_up=True,
            llm_up=True,
            voice_cloning_up=True,
            capacity=RealtimeCapacity(
                output_queue_size=10,
                emit_queue_size=10,
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
    async def test_admission_denied_when_tts_down(self):
        """Test that admission is denied when TTS service is down."""
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
                tts_active_sessions=3,
            ),
        )

        with patch("unmute.services.health_service.get_health", return_value=mock_health):
            can_admit, reason = await check_session_admission()

        assert can_admit is False
        assert reason == "TTS service unavailable"

    @pytest.mark.asyncio
    async def test_admission_denied_when_stt_down(self):
        """Test that admission is denied when STT service is down."""
        mock_health = HealthStatus(
            tts_up=True,
            stt_up=False,
            llm_up=True,
            voice_cloning_up=True,
            capacity=RealtimeCapacity(
                output_queue_size=10,
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
        assert reason == "STT service unavailable"

    @pytest.mark.asyncio
    async def test_admission_denied_when_llm_down(self):
        """Test that admission is denied when LLM service is down."""
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
                vllm_active_sessions=2,
                tts_active_sessions=3,
            ),
        )

        with patch("unmute.services.health_service.get_health", return_value=mock_health):
            can_admit, reason = await check_session_admission()

        assert can_admit is False
        assert reason == "LLM service unavailable"

    @pytest.mark.asyncio
    async def test_admission_denied_when_output_queue_full(self):
        """Test that admission is denied when output queue is full."""
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
        assert f"Output queue size ({MAX_OUTPUT_QUEUE_SIZE})" in reason

    @pytest.mark.asyncio
    async def test_admission_denied_when_emit_queue_full(self):
        """Test that admission is denied when emit queue is full."""
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
        assert f"Emit queue size ({MAX_EMIT_QUEUE_SIZE})" in reason

    @pytest.mark.asyncio
    async def test_admission_denied_when_sessions_at_max(self):
        """Test that admission is denied when active sessions at max."""
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
        assert f"Active sessions ({MAX_ACTIVE_SESSIONS})" in reason

    @pytest.mark.asyncio
    async def test_admission_allowed_when_capacity_none(self):
        """Test that admission is allowed when capacity metrics unavailable."""
        mock_health = HealthStatus(
            tts_up=True,
            stt_up=True,
            llm_up=True,
            voice_cloning_up=True,
            capacity=None,
        )

        with patch("unmute.services.health_service.get_health", return_value=mock_health):
            can_admit, reason = await check_session_admission()

        assert can_admit is True
        assert reason == ""
