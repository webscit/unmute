"""Health check service for monitoring backend service availability and realtime capacity."""

import asyncio
import logging
from functools import partial

import requests
from pydantic import BaseModel, computed_field

from unmute import metrics as mt
from unmute.kyutai_constants import (
    KYUTAI_LLM_API_KEY,
    LLM_SERVER,
    STT_SERVER,
    TTS_SERVER,
    VOICE_CLONING_SERVER,
)
from unmute.service_discovery import async_ttl_cached

logger = logging.getLogger(__name__)

# Health thresholds for capacity management
MAX_OUTPUT_QUEUE_SIZE = 75  # Between warning (50) and error (100) thresholds
MAX_EMIT_QUEUE_SIZE = 50
MAX_ACTIVE_SESSIONS = 10  # Adjust based on deployment capacity


def _ws_to_http(ws_url: str) -> str:
    """Convert a WebSocket URL to an HTTP URL."""
    return ws_url.replace("ws://", "http://").replace("wss://", "https://")


def _check_server_status(server_url: str, headers: dict | None = None) -> bool:
    """Check if the server is up by sending a GET request."""
    try:
        response = requests.get(
            server_url,
            timeout=2,
            headers=headers or {},
        )
        logger.info(f"Response from {server_url}: {response}")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        logger.info(f"Couldn't connect to {server_url}: {e}")
        return False


class RealtimeCapacity(BaseModel):
    """Realtime pipeline capacity metrics."""

    output_queue_size: int
    emit_queue_size: int
    active_sessions: int
    stt_active_sessions: int
    vllm_active_sessions: int
    tts_active_sessions: int

    @computed_field
    @property
    def has_capacity(self) -> bool:
        """Check if system has capacity for new sessions."""
        return (
            self.output_queue_size < MAX_OUTPUT_QUEUE_SIZE
            and self.emit_queue_size < MAX_EMIT_QUEUE_SIZE
            and self.active_sessions < MAX_ACTIVE_SESSIONS
        )


class HealthStatus(BaseModel):
    tts_up: bool
    stt_up: bool
    llm_up: bool
    voice_cloning_up: bool
    capacity: RealtimeCapacity | None = None

    @computed_field
    @property
    def ok(self) -> bool:
        # Note that voice cloning is not required for the server to be healthy.
        services_ok = self.tts_up and self.stt_up and self.llm_up
        # If capacity info is available, also check capacity
        if self.capacity is not None:
            return services_ok and self.capacity.has_capacity
        return services_ok


def _get_realtime_capacity() -> RealtimeCapacity:
    """Gather current realtime capacity metrics from Prometheus gauges."""
    return RealtimeCapacity(
        output_queue_size=int(mt.OUTPUT_QUEUE_SIZE._value.get()),  # type: ignore[attr-defined]
        emit_queue_size=int(mt.EMIT_QUEUE_SIZE._value.get()),  # type: ignore[attr-defined]
        active_sessions=int(mt.ACTIVE_SESSIONS._value.get()),  # type: ignore[attr-defined]
        stt_active_sessions=int(mt.STT_ACTIVE_SESSIONS._value.get()),  # type: ignore[attr-defined]
        vllm_active_sessions=int(mt.VLLM_ACTIVE_SESSIONS._value.get()),  # type: ignore[attr-defined]
        tts_active_sessions=int(mt.TTS_ACTIVE_SESSIONS._value.get()),  # type: ignore[attr-defined]
    )


@partial(async_ttl_cached, ttl_sec=0.5)
async def _get_health(
    _none: None,
):  # dummy param _none because caching function expects a single param as cache key.
    async with asyncio.TaskGroup() as tg:
        tts_up = tg.create_task(
            asyncio.to_thread(
                _check_server_status, _ws_to_http(TTS_SERVER) + "/api/build_info"
            )
        )
        stt_up = tg.create_task(
            asyncio.to_thread(
                _check_server_status, _ws_to_http(STT_SERVER) + "/api/build_info"
            )
        )
        llm_up = tg.create_task(
            asyncio.to_thread(
                _check_server_status,
                _ws_to_http(LLM_SERVER) + "/v1/models",
                # The default vLLM server doesn't use auth, but this is needed if you
                # use OpenAI or another LLM server.
                headers={"Authorization": f"Bearer {KYUTAI_LLM_API_KEY}"},
            )
        )
        voice_cloning_up = tg.create_task(
            asyncio.to_thread(
                _check_server_status,
                _ws_to_http(VOICE_CLONING_SERVER) + "/api/build_info",
            )
        )
        tts_up_res = await tts_up
        stt_up_res = await stt_up
        llm_up_res = await llm_up
        voice_cloning_up_res = await voice_cloning_up

    # Gather realtime capacity metrics
    capacity = _get_realtime_capacity()

    return HealthStatus(
        tts_up=tts_up_res,
        stt_up=stt_up_res,
        llm_up=llm_up_res,
        voice_cloning_up=voice_cloning_up_res,
        capacity=capacity,
    )


async def get_health() -> HealthStatus:
    """Get the current health status of all backend services and realtime capacity."""
    return await _get_health(None)


async def check_session_admission() -> tuple[bool, str]:
    """
    Check if the system can admit a new session based on health probes.

    Returns:
        tuple[bool, str]: (can_admit, reason)
            - can_admit: True if system can accept new session
            - reason: Empty string if can admit, otherwise reason for rejection
    """
    health = await get_health()

    # Check external service availability
    if not health.tts_up:
        return False, "TTS service unavailable"
    if not health.stt_up:
        return False, "STT service unavailable"
    if not health.llm_up:
        return False, "LLM service unavailable"

    # Check realtime capacity
    if health.capacity is None:
        logger.warning("Capacity metrics unavailable, allowing admission")
        return True, ""

    capacity = health.capacity

    if capacity.output_queue_size >= MAX_OUTPUT_QUEUE_SIZE:
        return (
            False,
            f"Output queue size ({capacity.output_queue_size}) exceeds threshold ({MAX_OUTPUT_QUEUE_SIZE})",
        )

    if capacity.emit_queue_size >= MAX_EMIT_QUEUE_SIZE:
        return (
            False,
            f"Emit queue size ({capacity.emit_queue_size}) exceeds threshold ({MAX_EMIT_QUEUE_SIZE})",
        )

    if capacity.active_sessions >= MAX_ACTIVE_SESSIONS:
        return (
            False,
            f"Active sessions ({capacity.active_sessions}) at capacity ({MAX_ACTIVE_SESSIONS})",
        )

    return True, ""
