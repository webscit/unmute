"""Health check service for monitoring backend service availability."""

import asyncio
import logging
from functools import partial

import requests
from pydantic import BaseModel, computed_field

from unmute.kyutai_constants import (
    KYUTAI_LLM_API_KEY,
    LLM_SERVER,
    STT_SERVER,
    TTS_SERVER,
    VOICE_CLONING_SERVER,
)
from unmute.service_discovery import async_ttl_cached

logger = logging.getLogger(__name__)


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


class HealthStatus(BaseModel):
    tts_up: bool
    stt_up: bool
    llm_up: bool
    voice_cloning_up: bool

    @computed_field
    @property
    def ok(self) -> bool:
        # Note that voice cloning is not required for the server to be healthy.
        return self.tts_up and self.stt_up and self.llm_up


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

    return HealthStatus(
        tts_up=tts_up_res,
        stt_up=stt_up_res,
        llm_up=llm_up_res,
        voice_cloning_up=voice_cloning_up_res,
    )


async def get_health() -> HealthStatus:
    """Get the current health status of all backend services."""
    return await _get_health(None)
