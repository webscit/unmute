"""Unit tests for unmute/service_discovery.py."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unmute.exceptions import MissingServiceAtCapacity, MissingServiceTimeout
from unmute.service_discovery import (
    async_ttl_cached,
    find_instance,
    get_instances,
)


# ---------------------------------------------------------------------------
# async_ttl_cached
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ttl_cached_returns_value():
    call_count = 0

    async def func(key: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"result-{key}"

    cached = async_ttl_cached(func, ttl_sec=10)
    result = await cached("a")
    assert result == "result-a"
    assert call_count == 1


@pytest.mark.asyncio
async def test_ttl_cached_returns_cached_within_ttl():
    call_count = 0

    async def func(key: str) -> int:
        nonlocal call_count
        call_count += 1
        return call_count

    cached = async_ttl_cached(func, ttl_sec=10)
    r1 = await cached("x")
    r2 = await cached("x")
    assert r1 == r2 == 1
    assert call_count == 1


@pytest.mark.asyncio
async def test_ttl_cached_expires_after_ttl():
    call_count = 0

    async def func(key: str) -> int:
        nonlocal call_count
        call_count += 1
        return call_count

    cached = async_ttl_cached(func, ttl_sec=0.05)
    r1 = await cached("x")
    assert r1 == 1
    await asyncio.sleep(0.06)
    r2 = await cached("x")
    assert r2 == 2
    assert call_count == 2


@pytest.mark.asyncio
async def test_ttl_cached_separate_keys():
    call_count = 0

    async def func(key: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"v-{key}"

    cached = async_ttl_cached(func, ttl_sec=10)
    assert await cached("a") == "v-a"
    assert await cached("b") == "v-b"
    assert call_count == 2


# ---------------------------------------------------------------------------
# get_instances
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_instances_returns_urls():
    with patch(
        "unmute.service_discovery._resolve.__wrapped__",
        new_callable=AsyncMock,
        return_value=["10.0.0.1", "10.0.0.2"],
    ), patch(
        "unmute.service_discovery._resolve",
        new_callable=AsyncMock,
        return_value=["10.0.0.1", "10.0.0.2"],
    ):
        instances = await get_instances("tts")
    # All returned URLs should have the right port and protocol
    assert len(instances) == 2
    for url in instances:
        assert ":8089" in url
        assert url.startswith("ws://")


@pytest.mark.asyncio
async def test_get_instances_unknown_service():
    with pytest.raises(KeyError):
        await get_instances("unknown_service")


# ---------------------------------------------------------------------------
# find_instance — success path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_instance_success_first_try():
    mock_client = MagicMock()
    mock_client.start_up = AsyncMock()

    factory = MagicMock(return_value=mock_client)

    with patch(
        "unmute.service_discovery.get_instances",
        new_callable=AsyncMock,
        return_value=["http://10.0.0.1:8091"],
    ):
        result = await find_instance("llm", factory)

    assert result is mock_client
    factory.assert_called_once_with("http://10.0.0.1:8091")
    mock_client.start_up.assert_awaited_once()


# ---------------------------------------------------------------------------
# find_instance — failover
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_instance_failover_to_second():
    """First instance fails, second succeeds."""
    bad_client = MagicMock()
    bad_client.start_up = AsyncMock(side_effect=ConnectionError("down"))

    good_client = MagicMock()
    good_client.start_up = AsyncMock()

    def factory(url: str):
        if url == "http://10.0.0.1:8091":
            return bad_client
        return good_client

    with patch(
        "unmute.service_discovery.get_instances",
        new_callable=AsyncMock,
        return_value=["http://10.0.0.1:8091", "http://10.0.0.2:8091"],
    ):
        result = await find_instance("llm", factory, max_trials=3)

    assert result is good_client


# ---------------------------------------------------------------------------
# find_instance — all fail
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_instance_all_timeout():
    """All instances time out → MissingServiceTimeout."""
    client = MagicMock()
    client.start_up = AsyncMock(side_effect=TimeoutError)

    with patch(
        "unmute.service_discovery.get_instances",
        new_callable=AsyncMock,
        return_value=["http://10.0.0.1:8091", "http://10.0.0.2:8091"],
    ):
        with pytest.raises(MissingServiceTimeout):
            await find_instance("llm", lambda _: client, max_trials=2)


@pytest.mark.asyncio
async def test_find_instance_all_at_capacity():
    """All instances at capacity → MissingServiceAtCapacity."""
    client = MagicMock()
    client.start_up = AsyncMock(side_effect=MissingServiceAtCapacity("tts"))

    with patch(
        "unmute.service_discovery.get_instances",
        new_callable=AsyncMock,
        return_value=["ws://10.0.0.1:8089"],
    ):
        with pytest.raises(MissingServiceAtCapacity):
            await find_instance("tts", lambda _: client, max_trials=1)


@pytest.mark.asyncio
async def test_find_instance_generic_error_propagates():
    """Non-timeout, non-capacity errors propagate directly."""
    client = MagicMock()
    client.start_up = AsyncMock(side_effect=RuntimeError("something broke"))

    with patch(
        "unmute.service_discovery.get_instances",
        new_callable=AsyncMock,
        return_value=["ws://10.0.0.1:8089"],
    ):
        with pytest.raises(RuntimeError, match="something broke"):
            await find_instance("tts", lambda _: client, max_trials=1)


# ---------------------------------------------------------------------------
# find_instance — max_trials capped by instance count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_instance_max_trials_capped():
    """max_trials is min(len(instances), max_trials)."""
    call_count = 0

    def factory(url: str):
        nonlocal call_count
        call_count += 1
        client = MagicMock()
        client.start_up = AsyncMock(side_effect=ConnectionError)
        return client

    with patch(
        "unmute.service_discovery.get_instances",
        new_callable=AsyncMock,
        return_value=["http://a:1", "http://b:1"],
    ):
        with pytest.raises(ConnectionError):
            await find_instance("llm", factory, max_trials=10)

    # Only 2 instances, so only 2 attempts even though max_trials=10
    assert call_count == 2
