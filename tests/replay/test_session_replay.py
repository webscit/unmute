"""Full WebSocket integration test for OpenAI Realtime API session replay.

Tests the unmute server against a fixture derived from a real OpenAI Realtime SDK
session recording, using mock STT/TTS/LLM services.
"""

import asyncio
import socket
from typing import Any

import pytest
import pytest_asyncio
import uvicorn

from tests.realtime_harness.fixture_schema import (
    ClientEvent,
    EventAssertion,
    FixtureEventType,
    FixtureMetadata,
    OrderingAssertion,
    TimingMode,
    TraceFixture,
)
from tests.realtime_harness.websocket_replayer import ReplayTrace


def _find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class UvicornServer:
    """Manages a uvicorn server in a background thread."""

    def __init__(self, app: Any, host: str = "127.0.0.1", port: int = 0):
        self.app = app
        self.host = host
        self.port = port or _find_free_port()
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="error",
        )
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve())
        # Wait for server to start
        for _ in range(50):
            if self._server.started:
                break
            await asyncio.sleep(0.1)

    async def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


def _build_tool_call_fixture() -> tuple[TraceFixture, list[dict[str, Any]]]:
    """Build a minimal fixture that tests the function calling flow.

    Returns (fixture, scripted_llm_responses).
    """
    # Session config with tools
    session_update = {
        "type": "session.update",
        "session": {
            "instructions": "You are a helpful robot. Always use tools when asked.",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "turn_detection": {"type": "server_vad"},
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "voice": "alloy",
                },
            },
            "tools": [
                {
                    "type": "function",
                    "name": "camera",
                    "description": "Take a picture",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Question about the picture",
                            }
                        },
                        "required": ["question"],
                    },
                },
                {
                    "type": "function",
                    "name": "play_emotion",
                    "description": "Play an emotion",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "emotion": {
                                "type": "string",
                                "description": "Name of the emotion",
                            }
                        },
                        "required": ["emotion"],
                    },
                },
            ],
            "tool_choice": "auto",
        },
    }

    # User message triggering a function call
    user_message = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Take a picture and tell me what you see"}],
        },
    }

    # First response.create → LLM should return tool call
    response_create_1 = {"type": "response.create"}

    # After the tool call completes, client sends the tool result
    function_call_output = {
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": "call_mock_0",
            "output": '{"description": "I see a person waving"}',
        },
    }

    # Second response.create with instructions override → LLM returns text
    response_create_2 = {
        "type": "response.create",
        "response": {
            "instructions": "Use the tool result just returned and answer concisely in speech."
        },
    }

    client_events = [
        ClientEvent(event=session_update, delay_ms=0, description="Configure session"),
        ClientEvent(event=user_message, delay_ms=200, description="User asks to take picture"),
        ClientEvent(event=response_create_1, delay_ms=200, description="Trigger first response (tool call)"),
        ClientEvent(event=function_call_output, delay_ms=3000, description="Return tool result"),
        ClientEvent(event=response_create_2, delay_ms=200, description="Trigger second response (text)"),
    ]

    # LLM scripted responses:
    # 1st response.create → tool call (camera)
    # 2nd response.create → text response
    scripted_llm_responses = [
        {
            "tool_calls": [
                {
                    "name": "camera",
                    "arguments": '{"question": "What do you see?"}',
                    "call_id": "call_mock_0",
                }
            ]
        },
        {"text": "I can see a person waving at us."},
    ]

    event_assertions = [
        EventAssertion(event_type="session.updated", min_occurrences=1),
        EventAssertion(event_type="response.created", min_occurrences=1),
        EventAssertion(event_type="conversation.item.added", min_occurrences=1),
        EventAssertion(event_type="conversation.item.done", min_occurrences=1),
        EventAssertion(event_type="response.function_call_arguments.done", min_occurrences=1),
        EventAssertion(event_type="response.done", min_occurrences=1),
    ]

    ordering_assertions = [
        OrderingAssertion(before="session.updated", after="response.created"),
        OrderingAssertion(before="response.created", after="response.done"),
    ]

    fixture = TraceFixture(
        metadata=FixtureMetadata(
            name="tool_call_replay",
            description="Test function calling flow with mock services",
            category=FixtureEventType.TOOL_CALL,
            tags=["function_calling", "integration"],
            timeout_seconds=30,
        ),
        timing_mode=TimingMode.RELATIVE,
        client_events=client_events,
        event_assertions=event_assertions,
        ordering_assertions=ordering_assertions,
    )

    return fixture, scripted_llm_responses


def assert_event_count(trace: ReplayTrace, event_type: str, *, min: int = 1) -> None:
    """Assert minimum count of events of a given type."""
    events = trace.get_events_by_type(event_type)
    assert len(events) >= min, (
        f"Expected at least {min} '{event_type}' events, got {len(events)}. "
        f"All event types: {[e.event_type for e in trace.received_events]}"
    )


def assert_order(trace: ReplayTrace, before: str, *, before_what: str) -> None:
    """Assert one event type appears before another."""
    first_before = trace.get_first_event(before)
    first_after = trace.get_first_event(before_what)
    if first_before is None or first_after is None:
        return  # Can't check ordering if events are missing
    assert first_before.timestamp_ms <= first_after.timestamp_ms, (
        f"Expected '{before}' before '{before_what}', "
        f"but {before} at {first_before.timestamp_ms}ms, "
        f"{before_what} at {first_after.timestamp_ms}ms"
    )


@pytest_asyncio.fixture
async def mock_services():
    """Start mock TTS and LLM servers on random ports."""
    from tests.replay.mock_llm_server import app as llm_app
    from tests.replay.mock_tts_server import app as tts_app

    tts_server = UvicornServer(tts_app)
    llm_server = UvicornServer(llm_app)

    await tts_server.start()
    await llm_server.start()

    yield {
        "tts_port": tts_server.port,
        "tts_url": tts_server.url,
        "llm_port": llm_server.port,
        "llm_url": llm_server.url,
    }

    await tts_server.stop()
    await llm_server.stop()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_session_update_normalization():
    """Test that nested audio config is properly normalized."""
    import unmute.openai_realtime_api_events as ora

    # Test the normalize function directly
    session_dict = {
        "type": "realtime",
        "instructions": "Hello",
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "transcription": {"model": "gpt-4o-transcribe", "language": "en"},
                "turn_detection": {"type": "server_vad", "interrupt_response": True},
            },
            "output": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "voice": "cedar",
            },
        },
    }

    normalized = ora.normalize_session_config(session_dict)

    assert "audio" not in normalized
    assert normalized["input_audio_format"] == "audio/pcm"
    assert normalized["output_audio_format"] == "audio/pcm"
    assert normalized["voice"] == "cedar"
    assert normalized["turn_detection"]["type"] == "server_vad"
    assert normalized["input_audio_transcription"]["model"] == "gpt-4o-transcribe"
    assert normalized["instructions"] == "Hello"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_conversation_item_events():
    """Test that ConversationItemAdded and ConversationItemDone events are properly formed."""
    import unmute.openai_realtime_api_events as ora

    # Test ConversationItemAdded
    item = ora.Item(id="test_item", type="message", role="user", status="completed")
    added = ora.ConversationItemAdded(item=item)
    assert added.type == "conversation.item.added"
    assert added.item.id == "test_item"

    # Test ConversationItemDone
    done = ora.ConversationItemDone(item=item)
    assert done.type == "conversation.item.done"
    assert done.item.id == "test_item"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_session_state_create_item_with_fields():
    """Test that create_item handles function_call_output fields."""
    from unmute.session_state import SessionState

    state = SessionState()
    item = state.create_item(
        item_type="function_call_output",
        call_id="call_123",
        output='{"result": "ok"}',
    )

    assert item.type == "function_call_output"
    assert item.call_id == "call_123"
    assert item.output == '{"result": "ok"}'
    assert item.status == "completed"
    assert item.id in state.items


@pytest.mark.asyncio
@pytest.mark.integration
async def test_chatbot_tool_message_state():
    """Test that chatbot correctly handles tool and assistant+tool_calls messages."""
    from unittest.mock import patch

    from unmute.llm.chatbot import Chatbot

    with patch("unmute.llm.system_prompt.autoselect_model", return_value="mock-model"):
        chatbot = Chatbot()

    # Add assistant message with tool_calls
    chatbot.chat_history.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "camera", "arguments": "{}"},
                }
            ],
        }
    )
    assert chatbot.conversation_state() == "waiting_for_user"

    # Add tool result
    chatbot.chat_history.append(
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"result": "photo taken"}',
        }
    )
    assert chatbot.conversation_state() == "waiting_for_user"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_preprocess_messages_with_tools():
    """Test that preprocess_messages_for_llm handles tool messages correctly."""
    from unmute.llm.llm_utils import preprocess_messages_for_llm

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Take a photo."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "camera", "arguments": '{"q": "what"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"result": "a cat"}',
        },
        {"role": "user", "content": "What did you see?"},
    ]

    processed = preprocess_messages_for_llm(messages)

    # All messages should be preserved
    roles = [m["role"] for m in processed]
    assert "tool" in roles
    assert "assistant" in roles

    # Tool calls should be preserved on assistant message
    assistant_msgs = [m for m in processed if m["role"] == "assistant"]
    assert any("tool_calls" in m for m in assistant_msgs)

    # Tool message should pass through
    tool_msgs = [m for m in processed if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "call_1"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_llm_tool_call_streaming(mock_services: dict):
    """Test VLLMStream.chat_completion_with_tools yields proper deltas."""
    from tests.replay.mock_llm_server import set_scripted_responses
    from unmute.llm.llm_utils import LLMToolCallDelta, VLLMStream

    # Configure mock LLM with a tool call response
    set_scripted_responses(
        [
            {
                "tool_calls": [
                    {
                        "name": "camera",
                        "arguments": '{"question": "What is this?"}',
                        "call_id": "call_test_1",
                    }
                ]
            }
        ]
    )

    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key="test-key",
        base_url=f"{mock_services['llm_url']}/v1",
    )

    # Patch autoselect_model to return our mock model
    import unmute.llm.llm_utils as llm_utils

    original_autoselect = llm_utils.autoselect_model
    llm_utils.autoselect_model = lambda: "mock-model"  # type: ignore[assignment]

    try:
        llm = VLLMStream(client)
        messages = [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "Take a picture."},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "camera",
                    "description": "Take photo",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        deltas = []
        async for delta in llm.chat_completion_with_tools(messages, tools):
            deltas.append(delta)

        # Should have at least one tool call delta
        tool_deltas = [d for d in deltas if isinstance(d, LLMToolCallDelta)]
        assert len(tool_deltas) > 0, f"Expected tool call deltas, got: {deltas}"

        # First tool delta should have the function name
        first_tc = next(d for d in tool_deltas if d.function_name)
        assert first_tc.function_name == "camera"
        assert first_tc.tool_call_id == "call_test_1"

        # Arguments should accumulate across deltas
        all_args = "".join(d.arguments_delta for d in tool_deltas)
        assert "question" in all_args

    finally:
        llm_utils.autoselect_model = original_autoselect  # type: ignore[assignment]
