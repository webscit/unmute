"""Mock LLM server implementing OpenAI-compatible chat completions API.

Provides scripted responses based on message content, including tool/function
call responses for testing the function calling flow.
"""

import asyncio
import json
import time
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()

# Scripted responses: list of dicts with either "text" or "tool_calls" key.
# Responses are consumed in order; if exhausted, returns a default text.
scripted_responses: list[dict[str, Any]] = []


class ScriptedResponse(BaseModel):
    """A single scripted response for the mock LLM."""

    text: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


def set_scripted_responses(responses: list[dict[str, Any]]) -> None:
    """Set the scripted responses the mock will return."""
    global scripted_responses
    scripted_responses = list(responses)


def _pop_response() -> dict[str, Any]:
    """Pop the next scripted response, or return a default."""
    if scripted_responses:
        return scripted_responses.pop(0)
    return {"text": "I'm a mock LLM response."}


def _make_text_chunks(text: str) -> list[dict[str, Any]]:
    """Generate SSE chunks for a text response."""
    response_id = f"chatcmpl-mock-{int(time.time())}"
    chunks = []

    # First chunk: role
    chunks.append(
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        }
    )

    # Content chunks: word by word
    words = text.split()
    for i, word in enumerate(words):
        prefix = " " if i > 0 else ""
        chunks.append(
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "mock-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": prefix + word},
                        "finish_reason": None,
                    }
                ],
            }
        )

    # Final chunk: finish
    chunks.append(
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
    )

    return chunks


def _make_tool_call_chunks(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate SSE chunks for a tool call response."""
    response_id = f"chatcmpl-mock-{int(time.time())}"
    chunks = []

    # First chunk: role
    chunks.append(
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": None},
                    "finish_reason": None,
                }
            ],
        }
    )

    for tc_index, tc in enumerate(tool_calls):
        func_name = tc.get("name", "unknown")
        arguments = tc.get("arguments", "{}")
        call_id = tc.get("call_id", f"call_mock_{tc_index}")

        # First delta for this tool call: includes id, type, and function name
        chunks.append(
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "mock-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": tc_index,
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": func_name,
                                        "arguments": "",
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
        )

        # Stream arguments in small chunks
        arg_str = arguments if isinstance(arguments, str) else json.dumps(arguments)
        chunk_size = 10
        for i in range(0, len(arg_str), chunk_size):
            arg_chunk = arg_str[i : i + chunk_size]
            chunks.append(
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "mock-model",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": tc_index,
                                        "function": {"arguments": arg_chunk},
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            )

    # Final chunk
    chunks.append(
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls",
                }
            ],
        }
    )

    return chunks


@app.post("/v1/chat/completions")
async def chat_completions(request: dict[str, Any]) -> StreamingResponse:
    """OpenAI-compatible streaming chat completions endpoint."""
    is_streaming = request.get("stream", False)
    scripted = _pop_response()

    if not is_streaming:
        # Non-streaming response
        text = scripted.get("text", "Mock response.")
        return StreamingResponse(
            content=json.dumps(
                {
                    "id": f"chatcmpl-mock-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "mock-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ),
            media_type="application/json",
        )

    # Streaming response
    if "tool_calls" in scripted and scripted["tool_calls"]:
        chunks = _make_tool_call_chunks(scripted["tool_calls"])
    else:
        text = scripted.get("text", "Mock response.")
        chunks = _make_text_chunks(text)

    async def generate():
        for chunk in chunks:
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.01)  # Small delay for realism
        yield "data: [DONE]\n\n"

    return StreamingResponse(content=generate(), media_type="text/event-stream")


@app.get("/v1/models")
async def list_models():
    """Return a single mock model for autoselect_model()."""
    return {
        "object": "list",
        "data": [
            {
                "id": "mock-model",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mock",
            }
        ],
    }
