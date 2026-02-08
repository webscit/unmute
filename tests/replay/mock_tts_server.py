"""Mock TTS WebSocket server for integration testing.

Based on unmute/loadtest/dummy_tts_server.py.
Accepts text messages and returns minimal sine wave audio.
"""

import asyncio
import logging

import msgpack
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 480

app = FastAPI()
logger = logging.getLogger(__name__)


def generate_sine_chunk(
    num_samples: int = SAMPLES_PER_FRAME, frequency: float = 440.0
) -> list[float]:
    """Generate a single chunk of sine wave audio."""
    t = np.linspace(0, num_samples / SAMPLE_RATE, num_samples, endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * frequency * t)
    return audio.tolist()


@app.get("/api/build_info")
def get_build_info():
    return {"note": "mock TTS for testing"}


@app.websocket("/api/tts_streaming")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        current_time = 0.0

        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=2.0)
            except asyncio.TimeoutError:
                await asyncio.sleep(0.01)
                continue

            if "text" in message:
                text = message["text"]
            elif "bytes" in message:
                if message["bytes"] == b"\0":
                    break
                else:
                    continue
            else:
                continue

            if not text.strip():
                continue

            words = text.strip().split()
            chunk_duration = SAMPLES_PER_FRAME / SAMPLE_RATE

            for word in words:
                word_duration = chunk_duration * max(len(word), 1)
                start_time = current_time
                stop_time = current_time + word_duration

                # Send text message
                text_msg = {
                    "type": "Text",
                    "text": word,
                    "start_s": start_time,
                    "stop_s": stop_time,
                }
                await websocket.send_bytes(msgpack.packb(text_msg))

                # Send a single audio chunk per word (fast for testing)
                audio_msg = {
                    "type": "Audio",
                    "pcm": generate_sine_chunk(),
                }
                await websocket.send_bytes(msgpack.packb(audio_msg))

                current_time += word_duration

    except WebSocketDisconnect:
        logger.debug("Mock TTS client disconnected")
    except Exception as e:
        logger.debug(f"Mock TTS error: {e}")

    try:
        await websocket.close()
    except Exception:
        pass
