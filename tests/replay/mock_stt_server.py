"""Mock STT WebSocket server for integration testing.

Based on unmute/stt/dummy_speech_to_text.py pattern.
Implements the STT msgpack protocol and yields scripted Word messages.
"""

import asyncio
import logging

import msgpack
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 480
FRAME_TIME_SEC = SAMPLES_PER_FRAME / SAMPLE_RATE

app = FastAPI()
logger = logging.getLogger(__name__)

# Scripted transcription words to emit when audio is received.
# Each entry: {"text": "word", "delay_frames": N} - emit after N audio frames
scripted_words: list[dict] = []


def set_scripted_words(words: list[dict]) -> None:
    """Set the scripted words the mock STT will return."""
    global scripted_words
    scripted_words = list(words)


@app.get("/api/build_info")
def get_build_info():
    return {"note": "mock STT for testing"}


@app.websocket("/api/stt_streaming")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    audio_frames_received = 0
    word_index = 0
    current_time = 0.0

    try:
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_bytes(), timeout=2.0
                )
            except asyncio.TimeoutError:
                await asyncio.sleep(0.01)
                continue

            msg = msgpack.unpackb(data)
            msg_type = msg.get("type", "")

            if msg_type == "Audio":
                audio_frames_received += 1
                current_time += FRAME_TIME_SEC

                # Send Step message with pause prediction
                # Default: high pause score (no speech detected)
                pause_score = 1.0

                # Check if we should emit a word
                if word_index < len(scripted_words):
                    word_entry = scripted_words[word_index]
                    trigger_frame = word_entry.get("delay_frames", 10)

                    if audio_frames_received >= trigger_frame:
                        # Low pause score while speaking
                        pause_score = 0.1

                        # Emit the word
                        word_msg = {
                            "type": "Word",
                            "text": word_entry["text"],
                            "start_time": current_time - FRAME_TIME_SEC,
                            "end_time": current_time,
                        }
                        await websocket.send_bytes(msgpack.packb(word_msg))
                        word_index += 1

                        # Reset frame counter for next word
                        audio_frames_received = 0

                step_msg = {
                    "type": "Step",
                    "pause_prediction": pause_score,
                }
                await websocket.send_bytes(msgpack.packb(step_msg))

            elif msg_type == "Marker":
                # Echo back marker
                marker_msg = {"type": "Marker", "id": msg.get("id", 0)}
                await websocket.send_bytes(msgpack.packb(marker_msg))

    except WebSocketDisconnect:
        logger.debug("Mock STT client disconnected")
    except Exception as e:
        logger.debug(f"Mock STT error: {e}")

    try:
        await websocket.close()
    except Exception:
        pass
