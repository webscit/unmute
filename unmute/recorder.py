import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import aiofiles
from pydantic import BaseModel, Field

import unmute.openai_realtime_api_events as ora

if TYPE_CHECKING:
    from unmute.session_state import SessionState

logger = logging.getLogger(__name__)

EventSender = Literal["client", "server"]


class RecorderEvent(BaseModel):
    timestamp_wall: float
    event_sender: EventSender
    data: Annotated[ora.Event, Field(discriminator="type")]


class SessionStateSnapshot(BaseModel):
    """Snapshot of session state for replay/reconnect."""

    type: Literal["session_state_snapshot"] = "session_state_snapshot"
    timestamp_wall: float
    items: dict[str, dict[str, Any]]
    item_order: list[str]
    current_response_id: str | None
    current_item_id: str | None
    input_buffer_samples: int
    input_buffer_committed: bool
    pending_audio_item_id: str | None
    response_queue: list[str]


class Recorder:
    """Record the events sent between the client and the server to a file.

    Doesn't include the user audio for privacy reasons.
    """

    def __init__(self, recordings_dir: Path):
        self.path = recordings_dir / (make_filename() + ".jsonl")
        recordings_dir.mkdir(exist_ok=True)
        # We use aiofiles to avoid blocking the event loop when writing to the file.
        self.opened_file = None

    async def add_event(self, event_sender: EventSender, data: ora.Event):
        """If the recorder is not actually running, the event will be ignored."""
        if self.opened_file is None:
            self.opened_file = await aiofiles.open(self.path, "a")

        await self.opened_file.write(
            RecorderEvent(
                timestamp_wall=datetime.now().timestamp(),
                event_sender=event_sender,
                data=data,
            ).model_dump_json()
            + "\n"
        )

    async def add_state_snapshot(self, session_state: "SessionState") -> None:
        """Record a state snapshot for replay/reconnect.

        Snapshots are recorded at key points:
        - After SessionUpdated
        - After ResponseDone
        - After ConversationItemCreated/Deleted
        """
        if self.opened_file is None:
            self.opened_file = await aiofiles.open(self.path, "a")

        snapshot = session_state.snapshot()
        snapshot_event = SessionStateSnapshot(
            timestamp_wall=datetime.now().timestamp(),
            items=snapshot.get("items", {}),
            item_order=snapshot.get("item_order", []),
            current_response_id=snapshot.get("current_response_id"),
            current_item_id=snapshot.get("current_item_id"),
            input_buffer_samples=snapshot.get("input_buffer_samples", 0),
            input_buffer_committed=snapshot.get("input_buffer_committed", False),
            pending_audio_item_id=snapshot.get("pending_audio_item_id"),
            response_queue=snapshot.get("response_queue", []),
        )
        await self.opened_file.write(snapshot_event.model_dump_json() + "\n")

    async def shutdown(self, keep_recording: bool = True):
        """Flush any remaining events to the file and close the recorder.

        If `keep_recording` is False, the file will be deleted if it exists.
        This is because we get the user consent after we've already started recording,
        so if the user doesn't consent, we delete the file afterwards.
        """
        if self.opened_file is not None:
            await self.opened_file.close()
            if keep_recording:
                logger.info(f"Recording stored into {self.path}.")
            else:
                try:
                    self.path.unlink()
                    logger.info(
                        f"Deleted recording {self.path} due to lack of consent."
                    )
                except Exception as e:
                    logger.error(f"Failed to delete recording file {self.path}: {e}")


def make_filename() -> str:
    """Create a unique filename based on the current timestamp and a short UUID, without a suffix."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    unique_id = uuid.uuid4().hex[:4]
    return f"{timestamp}_{unique_id}"
