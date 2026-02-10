"""Event routing and lifecycle management for UnmuteHandler."""

import asyncio
from logging import getLogger
from typing import Any, Callable

from unmute import metrics as mt
from unmute.exceptions import make_ora_error
from unmute.session_state import SessionState

# Backpressure thresholds - queue sizes that trigger warnings/errors
OUTPUT_QUEUE_WARNING_THRESHOLD = 50
OUTPUT_QUEUE_ERROR_THRESHOLD = 100
BACKPRESSURE_CHECK_INTERVAL = 5.0  # seconds between checks

logger = getLogger(__name__)


class EventRouter:
    """Routes events and manages response lifecycle."""

    def __init__(
        self,
        session_state: SessionState,
        output_queue: asyncio.Queue,
        get_audio_time_callback: Callable[[], float],
    ) -> None:
        """Initialize event router.

        Args:
            session_state: Session state for tracking responses and items
            output_queue: Queue for emitting events
            get_audio_time_callback: Callback to get current audio time in seconds
        """
        self.session_state = session_state
        self.output_queue = output_queue
        self.get_audio_time = get_audio_time_callback

        # Backpressure monitoring
        self.last_backpressure_check: float = 0
        self.backpressure_error_emitted: bool = False

    async def check_backpressure(self) -> None:
        """Monitor output queue size and emit errors if backpressure threshold exceeded."""
        current_time = self.get_audio_time()

        # Only check periodically to avoid overhead
        if current_time - self.last_backpressure_check < BACKPRESSURE_CHECK_INTERVAL:
            return

        self.last_backpressure_check = current_time
        queue_size = self.output_queue.qsize()

        # Update metrics
        mt.OUTPUT_QUEUE_SIZE.set(queue_size)

        if queue_size >= OUTPUT_QUEUE_ERROR_THRESHOLD:
            if not self.backpressure_error_emitted:
                logger.error(
                    f"Backpressure error: output queue size {queue_size} exceeds threshold {OUTPUT_QUEUE_ERROR_THRESHOLD}"
                )
                error = make_ora_error(
                    type="backpressure_error",
                    message=f"Output queue backpressure: {queue_size} items queued (threshold: {OUTPUT_QUEUE_ERROR_THRESHOLD})",
                )
                await self.output_queue.put(error)
                mt.BACKPRESSURE_ERRORS.inc()
                self.backpressure_error_emitted = True
        elif queue_size >= OUTPUT_QUEUE_WARNING_THRESHOLD:
            logger.warning(
                f"Backpressure warning: output queue size {queue_size} exceeds warning threshold {OUTPUT_QUEUE_WARNING_THRESHOLD}"
            )
        else:
            # Reset error flag when queue drains below threshold
            self.backpressure_error_emitted = False

    def _extract_text_content(self, content: list[dict[str, Any]]) -> str:
        """Extract text from content parts."""
        texts = []
        for part in content:
            if part.get("type") == "text":
                texts.append(part.get("text", ""))
            elif part.get("type") == "input_text":
                texts.append(part.get("text", ""))
            elif part.get("type") in ("audio", "input_audio"):
                transcript = part.get("transcript")
                if transcript:
                    texts.append(transcript)
        return " ".join(texts)
