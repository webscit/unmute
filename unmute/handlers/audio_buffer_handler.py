"""Audio buffer management handler for conversation audio tracking."""

from logging import getLogger

from unmute.audio.realtime_buffer import RealtimeAudioBuffer
from unmute.llm.chatbot import Chatbot
from unmute.session_state import SessionState

logger = getLogger(__name__)


class AudioBufferHandler:
    """Handles audio buffer lifecycle and management."""

    def __init__(
        self,
        sample_rate: int,
        session_state: SessionState,
        chatbot: Chatbot,
    ) -> None:
        """Initialize audio buffer handler.

        Args:
            sample_rate: Audio sample rate
            session_state: Session state for tracking items
            chatbot: Chatbot instance for getting transcripts
        """
        self.audio_buffer = RealtimeAudioBuffer(sample_rate=sample_rate)
        self.session_state = session_state
        self.chatbot = chatbot

    def add_samples(self, num_samples: int) -> None:
        """Track incoming audio samples.

        Args:
            num_samples: Number of samples to add
        """
        self.session_state.add_input_samples(num_samples)

    async def commit_audio_buffer(self) -> tuple[str, str | None]:
        """Commit accumulated audio as a conversation item.

        Returns:
            Tuple of (item_id, previous_item_id).
        """
        previous_item_id = self.session_state.get_previous_item_id()
        item_id = self.session_state.commit_input_buffer()

        # Commit audio buffer and capture latency metrics
        total_samples, latency_metrics = self.audio_buffer.commit(item_id)
        logger.debug(
            f"Audio buffer committed: {total_samples} samples, "
            f"latency={latency_metrics.arrival_to_flush_ms}ms"
        )

        # Update transcript from last user message if available
        last_user_msg = self.chatbot.last_message("user")
        if last_user_msg and item_id in self.session_state.items:
            item = self.session_state.items[item_id]
            if item.content:
                item.content[0]["transcript"] = last_user_msg

        # Reset audio buffer for next segment
        self.audio_buffer.reset()

        return item_id, previous_item_id

    async def clear_audio_buffer(self) -> None:
        """Clear the input audio buffer without committing."""
        self.session_state.clear_input_buffer()
        cleared_samples = self.audio_buffer.clear()
        logger.debug(f"Audio buffer cleared: {cleared_samples} samples discarded")
        self.audio_buffer.reset()

    def get_duration_sec(self) -> float:
        """Get the duration of buffered audio in seconds.

        Returns:
            Duration in seconds
        """
        return self.audio_buffer.duration_sec
