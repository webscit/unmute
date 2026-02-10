"""Voice Activity Detection handler for speech start/stop and pause detection."""

import asyncio
import math
from logging import getLogger
from typing import Literal

import numpy as np

import unmute.openai_realtime_api_events as ora
from unmute import metrics as mt
from unmute.kyutai_constants import FRAME_TIME_SEC, SAMPLES_PER_FRAME
from unmute.session_state import SessionState
from unmute.stt.speech_to_text import SpeechToText
from unmute.timer import Stopwatch
from unmute.tracing import STT_FLUSH_DURATION, trace_span

# For this much time, the VAD does not interrupt the bot. This is needed because at
# least on Mac, the echo cancellation takes a while to kick in, at the start, so the ASR
# sometimes hears a bit of the TTS audio and interrupts the bot. Only happens on the
# first message.
# A word from the ASR can still interrupt the bot.
UNINTERRUPTIBLE_BY_VAD_TIME_SEC = 3

logger = getLogger(__name__)


class VADHandler:
    """Handles voice activity detection logic including speech start/stop and pauses."""

    def __init__(
        self,
        session_state: SessionState,
        output_queue: asyncio.Queue,
        sample_rate: int,
    ) -> None:
        """Initialize VAD handler.

        Args:
            session_state: Session state for tracking conversation items
            output_queue: Queue for emitting VAD events
            sample_rate: Audio sample rate for time calculations
        """
        self.session_state = session_state
        self.output_queue = output_queue
        self.sample_rate = sample_rate

        # Speech tracking state
        self.speech_started: bool = False
        self.speech_start_time: float | None = None
        self.current_speech_item_id: str | None = None

        # STT timing state
        self.stt_last_message_time: float = 0
        self.stt_end_of_flush_time: float | None = None
        self.stt_flush_timer = Stopwatch()

    def reset_speech_tracking(self) -> None:
        """Reset speech tracking state after interruption or completion."""
        self.speech_started = False
        self.speech_start_time = None
        # Note: Keep current_speech_item_id for the interrupted segment

    async def mark_speech_started(
        self,
        audio_start_time: float,
        item_id: str | None = None,
    ) -> ora.InputAudioBufferSpeechStarted:
        """Mark speech as started and emit event.

        Args:
            audio_start_time: Time in seconds when speech started
            item_id: Optional item ID (will create new one if not provided)

        Returns:
            The speech started event that was emitted
        """
        self.speech_started = True
        self.speech_start_time = audio_start_time

        # Calculate audio_start_ms relative to session start
        audio_start_ms = int(audio_start_time * 1000)

        # Get or create pending item_id
        if item_id is None:
            item_id = self.session_state.get_pending_input_item_id()
        self.current_speech_item_id = item_id

        # Emit speech_started with metadata
        event = ora.InputAudioBufferSpeechStarted(
            item_id=item_id,
            audio_start_ms=audio_start_ms,
        )
        await self.output_queue.put(event)
        mt.VAD_SPEECH_STARTED.inc()
        logger.info(
            f"Speech started: item_id={item_id}, audio_start_ms={audio_start_ms}"
        )
        return event

    async def mark_speech_stopped(
        self,
        audio_end_time: float,
    ) -> ora.InputAudioBufferSpeechStopped:
        """Mark speech as stopped and emit event.

        Args:
            audio_end_time: Time in seconds when speech stopped

        Returns:
            The speech stopped event that was emitted
        """
        # Calculate audio_end_ms relative to session start
        audio_end_ms = int(audio_end_time * 1000)

        # Use current speech item_id if available
        item_id = self.current_speech_item_id

        # Emit speech_stopped with metadata
        event = ora.InputAudioBufferSpeechStopped(
            item_id=item_id,
            audio_end_ms=audio_end_ms,
        )
        await self.output_queue.put(event)
        mt.VAD_SPEECH_STOPPED.inc()

        # Track latency from speech start to stop
        if self.speech_start_time is not None:
            speech_duration = audio_end_time - self.speech_start_time
            mt.VAD_SPEECH_DURATION.observe(speech_duration)
            logger.info(
                f"Speech stopped: item_id={item_id}, audio_end_ms={audio_end_ms}, "
                f"duration={speech_duration:.2f}s"
            )

        # Reset speech tracking
        self.reset_speech_tracking()
        return event

    def determine_pause(
        self,
        stt: SpeechToText,
        conversation_state: Literal[
            "waiting_for_user", "user_speaking", "bot_speaking"
        ],
        debug_dict: dict,
    ) -> bool:
        """Determine if a pause has been detected.

        Args:
            stt: Speech-to-text service for pause predictions
            conversation_state: Current conversation state
            debug_dict: Debug dictionary for timing info

        Returns:
            True if pause detected and response should be generated
        """
        if conversation_state != "user_speaking":
            return False

        # This is how much wall clock time has passed since we received the last ASR
        # message. Assumes the ASR connection is healthy, so that stt.sent_samples is up
        # to date.
        time_since_last_message = (
            stt.sent_samples / self.sample_rate
        ) - self.stt_last_message_time
        debug_dict["time_since_last_message"] = time_since_last_message

        if stt.pause_prediction.value > 0.6:
            debug_dict["timing"]["pause_detection"] = time_since_last_message
            return True
        else:
            return False

    async def flush_stt(
        self,
        stt: SpeechToText,
        audio_end_time: float,
    ) -> None:
        """Flush STT pipeline after pause detected.

        Args:
            stt: Speech-to-text service to flush
            audio_end_time: Current audio time in seconds
        """
        async with trace_span(
            "stt_flush_pipeline",
            histogram=STT_FLUSH_DURATION,
            attributes={"pause_score": stt.pause_prediction.value},
        ):
            logger.info("Pause detected")

            # Emit speech_stopped event
            await self.mark_speech_stopped(audio_end_time)

            # Set up flush timing
            self.stt_end_of_flush_time = stt.current_time + stt.delay_sec
            self.stt_flush_timer = Stopwatch()

            # Send silence to flush the STT pipeline
            num_frames = (
                int(math.ceil(stt.delay_sec / FRAME_TIME_SEC)) + 1
            )  # some safety margin.
            zero = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)
            for _ in range(num_frames):
                await stt.send_audio(zero)

    def is_flush_complete(self, stt: SpeechToText) -> bool:
        """Check if STT flush is complete.

        Args:
            stt: Speech-to-text service

        Returns:
            True if flush is complete and response generation should start
        """
        if self.stt_end_of_flush_time is None:
            return False

        if stt.current_time > self.stt_end_of_flush_time:
            elapsed = self.stt_flush_timer.time()
            rtf = stt.delay_sec / elapsed
            logger.info(
                "Flushing finished, took %.1f ms, RTF: %.1f", elapsed * 1000, rtf
            )
            self.stt_end_of_flush_time = None
            return True

        return False

    def should_interrupt_by_vad(
        self,
        conversation_state: Literal[
            "waiting_for_user", "user_speaking", "bot_speaking"
        ],
        pause_score: float,
        audio_time: float,
    ) -> bool:
        """Determine if VAD should interrupt the bot.

        Args:
            conversation_state: Current conversation state
            pause_score: Current pause prediction score
            audio_time: Current audio time in seconds

        Returns:
            True if bot should be interrupted by VAD
        """
        return (
            conversation_state == "bot_speaking"
            and pause_score < 0.4
            and audio_time > UNINTERRUPTIBLE_BY_VAD_TIME_SEC
        )

    def update_stt_message_time(self, message_time: float) -> None:
        """Update the last STT message time.

        Args:
            message_time: Time of the last STT message
        """
        self.stt_last_message_time = message_time

    def is_flushing(self) -> bool:
        """Check if currently flushing the STT pipeline.

        Returns:
            True if currently flushing
        """
        return self.stt_end_of_flush_time is not None
