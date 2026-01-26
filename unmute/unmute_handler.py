import asyncio
import math
from functools import partial
from logging import getLogger
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import websockets
from fastrtc import (
    AdditionalOutputs,
    AsyncStreamHandler,
    CloseStream,
    audio_to_float32,
    wait_for_item,
)
from pydantic import BaseModel

import unmute.openai_realtime_api_events as ora
from unmute import metrics as mt
from unmute.audio.realtime_buffer import RealtimeAudioBuffer
from unmute.audio_input_override import AudioInputOverride
from unmute.exceptions import make_ora_error
from unmute.kyutai_constants import (
    FRAME_TIME_SEC,
    RECORDINGS_DIR,
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
)
from unmute.llm.chatbot import Chatbot
from unmute.llm.llm_utils import (
    INTERRUPTION_CHAR,
    USER_SILENCE_MARKER,
    VLLMStream,
    get_openai_client,
    rechunk_to_words,
)
from unmute.quest_manager import Quest, QuestManager
from unmute.recorder import Recorder
from unmute.service_discovery import find_instance
from unmute.session_state import SessionState
from unmute.stt.speech_to_text import SpeechToText, STTMarkerMessage
from unmute.timer import Stopwatch
from unmute.tts.text_to_speech import (
    TextToSpeech,
    TTSAudioMessage,
    TTSClientEosMessage,
    TTSTextMessage,
)

# TTS_DEBUGGING_TEXT: str | None = "What's 'Hello world'?"
# TTS_DEBUGGING_TEXT: str | None = "What's the difference between a bagel and a donut?"
TTS_DEBUGGING_TEXT = None

# AUDIO_INPUT_OVERRIDE: Path | None = Path.home() / "audio/dog-or-cat-3.mp3"
AUDIO_INPUT_OVERRIDE: Path | None = None
DEBUG_PLOT_HISTORY_SEC = 10.0

USER_SILENCE_TIMEOUT = 7.0
FIRST_MESSAGE_TEMPERATURE = 0.7
FURTHER_MESSAGES_TEMPERATURE = 0.3
# For this much time, the VAD does not interrupt the bot. This is needed because at
# least on Mac, the echo cancellation takes a while to kick in, at the start, so the ASR
# sometimes hears a bit of the TTS audio and interrupts the bot. Only happens on the
# first message.
# A word from the ASR can still interrupt the bot.
UNINTERRUPTIBLE_BY_VAD_TIME_SEC = 3

# Backpressure thresholds - queue sizes that trigger warnings/errors
OUTPUT_QUEUE_WARNING_THRESHOLD = 50
OUTPUT_QUEUE_ERROR_THRESHOLD = 100
BACKPRESSURE_CHECK_INTERVAL = 5.0  # seconds between checks

logger = getLogger(__name__)

HandlerOutput = (
    tuple[int, np.ndarray] | AdditionalOutputs | ora.ServerEvent | CloseStream
)


class GradioUpdate(BaseModel):
    chat_history: list[dict[str, str]]
    debug_dict: dict[str, Any]
    debug_plot_data: list[dict]


class UnmuteHandler(AsyncStreamHandler):
    def __init__(self) -> None:
        super().__init__(
            input_sample_rate=SAMPLE_RATE,
            # IMPORTANT! If set to a higher value, will lead to choppy audio. 🤷‍♂️
            output_frame_size=480,
            output_sample_rate=SAMPLE_RATE,
        )
        self.n_samples_received = 0  # Used for measuring time
        self.output_queue: asyncio.Queue[HandlerOutput] = asyncio.Queue()
        self.recorder = Recorder(RECORDINGS_DIR) if RECORDINGS_DIR else None
        self.session_state = SessionState()
        self.audio_buffer = RealtimeAudioBuffer(sample_rate=SAMPLE_RATE)

        self.quest_manager = QuestManager()

        self.stt_last_message_time: float = 0
        self.stt_end_of_flush_time: float | None = None
        self.stt_flush_timer = Stopwatch()

        self.tts_voice: str | None = None  # Stored separately because TTS is restarted
        self.tts_output_stopwatch = Stopwatch()

        self.chatbot = Chatbot()
        self.openai_client = get_openai_client()

        self.turn_transition_lock = asyncio.Lock()

        # VAD event tracking
        self.speech_started: bool = False
        self.speech_start_time: float | None = None
        self.current_speech_item_id: str | None = None

        # Backpressure monitoring
        self.last_backpressure_check: float = 0
        self.backpressure_error_emitted: bool = False

        self.debug_dict: dict[str, Any] = {
            "timing": {},
            "connection": {},
            "chatbot": {},
        }
        self.debug_plot_data: list[dict] = []
        self.last_additional_output_update = self.audio_received_sec()

        if AUDIO_INPUT_OVERRIDE is not None:
            self.audio_input_override = AudioInputOverride(AUDIO_INPUT_OVERRIDE)
        else:
            self.audio_input_override = None

    async def cleanup(self):
        if self.recorder is not None:
            await self.recorder.shutdown()

    @property
    def stt(self) -> SpeechToText | None:
        try:
            quest = self.quest_manager.quests["stt"]
        except KeyError:
            return None
        return cast(Quest[SpeechToText], quest).get_nowait()

    @property
    def tts(self) -> TextToSpeech | None:
        try:
            quest = self.quest_manager.quests["tts"]
        except KeyError:
            return None
        return cast(Quest[TextToSpeech], quest).get_nowait()

    def get_gradio_update(self):
        self.debug_dict["conversation_state"] = self.chatbot.conversation_state()
        self.debug_dict["connection"]["stt"] = self.stt.state() if self.stt else "none"
        self.debug_dict["connection"]["tts"] = self.tts.state() if self.tts else "none"
        self.debug_dict["tts_voice"] = self.tts.voice if self.tts else "none"
        self.debug_dict["stt_pause_prediction"] = (
            self.stt.pause_prediction.value if self.stt else -1
        )

        # This gets verbose
        # cutoff_time = self.audio_received_sec() - DEBUG_PLOT_HISTORY_SEC
        # self.debug_plot_data = [x for x in self.debug_plot_data if x["t"] > cutoff_time]

        return AdditionalOutputs(
            GradioUpdate(
                chat_history=[
                    # Not trying to hide the system prompt, just making it less verbose
                    m
                    for m in self.chatbot.chat_history
                    if m["role"] != "system"
                ],
                debug_dict=self.debug_dict,
                debug_plot_data=[],
            )
        )

    async def add_chat_message_delta(
        self,
        delta: str,
        role: Literal["user", "assistant"],
        generating_message_i: int | None = None,  # Avoid race conditions
    ):
        is_new_message = await self.chatbot.add_chat_message_delta(
            delta, role, generating_message_i=generating_message_i
        )

        return is_new_message

    async def _generate_response(self):
        # Empty message to signal we've started responding.
        # Do it here in the lock to avoid race conditions
        await self.add_chat_message_delta("", "assistant")
        quest = Quest.from_run_step("llm", self._generate_response_task)
        await self.quest_manager.add(quest)

    async def _generate_response_task(self):
        generating_message_i = len(self.chatbot.chat_history)

        # Generate IDs for response and output item
        response_id = ora.random_id("resp")
        item_id = ora.random_id("item")

        # Initialize response in session state
        response = self.session_state.start_response(response_id)
        response.voice = self.tts_voice or "missing"
        response.chat_history = self.chatbot.chat_history

        # Emit ResponseCreated
        await self.output_queue.put(ora.ResponseCreated(response=response))

        # Create and track output item
        output_item = ora.Item(
            id=item_id,
            type="message",
            role="assistant",
            status="in_progress",
            content=[],
        )
        self.session_state.add_output_item(output_item, output_index=0)

        # Emit ResponseOutputItemAdded
        await self.output_queue.put(
            ora.ResponseOutputItemAdded(
                response_id=response_id,
                output_index=0,
                item=output_item,
            )
        )

        # Emit ResponseContentPartAdded for text content
        await self.output_queue.put(
            ora.ResponseContentPartAdded(
                response_id=response_id,
                item_id=item_id,
                output_index=0,
                content_index=0,
                part={"type": "text", "text": ""},
            )
        )

        llm_stopwatch = Stopwatch()

        quest = await self.start_up_tts(generating_message_i)
        llm = VLLMStream(
            # if generating_message_i is 2, then we have a system prompt + an empty
            # assistant message signalling that we are generating a response.
            self.openai_client,
            temperature=FIRST_MESSAGE_TEMPERATURE
            if generating_message_i == 2
            else FURTHER_MESSAGES_TEMPERATURE,
        )

        messages = self.chatbot.preprocessed_messages()

        self.tts_output_stopwatch = Stopwatch(autostart=False)
        tts = None

        response_words = []
        error_from_tts = False
        time_to_first_token = None
        num_words_sent = sum(
            len(message.get("content", "").split()) for message in messages
        )
        mt.VLLM_SENT_WORDS.inc(num_words_sent)
        mt.VLLM_REQUEST_LENGTH.observe(num_words_sent)
        mt.VLLM_ACTIVE_SESSIONS.inc()

        final_status = "completed"
        try:
            async for delta in rechunk_to_words(llm.chat_completion(messages)):
                await self.output_queue.put(
                    ora.UnmuteResponseTextDeltaReady(delta=delta)
                )

                mt.VLLM_RECV_WORDS.inc()
                response_words.append(delta)

                if time_to_first_token is None:
                    time_to_first_token = llm_stopwatch.time()
                    self.debug_dict["timing"]["to_first_token"] = time_to_first_token
                    mt.VLLM_TTFT.observe(time_to_first_token)
                    logger.info("Sending first word to TTS: %s", delta)

                self.tts_output_stopwatch.start_if_not_started()
                try:
                    tts = await quest.get()
                except Exception:
                    error_from_tts = True
                    raise

                if len(self.chatbot.chat_history) > generating_message_i:
                    break  # We've been interrupted

                assert isinstance(delta, str)  # make Pyright happy
                await tts.send(delta)

            # Emit ResponseTextDone with IDs
            full_text = "".join(response_words)
            await self.output_queue.put(
                ora.ResponseTextDone(
                    text=full_text,
                    response_id=response_id,
                    item_id=item_id,
                    output_index=0,
                    content_index=0,
                )
            )

            # Emit ResponseContentPartDone
            await self.output_queue.put(
                ora.ResponseContentPartDone(
                    response_id=response_id,
                    item_id=item_id,
                    output_index=0,
                    content_index=0,
                    part={"type": "text", "text": full_text},
                )
            )

            if tts is not None:
                logger.info("Sending TTS EOS.")
                await tts.send(TTSClientEosMessage())
        except asyncio.CancelledError:
            mt.VLLM_INTERRUPTS.inc()
            final_status = "cancelled"
            raise
        except Exception:
            if not error_from_tts:
                mt.VLLM_HARD_ERRORS.inc()
            final_status = "failed"
            raise
        finally:
            logger.info("End of VLLM, after %d words.", len(response_words))
            mt.VLLM_ACTIVE_SESSIONS.dec()
            mt.VLLM_REPLY_LENGTH.observe(len(response_words))
            mt.VLLM_GEN_DURATION.observe(llm_stopwatch.time())

            # Update output item to completed
            if item_id in self.session_state.items:
                self.session_state.items[item_id].status = "completed"
                # Emit ResponseOutputItemDone
                await self.output_queue.put(
                    ora.ResponseOutputItemDone(
                        response_id=response_id,
                        output_index=0,
                        item=self.session_state.items[item_id],
                    )
                )

            # Complete response in session state and emit ResponseDone
            if self.session_state.has_active_response():
                final_response = self.session_state.complete_response(final_status)
                await self.output_queue.put(ora.ResponseDone(response=final_response))

                # Record state snapshot after response completion
                if self.recorder is not None:
                    await self.recorder.add_state_snapshot(self.session_state)

    def audio_received_sec(self) -> float:
        """How much audio has been received in seconds. Used instead of time.time().

        This is so that we aren't tied to real-time streaming.
        """
        return self.n_samples_received / self.input_sample_rate

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        stt = self.stt
        assert stt is not None
        sr = frame[0]
        assert sr == self.input_sample_rate

        assert frame[1].shape[0] == 1  # Mono
        array = frame[1][0]

        self.n_samples_received += array.shape[0]

        # Track samples in both session state and audio buffer for latency tracking
        self.session_state.add_input_samples(array.shape[0])

        # If this doesn't update, it means the receive loop isn't running because
        # the process is busy with something else, which is bad.
        self.debug_dict["last_receive_time"] = self.audio_received_sec()
        float_audio = audio_to_float32(array)

        self.debug_plot_data.append(
            {
                "t": self.audio_received_sec(),
                "amplitude": float(np.sqrt((float_audio**2).mean())),
                "pause_prediction": stt.pause_prediction.value,
            }
        )

        if self.chatbot.conversation_state() == "bot_speaking":
            # Periodically update this not to trigger the "long silence" accidentally.
            self.waiting_for_user_start_time = self.audio_received_sec()

        if TTS_DEBUGGING_TEXT is not None:
            assert self.audio_input_override is None, (
                "Can't use both TTS_DEBUGGING_TEXT and audio input override."
            )

            # Debugging mode: always send a fixed string when it's the user's turn.
            if self.chatbot.conversation_state() == "waiting_for_user":
                logger.info("Using TTS debugging text. Ignoring microphone.")
                self.chatbot.chat_history.append(
                    {"role": "user", "content": TTS_DEBUGGING_TEXT}
                )
                await self._generate_response()
            return

        if (
            len(self.chatbot.chat_history) == 1
            # Wait until the instructions are updated. A bit hacky
            and self.chatbot.get_instructions() is not None
        ):
            logger.info("Generating initial response.")
            await self._generate_response()

        if self.audio_input_override is not None:
            frame = (frame[0], self.audio_input_override.override(frame[1]))

        if self.chatbot.conversation_state() == "user_speaking":
            self.debug_dict["timing"] = {}

        await stt.send_audio(array)
        if self.stt_end_of_flush_time is None:
            await self.detect_long_silence()
            await self.check_backpressure()

            if self.determine_pause():
                logger.info("Pause detected")

                # Calculate audio_end_ms relative to session start
                audio_end_ms = int(self.audio_received_sec() * 1000)

                # Use current speech item_id if available
                item_id = self.current_speech_item_id

                # Emit speech_stopped with metadata
                await self.output_queue.put(
                    ora.InputAudioBufferSpeechStopped(
                        item_id=item_id,
                        audio_end_ms=audio_end_ms,
                    )
                )
                mt.VAD_SPEECH_STOPPED.inc()

                # Track latency from speech start to stop
                if self.speech_start_time is not None:
                    speech_duration = self.audio_received_sec() - self.speech_start_time
                    mt.VAD_SPEECH_DURATION.observe(speech_duration)
                    logger.info(
                        f"Speech stopped: item_id={item_id}, audio_end_ms={audio_end_ms}, "
                        f"duration={speech_duration:.2f}s"
                    )

                # Reset speech tracking
                self.speech_started = False
                self.speech_start_time = None

                self.stt_end_of_flush_time = stt.current_time + stt.delay_sec
                self.stt_flush_timer = Stopwatch()
                num_frames = (
                    int(math.ceil(stt.delay_sec / FRAME_TIME_SEC)) + 1
                )  # some safety margin.
                zero = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)
                for _ in range(num_frames):
                    await stt.send_audio(zero)
            elif (
                self.chatbot.conversation_state() == "bot_speaking"
                and stt.pause_prediction.value < 0.4
                and self.audio_received_sec() > UNINTERRUPTIBLE_BY_VAD_TIME_SEC
            ):
                logger.info("Interruption by STT-VAD")
                await self.interrupt_bot()
                await self.add_chat_message_delta("", "user")
        else:
            # We do not try to detect interruption here, the STT would be processing
            # a chunk full of 0, so there is little chance the pause score would indicate an interruption.
            if stt.current_time > self.stt_end_of_flush_time:
                self.stt_end_of_flush_time = None
                elapsed = self.stt_flush_timer.time()
                rtf = stt.delay_sec / elapsed
                logger.info(
                    "Flushing finished, took %.1f ms, RTF: %.1f", elapsed * 1000, rtf
                )
                await self._generate_response()

    def determine_pause(self) -> bool:
        stt = self.stt
        if stt is None:
            return False
        if self.chatbot.conversation_state() != "user_speaking":
            return False

        # This is how much wall clock time has passed since we received the last ASR
        # message. Assumes the ASR connection is healthy, so that stt.sent_samples is up
        # to date.
        time_since_last_message = (
            stt.sent_samples / self.input_sample_rate
        ) - self.stt_last_message_time
        self.debug_dict["time_since_last_message"] = time_since_last_message

        if stt.pause_prediction.value > 0.6:
            self.debug_dict["timing"]["pause_detection"] = time_since_last_message
            return True
        else:
            return False

    async def emit(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> HandlerOutput | None:
        output_queue_item = await wait_for_item(self.output_queue)

        if output_queue_item is not None:
            return output_queue_item
        else:
            if self.last_additional_output_update < self.audio_received_sec() - 1:
                # If we have nothing to emit, at least update the debug dict.
                # Don't update too often for performance reasons
                self.last_additional_output_update = self.audio_received_sec()
                return self.get_gradio_update()
            else:
                return None

    def copy(self):
        return UnmuteHandler()

    async def __aenter__(self) -> None:
        await self.quest_manager.__aenter__()

    async def start_up(self):
        await self.start_up_stt()
        self.waiting_for_user_start_time = self.audio_received_sec()

    async def __aexit__(self, *exc: Any) -> None:
        return await self.quest_manager.__aexit__(*exc)

    async def start_up_stt(self):
        async def _init() -> SpeechToText:
            return await find_instance("stt", SpeechToText)

        async def _run(stt: SpeechToText):
            await self._stt_loop(stt)

        async def _close(stt: SpeechToText):
            await stt.shutdown()

        quest = await self.quest_manager.add(Quest("stt", _init, _run, _close))
        # We want to be sure to have the STT before starting anything.
        await quest.get()

    async def _stt_loop(self, stt: SpeechToText):
        try:
            async for data in stt:
                if isinstance(data, STTMarkerMessage):
                    # Ignore the marker messages
                    continue

                await self.output_queue.put(
                    ora.ConversationItemInputAudioTranscriptionDelta(
                        delta=data.text,
                        start_time=data.start_time,
                    )
                )

                # The STT sends an empty string as the first message, but we
                # don't want to add that because it can trigger a pause even
                # if the user hasn't started speaking yet.
                if data.text == "":
                    continue

                if self.chatbot.conversation_state() == "bot_speaking":
                    logger.info("STT-based interruption")
                    await self.interrupt_bot()

                self.stt_last_message_time = data.start_time
                is_new_message = await self.add_chat_message_delta(data.text, "user")
                if is_new_message:
                    # Ensure we don't stop after the first word if the VAD didn't have
                    # time to react.
                    stt.pause_prediction.value = 0.0

                    # Mark speech as started and track timing
                    if not self.speech_started:
                        self.speech_started = True
                        self.speech_start_time = self.audio_received_sec()

                        # Calculate audio_start_ms relative to session start
                        audio_start_ms = int(self.speech_start_time * 1000)

                        # Get or create pending item_id
                        item_id = self.session_state.get_pending_input_item_id()
                        self.current_speech_item_id = item_id

                        # Emit speech_started with metadata
                        await self.output_queue.put(
                            ora.InputAudioBufferSpeechStarted(
                                item_id=item_id,
                                audio_start_ms=audio_start_ms,
                            )
                        )
                        mt.VAD_SPEECH_STARTED.inc()
                        logger.info(
                            f"Speech started: item_id={item_id}, audio_start_ms={audio_start_ms}"
                        )
        except websockets.ConnectionClosed:
            logger.info("STT connection closed while receiving messages.")

    async def start_up_tts(self, generating_message_i: int) -> Quest[TextToSpeech]:
        async def _init() -> TextToSpeech:
            factory = partial(
                TextToSpeech,
                recorder=self.recorder,
                get_time=self.audio_received_sec,
                voice=self.tts_voice,
            )
            sleep_time = 0.05
            sleep_growth = 1.5
            max_sleep = 1.0
            trials = 5
            for trial in range(trials):
                try:
                    tts = await find_instance("tts", factory)
                except Exception:
                    if trial == trials - 1:
                        raise
                    logger.warning("Will sleep for %.4f sec", sleep_time)
                    await asyncio.sleep(sleep_time)
                    sleep_time = min(max_sleep, sleep_time * sleep_growth)
                    error = make_ora_error(
                        type="warning",
                        message="Looking for the resources, expect some latency.",
                    )
                    await self.output_queue.put(error)
                else:
                    return tts
            raise AssertionError("Too many unexpected packets.")

        async def _run(tts: TextToSpeech):
            await self._tts_loop(tts, generating_message_i)

        async def _close(tts: TextToSpeech):
            await tts.shutdown()

        return await self.quest_manager.add(Quest("tts", _init, _run, _close))

    async def _tts_loop(self, tts: TextToSpeech, generating_message_i: int):
        # On interruption, we swap the output queue. This will ensure that this worker
        # can never accidentally push to the new queue if it's interrupted.
        output_queue = self.output_queue
        try:
            audio_started = None

            async for message in tts:
                if audio_started is not None:
                    time_since_start = self.audio_received_sec() - audio_started
                    time_received = tts.received_samples / self.input_sample_rate
                    time_received_yielded = (
                        tts.received_samples_yielded / self.input_sample_rate
                    )
                    assert self.input_sample_rate == SAMPLE_RATE
                    self.debug_dict["tts_throughput"] = {
                        "time_received": round(time_received, 2),
                        "time_received_yielded": round(time_received_yielded, 2),
                        "time_since_start": round(time_since_start, 2),
                        "ratio": round(
                            time_received_yielded / (time_since_start + 0.01), 2
                        ),
                    }

                if len(self.chatbot.chat_history) > generating_message_i:
                    break

                if isinstance(message, TTSAudioMessage):
                    t = self.tts_output_stopwatch.stop()
                    if t is not None:
                        self.debug_dict["timing"]["tts_audio"] = t

                    audio = np.array(message.pcm, dtype=np.float32)
                    assert self.output_sample_rate == SAMPLE_RATE

                    await output_queue.put((SAMPLE_RATE, audio))

                    if audio_started is None:
                        audio_started = self.audio_received_sec()
                elif isinstance(message, TTSTextMessage):
                    await output_queue.put(ora.ResponseTextDelta(delta=message.text))
                    await self.add_chat_message_delta(
                        message.text,
                        "assistant",
                        generating_message_i=generating_message_i,
                    )
                else:
                    logger.warning("Got unexpected message from TTS: %s", message.type)

        except websockets.ConnectionClosedError as e:
            logger.error(f"TTS connection closed with an error: {e}")

        # Push some silence to flush the Opus state.
        # Not sure that this is actually needed.
        await output_queue.put(
            (SAMPLE_RATE, np.zeros(SAMPLES_PER_FRAME, dtype=np.float32))
        )

        message = self.chatbot.last_message("assistant")
        if message is None:
            logger.warning("No message to send in TTS shutdown.")
            message = ""

        # It's convenient to have the whole chat history available in the client
        # after the response is done, so send the "gradio update"
        await self.output_queue.put(self.get_gradio_update())
        await self.output_queue.put(ora.ResponseAudioDone())

        # Signal that the turn is over by adding an empty message.
        await self.add_chat_message_delta("", "user")

        await asyncio.sleep(1)
        await self.check_for_bot_goodbye()
        self.waiting_for_user_start_time = self.audio_received_sec()

    async def interrupt_bot(self):
        if self.chatbot.conversation_state() != "bot_speaking":
            raise RuntimeError(
                "Can't interrupt bot when conversation state is "
                f"{self.chatbot.conversation_state()}"
            )

        await self.add_chat_message_delta(INTERRUPTION_CHAR, "assistant")

        if self._clear_queue is not None:
            # Clear any audio queued up by FastRTC's emit().
            # Not sure under what circumstatnces this is None.
            self._clear_queue()
        self.output_queue = asyncio.Queue()  # Clear our own queue too

        # Push some silence to flush the Opus state.
        # Not sure that this is actually needed.
        await self.output_queue.put(
            (SAMPLE_RATE, np.zeros(SAMPLES_PER_FRAME, dtype=np.float32))
        )

        await self.output_queue.put(ora.UnmuteInterruptedByVAD())

        # Reset speech tracking state after interruption
        self.speech_started = False
        self.speech_start_time = None
        # Note: Keep current_speech_item_id for the interrupted segment

        await self.quest_manager.remove("tts")
        await self.quest_manager.remove("llm")

    async def check_for_bot_goodbye(self):
        last_assistant_message = next(
            (
                msg
                for msg in reversed(self.chatbot.chat_history)
                if msg["role"] == "assistant"
            ),
            {"content": ""},
        )["content"]

        # Using function calling would be a more robust solution, but it would make it
        # harder to swap LLMs.
        if last_assistant_message.lower().endswith("bye!"):
            await self.output_queue.put(
                CloseStream("The assistant ended the conversation. Bye!")
            )

    async def detect_long_silence(self):
        """Handle situations where the user doesn't answer for a while."""
        if (
            self.chatbot.conversation_state() == "waiting_for_user"
            and (self.audio_received_sec() - self.waiting_for_user_start_time)
            > USER_SILENCE_TIMEOUT
        ):
            # This will trigger pause detection because it changes the conversation
            # state to "user_speaking".
            # The system prompt has a rule that tells it how to handle the "..."
            # messages.
            silence_duration = self.audio_received_sec() - self.waiting_for_user_start_time
            logger.info(f"Long silence detected: {silence_duration:.1f}s")

            # Emit compliant error event for silence timeout
            error = make_ora_error(
                type="silence_timeout",
                message=f"No user input detected for {silence_duration:.1f}s (timeout: {USER_SILENCE_TIMEOUT}s)",
            )
            await self.output_queue.put(error)
            mt.SILENCE_TIMEOUT_ERRORS.inc()

            await self.add_chat_message_delta(USER_SILENCE_MARKER, "user")

    async def check_backpressure(self):
        """Monitor output queue size and emit errors if backpressure threshold exceeded."""
        current_time = self.audio_received_sec()

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

    # === Client Event Handlers ===

    async def handle_response_create(
        self, event: ora.ResponseCreate
    ) -> None:
        """Handle explicit response.create request from client."""
        # If already generating, cancel current response first
        if self.session_state.has_active_response():
            await self.interrupt_bot()

        # Start new response generation
        await self._generate_response()

    async def handle_response_cancel(self) -> ora.ResponseDone | None:
        """Cancel in-progress response and return ResponseDone event."""
        if not self.session_state.has_active_response():
            return None

        try:
            await self.interrupt_bot()
        except RuntimeError:
            # Bot wasn't speaking - that's OK
            pass

        response = self.session_state.cancel_response()
        if response is not None:
            return ora.ResponseDone(response=response)
        return None

    async def handle_item_create(
        self, event: ora.ConversationItemCreate
    ) -> tuple[ora.Item, ora.ConversationItemCreated]:
        """Create a conversation item and return it with acknowledgement."""
        item_data = event.item
        item = self.session_state.create_item(
            item_type=item_data.get("type", "message"),
            role=item_data.get("role"),
            content=item_data.get("content"),
            previous_item_id=event.previous_item_id,
        )

        # Also add to chatbot history if it's a user or assistant message
        if item.role and item.content:
            text_content = self._extract_text_content(item.content)
            if text_content:
                self.chatbot.chat_history.append({
                    "role": item.role,
                    "content": text_content,
                })

        ack = ora.ConversationItemCreated(
            item=item, previous_item_id=event.previous_item_id
        )
        return item, ack

    async def handle_item_delete(self, item_id: str) -> ora.ConversationItemDeleted:
        """Delete a conversation item and return acknowledgement."""
        self.session_state.delete_item(item_id)
        return ora.ConversationItemDeleted(item_id=item_id)

    def handle_item_retrieve(self, item_id: str) -> ora.ConversationItemRetrieved | None:
        """Retrieve a conversation item."""
        item = self.session_state.get_item(item_id)
        if item is None:
            return None
        return ora.ConversationItemRetrieved(item=item)

    async def handle_item_truncate(
        self, event: ora.ConversationItemTruncate
    ) -> ora.ConversationItemTruncated | None:
        """Truncate audio in a conversation item."""
        success = self.session_state.truncate_item(
            event.item_id, event.content_index, event.audio_end_ms
        )
        if not success:
            return None
        return ora.ConversationItemTruncated(
            item_id=event.item_id,
            content_index=event.content_index,
            audio_end_ms=event.audio_end_ms,
        )

    async def commit_audio_buffer(self) -> tuple[str, str | None]:
        """Commit accumulated audio as a conversation item.

        Returns tuple of (item_id, previous_item_id).
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

    async def update_session(self, session: ora.Session | dict[str, Any]):
        # Handle both Session objects and dict-based configs
        if isinstance(session, dict):
            instructions = session.get("instructions")
            voice = session.get("voice")
            allow_recording = session.get("allow_recording", True)
        else:
            instructions = session.instructions
            voice = session.voice
            allow_recording = session.allow_recording if session.allow_recording is not None else True

        if instructions:
            self.chatbot.set_instructions(instructions)

        if voice:
            self.tts_voice = voice

        if not allow_recording and self.recorder:
            await self.recorder.add_event("client", ora.SessionUpdate(session=session))
            await self.recorder.shutdown(keep_recording=False)
            self.recorder = None
            logger.info("Recording disabled for a session.")
