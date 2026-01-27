"""See OpenAI's docs: https://platform.openai.com/docs/api-reference/realtime

https://platform.openai.com/docs/api-reference/realtime-client-events
https://platform.openai.com/docs/api-reference/realtime-server-events
"""

import random
from typing import (
    Any,
    Generic,
    Literal,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel, Field, model_validator

from unmute.llm.system_prompt import Instructions

T = TypeVar("T", bound=str)


def random_id(prefix: str) -> str:
    """e.g. event_BJhGUIswO2u7vA2Cxw3Jy"""
    n_characters = 21
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return prefix + "_" + "".join(random.choices(alphabet, k=n_characters))


class BaseEvent(BaseModel, Generic[T]):
    type: T = None  # type: ignore - will be set by validator below
    event_id: str = Field(default_factory=lambda: random_id("event"))

    @model_validator(mode="after")
    def set_type_from_generic(self) -> "BaseEvent":
        if type(self) == BaseEvent:  # noqa - we want to use type() == and not isinstance
            raise ValueError("Cannot instantiate BaseEvent directly")

        # e.g. Literal["session.update"]
        type_argument = self.__class__.model_fields["type"].annotation

        if get_origin(type_argument) is not Literal:
            # I don't know how to enforce this in the type system directly
            raise ValueError("Type argument is not a Literal")

        self.type = get_args(type_argument)[0]  # e.g. "session.update"

        return self


class ErrorDetails(BaseModel):
    type: str
    code: str | None = None
    message: str
    param: str | None = None
    # ours, not part of the OpenAI API:
    details: object | None = None


class Error(BaseEvent[Literal["error"]]):
    error: ErrorDetails


class Session(BaseModel):
    """Session configuration object."""
    id: str | None = None
    object: Literal["realtime.session"] = "realtime.session"
    model: str | None = None
    modalities: list[str] | None = None
    # The "Instructions" object is an Unmute extension
    instructions: Instructions | str | None = None
    voice: str | None = None
    input_audio_format: str | None = None
    output_audio_format: str | None = None
    input_audio_transcription: dict[str, Any] | None = None
    turn_detection: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | None = None
    temperature: float | None = None
    max_response_output_tokens: int | str | None = None
    # Unmute extensions
    allow_recording: bool | None = None


class SessionUpdate(BaseEvent[Literal["session.update"]]):
    """Update session configuration (client event)."""
    session: Session | dict[str, Any]


class SessionCreated(BaseEvent[Literal["session.created"]]):
    """First event after connection is established (server event)."""
    session: Session


class SessionUpdated(BaseEvent[Literal["session.updated"]]):
    """Session configuration was updated (server event)."""
    session: Session


class InputAudioBufferAppend(BaseEvent[Literal["input_audio_buffer.append"]]):
    audio: str  # Base64-encoded Opus data


class InputAudioBufferCommit(BaseEvent[Literal["input_audio_buffer.commit"]]):
    """Commit the input audio buffer to create a user message."""
    pass


class InputAudioBufferClear(BaseEvent[Literal["input_audio_buffer.clear"]]):
    """Clear the input audio buffer."""
    pass


class UnmuteInputAudioBufferAppendAnonymized(
    BaseEvent[Literal["unmute.input_audio_buffer.append_anonymized"]]
):
    """
    For recording, an anonymous version of InputAudioBufferAppend that only says
    how many samples were appended, not the actual audio data.
    """

    number_of_samples: int


class InputAudioBufferCommitted(
    BaseEvent[Literal["input_audio_buffer.committed"]]
):
    """Audio buffer was committed (server event)."""
    item_id: str
    previous_item_id: str | None = None


class InputAudioBufferCleared(BaseEvent[Literal["input_audio_buffer.cleared"]]):
    """Input buffer was cleared (server event)."""
    pass


class InputAudioBufferSpeechStarted(
    BaseEvent[Literal["input_audio_buffer.speech_started"]]
):
    """Speech started according to the STT.

    Note this is not symmetrical with `InputAudioBufferSpeechStopped` because it's based
    on the STT and not the VAD signal. This is because sometimes the VAD will think
    there is speech but then nothing will end up getting transcribed. If we were using
    the VAD for both events we might get a start event without a stop event.
    For VAD interruptions, see `UnmuteInterruptedByVAD`.
    """
    item_id: str | None = None
    audio_start_ms: int | None = None


class InputAudioBufferSpeechStopped(
    BaseEvent[Literal["input_audio_buffer.speech_stopped"]]
):
    """A pause was detected by the VAD."""
    item_id: str | None = None
    audio_end_ms: int | None = None




class UsageStats(BaseModel):
    """Token usage statistics."""
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class ContentPart(BaseModel):
    """Base content part for conversation items and responses."""
    type: str
    # Subclasses will define specific fields


class TextContentPart(ContentPart):
    """Text content part."""
    type: Literal["text"] = "text"
    text: str


class AudioContentPart(ContentPart):
    """Audio content part."""
    type: Literal["audio"] = "audio"
    audio: str | None = None  # Base64-encoded audio
    transcript: str | None = None


class InputTextContentPart(ContentPart):
    """User text input content part."""
    type: Literal["input_text"] = "input_text"
    text: str


class InputAudioContentPart(ContentPart):
    """User audio input content part."""
    type: Literal["input_audio"] = "input_audio"
    audio: str | None = None  # Base64-encoded audio
    transcript: str | None = None


class ImageContentPart(ContentPart):
    """Image content part for assistant responses."""
    type: Literal["image"] = "image"
    image_url: dict[str, str] | None = None  # {"url": "data:image/..."}
    detail: Literal["auto", "low", "high"] | None = None


class InputImageContentPart(ContentPart):
    """User image input content part."""
    type: Literal["input_image"] = "input_image"
    image_url: dict[str, str] | None = None  # {"url": "data:image/..."}
    detail: Literal["auto", "low", "high"] | None = None


class MetadataContentPart(ContentPart):
    """Metadata content part for sensor readings and structured context."""
    type: Literal["metadata"] = "metadata"
    key: str
    value: Any
    timestamp: float | None = None  # Unix timestamp


class FunctionCall(BaseModel):
    """Function call details."""
    call_id: str
    name: str
    arguments: str  # JSON string


class FunctionCallOutput(BaseModel):
    """Function call output details."""
    call_id: str
    output: str


class Item(BaseModel):
    """Conversation or response item."""
    id: str
    object: Literal["realtime.item"] = "realtime.item"
    type: Literal["message", "function_call", "function_call_output"]
    status: Literal["in_progress", "completed", "incomplete"] | None = None
    role: Literal["system", "user", "assistant"] | None = None
    content: list[dict[str, Any]] | None = None
    call_id: str | None = None
    name: str | None = None
    arguments: str | None = None
    output: str | None = None


class Response(BaseModel):
    """Response object."""
    id: str | None = None
    object: Literal["realtime.response"] = "realtime.response"
    # We currently only use in_progress
    status: Literal["in_progress", "completed", "cancelled", "failed", "incomplete"]
    status_details: dict[str, Any] | None = None
    output: list[Item] = Field(default_factory=list)
    usage: UsageStats | None = None
    # Unmute extensions
    voice: str | None = None
    chat_history: list[dict[str, Any]] = Field(default_factory=list)


class ResponseCreate(BaseEvent[Literal["response.create"]]):
    """Instruct the server to create a response via inference."""
    response: dict[str, Any] | None = None  # Optional configuration


class ResponseCancel(BaseEvent[Literal["response.cancel"]]):
    """Cancel an in-progress response."""
    pass


class ResponseCreated(BaseEvent[Literal["response.created"]]):
    response: Response


class ResponseDone(BaseEvent[Literal["response.done"]]):
    """Final response event with complete information."""
    response: Response


class ResponseOutputItemAdded(BaseEvent[Literal["response.output_item.added"]]):
    """New output item was added during response generation (server event)."""
    response_id: str
    output_index: int
    item: Item


class ResponseOutputItemDone(BaseEvent[Literal["response.output_item.done"]]):
    """Output item finished streaming (server event)."""
    response_id: str
    output_index: int
    item: Item


class ResponseContentPartAdded(BaseEvent[Literal["response.content_part.added"]]):
    """New content part was added (server event)."""
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    part: dict[str, Any]  # ContentPart


class ResponseContentPartDone(BaseEvent[Literal["response.content_part.done"]]):
    """Content part finished (server event)."""
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    part: dict[str, Any]  # ContentPart


class ResponseTextDelta(BaseEvent[Literal["response.text.delta"]]):
    """Incremental text response chunk (server event)."""
    delta: str
    response_id: str | None = None
    item_id: str | None = None
    output_index: int | None = None
    content_index: int | None = None


class ResponseTextDone(BaseEvent[Literal["response.text.done"]]):
    """Final text response (server event)."""
    text: str
    response_id: str | None = None
    item_id: str | None = None
    output_index: int | None = None
    content_index: int | None = None


class ResponseAudioTranscriptDelta(
    BaseEvent[Literal["response.audio_transcript.delta"]]
):
    """Incremental audio transcript chunk (server event)."""
    delta: str
    response_id: str | None = None
    item_id: str | None = None
    output_index: int | None = None
    content_index: int | None = None


class ResponseAudioTranscriptDone(
    BaseEvent[Literal["response.audio_transcript.done"]]
):
    """Final audio transcript (server event)."""
    transcript: str
    response_id: str | None = None
    item_id: str | None = None
    output_index: int | None = None
    content_index: int | None = None


class ResponseAudioDelta(BaseEvent[Literal["response.audio.delta"]]):
    """Incremental audio response chunk (server event)."""
    delta: str  # Base64-encoded Opus audio data
    response_id: str | None = None
    item_id: str | None = None
    output_index: int | None = None
    content_index: int | None = None


class ResponseAudioDone(BaseEvent[Literal["response.audio.done"]]):
    """Audio response finished (server event)."""
    response_id: str | None = None
    item_id: str | None = None
    output_index: int | None = None
    content_index: int | None = None


class TranscriptLogprob(BaseModel):
    bytes: bytes
    logprob: float
    token: str


class ResponseFunctionCallArgumentsDelta(
    BaseEvent[Literal["response.function_call_arguments.delta"]]
):
    """Incremental function call arguments chunk (server event)."""
    response_id: str
    item_id: str
    output_index: int
    call_id: str
    delta: str


class ResponseFunctionCallArgumentsDone(
    BaseEvent[Literal["response.function_call_arguments.done"]]
):
    """Final function call arguments (server event)."""
    response_id: str
    item_id: str
    output_index: int
    call_id: str
    arguments: str


class RateLimit(BaseModel):
    """Rate limit information."""
    name: str
    limit: int
    remaining: int
    reset_seconds: float


class RateLimitsUpdated(BaseEvent[Literal["rate_limits.updated"]]):
    """Rate limits were updated (server event)."""
    rate_limits: list[RateLimit]


class Conversation(BaseModel):
    """Conversation object."""
    id: str
    object: Literal["realtime.conversation"] = "realtime.conversation"


class ConversationCreated(BaseEvent[Literal["conversation.created"]]):
    """Conversation was created (server event)."""
    conversation: Conversation


class ConversationItemInputAudioTranscriptionDelta(
    BaseEvent[Literal["conversation.item.input_audio_transcription.delta"]]
):
    """Incremental transcription of user audio (server event)."""
    item_id: str
    content_index: int
    delta: str
    start_time: float | None = None  # Unmute extension


class ConversationItemInputAudioTranscriptionCompleted(
    BaseEvent[Literal["conversation.item.input_audio_transcription.completed"]]
):
    """Final transcription of user audio (server event)."""
    item_id: str
    content_index: int
    transcript: str


class ConversationItemInputAudioTranscriptionFailed(
    BaseEvent[Literal["conversation.item.input_audio_transcription.failed"]]
):
    """Transcription failed (server event)."""
    item_id: str
    content_index: int
    error: ErrorDetails


class UnmuteAdditionalOutputs(BaseEvent[Literal["unmute.additional_outputs"]]):
    args: Any


class UnmuteResponseTextDeltaReady(
    BaseEvent[Literal["unmute.response.text.delta.ready"]]
):
    delta: str


class UnmuteResponseAudioDeltaReady(
    BaseEvent[Literal["unmute.response.audio.delta.ready"]]
):
    number_of_samples: int


class UnmuteInterruptedByVAD(BaseEvent[Literal["unmute.interrupted_by_vad"]]):
    """The VAD interrupted the response generation."""


# Server events (from OpenAI to client)

# Conversation item events
class ConversationItemCreate(BaseEvent[Literal["conversation.item.create"]]):
    """Add a new item to the conversation (client event)."""
    item: dict
    previous_item_id: str | None = None


class ConversationItemCreated(BaseEvent[Literal["conversation.item.created"]]):
    """Item was added to conversation (server event)."""
    item: Item
    previous_item_id: str | None = None


class ConversationItemDelete(BaseEvent[Literal["conversation.item.delete"]]):
    """Remove an item from conversation history (client event)."""
    item_id: str


class ConversationItemDeleted(BaseEvent[Literal["conversation.item.deleted"]]):
    """Item was removed from conversation (server event)."""
    item_id: str


class ConversationItemRetrieve(BaseEvent[Literal["conversation.item.retrieve"]]):
    """Retrieve a specific item from conversation history (client event)."""
    item_id: str


class ConversationItemRetrieved(BaseEvent[Literal["conversation.item.retrieved"]]):
    """Item retrieved from conversation (server event)."""
    item: Item


class ConversationItemTruncate(BaseEvent[Literal["conversation.item.truncate"]]):
    """Truncate assistant message audio mid-stream (client event)."""
    item_id: str
    content_index: int
    audio_end_ms: int


class ConversationItemTruncated(BaseEvent[Literal["conversation.item.truncated"]]):
    """Item was truncated (server event)."""
    item_id: str
    content_index: int
    audio_end_ms: int



ServerEvent = Union[
    # Session events
    SessionCreated,
    SessionUpdated,
    # Error events
    Error,
    # Conversation events
    ConversationCreated,
    ConversationItemCreated,
    ConversationItemDeleted,
    ConversationItemRetrieved,
    ConversationItemTruncated,
    # Input audio buffer events
    InputAudioBufferCommitted,
    InputAudioBufferCleared,
    InputAudioBufferSpeechStarted,
    InputAudioBufferSpeechStopped,
    # Transcription events
    ConversationItemInputAudioTranscriptionDelta,
    ConversationItemInputAudioTranscriptionCompleted,
    ConversationItemInputAudioTranscriptionFailed,
    # Response events
    ResponseCreated,
    ResponseDone,
    ResponseOutputItemAdded,
    ResponseOutputItemDone,
    ResponseContentPartAdded,
    ResponseContentPartDone,
    # Response text events
    ResponseTextDelta,
    ResponseTextDone,
    # Response audio events
    ResponseAudioDelta,
    ResponseAudioDone,
    ResponseAudioTranscriptDelta,
    ResponseAudioTranscriptDone,
    # Function call events
    ResponseFunctionCallArgumentsDelta,
    ResponseFunctionCallArgumentsDone,
    # Rate limits
    RateLimitsUpdated,
    # Unmute extensions
    UnmuteAdditionalOutputs,
    UnmuteResponseTextDeltaReady,
    UnmuteResponseAudioDeltaReady,
    UnmuteInterruptedByVAD,
]

# Client events (from client to OpenAI)
ClientEvent = Union[
    # Session events
    SessionUpdate,
    # Input audio buffer events
    InputAudioBufferAppend,
    InputAudioBufferCommit,
    InputAudioBufferClear,
    # Conversation item events
    ConversationItemCreate,
    ConversationItemDelete,
    ConversationItemRetrieve,
    ConversationItemTruncate,
    # Response events
    ResponseCreate,
    ResponseCancel,
    # Unmute extensions (used internally for recording)
    UnmuteInputAudioBufferAppendAnonymized,
    UnmuteAdditionalOutputs,
]

Event = ClientEvent | ServerEvent
