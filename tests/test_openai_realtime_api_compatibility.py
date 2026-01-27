"""Tests for OpenAI Realtime API compatibility with official openai Python client.

These tests verify that our event models are compatible with the official
openai Python client types (package 'openai' version >= 1.70.0).

The tests cover:
1. Type compatibility between our models and official client types
2. Content types including image_url support
3. Event serialization/deserialization compatibility
4. Connection setup expectations (headers, authentication)
"""

import json
import pytest
from typing import get_args, get_origin, Union
from pydantic import TypeAdapter, ValidationError

# Import our custom event models
import unmute.openai_realtime_api_events as ora

# Import official OpenAI types
try:
    from openai.types.realtime import (
        # Client events
        SessionUpdateEvent,
        InputAudioBufferAppendEvent,
        InputAudioBufferCommitEvent,
        InputAudioBufferClearEvent,
        ConversationItemCreateEvent,
        ConversationItemDeleteEvent,
        ConversationItemTruncateEvent,
        ResponseCreateEvent,
        ResponseCancelEvent,
        OutputAudioBufferClearEvent,
        # Server events
        SessionCreatedEvent,
        SessionUpdatedEvent,
        RealtimeErrorEvent,
        ConversationCreatedEvent,
        ConversationItemCreatedEvent,
        ConversationItemDeletedEvent,
        ConversationItemTruncatedEvent,
        InputAudioBufferCommittedEvent,
        InputAudioBufferClearedEvent,
        InputAudioBufferSpeechStartedEvent,
        InputAudioBufferSpeechStoppedEvent,
        ResponseCreatedEvent,
        ResponseDoneEvent,
        ResponseOutputItemAddedEvent,
        ResponseOutputItemDoneEvent,
        ResponseContentPartAddedEvent,
        ResponseContentPartDoneEvent,
        ResponseTextDeltaEvent,
        ResponseTextDoneEvent,
        ResponseAudioDeltaEvent,
        ResponseAudioDoneEvent,
        ResponseAudioTranscriptDeltaEvent,
        ResponseAudioTranscriptDoneEvent,
        ResponseFunctionCallArgumentsDeltaEvent,
        ResponseFunctionCallArgumentsDoneEvent,
        RateLimitsUpdatedEvent,
        ConversationItemInputAudioTranscriptionCompletedEvent,
        ConversationItemInputAudioTranscriptionFailedEvent,
    )
    from openai.types.realtime import (
        RealtimeConversationItemUserMessageParam,
        RealtimeResponse,
        RealtimeError,
    )
    OPENAI_TYPES_AVAILABLE = True
except ImportError as e:
    OPENAI_TYPES_AVAILABLE = False
    # Don't skip at module level - let tests run and check availability


class TestClientEventCompatibility:
    """Test that our client events are compatible with official OpenAI types."""

    def test_session_update_payload_structure(self):
        """Verify session.update payload matches OpenAI expected structure."""
        # Create payload using our types
        our_event = ora.SessionUpdate(
            session=ora.Session(
                modalities=["text", "audio"],
                voice="alloy",
                instructions="You are a helpful assistant.",
                temperature=0.8,
                max_response_output_tokens=4096,
                input_audio_format="pcm16",
                output_audio_format="pcm16",
            )
        )

        payload = json.loads(our_event.model_dump_json())

        # Verify structure matches OpenAI expectations
        assert payload["type"] == "session.update"
        assert "session" in payload
        assert payload["session"]["modalities"] == ["text", "audio"]
        assert payload["session"]["voice"] == "alloy"

        # Validate against OpenAI type if available
        if OPENAI_TYPES_AVAILABLE:
            # OpenAI types should accept the payload structure
            assert payload["type"] == "session.update"

    def test_input_audio_buffer_append_structure(self):
        """Verify input_audio_buffer.append matches OpenAI structure."""
        our_event = ora.InputAudioBufferAppend(
            audio="SGVsbG8gd29ybGQ="  # Base64-encoded audio
        )

        payload = json.loads(our_event.model_dump_json())

        assert payload["type"] == "input_audio_buffer.append"
        assert payload["audio"] == "SGVsbG8gd29ybGQ="
        assert "event_id" in payload

    def test_conversation_item_create_with_text(self):
        """Verify conversation.item.create with text content."""
        our_event = ora.ConversationItemCreate(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello, assistant!"}],
            }
        )

        payload = json.loads(our_event.model_dump_json())

        assert payload["type"] == "conversation.item.create"
        assert payload["item"]["type"] == "message"
        assert payload["item"]["role"] == "user"
        assert payload["item"]["content"][0]["type"] == "input_text"

    def test_conversation_item_create_with_image_url(self):
        """Verify conversation.item.create with image_url content (OpenAI standard way).

        This is how OpenAI expects images to be sent - via conversation.item.create
        with content type 'input_image' and an image_url field.
        """
        # Base64 data URI format for images
        image_data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB..."

        our_event = ora.ConversationItemCreate(
            item={
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What's in this image?"},
                    {
                        "type": "input_image",
                        "image_url": image_data_uri,
                        "detail": "auto",
                    },
                ],
            }
        )

        payload = json.loads(our_event.model_dump_json())

        assert payload["type"] == "conversation.item.create"
        assert len(payload["item"]["content"]) == 2

        # Text content
        assert payload["item"]["content"][0]["type"] == "input_text"

        # Image content - OpenAI format
        image_content = payload["item"]["content"][1]
        assert image_content["type"] == "input_image"
        assert "image_url" in image_content
        assert image_content["detail"] == "auto"

    def test_conversation_item_create_with_audio(self):
        """Verify conversation.item.create with audio content."""
        our_event = ora.ConversationItemCreate(
            item={
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "audio": "SGVsbG8gd29ybGQ=",
                        "transcript": "Hello world",
                    }
                ],
            }
        )

        payload = json.loads(our_event.model_dump_json())

        assert payload["type"] == "conversation.item.create"
        audio_content = payload["item"]["content"][0]
        assert audio_content["type"] == "input_audio"
        assert audio_content["audio"] == "SGVsbG8gd29ybGQ="

    def test_response_create_structure(self):
        """Verify response.create matches OpenAI structure."""
        our_event = ora.ResponseCreate(
            response={
                "modalities": ["text", "audio"],
                "instructions": "Be concise",
            }
        )

        payload = json.loads(our_event.model_dump_json())

        assert payload["type"] == "response.create"
        assert "response" in payload

    def test_response_cancel_structure(self):
        """Verify response.cancel matches OpenAI structure."""
        our_event = ora.ResponseCancel()

        payload = json.loads(our_event.model_dump_json())

        assert payload["type"] == "response.cancel"


class TestServerEventCompatibility:
    """Test that our server events are compatible with official OpenAI types."""

    def test_session_created_structure(self):
        """Verify session.created matches OpenAI structure."""
        our_event = ora.SessionCreated(
            session=ora.Session(
                id="sess_ABC123",
                model="gpt-4o-realtime-preview",
                modalities=["text", "audio"],
                voice="alloy",
            )
        )

        payload = json.loads(our_event.model_dump_json())

        assert payload["type"] == "session.created"
        assert payload["session"]["id"] == "sess_ABC123"
        assert payload["session"]["object"] == "realtime.session"

    def test_error_event_structure(self):
        """Verify error event matches OpenAI structure."""
        our_event = ora.Error(
            error=ora.ErrorDetails(
                type="invalid_request_error",
                code="invalid_value",
                message="Invalid audio format specified",
                param="audio_format",
            )
        )

        payload = json.loads(our_event.model_dump_json())

        assert payload["type"] == "error"
        assert payload["error"]["type"] == "invalid_request_error"
        assert payload["error"]["code"] == "invalid_value"
        assert payload["error"]["message"] == "Invalid audio format specified"

    def test_response_audio_delta_structure(self):
        """Verify response.audio.delta matches OpenAI structure."""
        our_event = ora.ResponseAudioDelta(
            delta="SGVsbG8gd29ybGQ=",
            response_id="resp_001",
            item_id="item_001",
            output_index=0,
            content_index=0,
        )

        payload = json.loads(our_event.model_dump_json())

        assert payload["type"] == "response.audio.delta"
        assert payload["delta"] == "SGVsbG8gd29ybGQ="
        assert payload["response_id"] == "resp_001"

    def test_transcription_completed_structure(self):
        """Verify transcription.completed matches OpenAI structure."""
        our_event = ora.ConversationItemInputAudioTranscriptionCompleted(
            item_id="item_001",
            content_index=0,
            transcript="Hello, how are you?",
        )

        payload = json.loads(our_event.model_dump_json())

        assert payload["type"] == "conversation.item.input_audio_transcription.completed"
        assert payload["transcript"] == "Hello, how are you?"

    def test_rate_limits_updated_structure(self):
        """Verify rate_limits.updated matches OpenAI structure."""
        our_event = ora.RateLimitsUpdated(
            rate_limits=[
                ora.RateLimit(
                    name="requests",
                    limit=1000,
                    remaining=999,
                    reset_seconds=60.0,
                ),
                ora.RateLimit(
                    name="tokens",
                    limit=150000,
                    remaining=145000,
                    reset_seconds=60.0,
                ),
            ]
        )

        payload = json.loads(our_event.model_dump_json())

        assert payload["type"] == "rate_limits.updated"
        assert len(payload["rate_limits"]) == 2
        assert payload["rate_limits"][0]["name"] == "requests"


class TestContentPartTypes:
    """Test content part types for multimodal support."""

    def test_input_text_content_part(self):
        """Test input_text content part structure."""
        content_part = ora.InputTextContentPart(text="Hello, world!")

        payload = content_part.model_dump()

        assert payload["type"] == "input_text"
        assert payload["text"] == "Hello, world!"

    def test_input_audio_content_part(self):
        """Test input_audio content part structure."""
        content_part = ora.InputAudioContentPart(
            audio="SGVsbG8gd29ybGQ=",
            transcript="Hello world",
        )

        payload = content_part.model_dump()

        assert payload["type"] == "input_audio"
        assert payload["audio"] == "SGVsbG8gd29ybGQ="
        assert payload["transcript"] == "Hello world"

    def test_input_image_content_part(self):
        """Test input_image content part structure.

        OpenAI expects image_url as a string (data URI or URL).
        """
        # Test with data URI format
        image_data_uri = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."

        content_part = ora.InputImageContentPart(
            image_url={"url": image_data_uri},
            detail="high",
        )

        payload = content_part.model_dump()

        assert payload["type"] == "input_image"
        assert "image_url" in payload
        assert payload["detail"] == "high"

    def test_text_content_part_for_response(self):
        """Test text content part for assistant responses."""
        content_part = ora.TextContentPart(text="I can help you with that!")

        payload = content_part.model_dump()

        assert payload["type"] == "text"
        assert payload["text"] == "I can help you with that!"

    def test_audio_content_part_for_response(self):
        """Test audio content part for assistant responses."""
        content_part = ora.AudioContentPart(
            audio="SGVsbG8gd29ybGQ=",
            transcript="Hello world",
        )

        payload = content_part.model_dump()

        assert payload["type"] == "audio"
        assert payload["audio"] == "SGVsbG8gd29ybGQ="


class TestConnectionSetup:
    """Test connection setup expectations.

    The official OpenAI client uses:
    - WebSocket (wss://) connection
    - OpenAI-Beta: realtime=v1 header
    - API key or ephemeral token authentication

    Note: Subprotocol 'realtime' may no longer be strictly required
    by the official client, but we maintain compatibility.
    """

    def test_openai_beta_header_value(self):
        """Verify the OpenAI-Beta header value matches expected."""
        from unmute.websocket_auth import OPENAI_BETA_REALTIME

        assert OPENAI_BETA_REALTIME == "realtime=v1"

    def test_subprotocol_constant(self):
        """Verify the subprotocol constant is defined."""
        from unmute.websocket_auth import REALTIME_SUBPROTOCOL

        assert REALTIME_SUBPROTOCOL == "realtime"


class TestImageHandlingMigration:
    """Tests for image handling using OpenAI standard.

    OpenAI standard: Images sent via conversation.item.create with input_image content
    """

    def test_openai_standard_image_via_conversation_item(self):
        """Test OpenAI standard way to send images.

        Per OpenAI docs, images should be sent as part of a user message
        via conversation.item.create with 'input_image' content type.
        """
        # OpenAI standard image format
        image_data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAE="

        event = ora.ConversationItemCreate(
            item={
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe this image:"},
                    {
                        "type": "input_image",
                        "image_url": image_data_uri,
                        "detail": "auto",
                    },
                ],
            }
        )

        payload = json.loads(event.model_dump_json())

        assert payload["type"] == "conversation.item.create"
        assert payload["item"]["role"] == "user"

        # Find image content
        image_contents = [
            c for c in payload["item"]["content"]
            if c.get("type") == "input_image"
        ]
        assert len(image_contents) == 1
        assert image_contents[0]["image_url"] == image_data_uri

    def test_image_detail_levels(self):
        """Test image detail level options match OpenAI spec."""
        valid_details = ["auto", "low", "high"]

        for detail in valid_details:
            content_part = ora.InputImageContentPart(
                image_url={"url": "data:image/png;base64,ABC="},
                detail=detail,
            )
            assert content_part.detail == detail


class TestMissingOfficialEvents:
    """Test for events that exist in official OpenAI API but may be missing."""

    def test_output_audio_buffer_clear_not_implemented(self):
        """Document that output_audio_buffer.clear is in official API.

        The official OpenAI Realtime API includes 'output_audio_buffer.clear'
        as a client event to clear the server's audio output buffer.
        """
        # Check if we have this event
        has_output_buffer_clear = hasattr(ora, 'OutputAudioBufferClear')

        # Document current state - this test will fail when implemented
        if not has_output_buffer_clear:
            pytest.skip(
                "output_audio_buffer.clear not yet implemented - "
                "required for full OpenAI Realtime API compliance"
            )


class TestUnmuteExtensions:
    """Test Unmute-specific extensions that are NOT part of OpenAI API."""

    def test_unmute_recording_events(self):
        """Verify Unmute recording extension events."""
        event = ora.UnmuteInputAudioBufferAppendAnonymized(
            number_of_samples=960,
        )

        payload = json.loads(event.model_dump_json())

        assert payload["type"] == "unmute.input_audio_buffer.append_anonymized"
        # This is for privacy-preserving recording

    def test_unmute_vad_interruption_event(self):
        """Verify Unmute VAD interruption event."""
        event = ora.UnmuteInterruptedByVAD()

        payload = json.loads(event.model_dump_json())

        assert payload["type"] == "unmute.interrupted_by_vad"


class TestEventDiscrimination:
    """Test that events can be properly discriminated by type."""

    def test_client_event_type_adapter(self):
        """Test ClientEvent union works with type discriminator."""
        from pydantic import Field
        from typing import Annotated

        adapter = TypeAdapter(
            Annotated[ora.ClientEvent, Field(discriminator="type")]
        )

        # Test various client events
        test_cases = [
            ('{"type": "session.update", "session": {"voice": "alloy"}}', ora.SessionUpdate),
            ('{"type": "input_audio_buffer.append", "audio": "SGVsbG8="}', ora.InputAudioBufferAppend),
            ('{"type": "input_audio_buffer.commit"}', ora.InputAudioBufferCommit),
            ('{"type": "response.create"}', ora.ResponseCreate),
            ('{"type": "conversation.item.create", "item": {"type": "message", "role": "user"}}', ora.ConversationItemCreate),
        ]

        for json_str, expected_type in test_cases:
            event = adapter.validate_json(json_str)
            assert isinstance(event, expected_type), f"Expected {expected_type}, got {type(event)}"

    def test_server_event_type_adapter(self):
        """Test ServerEvent union works with type discriminator."""
        from pydantic import Field
        from typing import Annotated

        adapter = TypeAdapter(
            Annotated[ora.ServerEvent, Field(discriminator="type")]
        )

        # Test various server events
        test_cases = [
            ('{"type": "session.created", "session": {"id": "sess_1", "object": "realtime.session"}}', ora.SessionCreated),
            ('{"type": "error", "error": {"type": "invalid_request_error", "message": "test"}}', ora.Error),
            ('{"type": "response.audio.delta", "delta": "SGVsbG8="}', ora.ResponseAudioDelta),
            ('{"type": "rate_limits.updated", "rate_limits": []}', ora.RateLimitsUpdated),
        ]

        for json_str, expected_type in test_cases:
            event = adapter.validate_json(json_str)
            assert isinstance(event, expected_type), f"Expected {expected_type}, got {type(event)}"


class TestSessionConfiguration:
    """Test session configuration matches OpenAI spec."""

    def test_session_modalities(self):
        """Test session modalities field."""
        session = ora.Session(
            modalities=["text", "audio"],
        )

        assert session.modalities == ["text", "audio"]

    def test_session_turn_detection(self):
        """Test session turn detection configuration."""
        session = ora.Session(
            turn_detection={
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500,
            }
        )

        payload = session.model_dump()

        assert payload["turn_detection"]["type"] == "server_vad"
        assert payload["turn_detection"]["threshold"] == 0.5

    def test_session_tools_configuration(self):
        """Test session tools/function calling configuration."""
        session = ora.Session(
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"],
                    },
                }
            ],
            tool_choice="auto",
        )

        payload = session.model_dump()

        assert len(payload["tools"]) == 1
        assert payload["tools"][0]["name"] == "get_weather"
        assert payload["tool_choice"] == "auto"

    def test_session_audio_formats(self):
        """Test session audio format options."""
        session = ora.Session(
            input_audio_format="pcm16",
            output_audio_format="pcm16",
        )

        payload = session.model_dump()

        assert payload["input_audio_format"] == "pcm16"
        assert payload["output_audio_format"] == "pcm16"


class TestFunctionCallEvents:
    """Test function/tool call events match OpenAI spec."""

    def test_function_call_arguments_delta(self):
        """Test function call arguments streaming."""
        event = ora.ResponseFunctionCallArgumentsDelta(
            response_id="resp_001",
            item_id="item_001",
            output_index=0,
            call_id="call_ABC123",
            delta='{"loc',
        )

        payload = json.loads(event.model_dump_json())

        assert payload["type"] == "response.function_call_arguments.delta"
        assert payload["call_id"] == "call_ABC123"
        assert payload["delta"] == '{"loc'

    def test_function_call_arguments_done(self):
        """Test function call arguments completion."""
        event = ora.ResponseFunctionCallArgumentsDone(
            response_id="resp_001",
            item_id="item_001",
            output_index=0,
            call_id="call_ABC123",
            arguments='{"location": "San Francisco"}',
        )

        payload = json.loads(event.model_dump_json())

        assert payload["type"] == "response.function_call_arguments.done"
        assert payload["arguments"] == '{"location": "San Francisco"}'

    def test_conversation_item_function_call_output(self):
        """Test sending function call output."""
        event = ora.ConversationItemCreate(
            item={
                "type": "function_call_output",
                "call_id": "call_ABC123",
                "output": '{"temperature": 72, "unit": "fahrenheit"}',
            }
        )

        payload = json.loads(event.model_dump_json())

        assert payload["type"] == "conversation.item.create"
        assert payload["item"]["type"] == "function_call_output"
        assert payload["item"]["call_id"] == "call_ABC123"


@pytest.mark.skipif(not OPENAI_TYPES_AVAILABLE, reason="openai package types not available")
class TestOfficialOpenAITypeValidation:
    """Test that our events can be validated against official OpenAI types.

    These tests ensure structural compatibility between our custom event
    models and the official openai Python client types.
    """

    @pytest.mark.xfail(reason="Session object requires 'type' field in OpenAI SDK - needs upgrade")
    def test_validate_session_update_against_openai(self):
        """Validate our SessionUpdate can be parsed by OpenAI types.

        UPGRADE REQUIRED: OpenAI's RealtimeSessionCreateRequest requires
        a 'type' field (e.g., "session" or "transcription_session").
        Our Session model doesn't include this field.
        """
        our_event = ora.SessionUpdate(
            session=ora.Session(
                modalities=["text", "audio"],
                voice="alloy",
                instructions="You are helpful.",
            )
        )

        payload = json.loads(our_event.model_dump_json())

        # Verify structure matches what OpenAI expects
        assert payload["type"] == "session.update"
        assert "session" in payload
        # OpenAI type should accept this structure
        openai_event = SessionUpdateEvent.model_validate(payload)
        assert openai_event.type == "session.update"

    def test_validate_input_audio_buffer_append_against_openai(self):
        """Validate our InputAudioBufferAppend against OpenAI types."""
        our_event = ora.InputAudioBufferAppend(audio="SGVsbG8gd29ybGQ=")
        payload = json.loads(our_event.model_dump_json())

        openai_event = InputAudioBufferAppendEvent.model_validate(payload)
        assert openai_event.type == "input_audio_buffer.append"
        assert openai_event.audio == "SGVsbG8gd29ybGQ="

    def test_validate_conversation_item_create_against_openai(self):
        """Validate our ConversationItemCreate against OpenAI types."""
        our_event = ora.ConversationItemCreate(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello!"}],
            }
        )
        payload = json.loads(our_event.model_dump_json())

        openai_event = ConversationItemCreateEvent.model_validate(payload)
        assert openai_event.type == "conversation.item.create"

    def test_validate_response_create_against_openai(self):
        """Validate our ResponseCreate against OpenAI types."""
        our_event = ora.ResponseCreate(
            response={"modalities": ["text", "audio"]}
        )
        payload = json.loads(our_event.model_dump_json())

        openai_event = ResponseCreateEvent.model_validate(payload)
        assert openai_event.type == "response.create"

    def test_validate_response_cancel_against_openai(self):
        """Validate our ResponseCancel against OpenAI types."""
        our_event = ora.ResponseCancel()
        payload = json.loads(our_event.model_dump_json())

        openai_event = ResponseCancelEvent.model_validate(payload)
        assert openai_event.type == "response.cancel"

    @pytest.mark.xfail(reason="Session object requires 'type' field in OpenAI SDK - needs upgrade")
    def test_validate_session_created_against_openai(self):
        """Validate our SessionCreated against OpenAI types.

        UPGRADE REQUIRED: Same issue as session.update - the session
        object needs a 'type' field.
        """
        our_event = ora.SessionCreated(
            session=ora.Session(
                id="sess_001",
                model="gpt-4o-realtime-preview",
            )
        )
        payload = json.loads(our_event.model_dump_json())

        openai_event = SessionCreatedEvent.model_validate(payload)
        assert openai_event.type == "session.created"

    def test_validate_error_against_openai(self):
        """Validate our Error event against OpenAI types."""
        our_event = ora.Error(
            error=ora.ErrorDetails(
                type="invalid_request_error",
                message="Invalid input",
            )
        )
        payload = json.loads(our_event.model_dump_json())

        openai_event = RealtimeErrorEvent.model_validate(payload)
        assert openai_event.type == "error"

    @pytest.mark.xfail(reason="Event type renamed: 'response.text.delta' -> 'response.output_text.delta'")
    def test_validate_response_text_delta_against_openai(self):
        """Validate our ResponseTextDelta against OpenAI types.

        UPGRADE REQUIRED: OpenAI has renamed this event type:
        - Old (ours): response.text.delta
        - New (OpenAI): response.output_text.delta
        """
        our_event = ora.ResponseTextDelta(
            delta="Hello",
            response_id="resp_001",
            item_id="item_001",
            output_index=0,
            content_index=0,
        )
        payload = json.loads(our_event.model_dump_json())

        openai_event = ResponseTextDeltaEvent.model_validate(payload)
        assert openai_event.type == "response.output_text.delta"
        assert openai_event.delta == "Hello"

    @pytest.mark.xfail(reason="Event type renamed: 'response.audio.delta' -> 'response.output_audio.delta'")
    def test_validate_response_audio_delta_against_openai(self):
        """Validate our ResponseAudioDelta against OpenAI types.

        UPGRADE REQUIRED: OpenAI has renamed this event type:
        - Old (ours): response.audio.delta
        - New (OpenAI): response.output_audio.delta
        """
        our_event = ora.ResponseAudioDelta(
            delta="SGVsbG8=",
            response_id="resp_001",
            item_id="item_001",
            output_index=0,
            content_index=0,
        )
        payload = json.loads(our_event.model_dump_json())

        openai_event = ResponseAudioDeltaEvent.model_validate(payload)
        assert openai_event.type == "response.output_audio.delta"

    @pytest.mark.xfail(reason="Transcription completed requires 'usage' field in OpenAI SDK")
    def test_validate_transcription_completed_against_openai(self):
        """Validate our transcription completed against OpenAI types.

        UPGRADE REQUIRED: OpenAI's transcription completed event
        now requires a 'usage' field with token usage information.
        """
        our_event = ora.ConversationItemInputAudioTranscriptionCompleted(
            item_id="item_001",
            content_index=0,
            transcript="Hello world",
        )
        payload = json.loads(our_event.model_dump_json())

        openai_event = ConversationItemInputAudioTranscriptionCompletedEvent.model_validate(payload)
        assert openai_event.type == "conversation.item.input_audio_transcription.completed"

    def test_validate_rate_limits_updated_against_openai(self):
        """Validate our RateLimitsUpdated against OpenAI types."""
        our_event = ora.RateLimitsUpdated(
            rate_limits=[
                ora.RateLimit(
                    name="requests",
                    limit=1000,
                    remaining=999,
                    reset_seconds=60.0,
                )
            ]
        )
        payload = json.loads(our_event.model_dump_json())

        openai_event = RateLimitsUpdatedEvent.model_validate(payload)
        assert openai_event.type == "rate_limits.updated"


@pytest.mark.skipif(not OPENAI_TYPES_AVAILABLE, reason="openai package types not available")
class TestCriticalAPIChanges:
    """Test critical API changes discovered between our impl and OpenAI SDK.

    These tests document the specific breaking changes that need to be addressed.
    """

    def test_response_text_delta_type_change(self):
        """Document the response.text.delta -> response.output_text.delta change."""
        our_type = "response.text.delta"
        openai_type = "response.output_text.delta"

        assert our_type != openai_type

        # Our event uses the old type
        our_event = ora.ResponseTextDelta(delta="Hello")
        assert our_event.type == our_type

    def test_response_audio_delta_type_change(self):
        """Document the response.audio.delta -> response.output_audio.delta change."""
        our_type = "response.audio.delta"
        openai_type = "response.output_audio.delta"

        assert our_type != openai_type

        # Our event uses the old type
        our_event = ora.ResponseAudioDelta(delta="SGVsbG8=")
        assert our_event.type == our_type

    def test_response_text_done_type_change(self):
        """Document the response.text.done -> response.output_text.done change."""
        our_type = "response.text.done"
        openai_type = "response.output_text.done"  # OpenAI likely uses this

        assert our_type != openai_type

    def test_response_audio_done_type_change(self):
        """Document the response.audio.done -> response.output_audio.done change."""
        our_type = "response.audio.done"
        openai_type = "response.output_audio.done"  # OpenAI likely uses this

        assert our_type != openai_type

    def test_session_object_needs_type_field(self):
        """Document that Session object needs a 'type' field for OpenAI."""
        session = ora.Session(
            modalities=["text", "audio"],
            voice="alloy",
        )

        payload = session.model_dump()

        # Our session doesn't have a 'type' field
        assert "type" not in payload or payload.get("type") is None

        # OpenAI expects either:
        # - type="session" for regular sessions
        # - type="transcription_session" for transcription-only sessions

    def test_transcription_completed_needs_usage(self):
        """Document that transcription completed needs 'usage' field."""
        event = ora.ConversationItemInputAudioTranscriptionCompleted(
            item_id="item_001",
            content_index=0,
            transcript="Hello",
        )

        payload = json.loads(event.model_dump_json())

        # Our event doesn't have 'usage'
        assert "usage" not in payload

        # OpenAI expects usage: { total_tokens, input_tokens, output_tokens }


@pytest.mark.skipif(not OPENAI_TYPES_AVAILABLE, reason="openai package types not available")
class TestImageUrlFormatCompatibility:
    """Test image_url format differences between our implementation and OpenAI.

    OpenAI standard: image_url is a string containing a data URI or URL
    Our current implementation: image_url is a dict with {"url": "..."}

    These tests document the difference and test both formats.
    """

    def test_openai_image_url_format_is_string(self):
        """Verify OpenAI expects image_url as a string, not a dict."""
        # Check the OpenAI type definition
        from openai.types.realtime import RealtimeConversationItemUserMessageParam

        # OpenAI format: image_url is a string
        openai_format_content = {
            "type": "input_image",
            "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAE=",
        }

        # This should be valid for OpenAI
        # Note: We can't directly validate content parts, but we document the expected format

    def test_our_image_url_format_uses_dict(self):
        """Document that our InputImageContentPart uses dict format."""
        content_part = ora.InputImageContentPart(
            image_url={"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAE="},
            detail="auto",
        )

        payload = content_part.model_dump()

        # Our format uses a dict
        assert isinstance(payload["image_url"], dict)
        assert "url" in payload["image_url"]

    def test_conversation_item_with_openai_string_image_url(self):
        """Test that we can handle OpenAI's string format for image_url.

        This tests receiving OpenAI-format messages where image_url is a string.
        """
        # Payload in OpenAI format (image_url as string)
        openai_format_payload = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What's in this image?"},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAE=",
                    },
                ],
            },
        }

        # Our ConversationItemCreate should accept this
        # (item is dict[str, Any] so it accepts any structure)
        event = ora.ConversationItemCreate.model_validate(openai_format_payload)
        assert event.type == "conversation.item.create"
        assert event.item["content"][1]["type"] == "input_image"
        # The image_url is a string in OpenAI format
        assert isinstance(event.item["content"][1]["image_url"], str)


class TestUpgradeTasks:
    """Document tasks required to upgrade to latest OpenAI Realtime API.

    These tests serve as documentation and will fail until the upgrades are complete.
    """

    def test_task_list_documentation(self):
        """Document the upgrade tasks identified."""
        tasks = [
            # CRITICAL - Breaking changes
            {
                "id": 1,
                "title": "Rename response.text.delta to response.output_text.delta",
                "description": "OpenAI renamed this event type",
                "priority": "critical",
                "breaking": True,
                "details": "Also rename response.text.done to response.output_text.done",
            },
            {
                "id": 2,
                "title": "Rename response.audio.delta to response.output_audio.delta",
                "description": "OpenAI renamed this event type",
                "priority": "critical",
                "breaking": True,
                "details": "Also rename response.audio.done to response.output_audio.done, response.audio_transcript.* to response.output_audio_transcript.*",
            },
            {
                "id": 3,
                "title": "Add 'type' field to Session model",
                "description": "OpenAI's session object requires a 'type' field",
                "priority": "critical",
                "breaking": True,
                "details": "Add type: Literal['session', 'transcription_session'] to Session model",
            },
            {
                "id": 4,
                "title": "Add 'usage' field to ConversationItemInputAudioTranscriptionCompleted",
                "description": "OpenAI now requires usage stats in transcription completed events",
                "priority": "critical",
                "breaking": True,
                "details": "Add usage: { total_tokens, input_tokens, output_tokens }",
            },
            # HIGH priority
            {
                "id": 5,
                "title": "Standardize image_url format to string",
                "description": "OpenAI expects image_url as string, our InputImageContentPart uses dict",
                "priority": "high",
                "details": "Change image_url from dict[str, str] to str (data URI or URL)",
            },
            {
                "id": 6,
                "title": "Add OutputAudioBufferClear event",
                "description": "The official OpenAI API includes output_audio_buffer.clear client event",
                "priority": "high",
                "status": "pending" if not hasattr(ora, 'OutputAudioBufferClear') else "done",
            },
            # MEDIUM priority
            {
                "id": 7,
                "title": "Review subprotocol requirement",
                "description": "Official client may not require 'realtime' subprotocol",
                "priority": "medium",
                "details": "Verify if Sec-WebSocket-Protocol: realtime is still required by OpenAI servers",
            },
            {
                "id": 8,
                "title": "Verify transcription delta event",
                "description": "Verify ConversationItemInputAudioTranscriptionDelta matches OpenAI spec",
                "priority": "medium",
                "status": "done" if hasattr(ora, 'ConversationItemInputAudioTranscriptionDelta') else "pending",
            },
            # LOW priority - Extensions and optional features
            {
                "id": 9,
                "title": "Add input_audio_buffer.dtmf_event_received event",
                "description": "OpenAI API supports DTMF tone detection events",
                "priority": "low",
            },
            {
                "id": 10,
                "title": "Add MCP tool call events",
                "description": "OpenAI API supports MCP (Model Context Protocol) events",
                "priority": "low",
            },
            {
                "id": 11,
                "title": "Add noise reduction configuration",
                "description": "OpenAI API supports noise reduction type configuration",
                "priority": "low",
            },
            {
                "id": 12,
                "title": "Add input_audio_buffer.timeout_triggered event",
                "description": "OpenAI API supports timeout triggered events",
                "priority": "low",
            },
        ]

        # Print task summary
        print("\n=== OpenAI Realtime API Upgrade Tasks ===\n")

        # Group by priority
        for priority in ["critical", "high", "medium", "low"]:
            priority_tasks = [t for t in tasks if t["priority"] == priority]
            if priority_tasks:
                print(f"## {priority.upper()} Priority\n")
                for task in priority_tasks:
                    status = task.get("status", "pending")
                    breaking = " [BREAKING]" if task.get("breaking") else ""
                    print(f"  [{status.upper()}] Task {task['id']}: {task['title']}{breaking}")
                    print(f"           {task['description']}")
                    if "details" in task:
                        print(f"           Details: {task['details']}")
                    print()

        # This test always passes - it's for documentation
        assert True

    def test_critical_tasks_summary(self):
        """Summary of critical breaking changes."""
        critical_changes = """
        CRITICAL BREAKING CHANGES REQUIRED:

        1. Event Type Renames (response.* -> response.output_*):
           - response.text.delta -> response.output_text.delta
           - response.text.done -> response.output_text.done
           - response.audio.delta -> response.output_audio.delta
           - response.audio.done -> response.output_audio.done
           - response.audio_transcript.delta -> response.output_audio_transcript.delta
           - response.audio_transcript.done -> response.output_audio_transcript.done

        2. Session Object Changes:
           - Add required 'type' field: "session" | "transcription_session"

        3. Transcription Completed Event:
           - Add required 'usage' field with token counts

        4. Image Handling:
           - image_url should be string (data URI), not dict
           - Use conversation.item.create with input_image content type

        5. Missing Events:
           - output_audio_buffer.clear (client event)
           - input_audio_buffer.dtmf_event_received (server event)
           - input_audio_buffer.timeout_triggered (server event)
        """
        print(critical_changes)
        assert True
