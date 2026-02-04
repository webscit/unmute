"""Tests for OpenAI Realtime API event models.

Tests that all event models can validate JSON payloads and round-trip through
.model_validate_json() and .model_dump().
"""

import json
import pytest
from pydantic import ValidationError

import unmute.openai_realtime_api_events as ora


class TestClientEvents:
    """Test all client events (client -> server)."""

    def test_session_update(self):
        """Test session.update event."""
        payload = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "voice": "alloy",
                "instructions": "You are a helpful assistant.",
                "temperature": 0.8,
                "max_response_output_tokens": 4096,
            },
        }
        event = ora.SessionUpdate.model_validate(payload)
        assert event.type == "session.update"
        # session can be dict or Session object after validation
        session_voice = event.session.voice if hasattr(event.session, 'voice') else event.session["voice"]
        assert session_voice == "alloy"

        # Test round-trip
        dumped = json.loads(event.model_dump_json())
        assert dumped["type"] == "session.update"
        assert dumped["session"]["voice"] == "alloy"

    def test_input_audio_buffer_append(self):
        """Test input_audio_buffer.append event."""
        payload = {
            "type": "input_audio_buffer.append",
            "audio": "SGVsbG8gd29ybGQ=",  # Base64-encoded
        }
        event = ora.InputAudioBufferAppend.model_validate(payload)
        assert event.type == "input_audio_buffer.append"
        assert event.audio == "SGVsbG8gd29ybGQ="

        # Test round-trip
        dumped = json.loads(event.model_dump_json())
        assert dumped["type"] == "input_audio_buffer.append"
        assert dumped["audio"] == "SGVsbG8gd29ybGQ="

    def test_input_audio_buffer_commit(self):
        """Test input_audio_buffer.commit event."""
        payload = {"type": "input_audio_buffer.commit"}
        event = ora.InputAudioBufferCommit.model_validate(payload)
        assert event.type == "input_audio_buffer.commit"

        # Test round-trip
        dumped = json.loads(event.model_dump_json())
        assert dumped["type"] == "input_audio_buffer.commit"

    def test_input_audio_buffer_clear(self):
        """Test input_audio_buffer.clear event."""
        payload = {"type": "input_audio_buffer.clear"}
        event = ora.InputAudioBufferClear.model_validate(payload)
        assert event.type == "input_audio_buffer.clear"

        # Test round-trip
        dumped = json.loads(event.model_dump_json())
        assert dumped["type"] == "input_audio_buffer.clear"

    def test_conversation_item_create(self):
        """Test conversation.item.create event."""
        payload = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            },
            "previous_item_id": "item_123",
        }
        event = ora.ConversationItemCreate.model_validate(payload)
        assert event.type == "conversation.item.create"
        assert event.item["role"] == "user"

        # Test round-trip
        dumped = json.loads(event.model_dump_json())
        assert dumped["type"] == "conversation.item.create"

    def test_conversation_item_delete(self):
        """Test conversation.item.delete event."""
        payload = {
            "type": "conversation.item.delete",
            "item_id": "item_123",
        }
        event = ora.ConversationItemDelete.model_validate(payload)
        assert event.type == "conversation.item.delete"
        assert event.item_id == "item_123"

        # Test round-trip
        dumped = json.loads(event.model_dump_json())
        assert dumped["type"] == "conversation.item.delete"
        assert dumped["item_id"] == "item_123"

    def test_conversation_item_retrieve(self):
        """Test conversation.item.retrieve event."""
        payload = {
            "type": "conversation.item.retrieve",
            "item_id": "item_123",
        }
        event = ora.ConversationItemRetrieve.model_validate(payload)
        assert event.type == "conversation.item.retrieve"
        assert event.item_id == "item_123"

    def test_conversation_item_truncate(self):
        """Test conversation.item.truncate event."""
        payload = {
            "type": "conversation.item.truncate",
            "item_id": "item_123",
            "content_index": 0,
            "audio_end_ms": 1500,
        }
        event = ora.ConversationItemTruncate.model_validate(payload)
        assert event.type == "conversation.item.truncate"
        assert event.item_id == "item_123"
        assert event.audio_end_ms == 1500

    def test_response_create(self):
        """Test response.create event."""
        payload = {"type": "response.create"}
        event = ora.ResponseCreate.model_validate(payload)
        assert event.type == "response.create"

        # Test with configuration
        payload_with_config = {
            "type": "response.create",
            "response": {"modalities": ["text", "audio"]},
        }
        event2 = ora.ResponseCreate.model_validate(payload_with_config)
        assert event2.response["modalities"] == ["text", "audio"]

    def test_response_cancel(self):
        """Test response.cancel event."""
        payload = {"type": "response.cancel"}
        event = ora.ResponseCancel.model_validate(payload)
        assert event.type == "response.cancel"


class TestServerEvents:
    """Test all server events (server -> client)."""

    def test_session_created(self):
        """Test session.created event."""
        payload = {
            "type": "session.created",
            "session": {
                "id": "sess_001",
                "object": "realtime.session",
                "model": "gpt-4o-realtime-preview",
                "modalities": ["text", "audio"],
                "voice": "alloy",
                "temperature": 0.8,
            },
        }
        event = ora.SessionCreated.model_validate(payload)
        assert event.type == "session.created"
        assert event.session.id == "sess_001"
        assert event.session.voice == "alloy"

    def test_session_updated(self):
        """Test session.updated event."""
        payload = {
            "type": "session.updated",
            "session": {
                "id": "sess_001",
                "object": "realtime.session",
                "model": "gpt-4o-realtime-preview",
                "voice": "echo",
            },
        }
        event = ora.SessionUpdated.model_validate(payload)
        assert event.type == "session.updated"
        assert event.session.voice == "echo"

    def test_error(self):
        """Test error event."""
        payload = {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "code": "invalid_value",
                "message": "Invalid audio format",
                "param": "audio_format",
            },
        }
        event = ora.Error.model_validate(payload)
        assert event.type == "error"
        assert event.error.code == "invalid_value"
        assert event.error.message == "Invalid audio format"

    def test_conversation_created(self):
        """Test conversation.created event."""
        payload = {
            "type": "conversation.created",
            "conversation": {
                "id": "conv_001",
                "object": "realtime.conversation",
            },
        }
        event = ora.ConversationCreated.model_validate(payload)
        assert event.type == "conversation.created"
        assert event.conversation.id == "conv_001"

    def test_conversation_item_created(self):
        """Test conversation.item.created event."""
        payload = {
            "type": "conversation.item.created",
            "item": {
                "id": "item_001",
                "object": "realtime.item",
                "type": "message",
                "status": "completed",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            },
        }
        event = ora.ConversationItemCreated.model_validate(payload)
        assert event.type == "conversation.item.created"
        assert event.item.id == "item_001"
        assert event.item.type == "message"

    def test_conversation_item_deleted(self):
        """Test conversation.item.deleted event."""
        payload = {
            "type": "conversation.item.deleted",
            "item_id": "item_001",
        }
        event = ora.ConversationItemDeleted.model_validate(payload)
        assert event.type == "conversation.item.deleted"
        assert event.item_id == "item_001"

    def test_conversation_item_retrieved(self):
        """Test conversation.item.retrieved event."""
        payload = {
            "type": "conversation.item.retrieved",
            "item": {
                "id": "item_001",
                "object": "realtime.item",
                "type": "message",
                "role": "assistant",
            },
        }
        event = ora.ConversationItemRetrieved.model_validate(payload)
        assert event.type == "conversation.item.retrieved"
        assert event.item.id == "item_001"

    def test_conversation_item_truncated(self):
        """Test conversation.item.truncated event."""
        payload = {
            "type": "conversation.item.truncated",
            "item_id": "item_001",
            "content_index": 0,
            "audio_end_ms": 2000,
        }
        event = ora.ConversationItemTruncated.model_validate(payload)
        assert event.type == "conversation.item.truncated"
        assert event.item_id == "item_001"
        assert event.audio_end_ms == 2000

    def test_input_audio_buffer_committed(self):
        """Test input_audio_buffer.committed event."""
        payload = {
            "type": "input_audio_buffer.committed",
            "item_id": "item_001",
            "previous_item_id": "item_000",
        }
        event = ora.InputAudioBufferCommitted.model_validate(payload)
        assert event.type == "input_audio_buffer.committed"
        assert event.item_id == "item_001"

    def test_input_audio_buffer_cleared(self):
        """Test input_audio_buffer.cleared event."""
        payload = {"type": "input_audio_buffer.cleared"}
        event = ora.InputAudioBufferCleared.model_validate(payload)
        assert event.type == "input_audio_buffer.cleared"

    def test_input_audio_buffer_speech_started(self):
        """Test input_audio_buffer.speech_started event."""
        payload = {
            "type": "input_audio_buffer.speech_started",
            "item_id": "item_001",
            "audio_start_ms": 1000,
        }
        event = ora.InputAudioBufferSpeechStarted.model_validate(payload)
        assert event.type == "input_audio_buffer.speech_started"
        assert event.item_id == "item_001"
        assert event.audio_start_ms == 1000

    def test_input_audio_buffer_speech_stopped(self):
        """Test input_audio_buffer.speech_stopped event."""
        payload = {
            "type": "input_audio_buffer.speech_stopped",
            "item_id": "item_001",
            "audio_end_ms": 5000,
        }
        event = ora.InputAudioBufferSpeechStopped.model_validate(payload)
        assert event.type == "input_audio_buffer.speech_stopped"
        assert event.audio_end_ms == 5000

    def test_conversation_item_input_audio_transcription_delta(self):
        """Test conversation.item.input_audio_transcription.delta event."""
        payload = {
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "item_001",
            "content_index": 0,
            "delta": "Hello",
        }
        event = ora.ConversationItemInputAudioTranscriptionDelta.model_validate(payload)
        assert event.type == "conversation.item.input_audio_transcription.delta"
        assert event.delta == "Hello"

    def test_conversation_item_input_audio_transcription_completed(self):
        """Test conversation.item.input_audio_transcription.completed event."""
        payload = {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item_001",
            "content_index": 0,
            "transcript": "Hello, how are you?",
        }
        event = ora.ConversationItemInputAudioTranscriptionCompleted.model_validate(
            payload
        )
        assert event.type == "conversation.item.input_audio_transcription.completed"
        assert event.transcript == "Hello, how are you?"

    def test_conversation_item_input_audio_transcription_failed(self):
        """Test conversation.item.input_audio_transcription.failed event."""
        payload = {
            "type": "conversation.item.input_audio_transcription.failed",
            "item_id": "item_001",
            "content_index": 0,
            "error": {
                "type": "transcription_error",
                "message": "Failed to transcribe audio",
            },
        }
        event = ora.ConversationItemInputAudioTranscriptionFailed.model_validate(payload)
        assert event.type == "conversation.item.input_audio_transcription.failed"
        assert event.error.message == "Failed to transcribe audio"

    def test_response_created(self):
        """Test response.created event."""
        payload = {
            "type": "response.created",
            "response": {
                "id": "resp_001",
                "object": "realtime.response",
                "status": "in_progress",
                "output": [],
            },
        }
        event = ora.ResponseCreated.model_validate(payload)
        assert event.type == "response.created"
        assert event.response.status == "in_progress"

    def test_response_done(self):
        """Test response.done event."""
        payload = {
            "type": "response.done",
            "response": {
                "id": "resp_001",
                "object": "realtime.response",
                "status": "completed",
                "output": [],
                "usage": {
                    "total_tokens": 100,
                    "input_tokens": 50,
                    "output_tokens": 50,
                },
            },
        }
        event = ora.ResponseDone.model_validate(payload)
        assert event.type == "response.done"
        assert event.response.status == "completed"
        assert event.response.usage.total_tokens == 100

    def test_response_output_item_added(self):
        """Test response.output_item.added event."""
        payload = {
            "type": "response.output_item.added",
            "response_id": "resp_001",
            "output_index": 0,
            "item": {
                "id": "item_001",
                "object": "realtime.item",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        }
        event = ora.ResponseOutputItemAdded.model_validate(payload)
        assert event.type == "response.output_item.added"
        assert event.output_index == 0
        assert event.item.role == "assistant"

    def test_response_output_item_done(self):
        """Test response.output_item.done event."""
        payload = {
            "type": "response.output_item.done",
            "response_id": "resp_001",
            "output_index": 0,
            "item": {
                "id": "item_001",
                "object": "realtime.item",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello!"}],
            },
        }
        event = ora.ResponseOutputItemDone.model_validate(payload)
        assert event.type == "response.output_item.done"
        assert event.item.status == "completed"

    def test_response_content_part_added(self):
        """Test response.content_part.added event."""
        payload = {
            "type": "response.content_part.added",
            "response_id": "resp_001",
            "item_id": "item_001",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "text", "text": ""},
        }
        event = ora.ResponseContentPartAdded.model_validate(payload)
        assert event.type == "response.content_part.added"
        assert event.content_index == 0

    def test_response_content_part_done(self):
        """Test response.content_part.done event."""
        payload = {
            "type": "response.content_part.done",
            "response_id": "resp_001",
            "item_id": "item_001",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "text", "text": "Hello!"},
        }
        event = ora.ResponseContentPartDone.model_validate(payload)
        assert event.type == "response.content_part.done"
        assert event.part["text"] == "Hello!"

    def test_response_text_delta(self):
        """Test response.output_text.delta event."""
        payload = {
            "type": "response.output_text.delta",
            "delta": "Hello",
            "response_id": "resp_001",
            "item_id": "item_001",
            "output_index": 0,
            "content_index": 0,
        }
        event = ora.ResponseTextDelta.model_validate(payload)
        assert event.type == "response.output_text.delta"
        assert event.delta == "Hello"

        # Test backward compatibility (without optional fields)
        payload_minimal = {
            "type": "response.output_text.delta",
            "delta": "World",
        }
        event2 = ora.ResponseTextDelta.model_validate(payload_minimal)
        assert event2.delta == "World"

    def test_response_text_done(self):
        """Test response.output_text.done event."""
        payload = {
            "type": "response.output_text.done",
            "text": "Hello, how can I help you?",
            "response_id": "resp_001",
            "item_id": "item_001",
            "output_index": 0,
            "content_index": 0,
        }
        event = ora.ResponseTextDone.model_validate(payload)
        assert event.type == "response.output_text.done"
        assert event.text == "Hello, how can I help you?"

    def test_response_audio_transcript_delta(self):
        """Test response.output_audio_transcript.delta event."""
        payload = {
            "type": "response.output_audio_transcript.delta",
            "delta": "Hello",
            "response_id": "resp_001",
            "item_id": "item_001",
            "output_index": 0,
            "content_index": 0,
        }
        event = ora.ResponseAudioTranscriptDelta.model_validate(payload)
        assert event.type == "response.output_audio_transcript.delta"
        assert event.delta == "Hello"

    def test_response_audio_transcript_done(self):
        """Test response.output_audio_transcript.done event."""
        payload = {
            "type": "response.output_audio_transcript.done",
            "transcript": "Hello, how are you today?",
            "response_id": "resp_001",
            "item_id": "item_001",
            "output_index": 0,
            "content_index": 0,
        }
        event = ora.ResponseAudioTranscriptDone.model_validate(payload)
        assert event.type == "response.output_audio_transcript.done"
        assert event.transcript == "Hello, how are you today?"

    def test_response_audio_delta(self):
        """Test response.output_audio.delta event."""
        payload = {
            "type": "response.output_audio.delta",
            "delta": "SGVsbG8gd29ybGQ=",
            "response_id": "resp_001",
            "item_id": "item_001",
            "output_index": 0,
            "content_index": 0,
        }
        event = ora.ResponseAudioDelta.model_validate(payload)
        assert event.type == "response.output_audio.delta"
        assert event.delta == "SGVsbG8gd29ybGQ="

    def test_response_audio_done(self):
        """Test response.output_audio.done event."""
        payload = {
            "type": "response.output_audio.done",
            "response_id": "resp_001",
            "item_id": "item_001",
            "output_index": 0,
            "content_index": 0,
        }
        event = ora.ResponseAudioDone.model_validate(payload)
        assert event.type == "response.output_audio.done"

        # Test backward compatibility
        payload_minimal = {"type": "response.output_audio.done"}
        event2 = ora.ResponseAudioDone.model_validate(payload_minimal)
        assert event2.type == "response.output_audio.done"

    def test_response_function_call_arguments_delta(self):
        """Test response.function_call_arguments.delta event."""
        payload = {
            "type": "response.function_call_arguments.delta",
            "response_id": "resp_001",
            "item_id": "item_001",
            "output_index": 0,
            "call_id": "call_001",
            "delta": '{"name": "',
        }
        event = ora.ResponseFunctionCallArgumentsDelta.model_validate(payload)
        assert event.type == "response.function_call_arguments.delta"
        assert event.call_id == "call_001"

    def test_response_function_call_arguments_done(self):
        """Test response.function_call_arguments.done event."""
        payload = {
            "type": "response.function_call_arguments.done",
            "response_id": "resp_001",
            "item_id": "item_001",
            "output_index": 0,
            "call_id": "call_001",
            "arguments": '{"name": "John", "age": 30}',
        }
        event = ora.ResponseFunctionCallArgumentsDone.model_validate(payload)
        assert event.type == "response.function_call_arguments.done"
        assert event.arguments == '{"name": "John", "age": 30}'

    def test_rate_limits_updated(self):
        """Test rate_limits.updated event."""
        payload = {
            "type": "rate_limits.updated",
            "rate_limits": [
                {
                    "name": "requests",
                    "limit": 1000,
                    "remaining": 999,
                    "reset_seconds": 60.0,
                },
                {
                    "name": "tokens",
                    "limit": 150000,
                    "remaining": 145000,
                    "reset_seconds": 60.0,
                },
            ],
        }
        event = ora.RateLimitsUpdated.model_validate(payload)
        assert event.type == "rate_limits.updated"
        assert len(event.rate_limits) == 2
        assert event.rate_limits[0].name == "requests"
        assert event.rate_limits[0].remaining == 999


class TestUnmuteExtensions:
    """Test Unmute-specific extension events."""

    def test_unmute_input_audio_buffer_append_anonymized(self):
        """Test unmute.input_audio_buffer.append_anonymized event."""
        payload = {
            "type": "unmute.input_audio_buffer.append_anonymized",
            "number_of_samples": 960,
        }
        event = ora.UnmuteInputAudioBufferAppendAnonymized.model_validate(payload)
        assert event.type == "unmute.input_audio_buffer.append_anonymized"
        assert event.number_of_samples == 960

    def test_unmute_additional_outputs(self):
        """Test unmute.additional_outputs event."""
        payload = {
            "type": "unmute.additional_outputs",
            "args": {"debug": "info"},
        }
        event = ora.UnmuteAdditionalOutputs.model_validate(payload)
        assert event.type == "unmute.additional_outputs"

    def test_unmute_response_text_delta_ready(self):
        """Test unmute.response.text.delta.ready event."""
        payload = {
            "type": "unmute.response.text.delta.ready",
            "delta": "Hello",
        }
        event = ora.UnmuteResponseTextDeltaReady.model_validate(payload)
        assert event.type == "unmute.response.text.delta.ready"
        assert event.delta == "Hello"

    def test_unmute_response_audio_delta_ready(self):
        """Test unmute.response.audio.delta.ready event."""
        payload = {
            "type": "unmute.response.audio.delta.ready",
            "number_of_samples": 480,
        }
        event = ora.UnmuteResponseAudioDeltaReady.model_validate(payload)
        assert event.type == "unmute.response.audio.delta.ready"
        assert event.number_of_samples == 480

    def test_unmute_interrupted_by_vad(self):
        """Test unmute.interrupted_by_vad event."""
        payload = {"type": "unmute.interrupted_by_vad"}
        event = ora.UnmuteInterruptedByVAD.model_validate(payload)
        assert event.type == "unmute.interrupted_by_vad"


class TestDiscriminatedUnions:
    """Test that discriminated unions work correctly."""

    def test_client_event_validation(self):
        """Test that ClientEvent union validates correctly."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(ora.ClientEvent)

        # Test session.update
        payload = {
            "type": "session.update",
            "session": {"voice": "alloy"},
        }
        event = adapter.validate_python(payload)
        assert isinstance(event, ora.SessionUpdate)

        # Test input_audio_buffer.commit
        payload = {"type": "input_audio_buffer.commit"}
        event = adapter.validate_python(payload)
        assert isinstance(event, ora.InputAudioBufferCommit)

        # Test response.create
        payload = {"type": "response.create"}
        event = adapter.validate_python(payload)
        assert isinstance(event, ora.ResponseCreate)

    def test_server_event_validation(self):
        """Test that ServerEvent union validates correctly."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(ora.ServerEvent)

        # Test session.created
        payload = {
            "type": "session.created",
            "session": {
                "id": "sess_001",
                "object": "realtime.session",
                "model": "gpt-4o-realtime-preview",
            },
        }
        event = adapter.validate_python(payload)
        assert isinstance(event, ora.SessionCreated)

        # Test response.done
        payload = {
            "type": "response.done",
            "response": {
                "id": "resp_001",
                "object": "realtime.response",
                "status": "completed",
                "output": [],
            },
        }
        event = adapter.validate_python(payload)
        assert isinstance(event, ora.ResponseDone)

        # Test rate_limits.updated
        payload = {
            "type": "rate_limits.updated",
            "rate_limits": [
                {
                    "name": "requests",
                    "limit": 1000,
                    "remaining": 999,
                    "reset_seconds": 60.0,
                }
            ],
        }
        event = adapter.validate_python(payload)
        assert isinstance(event, ora.RateLimitsUpdated)

    def test_event_union_validation(self):
        """Test that Event union validates both client and server events."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(ora.Event)

        # Test client event
        client_payload = {"type": "input_audio_buffer.commit"}
        event = adapter.validate_python(client_payload)
        assert isinstance(event, ora.InputAudioBufferCommit)

        # Test server event
        server_payload = {"type": "input_audio_buffer.committed", "item_id": "item_001"}
        event = adapter.validate_python(server_payload)
        assert isinstance(event, ora.InputAudioBufferCommitted)


class TestRoundTrip:
    """Test that all events can round-trip through JSON."""

    @pytest.mark.parametrize(
        "event_class,payload",
        [
            (
                ora.SessionUpdate,
                {"type": "session.update", "session": {"voice": "alloy"}},
            ),
            (
                ora.InputAudioBufferAppend,
                {"type": "input_audio_buffer.append", "audio": "SGVsbG8="},
            ),
            (ora.InputAudioBufferCommit, {"type": "input_audio_buffer.commit"}),
            (ora.InputAudioBufferClear, {"type": "input_audio_buffer.clear"}),
            (ora.ResponseCreate, {"type": "response.create"}),
            (ora.ResponseCancel, {"type": "response.cancel"}),
            (
                ora.SessionCreated,
                {
                    "type": "session.created",
                    "session": {
                        "id": "sess_001",
                        "object": "realtime.session",
                        "model": "gpt-4o-realtime-preview",
                    },
                },
            ),
            (
                ora.ConversationCreated,
                {
                    "type": "conversation.created",
                    "conversation": {
                        "id": "conv_001",
                        "object": "realtime.conversation",
                    },
                },
            ),
            (
                ora.InputAudioBufferCommitted,
                {"type": "input_audio_buffer.committed", "item_id": "item_001"},
            ),
            (ora.InputAudioBufferCleared, {"type": "input_audio_buffer.cleared"}),
            (
                ora.ResponseTextDelta,
                {"type": "response.output_text.delta", "delta": "Hello"},
            ),
            (
                ora.ResponseTextDone,
                {"type": "response.output_text.done", "text": "Hello!"},
            ),
            (
                ora.ResponseAudioDelta,
                {"type": "response.output_audio.delta", "delta": "SGVsbG8="},
            ),
            (ora.ResponseAudioDone, {"type": "response.output_audio.done"}),
        ],
    )
    def test_round_trip(self, event_class, payload):
        """Test that event can validate JSON and dump back to equivalent JSON."""
        # Validate from JSON
        event = event_class.model_validate(payload)

        # Dump back to JSON
        dumped = json.loads(event.model_dump_json())

        # Verify type is preserved
        assert dumped["type"] == payload["type"]

        # Re-validate to ensure round-trip works
        event2 = event_class.model_validate(dumped)
        assert event2.type == event.type


class TestAnonymizedAudioEvents:
    """Test that anonymized audio events work for recorder."""

    def test_anonymized_append_replaces_audio_data(self):
        """Test that anonymized events preserve count but not actual audio."""
        # Regular event with audio
        regular = ora.InputAudioBufferAppend(audio="SGVsbG8gd29ybGQ=")
        assert regular.audio == "SGVsbG8gd29ybGQ="

        # Anonymized version
        anonymized = ora.UnmuteInputAudioBufferAppendAnonymized(number_of_samples=960)
        assert anonymized.number_of_samples == 960

        # Verify they serialize differently
        regular_json = json.loads(regular.model_dump_json())
        anon_json = json.loads(anonymized.model_dump_json())

        assert regular_json["type"] == "input_audio_buffer.append"
        assert "audio" in regular_json

        assert anon_json["type"] == "unmute.input_audio_buffer.append_anonymized"
        assert "number_of_samples" in anon_json
        assert "audio" not in anon_json
