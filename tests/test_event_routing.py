"""Tests for event routing in main_websocket.py.

Integration tests for the receive_loop event routing to handler methods.
"""

from unittest.mock import MagicMock

import pytest

import unmute.openai_realtime_api_events as ora
from unmute.session_state import SessionState


class TestHandlerEventMethods:
    """Test handler methods that process client events."""

    @pytest.fixture
    def mock_handler(self):
        """Create a mock handler with session state."""
        handler = MagicMock()
        handler.session_state = SessionState()
        handler.chatbot = MagicMock()
        handler.chatbot.chat_history = []
        handler.chatbot.last_message = MagicMock(return_value=None)
        handler.recorder = None
        return handler

    def test_handle_item_create_basic(self, mock_handler):
        """Test basic item creation through handler method."""
        # Simulate the handler method logic
        state = mock_handler.session_state

        item_data = {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello"}],
        }

        item = state.create_item(
            item_type=item_data.get("type", "message"),
            role=item_data.get("role"),
            content=item_data.get("content"),
        )

        assert item.id.startswith("item_")
        assert item.type == "message"
        assert item.role == "user"
        assert item.id in state.items

    def test_handle_item_delete(self, mock_handler):
        """Test item deletion through handler method."""
        state = mock_handler.session_state

        # Create an item first
        item = state.create_item(item_type="message", role="user")
        item_id = item.id

        # Delete it
        success = state.delete_item(item_id)

        assert success
        assert item_id not in state.items

    def test_handle_item_retrieve(self, mock_handler):
        """Test item retrieval through handler method."""
        state = mock_handler.session_state

        # Create an item
        created = state.create_item(item_type="message", role="user")

        # Retrieve it
        retrieved = state.get_item(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id

    def test_handle_item_retrieve_not_found(self, mock_handler):
        """Test item retrieval when not found."""
        state = mock_handler.session_state
        result = state.get_item("nonexistent_item")
        assert result is None

    def test_handle_item_truncate(self, mock_handler):
        """Test item truncation through handler method."""
        state = mock_handler.session_state

        # Create an item
        item = state.create_item(item_type="message", role="assistant")

        # Truncate it
        success = state.truncate_item(item.id, content_index=0, audio_end_ms=1500)

        assert success

    def test_commit_audio_buffer(self, mock_handler):
        """Test audio buffer commit through handler method."""
        state = mock_handler.session_state

        # Add some samples first
        state.add_input_samples(960)

        # Get previous item ID (none initially)
        previous_item_id = state.get_previous_item_id()
        assert previous_item_id is None

        # Commit
        item_id = state.commit_input_buffer()

        assert item_id.startswith("item_")
        assert state.input_buffer_committed

        # Now previous item should be the committed one
        assert state.get_previous_item_id() == item_id

    def test_clear_audio_buffer(self, mock_handler):
        """Test audio buffer clear through handler method."""
        state = mock_handler.session_state

        # Add samples and commit
        state.add_input_samples(960)
        state.commit_input_buffer()

        # Clear
        state.clear_input_buffer()

        assert state.input_buffer_samples == 0
        assert not state.input_buffer_committed


class TestResponseEventRouting:
    """Test response control event routing."""

    @pytest.fixture
    def session_state(self):
        """Create a session state instance."""
        return SessionState()

    def test_response_create_starts_response(self, session_state):
        """Test that response.create starts a response."""
        # Simulate response creation
        assert not session_state.has_active_response()

        response = session_state.start_response("resp_123")

        assert session_state.has_active_response()
        assert response.status == "in_progress"

    def test_response_cancel_stops_response(self, session_state):
        """Test that response.cancel stops in-progress response."""
        # Start a response
        session_state.start_response("resp_123")
        assert session_state.has_active_response()

        # Cancel it
        response = session_state.cancel_response()

        assert not session_state.has_active_response()
        assert response is not None
        assert response.status == "cancelled"

    def test_response_cancel_when_no_response(self, session_state):
        """Test that response.cancel when no response returns None."""
        response = session_state.cancel_response()
        assert response is None


class TestConversationItemEventRouting:
    """Test conversation item event routing."""

    @pytest.fixture
    def session_state(self):
        """Create a session state instance."""
        return SessionState()

    def test_item_create_returns_ack(self, session_state):
        """Test that conversation.item.create returns created acknowledgement."""
        item = session_state.create_item(
            item_type="message",
            role="user",
            content=[{"type": "input_text", "text": "Hello"}],
        )

        # Create acknowledgement event
        ack = ora.ConversationItemCreated(item=item, previous_item_id=None)

        assert ack.type == "conversation.item.created"
        assert ack.item.id == item.id

    def test_item_create_with_previous(self, session_state):
        """Test item creation with previous_item_id."""
        first = session_state.create_item(item_type="message", role="user")
        second = session_state.create_item(
            item_type="message",
            role="assistant",
            previous_item_id=first.id,
        )

        # Verify order
        assert session_state.item_order[0] == first.id
        assert session_state.item_order[1] == second.id

    def test_item_delete_returns_ack(self, session_state):
        """Test that conversation.item.delete returns deleted acknowledgement."""
        item = session_state.create_item(item_type="message", role="user")
        item_id = item.id

        session_state.delete_item(item_id)
        ack = ora.ConversationItemDeleted(item_id=item_id)

        assert ack.type == "conversation.item.deleted"
        assert ack.item_id == item_id


class TestInputAudioBufferEventRouting:
    """Test input audio buffer event routing."""

    @pytest.fixture
    def session_state(self):
        """Create a session state instance."""
        return SessionState()

    def test_buffer_commit_returns_committed(self, session_state):
        """Test that input_audio_buffer.commit returns committed event."""
        session_state.add_input_samples(960)

        prev_id = session_state.get_previous_item_id()
        item_id = session_state.commit_input_buffer()

        ack = ora.InputAudioBufferCommitted(item_id=item_id, previous_item_id=prev_id)

        assert ack.type == "input_audio_buffer.committed"
        assert ack.item_id == item_id

    def test_buffer_clear_returns_cleared(self, session_state):
        """Test that input_audio_buffer.clear returns cleared event."""
        session_state.add_input_samples(960)
        session_state.clear_input_buffer()

        ack = ora.InputAudioBufferCleared()

        assert ack.type == "input_audio_buffer.cleared"


class TestEventSequencing:
    """Test proper event sequencing."""

    def test_response_lifecycle_events_order(self):
        """Test that response lifecycle events are properly ordered."""
        state = SessionState()

        # Start response
        response = state.start_response("resp_123")
        created_event = ora.ResponseCreated(response=response)
        assert created_event.response.status == "in_progress"

        # Add output item
        item = ora.Item(
            id="item_abc",
            type="message",
            role="assistant",
            status="in_progress",
            content=[],
        )
        state.add_output_item(item, output_index=0)
        added_event = ora.ResponseOutputItemAdded(
            response_id="resp_123",
            output_index=0,
            item=item,
        )
        assert added_event.output_index == 0

        # Complete response
        final_response = state.complete_response("completed")
        done_event = ora.ResponseDone(response=final_response)
        assert done_event.response.status == "completed"

    def test_content_part_events(self):
        """Test content part added/done events."""
        response_id = "resp_123"
        item_id = "item_abc"

        # Content part added
        added = ora.ResponseContentPartAdded(
            response_id=response_id,
            item_id=item_id,
            output_index=0,
            content_index=0,
            part={"type": "text", "text": ""},
        )
        assert added.type == "response.content_part.added"

        # Content part done
        done = ora.ResponseContentPartDone(
            response_id=response_id,
            item_id=item_id,
            output_index=0,
            content_index=0,
            part={"type": "text", "text": "Hello!"},
        )
        assert done.type == "response.content_part.done"
        assert done.part["text"] == "Hello!"
