"""Tests for SessionState class.

Tests the per-session state machine for tracking responses, conversation items,
and audio buffer state.
"""

import pytest

import unmute.openai_realtime_api_events as ora
from unmute.session_state import SessionState


class TestResponseLifecycle:
    """Test response lifecycle tracking."""

    def test_no_active_response_initially(self):
        """Test that there's no active response initially."""
        state = SessionState()
        assert not state.has_active_response()
        assert state.current_response is None

    def test_start_response(self):
        """Test starting a response."""
        state = SessionState()
        response = state.start_response("resp_123")

        assert state.has_active_response()
        assert response.id == "resp_123"
        assert response.status == "in_progress"
        assert state.current_response_id == "resp_123"

    def test_complete_response(self):
        """Test completing a response."""
        state = SessionState()
        state.start_response("resp_123")

        response = state.complete_response(status="completed")

        assert not state.has_active_response()
        assert response.id == "resp_123"
        assert response.status == "completed"
        assert state.current_response_id is None

    def test_cancel_response(self):
        """Test cancelling a response."""
        state = SessionState()
        state.start_response("resp_123")

        response = state.cancel_response()

        assert not state.has_active_response()
        assert response is not None
        assert response.status == "cancelled"

    def test_cancel_when_no_response(self):
        """Test that cancelling when no response returns None."""
        state = SessionState()
        response = state.cancel_response()
        assert response is None

    def test_complete_without_start_raises(self):
        """Test that completing without starting raises an error."""
        state = SessionState()
        with pytest.raises(RuntimeError, match="No response in progress"):
            state.complete_response()

    def test_add_output_item(self):
        """Test adding an output item to current response."""
        state = SessionState()
        state.start_response("resp_123")

        item = ora.Item(
            id="item_abc",
            type="message",
            role="assistant",
            status="in_progress",
            content=[],
        )
        state.add_output_item(item, output_index=0)

        assert state.current_item_id == "item_abc"
        assert state.output_items_in_progress["item_abc"] == 0
        assert "item_abc" in state.items
        assert "item_abc" in state.item_order


class TestConversationItems:
    """Test conversation item management."""

    def test_create_item(self):
        """Test creating a conversation item."""
        state = SessionState()
        item = state.create_item(
            item_type="message",
            role="user",
            content=[{"type": "input_text", "text": "Hello"}],
        )

        assert item.id.startswith("item_")
        assert item.type == "message"
        assert item.role == "user"
        assert item.status == "completed"
        assert item.id in state.items
        assert item.id in state.item_order

    def test_create_item_with_previous(self):
        """Test creating an item after a specific previous item."""
        state = SessionState()
        first = state.create_item(item_type="message", role="user")
        second = state.create_item(item_type="message", role="assistant")
        # Insert between first and second
        middle = state.create_item(
            item_type="message", role="user", previous_item_id=first.id
        )

        assert state.item_order == [first.id, middle.id, second.id]

    def test_delete_item(self):
        """Test deleting a conversation item."""
        state = SessionState()
        item = state.create_item(item_type="message", role="user")
        item_id = item.id

        success = state.delete_item(item_id)

        assert success
        assert item_id not in state.items
        assert item_id not in state.item_order

    def test_delete_nonexistent_item(self):
        """Test deleting a nonexistent item returns False."""
        state = SessionState()
        success = state.delete_item("nonexistent")
        assert not success

    def test_get_item(self):
        """Test retrieving an item."""
        state = SessionState()
        created = state.create_item(item_type="message", role="user")

        retrieved = state.get_item(created.id)
        assert retrieved is not None
        assert retrieved.id == created.id

    def test_get_nonexistent_item(self):
        """Test retrieving a nonexistent item returns None."""
        state = SessionState()
        assert state.get_item("nonexistent") is None

    def test_truncate_item(self):
        """Test truncating an item."""
        state = SessionState()
        item = state.create_item(item_type="message", role="assistant")

        success = state.truncate_item(item.id, content_index=0, audio_end_ms=1500)
        assert success

    def test_truncate_nonexistent_item(self):
        """Test truncating a nonexistent item returns False."""
        state = SessionState()
        success = state.truncate_item("nonexistent", 0, 1500)
        assert not success

    def test_get_previous_item_id(self):
        """Test getting the previous item ID."""
        state = SessionState()
        assert state.get_previous_item_id() is None

        first = state.create_item(item_type="message", role="user")
        assert state.get_previous_item_id() == first.id

        second = state.create_item(item_type="message", role="assistant")
        assert state.get_previous_item_id() == second.id

    def test_item_create_delete_idempotent(self):
        """Test that item operations are idempotent."""
        state = SessionState()
        item = state.create_item(item_type="message", role="user")
        item_id = item.id

        # First delete
        assert state.delete_item(item_id)
        # Second delete should return False but not raise
        assert not state.delete_item(item_id)


class TestInputAudioBuffer:
    """Test input audio buffer state tracking."""

    def test_initial_buffer_state(self):
        """Test initial buffer state."""
        state = SessionState()
        assert state.input_buffer_samples == 0
        assert not state.input_buffer_committed
        assert state.pending_audio_item_id is None

    def test_add_input_samples(self):
        """Test adding input samples."""
        state = SessionState()
        state.add_input_samples(960)
        assert state.input_buffer_samples == 960
        assert not state.input_buffer_committed

        state.add_input_samples(480)
        assert state.input_buffer_samples == 1440

    def test_commit_input_buffer(self):
        """Test committing the input buffer."""
        state = SessionState()
        state.add_input_samples(960)

        item_id = state.commit_input_buffer()

        assert item_id.startswith("item_")
        assert state.input_buffer_committed
        assert state.pending_audio_item_id == item_id
        assert item_id in state.items
        assert state.items[item_id].type == "message"
        assert state.items[item_id].role == "user"

    def test_clear_input_buffer(self):
        """Test clearing the input buffer."""
        state = SessionState()
        state.add_input_samples(960)
        state.commit_input_buffer()

        state.clear_input_buffer()

        assert state.input_buffer_samples == 0
        assert not state.input_buffer_committed
        assert state.pending_audio_item_id is None

    def test_audio_buffer_commit_creates_item(self):
        """Test that committing audio buffer creates a conversation item."""
        state = SessionState()
        state.add_input_samples(1920)

        item_id = state.commit_input_buffer()
        item = state.get_item(item_id)

        assert item is not None
        assert item.type == "message"
        assert item.role == "user"
        assert item.content is not None
        assert item.content[0]["type"] == "input_audio"


class TestSnapshotRestore:
    """Test state snapshot and restore functionality."""

    def test_snapshot_empty_state(self):
        """Test snapshotting empty state."""
        state = SessionState()
        snapshot = state.snapshot()

        assert snapshot["items"] == {}
        assert snapshot["item_order"] == []
        assert snapshot["current_response_id"] is None
        assert snapshot["input_buffer_samples"] == 0

    def test_snapshot_with_items(self):
        """Test snapshotting state with items."""
        state = SessionState()
        item = state.create_item(item_type="message", role="user")
        state.add_input_samples(960)

        snapshot = state.snapshot()

        assert item.id in snapshot["items"]
        assert snapshot["item_order"] == [item.id]
        assert snapshot["input_buffer_samples"] == 960

    def test_restore_from_snapshot(self):
        """Test restoring state from snapshot."""
        original = SessionState()
        item = original.create_item(item_type="message", role="user")
        original.add_input_samples(960)
        snapshot = original.snapshot()

        restored = SessionState()
        restored.restore(snapshot)

        assert item.id in restored.items
        assert restored.item_order == [item.id]
        assert restored.input_buffer_samples == 960

    def test_snapshot_restore_roundtrip(self):
        """Test that snapshot/restore roundtrip preserves state."""
        state = SessionState()

        # Create some state
        user_item = state.create_item(
            item_type="message",
            role="user",
            content=[{"type": "input_text", "text": "Hello"}],
        )
        state.create_item(
            item_type="message",
            role="assistant",
            content=[{"type": "text", "text": "Hi there!"}],
        )
        state.add_input_samples(960)
        state.commit_input_buffer()

        # Snapshot
        snapshot = state.snapshot()

        # Restore to new state
        new_state = SessionState()
        new_state.restore(snapshot)

        # Verify
        assert len(new_state.items) == len(state.items)
        assert new_state.item_order == state.item_order
        assert new_state.input_buffer_samples == state.input_buffer_samples
        assert new_state.input_buffer_committed == state.input_buffer_committed

        # Verify item content
        restored_user = new_state.get_item(user_item.id)
        assert restored_user is not None
        assert restored_user.role == "user"

    def test_snapshot_before_after_modification(self):
        """Test that snapshots correctly capture state changes."""
        state = SessionState()

        # Take snapshot before modification
        before = state.snapshot()
        assert len(before["items"]) == 0

        # Modify state
        item = state.create_item(item_type="message", role="user")

        # Take snapshot after modification
        after = state.snapshot()
        assert len(after["items"]) == 1
        assert item.id in after["items"]

        # Before snapshot should still show empty
        assert len(before["items"]) == 0


class TestRepr:
    """Test string representation."""

    def test_repr_empty(self):
        """Test repr of empty state."""
        state = SessionState()
        repr_str = repr(state)
        assert "items=0" in repr_str
        assert "response=none" in repr_str
        assert "buffer_samples=0" in repr_str

    def test_repr_with_data(self):
        """Test repr with data."""
        state = SessionState()
        state.create_item(item_type="message", role="user")
        state.start_response("resp_123")
        state.add_input_samples(960)

        repr_str = repr(state)
        assert "items=1" in repr_str
        assert "response=active" in repr_str
        assert "buffer_samples=960" in repr_str
