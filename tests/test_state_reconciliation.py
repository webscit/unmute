"""Tests for state reconciliation and replay helpers.

Tests conflict detection, event replay, and state consistency.
"""

import json
from datetime import datetime

import unmute.openai_realtime_api_events as ora
from unmute.recorder import RecorderEvent, SessionStateSnapshot
from unmute.session_state import SessionState


class TestConflictDetection:
    """Test conflict detection in state management."""

    def test_duplicate_item_detection(self):
        """Test that duplicate item creation is handled."""
        state = SessionState()

        # Create first item
        first = state.create_item(item_type="message", role="user")

        # Creating another item should generate a different ID
        second = state.create_item(item_type="message", role="user")

        assert first.id != second.id
        assert len(state.items) == 2

    def test_delete_already_deleted_item(self):
        """Test deleting an already deleted item."""
        state = SessionState()
        item = state.create_item(item_type="message", role="user")
        item_id = item.id

        # First delete succeeds
        assert state.delete_item(item_id)

        # Second delete returns False (idempotent)
        assert not state.delete_item(item_id)

    def test_truncate_already_truncated_item(self):
        """Test truncating an already truncated item is idempotent."""
        state = SessionState()
        item = state.create_item(item_type="message", role="assistant")

        # First truncation
        assert state.truncate_item(item.id, 0, 1000)

        # Second truncation (should still work)
        assert state.truncate_item(item.id, 0, 500)

    def test_response_conflict(self):
        """Test handling response conflicts."""
        state = SessionState()

        # Start first response
        state.start_response("resp_1")
        assert state.has_active_response()

        # Cancel first response
        cancelled = state.cancel_response()
        assert cancelled is not None
        assert cancelled.status == "cancelled"

        # Start second response
        state.start_response("resp_2")
        assert state.current_response_id == "resp_2"


class TestEventReplay:
    """Test replaying recorded events to restore state."""

    def test_replay_item_create_events(self):
        """Test replaying item create events restores items."""
        # Simulate recorded events
        events = [
            {
                "item_type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            },
            {
                "item_type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi!"}],
            },
        ]

        state = SessionState()

        # Replay events
        for event in events:
            state.create_item(
                item_type=event["item_type"],
                role=event.get("role"),
                content=event.get("content"),
            )

        assert len(state.items) == 2
        assert len(state.item_order) == 2

    def test_replay_events_restore_state(self):
        """Test that replaying recorded events restores session state."""
        # Create original state
        original = SessionState()
        user_item = original.create_item(
            item_type="message",
            role="user",
            content=[{"type": "input_text", "text": "Hello"}],
        )
        assistant_item = original.create_item(
            item_type="message",
            role="assistant",
            content=[{"type": "text", "text": "Hi there!"}],
        )

        # Take snapshot
        snapshot = original.snapshot()

        # Restore to new state
        restored = SessionState()
        restored.restore(snapshot)

        # Verify restoration
        assert user_item.id in restored.items
        assert assistant_item.id in restored.items
        assert restored.item_order == [user_item.id, assistant_item.id]


class TestStateSnapshotSerialization:
    """Test state snapshot serialization for recording."""

    def test_snapshot_model_creation(self):
        """Test creating SessionStateSnapshot model."""
        state = SessionState()
        item = state.create_item(item_type="message", role="user")
        state.add_input_samples(960)

        snapshot_data = state.snapshot()

        snapshot = SessionStateSnapshot(
            timestamp_wall=datetime.now().timestamp(),
            items=snapshot_data["items"],
            item_order=snapshot_data["item_order"],
            current_response_id=snapshot_data.get("current_response_id"),
            current_item_id=snapshot_data.get("current_item_id"),
            input_buffer_samples=snapshot_data.get("input_buffer_samples", 0),
            input_buffer_committed=snapshot_data.get("input_buffer_committed", False),
            pending_audio_item_id=snapshot_data.get("pending_audio_item_id"),
            response_queue=snapshot_data.get("response_queue", []),
        )

        assert snapshot.type == "session_state_snapshot"
        assert item.id in snapshot.items
        assert snapshot.input_buffer_samples == 960

    def test_snapshot_json_serialization(self):
        """Test snapshot can be serialized to JSON."""
        state = SessionState()
        state.create_item(item_type="message", role="user")
        snapshot_data = state.snapshot()

        snapshot = SessionStateSnapshot(
            timestamp_wall=datetime.now().timestamp(),
            items=snapshot_data["items"],
            item_order=snapshot_data["item_order"],
            current_response_id=None,
            current_item_id=None,
            input_buffer_samples=0,
            input_buffer_committed=False,
            pending_audio_item_id=None,
            response_queue=[],
        )

        json_str = snapshot.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["type"] == "session_state_snapshot"
        assert "items" in parsed
        assert "item_order" in parsed

    def test_recorder_event_with_snapshot(self):
        """Test that recorder can handle various event types."""
        # RecorderEvent for regular events
        event = ora.SessionUpdated(
            session=ora.Session(
                id="sess_001",
                model="test-model",
            )
        )

        recorder_event = RecorderEvent(
            timestamp_wall=datetime.now().timestamp(),
            event_sender="server",
            data=event,
        )

        json_str = recorder_event.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["event_sender"] == "server"
        assert parsed["data"]["type"] == "session.updated"


class TestStateConsistency:
    """Test state consistency invariants."""

    def test_item_order_matches_items(self):
        """Test that item_order always matches items dict keys."""
        state = SessionState()

        # Create items
        items = []
        for _ in range(5):
            item = state.create_item(item_type="message", role="user")
            items.append(item)

        # Verify consistency
        assert set(state.item_order) == set(state.items.keys())

        # Delete some items
        state.delete_item(items[1].id)
        state.delete_item(items[3].id)

        # Still consistent
        assert set(state.item_order) == set(state.items.keys())

    def test_response_state_consistency(self):
        """Test response state is consistent."""
        state = SessionState()

        # No response initially
        assert state.current_response is None
        assert state.current_response_id is None
        assert not state.has_active_response()

        # Start response
        state.start_response("resp_123")
        assert state.current_response is not None
        assert state.current_response_id == "resp_123"
        assert state.has_active_response()

        # Complete response
        state.complete_response()
        assert state.current_response is None
        assert state.current_response_id is None
        assert not state.has_active_response()

    def test_output_items_cleared_on_response_complete(self):
        """Test output items tracking is cleared when response completes."""
        state = SessionState()
        state.start_response("resp_123")

        # Add output item
        item = ora.Item(
            id="item_abc",
            type="message",
            role="assistant",
            status="in_progress",
            content=[],
        )
        state.add_output_item(item, output_index=0)
        assert len(state.output_items_in_progress) == 1

        # Complete response
        state.complete_response()

        # Output items tracking should be cleared
        assert len(state.output_items_in_progress) == 0

    def test_buffer_state_consistency(self):
        """Test audio buffer state is consistent."""
        state = SessionState()

        # Initial state
        assert state.input_buffer_samples == 0
        assert not state.input_buffer_committed

        # Add samples
        state.add_input_samples(960)
        assert state.input_buffer_samples == 960
        assert not state.input_buffer_committed

        # Commit
        item_id = state.commit_input_buffer()
        assert state.input_buffer_committed
        assert state.pending_audio_item_id == item_id
        assert item_id in state.items

        # Clear
        state.clear_input_buffer()
        assert state.input_buffer_samples == 0
        assert not state.input_buffer_committed
        assert state.pending_audio_item_id is None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_content_item(self):
        """Test creating item with empty content."""
        state = SessionState()
        item = state.create_item(item_type="message", role="user", content=[])

        assert item.content == []

    def test_none_content_item(self):
        """Test creating item with None content."""
        state = SessionState()
        item = state.create_item(item_type="message", role="user", content=None)

        assert item.content is None

    def test_many_items_ordering(self):
        """Test ordering with many items."""
        state = SessionState()
        items = []

        for _ in range(100):
            item = state.create_item(item_type="message", role="user")
            items.append(item)

        # Verify order is preserved
        for i, item in enumerate(items):
            assert state.item_order[i] == item.id

    def test_snapshot_with_active_response(self):
        """Test snapshotting during active response."""
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

        snapshot = state.snapshot()

        assert snapshot["current_response_id"] == "resp_123"
        assert snapshot["current_item_id"] == "item_abc"

    def test_restore_clears_active_response(self):
        """Test that restore clears any active response."""
        original = SessionState()
        original.create_item(item_type="message", role="user")
        snapshot = original.snapshot()

        # Start a response in target state before restore
        target = SessionState()
        target.start_response("resp_456")
        assert target.has_active_response()

        # Restore should clear the active response
        target.restore(snapshot)
        assert not target.has_active_response()
