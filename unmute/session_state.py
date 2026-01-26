"""
Session State Machine for OpenAI Realtime protocol.

This module provides per-session state tracking for responses, conversation items,
and audio buffer management.

Session State Machine Invariants:
================================

1. Response Lifecycle:
   - Only one response can be in_progress at a time
   - State transitions: None -> in_progress -> completed|cancelled|failed
   - ResponseCreated MUST precede any ResponseTextDelta/AudioDelta
   - ResponseDone MUST follow all content events

2. Conversation Items:
   - item_order maintains insertion order
   - items dict is source of truth for item content
   - Deleted items are removed from both

3. Input Audio Buffer:
   - Samples accumulate until commit or clear
   - Commit creates a user message item
   - Clear discards without creating item

4. Event Ordering:
   - Acknowledgement events (*.created, *.committed, etc.) sent immediately
   - Content events (*.delta) may be batched
   - Lifecycle events (*.done) mark completion
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import unmute.openai_realtime_api_events as ora


@dataclass
class SessionState:
    """Per-session state machine for OpenAI Realtime protocol."""

    # Response tracking
    current_response: ora.Response | None = None
    response_queue: list[str] = field(default_factory=list)  # response_ids

    # Conversation items registry
    items: dict[str, ora.Item] = field(default_factory=dict)  # item_id -> Item
    item_order: list[str] = field(default_factory=list)  # ordered item_ids

    # Input audio buffer state
    input_buffer_samples: int = 0
    input_buffer_committed: bool = False
    pending_audio_item_id: str | None = None

    # Output tracking - maps item_id to output_index in current response
    output_items_in_progress: dict[str, int] = field(default_factory=dict)

    # Current response IDs for tracking
    current_response_id: str | None = None
    current_item_id: str | None = None

    def has_active_response(self) -> bool:
        """Check if there's an active response being generated."""
        return self.current_response is not None

    def start_response(self, response_id: str) -> ora.Response:
        """Start a new response, returning the Response object."""
        self.current_response_id = response_id
        self.current_response = ora.Response(
            id=response_id,
            status="in_progress",
            output=[],
        )
        self.output_items_in_progress.clear()
        return self.current_response

    def add_output_item(self, item: ora.Item, output_index: int) -> None:
        """Track an output item added to current response."""
        self.output_items_in_progress[item.id] = output_index
        self.current_item_id = item.id
        # Also add to conversation items
        self.items[item.id] = item
        self.item_order.append(item.id)

    def complete_response(
        self, status: str = "completed"
    ) -> ora.Response:
        """Mark current response as complete and return final Response object."""
        if self.current_response is None:
            raise RuntimeError("No response in progress to complete")

        self.current_response.status = status  # type: ignore[assignment]
        response = self.current_response

        # Reset tracking
        self.current_response = None
        self.current_response_id = None
        self.current_item_id = None
        self.output_items_in_progress.clear()

        return response

    def cancel_response(self) -> ora.Response | None:
        """Cancel current response if any."""
        if self.current_response is None:
            return None
        return self.complete_response(status="cancelled")

    def create_item(
        self,
        item_type: str,
        role: str | None = None,
        content: list[dict[str, Any]] | None = None,
        previous_item_id: str | None = None,
    ) -> ora.Item:
        """Create a new conversation item and add to tracking."""
        item_id = ora.random_id("item")
        item = ora.Item(
            id=item_id,
            type=item_type,  # type: ignore[arg-type]
            role=role,  # type: ignore[arg-type]
            content=content,
            status="completed",
        )

        # Insert at correct position
        if previous_item_id is not None and previous_item_id in self.items:
            try:
                idx = self.item_order.index(previous_item_id)
                self.item_order.insert(idx + 1, item_id)
            except ValueError:
                self.item_order.append(item_id)
        else:
            self.item_order.append(item_id)

        self.items[item_id] = item
        return item

    def delete_item(self, item_id: str) -> bool:
        """Delete an item from conversation. Returns True if item existed."""
        if item_id not in self.items:
            return False

        del self.items[item_id]
        try:
            self.item_order.remove(item_id)
        except ValueError:
            pass
        return True

    def get_item(self, item_id: str) -> ora.Item | None:
        """Retrieve an item by ID."""
        return self.items.get(item_id)

    def truncate_item(
        self, item_id: str, content_index: int, audio_end_ms: int
    ) -> bool:
        """Mark truncation point for an item. Returns True if item existed."""
        if item_id not in self.items:
            return False
        # In a full implementation, we'd modify the item's audio content
        # For now, we just track that truncation was requested
        return True

    def add_input_samples(self, num_samples: int) -> None:
        """Track input audio buffer samples."""
        self.input_buffer_samples += num_samples
        self.input_buffer_committed = False

    def commit_input_buffer(self) -> str:
        """Commit input buffer and return the created item ID."""
        item_id = ora.random_id("item")
        item = ora.Item(
            id=item_id,
            type="message",
            role="user",
            content=[{"type": "input_audio", "transcript": None}],
            status="completed",
        )
        self.items[item_id] = item
        self.item_order.append(item_id)

        self.input_buffer_committed = True
        self.pending_audio_item_id = item_id

        return item_id

    def clear_input_buffer(self) -> None:
        """Clear input audio buffer without committing."""
        self.input_buffer_samples = 0
        self.input_buffer_committed = False
        self.pending_audio_item_id = None

    def get_previous_item_id(self) -> str | None:
        """Get the ID of the last item in the conversation."""
        if not self.item_order:
            return None
        return self.item_order[-1]

    def snapshot(self) -> dict[str, Any]:
        """Create a serializable snapshot of session state for recording/replay."""
        return {
            "items": {
                item_id: item.model_dump() for item_id, item in self.items.items()
            },
            "item_order": list(self.item_order),
            "current_response_id": self.current_response_id,
            "current_item_id": self.current_item_id,
            "input_buffer_samples": self.input_buffer_samples,
            "input_buffer_committed": self.input_buffer_committed,
            "pending_audio_item_id": self.pending_audio_item_id,
            "response_queue": list(self.response_queue),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore session state from a snapshot."""
        self.items = {
            item_id: ora.Item(**item_data)
            for item_id, item_data in snapshot.get("items", {}).items()
        }
        self.item_order = list(snapshot.get("item_order", []))
        self.current_response_id = snapshot.get("current_response_id")
        self.current_item_id = snapshot.get("current_item_id")
        self.input_buffer_samples = snapshot.get("input_buffer_samples", 0)
        self.input_buffer_committed = snapshot.get("input_buffer_committed", False)
        self.pending_audio_item_id = snapshot.get("pending_audio_item_id")
        self.response_queue = list(snapshot.get("response_queue", []))

        # Reset current response - would need to be re-established
        self.current_response = None
        self.output_items_in_progress.clear()

    def __repr__(self) -> str:
        return (
            f"SessionState("
            f"items={len(self.items)}, "
            f"response={'active' if self.current_response else 'none'}, "
            f"buffer_samples={self.input_buffer_samples})"
        )
