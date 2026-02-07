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
from unmute.media_store import MediaStore


@dataclass
class SessionState:
    """Per-session state machine for OpenAI Realtime protocol."""

    # Session configuration
    session: ora.Session = field(default_factory=lambda: ora.Session())

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

    # Input image buffer state
    pending_image_data: str | None = None  # Base64-encoded image
    pending_image_format: str | None = None
    pending_image_item_id: str | None = None

    # Media storage
    media_store: MediaStore = field(default_factory=MediaStore)

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

    def complete_response(self, status: str = "completed") -> ora.Response:
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
        """Commit input buffer and return the created item ID.

        If a pending item ID was pre-allocated (e.g., for speech_started event),
        use that ID. Otherwise create a new one.
        """
        # Use pre-allocated item ID if available, otherwise create new
        item_id = self.pending_audio_item_id or ora.random_id("item")

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

    def get_pending_input_item_id(self) -> str:
        """Get or create a pending item ID for the current input audio buffer.

        This is used to pre-allocate an item ID when speech starts,
        before the buffer is actually committed.
        """
        if self.pending_audio_item_id is None:
            self.pending_audio_item_id = ora.random_id("item")
        return self.pending_audio_item_id

    def append_image_buffer(self, image_b64: str, format: str | None = None) -> None:
        """Append image data to the input image buffer.

        Args:
            image_b64: Base64-encoded image data.
            format: Image format (jpeg, png, webp, gif). If None, will try to detect.
        """
        self.pending_image_data = image_b64
        self.pending_image_format = format or "jpeg"

    def commit_image_buffer(self) -> str:
        """Commit image buffer and return the created item ID.

        Stores the image in the media store and creates a conversation item.

        Returns:
            Item ID of the created image item.

        Raises:
            RuntimeError: If no image data is pending.
            ValueError: If image storage fails.
        """
        if self.pending_image_data is None:
            raise RuntimeError("No image data in buffer to commit")

        # Generate IDs
        item_id = self.pending_image_item_id or ora.random_id("item")
        image_id = ora.random_id("img")

        # Store image in media store
        try:
            self.media_store.store_image(
                image_id=image_id,
                format=self.pending_image_format or "jpeg",
                data_b64=self.pending_image_data,
                item_id=item_id,
            )
        except ValueError as e:
            # Clear buffer and re-raise
            self.clear_image_buffer()
            raise e

        # Create conversation item with image reference
        image_url = self.media_store.get_image_url(image_id)
        item = ora.Item(
            id=item_id,
            type="message",
            role="user",
            content=[{"type": "input_image", "image_url": {"url": image_url}}],
            status="completed",
        )
        self.items[item_id] = item
        self.item_order.append(item_id)

        # Clear buffer
        self.pending_image_data = None
        self.pending_image_format = None
        self.pending_image_item_id = None

        return item_id

    def clear_image_buffer(self) -> None:
        """Clear input image buffer without committing."""
        self.pending_image_data = None
        self.pending_image_format = None
        self.pending_image_item_id = None

    def append_metadata(
        self, key: str, value: Any, timestamp: float | None = None
    ) -> str:
        """Append metadata item to conversation.

        Unlike images, metadata items are committed immediately without buffering.

        Args:
            key: Metadata key (e.g., "sensor.depth").
            value: Metadata value.
            timestamp: Optional timestamp.

        Returns:
            Item ID of the created metadata item.
        """
        item_id = ora.random_id("item")
        metadata_id = ora.random_id("meta")

        # Store in media store
        self.media_store.store_metadata(
            metadata_id=metadata_id,
            key=key,
            value=value,
            timestamp=timestamp,
            item_id=item_id,
        )

        # Create conversation item
        item = ora.Item(
            id=item_id,
            type="message",
            role="user",
            content=[
                {"type": "metadata", "key": key, "value": value, "timestamp": timestamp}
            ],
            status="completed",
        )
        self.items[item_id] = item
        self.item_order.append(item_id)

        return item_id

    def delete_item(self, item_id: str) -> bool:
        """Delete an item from conversation and clean up associated media.

        Returns True if item existed.
        """
        if item_id not in self.items:
            return False

        # Clean up associated media
        self.media_store.cleanup_item_media(item_id)

        # Delete item
        del self.items[item_id]
        try:
            self.item_order.remove(item_id)
        except ValueError:
            pass
        return True

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
            "pending_image_format": self.pending_image_format,
            "pending_image_item_id": self.pending_image_item_id,
            "response_queue": list(self.response_queue),
            "media_store": self.media_store.to_snapshot(),
            "session": self.session.model_dump(),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore session state from a snapshot.

        Note: Media store restoration is partial - only references are restored,
        not full image data. Clients must re-upload images after reconnect.
        """
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
        self.pending_image_format = snapshot.get("pending_image_format")
        self.pending_image_item_id = snapshot.get("pending_image_item_id")
        self.response_queue = list(snapshot.get("response_queue", []))

        # Restore session config
        session_data = snapshot.get("session")
        if session_data:
            self.session = ora.Session(**session_data)

        # Note: media_store is NOT fully restored - only metadata references
        # Actual image data would need to be re-uploaded after reconnect
        # This is intentional to avoid storing large amounts of binary data

        # Reset current response - would need to be re-established
        self.current_response = None
        self.output_items_in_progress.clear()

    def __repr__(self) -> str:
        media_stats = self.media_store.get_storage_stats()
        return (
            f"SessionState("
            f"items={len(self.items)}, "
            f"response={'active' if self.current_response else 'none'}, "
            f"buffer_samples={self.input_buffer_samples}, "
            f"images={media_stats['num_images']}, "
            f"metadata={media_stats['num_metadata']})"
        )
