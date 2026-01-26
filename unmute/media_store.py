"""Per-session media storage for images and metadata.

Manages multimodal content with lifecycle tracking, storage quotas,
and efficient serialization for session snapshots and reconnects.
"""

import base64
import hashlib
import time
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

logger = getLogger(__name__)

# Storage limits to ensure Jetson compatibility
MAX_IMAGES_PER_SESSION = 20
MAX_METADATA_ITEMS_PER_SESSION = 100
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB
MAX_TOTAL_IMAGE_SIZE_BYTES = 20 * 1024 * 1024  # 20MB total


@dataclass
class ImageData:
    """Stored image with metadata."""

    image_id: str
    format: str  # jpeg, png, webp, gif
    data_b64: str  # Base64-encoded image
    size_bytes: int
    data_hash: str  # SHA256 hash for deduplication
    timestamp: float
    item_id: str | None = None  # Associated conversation item


@dataclass
class MetadataItem:
    """Stored metadata item."""

    metadata_id: str
    key: str
    value: Any
    timestamp: float
    item_id: str | None = None  # Associated conversation item


@dataclass
class MediaStore:
    """Per-session media storage manager.

    Tracks images and metadata with lifecycle management,
    deduplication, and storage quotas.
    """

    images: dict[str, ImageData] = field(default_factory=dict)
    metadata: dict[str, MetadataItem] = field(default_factory=dict)

    # Mapping from conversation item IDs to media IDs
    item_to_images: dict[str, list[str]] = field(default_factory=dict)
    item_to_metadata: dict[str, list[str]] = field(default_factory=dict)

    # Hash-based deduplication
    image_hash_to_id: dict[str, str] = field(default_factory=dict)

    def store_image(
        self,
        image_id: str,
        format: str,
        data_b64: str,
        item_id: str | None = None,
    ) -> str:
        """Store an image and return its ID.

        Args:
            image_id: Unique identifier for the image.
            format: Image format (jpeg, png, webp, gif).
            data_b64: Base64-encoded image data.
            item_id: Optional associated conversation item ID.

        Returns:
            Image ID (same as input if new, or existing ID if duplicate).

        Raises:
            ValueError: If storage limits are exceeded or image is invalid.
        """
        # Validate format
        if format not in ("jpeg", "png", "webp", "gif"):
            raise ValueError(f"Unsupported image format: {format}")

        # Decode to check size
        try:
            data_bytes = base64.b64decode(data_b64)
        except Exception as e:
            raise ValueError(f"Invalid base64 image data: {e}") from e

        size_bytes = len(data_bytes)

        # Check individual image size
        if size_bytes > MAX_IMAGE_SIZE_BYTES:
            raise ValueError(
                f"Image too large: {size_bytes} bytes "
                f"(max {MAX_IMAGE_SIZE_BYTES} bytes)"
            )

        # Compute hash for deduplication
        data_hash = hashlib.sha256(data_bytes).hexdigest()

        # Check if this exact image already exists
        if data_hash in self.image_hash_to_id:
            existing_id = self.image_hash_to_id[data_hash]
            logger.info(f"Image deduplicated: {image_id} -> {existing_id}")

            # Associate with item if provided
            if item_id is not None:
                if item_id not in self.item_to_images:
                    self.item_to_images[item_id] = []
                if existing_id not in self.item_to_images[item_id]:
                    self.item_to_images[item_id].append(existing_id)

            return existing_id

        # Check total storage limits
        if len(self.images) >= MAX_IMAGES_PER_SESSION:
            raise ValueError(
                f"Too many images in session "
                f"(max {MAX_IMAGES_PER_SESSION})"
            )

        total_size = sum(img.size_bytes for img in self.images.values())
        if total_size + size_bytes > MAX_TOTAL_IMAGE_SIZE_BYTES:
            raise ValueError(
                f"Total image storage exceeded: {total_size + size_bytes} bytes "
                f"(max {MAX_TOTAL_IMAGE_SIZE_BYTES} bytes)"
            )

        # Store the image
        image_data = ImageData(
            image_id=image_id,
            format=format,
            data_b64=data_b64,
            size_bytes=size_bytes,
            data_hash=data_hash,
            timestamp=time.time(),
            item_id=item_id,
        )

        self.images[image_id] = image_data
        self.image_hash_to_id[data_hash] = image_id

        # Associate with item
        if item_id is not None:
            if item_id not in self.item_to_images:
                self.item_to_images[item_id] = []
            self.item_to_images[item_id].append(image_id)

        logger.info(
            f"Stored image {image_id}: {format}, {size_bytes} bytes, "
            f"total images: {len(self.images)}"
        )

        return image_id

    def get_image(self, image_id: str) -> ImageData | None:
        """Retrieve image data by ID."""
        return self.images.get(image_id)

    def get_image_url(self, image_id: str) -> str | None:
        """Get data URL for an image suitable for LLM context.

        Returns a data URL like: data:image/jpeg;base64,/9j/4AAQ...
        """
        image = self.images.get(image_id)
        if image is None:
            return None

        return f"data:image/{image.format};base64,{image.data_b64}"

    def store_metadata(
        self,
        metadata_id: str,
        key: str,
        value: Any,
        timestamp: float | None = None,
        item_id: str | None = None,
    ) -> str:
        """Store a metadata item.

        Args:
            metadata_id: Unique identifier for the metadata.
            key: Metadata key (e.g., "sensor.depth", "robot.position").
            value: Metadata value (any JSON-serializable type).
            timestamp: Optional timestamp (defaults to current time).
            item_id: Optional associated conversation item ID.

        Returns:
            Metadata ID.

        Raises:
            ValueError: If storage limits are exceeded.
        """
        if len(self.metadata) >= MAX_METADATA_ITEMS_PER_SESSION:
            raise ValueError(
                f"Too many metadata items in session "
                f"(max {MAX_METADATA_ITEMS_PER_SESSION})"
            )

        metadata_item = MetadataItem(
            metadata_id=metadata_id,
            key=key,
            value=value,
            timestamp=timestamp if timestamp is not None else time.time(),
            item_id=item_id,
        )

        self.metadata[metadata_id] = metadata_item

        # Associate with item
        if item_id is not None:
            if item_id not in self.item_to_metadata:
                self.item_to_metadata[item_id] = []
            self.item_to_metadata[item_id].append(metadata_id)

        logger.info(
            f"Stored metadata {metadata_id}: {key}={value}, "
            f"total metadata: {len(self.metadata)}"
        )

        return metadata_id

    def get_metadata(self, metadata_id: str) -> MetadataItem | None:
        """Retrieve metadata by ID."""
        return self.metadata.get(metadata_id)

    def cleanup_item_media(self, item_id: str) -> None:
        """Remove media associated with a deleted conversation item.

        Args:
            item_id: Conversation item ID.
        """
        # Remove images
        if item_id in self.item_to_images:
            for image_id in self.item_to_images[item_id]:
                if image_id in self.images:
                    image = self.images[image_id]
                    # Remove from hash index
                    if image.data_hash in self.image_hash_to_id:
                        del self.image_hash_to_id[image.data_hash]
                    # Remove image
                    del self.images[image_id]
                    logger.info(f"Removed image {image_id} for item {item_id}")

            del self.item_to_images[item_id]

        # Remove metadata
        if item_id in self.item_to_metadata:
            for metadata_id in self.item_to_metadata[item_id]:
                if metadata_id in self.metadata:
                    del self.metadata[metadata_id]
                    logger.info(f"Removed metadata {metadata_id} for item {item_id}")

            del self.item_to_metadata[item_id]

    def get_item_images(self, item_id: str) -> list[ImageData]:
        """Get all images associated with a conversation item."""
        if item_id not in self.item_to_images:
            return []

        return [
            self.images[img_id]
            for img_id in self.item_to_images[item_id]
            if img_id in self.images
        ]

    def get_item_metadata(self, item_id: str) -> list[MetadataItem]:
        """Get all metadata items associated with a conversation item."""
        if item_id not in self.item_to_metadata:
            return []

        return [
            self.metadata[meta_id]
            for meta_id in self.item_to_metadata[item_id]
            if meta_id in self.metadata
        ]

    def get_storage_stats(self) -> dict[str, Any]:
        """Get current storage statistics."""
        total_image_size = sum(img.size_bytes for img in self.images.values())

        return {
            "num_images": len(self.images),
            "max_images": MAX_IMAGES_PER_SESSION,
            "total_image_size_bytes": total_image_size,
            "max_total_image_size_bytes": MAX_TOTAL_IMAGE_SIZE_BYTES,
            "num_metadata": len(self.metadata),
            "max_metadata": MAX_METADATA_ITEMS_PER_SESSION,
        }

    def to_snapshot(self) -> dict[str, Any]:
        """Serialize to a snapshot for session persistence.

        Returns image references (not full data) for efficient storage.
        """
        return {
            "image_refs": [
                {
                    "id": img.image_id,
                    "format": img.format,
                    "size_bytes": img.size_bytes,
                    "data_hash": img.data_hash,
                    "timestamp": img.timestamp,
                    "item_id": img.item_id,
                }
                for img in self.images.values()
            ],
            "metadata_items": [
                {
                    "id": meta.metadata_id,
                    "key": meta.key,
                    "value": meta.value,
                    "timestamp": meta.timestamp,
                    "item_id": meta.item_id,
                }
                for meta in self.metadata.values()
            ],
        }

    def clear(self) -> None:
        """Clear all stored media."""
        self.images.clear()
        self.metadata.clear()
        self.item_to_images.clear()
        self.item_to_metadata.clear()
        self.image_hash_to_id.clear()
        logger.info("Cleared media store")
