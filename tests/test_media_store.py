"""Tests for media store functionality."""

import base64
import hashlib

import pytest

from unmute.media_store import (
    MAX_IMAGES_PER_SESSION,
    MAX_IMAGE_SIZE_BYTES,
    MAX_METADATA_ITEMS_PER_SESSION,
    MAX_TOTAL_IMAGE_SIZE_BYTES,
    MediaStore,
)


class TestImageStorage:
    """Tests for image storage functionality."""

    def test_store_valid_image(self):
        """Test storing a valid image."""
        store = MediaStore()
        image_data = base64.b64encode(b"fake_image_data").decode()

        image_id = store.store_image(
            image_id="img_123",
            format="jpeg",
            data_b64=image_data,
            item_id="item_456",
        )

        assert image_id == "img_123"
        assert "img_123" in store.images
        assert store.images["img_123"].format == "jpeg"
        assert store.images["img_123"].item_id == "item_456"

    def test_get_image_url(self):
        """Test getting data URL for an image."""
        store = MediaStore()
        image_data = base64.b64encode(b"fake_image_data").decode()

        store.store_image(
            image_id="img_123",
            format="png",
            data_b64=image_data,
        )

        url = store.get_image_url("img_123")
        assert url is not None
        assert url.startswith("data:image/png;base64,")
        assert image_data in url

    def test_get_nonexistent_image(self):
        """Test getting a nonexistent image."""
        store = MediaStore()
        assert store.get_image("nonexistent") is None
        assert store.get_image_url("nonexistent") is None

    def test_unsupported_image_format(self):
        """Test storing an image with unsupported format."""
        store = MediaStore()
        image_data = base64.b64encode(b"fake_image").decode()

        with pytest.raises(ValueError, match="Unsupported image format"):
            store.store_image(
                image_id="img_123",
                format="bmp",  # Not supported
                data_b64=image_data,
            )

    def test_invalid_base64(self):
        """Test storing invalid base64 data."""
        store = MediaStore()

        with pytest.raises(ValueError, match="Invalid base64"):
            store.store_image(
                image_id="img_123",
                format="jpeg",
                data_b64="not_valid_base64!!!",
            )

    def test_image_too_large(self):
        """Test storing an image that exceeds size limit."""
        store = MediaStore()
        # Create image larger than limit
        large_data = b"x" * (MAX_IMAGE_SIZE_BYTES + 1)
        image_data = base64.b64encode(large_data).decode()

        with pytest.raises(ValueError, match="Image too large"):
            store.store_image(
                image_id="img_123",
                format="jpeg",
                data_b64=image_data,
            )

    def test_too_many_images(self):
        """Test exceeding maximum number of images."""
        store = MediaStore()

        # Fill up to the limit
        for i in range(MAX_IMAGES_PER_SESSION):
            # Use different data to avoid deduplication
            image_data = base64.b64encode(f"image_{i}".encode()).decode()
            store.store_image(
                image_id=f"img_{i}",
                format="jpeg",
                data_b64=image_data,
            )

        # Try to add one more
        with pytest.raises(ValueError, match="Too many images"):
            image_data = base64.b64encode(b"one_more").decode()
            store.store_image(
                image_id="img_overflow",
                format="jpeg",
                data_b64=image_data,
            )

    def test_total_size_limit(self):
        """Test exceeding total storage size limit."""
        store = MediaStore()

        # Use images that are 2MB each (within individual limit of 5MB)
        # Total limit is 20MB, so store 9 images to fill 18MB,
        # then 10th image will exceed the 20MB limit
        single_image_size = 2 * 1024 * 1024  # 2MB
        large_data = b"x" * single_image_size

        # Store images until we're close to the limit
        # Store one fewer than would fill it completely
        num_to_store = (MAX_TOTAL_IMAGE_SIZE_BYTES // single_image_size) - 1
        for i in range(num_to_store):
            # Make each image unique to avoid deduplication
            unique_data = large_data + f"_image_{i}_".encode()
            image_data = base64.b64encode(unique_data).decode()
            store.store_image(
                image_id=f"img_{i}",
                format="jpeg",
                data_b64=image_data,
            )

        # Now try to store another large image that will exceed total
        with pytest.raises(ValueError, match="Total image storage exceeded"):
            overflow_data = large_data + b"_overflow_"
            image_data = base64.b64encode(overflow_data).decode()
            store.store_image(
                image_id="img_overflow",
                format="jpeg",
                data_b64=image_data,
            )

    def test_image_deduplication(self):
        """Test that identical images are deduplicated."""
        store = MediaStore()
        image_data = base64.b64encode(b"same_image").decode()

        # Store same image twice with different IDs
        id1 = store.store_image(
            image_id="img_1",
            format="jpeg",
            data_b64=image_data,
            item_id="item_1",
        )

        id2 = store.store_image(
            image_id="img_2",
            format="jpeg",
            data_b64=image_data,
            item_id="item_2",
        )

        # Should return the same ID (first one)
        assert id1 == id2 == "img_1"

        # Only one image should be stored
        assert len(store.images) == 1

        # Both items should reference the same image
        assert "item_1" in store.item_to_images
        assert "item_2" in store.item_to_images
        assert "img_1" in store.item_to_images["item_2"]


class TestMetadataStorage:
    """Tests for metadata storage functionality."""

    def test_store_metadata(self):
        """Test storing metadata."""
        store = MediaStore()

        metadata_id = store.store_metadata(
            metadata_id="meta_123",
            key="sensor.depth",
            value=1.234,
            timestamp=1234567890.0,
            item_id="item_456",
        )

        assert metadata_id == "meta_123"
        assert "meta_123" in store.metadata
        assert store.metadata["meta_123"].key == "sensor.depth"
        assert store.metadata["meta_123"].value == 1.234
        assert store.metadata["meta_123"].timestamp == 1234567890.0

    def test_get_metadata(self):
        """Test retrieving metadata."""
        store = MediaStore()

        store.store_metadata(
            metadata_id="meta_123",
            key="robot.position",
            value={"x": 1, "y": 2, "z": 3},
        )

        metadata = store.get_metadata("meta_123")
        assert metadata is not None
        assert metadata.key == "robot.position"
        assert metadata.value == {"x": 1, "y": 2, "z": 3}

    def test_get_nonexistent_metadata(self):
        """Test getting nonexistent metadata."""
        store = MediaStore()
        assert store.get_metadata("nonexistent") is None

    def test_too_many_metadata_items(self):
        """Test exceeding maximum metadata items."""
        store = MediaStore()

        # Fill up to the limit
        for i in range(MAX_METADATA_ITEMS_PER_SESSION):
            store.store_metadata(
                metadata_id=f"meta_{i}",
                key=f"sensor.{i}",
                value=i,
            )

        # Try to add one more
        with pytest.raises(ValueError, match="Too many metadata items"):
            store.store_metadata(
                metadata_id="meta_overflow",
                key="sensor.overflow",
                value=999,
            )

    def test_metadata_default_timestamp(self):
        """Test that metadata gets default timestamp if not provided."""
        store = MediaStore()

        store.store_metadata(
            metadata_id="meta_123",
            key="test.key",
            value="test_value",
        )

        metadata = store.get_metadata("meta_123")
        assert metadata is not None
        assert metadata.timestamp is not None
        assert metadata.timestamp > 0


class TestItemMediaManagement:
    """Tests for managing media associated with conversation items."""

    def test_get_item_images(self):
        """Test getting all images for an item."""
        store = MediaStore()

        # Store images for an item
        img1_data = base64.b64encode(b"image_1").decode()
        img2_data = base64.b64encode(b"image_2").decode()

        store.store_image("img_1", "jpeg", img1_data, item_id="item_123")
        store.store_image("img_2", "png", img2_data, item_id="item_123")

        images = store.get_item_images("item_123")
        assert len(images) == 2
        assert {img.image_id for img in images} == {"img_1", "img_2"}

    def test_get_item_metadata(self):
        """Test getting all metadata for an item."""
        store = MediaStore()

        store.store_metadata("meta_1", "sensor.temp", 25.0, item_id="item_123")
        store.store_metadata("meta_2", "sensor.humid", 60.0, item_id="item_123")

        metadata_items = store.get_item_metadata("item_123")
        assert len(metadata_items) == 2
        assert {m.metadata_id for m in metadata_items} == {"meta_1", "meta_2"}

    def test_cleanup_item_media(self):
        """Test cleaning up media when an item is deleted."""
        store = MediaStore()

        # Store image and metadata for an item
        img_data = base64.b64encode(b"image").decode()
        store.store_image("img_1", "jpeg", img_data, item_id="item_123")
        store.store_metadata("meta_1", "sensor.temp", 25.0, item_id="item_123")

        assert len(store.images) == 1
        assert len(store.metadata) == 1

        # Cleanup
        store.cleanup_item_media("item_123")

        assert len(store.images) == 0
        assert len(store.metadata) == 0
        assert "item_123" not in store.item_to_images
        assert "item_123" not in store.item_to_metadata

    def test_cleanup_nonexistent_item(self):
        """Test cleaning up media for nonexistent item."""
        store = MediaStore()
        # Should not raise error
        store.cleanup_item_media("nonexistent")


class TestStorageStats:
    """Tests for storage statistics."""

    def test_get_storage_stats(self):
        """Test getting storage statistics."""
        store = MediaStore()

        # Add some images and metadata
        img_data = base64.b64encode(b"test_image").decode()
        store.store_image("img_1", "jpeg", img_data)
        store.store_metadata("meta_1", "test.key", "test_value")

        stats = store.get_storage_stats()

        assert stats["num_images"] == 1
        assert stats["max_images"] == MAX_IMAGES_PER_SESSION
        assert stats["total_image_size_bytes"] > 0
        assert stats["max_total_image_size_bytes"] == MAX_TOTAL_IMAGE_SIZE_BYTES
        assert stats["num_metadata"] == 1
        assert stats["max_metadata"] == MAX_METADATA_ITEMS_PER_SESSION


class TestSnapshotSerialization:
    """Tests for snapshot serialization."""

    def test_to_snapshot(self):
        """Test creating a snapshot."""
        store = MediaStore()

        # Add data
        img_data = base64.b64encode(b"test_image").decode()
        store.store_image("img_1", "jpeg", img_data, item_id="item_123")
        store.store_metadata("meta_1", "sensor.temp", 25.0, item_id="item_123")

        snapshot = store.to_snapshot()

        assert "image_refs" in snapshot
        assert "metadata_items" in snapshot

        # Check image references (not full data)
        assert len(snapshot["image_refs"]) == 1
        img_ref = snapshot["image_refs"][0]
        assert img_ref["id"] == "img_1"
        assert img_ref["format"] == "jpeg"
        assert "data_hash" in img_ref
        assert "size_bytes" in img_ref

        # Check metadata
        assert len(snapshot["metadata_items"]) == 1
        meta_item = snapshot["metadata_items"][0]
        assert meta_item["id"] == "meta_1"
        assert meta_item["key"] == "sensor.temp"
        assert meta_item["value"] == 25.0

    def test_snapshot_no_image_data(self):
        """Test that snapshots don't include full image data."""
        store = MediaStore()

        # Store a large image
        large_data = b"x" * 10000
        img_data = base64.b64encode(large_data).decode()
        store.store_image("img_1", "jpeg", img_data)

        snapshot = store.to_snapshot()

        # Snapshot should be small (no image data)
        import json
        snapshot_json = json.dumps(snapshot)
        assert len(snapshot_json) < 1000  # Much smaller than original image


class TestClear:
    """Tests for clearing media store."""

    def test_clear(self):
        """Test clearing all media."""
        store = MediaStore()

        # Add data
        img_data = base64.b64encode(b"test_image").decode()
        store.store_image("img_1", "jpeg", img_data, item_id="item_123")
        store.store_metadata("meta_1", "sensor.temp", 25.0, item_id="item_123")

        assert len(store.images) == 1
        assert len(store.metadata) == 1

        # Clear
        store.clear()

        assert len(store.images) == 0
        assert len(store.metadata) == 0
        assert len(store.item_to_images) == 0
        assert len(store.item_to_metadata) == 0
        assert len(store.image_hash_to_id) == 0
