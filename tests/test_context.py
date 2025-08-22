"""Tests for ImageContext class."""

import pytest
from pydantic import ValidationError

from core.context import ImageContext


class TestImageContext:
    """Test ImageContext functionality."""

    def test_valid_context_creation(self):
        """Test creating valid ImageContext."""
        context = ImageContext(width=100, height=50, channels=3, dtype="uint8")
        assert context.width == 100
        assert context.height == 50
        assert context.channels == 3
        assert context.dtype == "uint8"
        assert context.warnings == []
        assert context.metadata == {}

    def test_context_with_warnings_and_metadata(self):
        """Test context with warnings and metadata."""
        warnings = ["Test warning"]
        metadata = {"test_key": "test_value"}

        context = ImageContext(
            width=10, height=10, channels=4, dtype="float32", warnings=warnings, metadata=metadata
        )

        assert context.warnings == warnings
        assert context.metadata == metadata

    def test_invalid_dimensions(self):
        """Test that invalid dimensions raise ValidationError."""
        with pytest.raises(ValidationError):
            ImageContext(width=0, height=10, channels=3, dtype="uint8")

        with pytest.raises(ValidationError):
            ImageContext(width=10, height=-5, channels=3, dtype="uint8")

    def test_invalid_channels(self):
        """Test that invalid channel counts raise ValidationError."""
        with pytest.raises(ValidationError):
            ImageContext(width=10, height=10, channels=0, dtype="uint8")

        with pytest.raises(ValidationError):
            ImageContext(width=10, height=10, channels=5, dtype="uint8")

        with pytest.raises(ValidationError):
            ImageContext(width=10, height=10, channels=2, dtype="uint8")

    def test_valid_channels(self):
        """Test that valid channel counts work."""
        # Grayscale
        context1 = ImageContext(width=10, height=10, channels=1, dtype="uint8")
        assert context1.channels == 1

        # RGB
        context3 = ImageContext(width=10, height=10, channels=3, dtype="uint8")
        assert context3.channels == 3

        # RGBA
        context4 = ImageContext(width=10, height=10, channels=4, dtype="uint8")
        assert context4.channels == 4

    def test_invalid_dtype(self):
        """Test that invalid dtypes raise ValidationError."""
        with pytest.raises(ValidationError):
            ImageContext(width=10, height=10, channels=3, dtype="int16")

    def test_valid_dtypes(self):
        """Test that valid dtypes work."""
        context1 = ImageContext(width=10, height=10, channels=3, dtype="uint8")
        assert context1.dtype == "uint8"

        context2 = ImageContext(width=10, height=10, channels=3, dtype="float32")
        assert context2.dtype == "float32"

    def test_add_warning(self):
        """Test adding warnings to context."""
        context = ImageContext(width=10, height=10, channels=3, dtype="uint8")

        assert len(context.warnings) == 0

        context.add_warning("First warning")
        assert len(context.warnings) == 1
        assert context.warnings[0] == "First warning"

        context.add_warning("Second warning")
        assert len(context.warnings) == 2
        assert context.warnings[1] == "Second warning"

    def test_copy_with_updates(self):
        """Test copying context with updates."""
        original = ImageContext(
            width=10,
            height=10,
            channels=3,
            dtype="uint8",
            warnings=["original warning"],
            metadata={"original": "data"},
        )

        # Copy with dimension changes
        updated = original.copy_with_updates(width=20, height=15)

        assert updated.width == 20
        assert updated.height == 15
        assert updated.channels == 3  # Unchanged
        assert updated.dtype == "uint8"  # Unchanged
        assert updated.warnings == ["original warning"]  # Preserved
        assert updated.metadata == {"original": "data"}  # Preserved

        # Original should be unchanged
        assert original.width == 10
        assert original.height == 10

    def test_total_pixels_property(self):
        """Test total_pixels property calculation."""
        context = ImageContext(width=100, height=50, channels=3, dtype="uint8")
        assert context.total_pixels == 5000

        context_square = ImageContext(width=10, height=10, channels=4, dtype="float32")
        assert context_square.total_pixels == 100

    def test_memory_estimate_property(self):
        """Test memory_estimate property calculation."""
        # uint8 RGB image: 100 * 50 * 3 * 1 = 15000 bytes
        context_uint8 = ImageContext(width=100, height=50, channels=3, dtype="uint8")
        assert context_uint8.memory_estimate == 15000

        # float32 RGBA image: 10 * 10 * 4 * 4 = 1600 bytes
        context_float32 = ImageContext(width=10, height=10, channels=4, dtype="float32")
        assert context_float32.memory_estimate == 1600

        # Grayscale uint8: 20 * 20 * 1 * 1 = 400 bytes
        context_gray = ImageContext(width=20, height=20, channels=1, dtype="uint8")
        assert context_gray.memory_estimate == 400

    def test_context_immutability_after_copy(self):
        """Test that copies are independent."""
        original = ImageContext(
            width=10,
            height=10,
            channels=3,
            dtype="uint8",
            warnings=["warning"],
            metadata={"key": "value"},
        )

        copy = original.copy_with_updates(width=20)

        # Modify copy's mutable fields
        copy.add_warning("new warning")
        copy.metadata["new_key"] = "new_value"

        # Original should be unchanged
        assert len(original.warnings) == 1
        assert "new_key" not in original.metadata

        # Copy should have changes
        assert len(copy.warnings) == 2
        assert copy.metadata["new_key"] == "new_value"

    @pytest.mark.edge_case
    def test_edge_case_minimum_image(self):
        """Test context for minimum possible image (1x1)."""
        context = ImageContext(width=1, height=1, channels=1, dtype="uint8")
        assert context.total_pixels == 1
        assert context.memory_estimate == 1

    @pytest.mark.edge_case
    def test_edge_case_large_dimensions(self):
        """Test context with large dimensions."""
        context = ImageContext(width=10000, height=10000, channels=4, dtype="float32")
        assert context.total_pixels == 100_000_000
        assert context.memory_estimate == 1_600_000_000  # 1.6GB (10000*10000*4*4)

    def test_channel_validation_error_message(self):
        """Test that channel validation provides clear error messages."""
        with pytest.raises(ValidationError) as exc_info:
            ImageContext(width=10, height=10, channels=2, dtype="uint8")

        error_message = str(exc_info.value)
        assert "Channels must be 1" in error_message
        assert "RGB" in error_message or "RGBA" in error_message
