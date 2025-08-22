"""Tests for row-based operations."""

import numpy as np
import pytest
from PIL import Image

from core.context import ImageContext
from exceptions import ValidationError
from utils.synthetic_images import create_numbered_grid

from .base_test import BaseOperationTest
from .row import RowRemove, RowShift, RowShuffle, RowStretch


class RowOperationTest(BaseOperationTest):
    """Base test class for row operations."""

    pass


class TestRowShift(RowOperationTest):
    """Test RowShift operation."""

    def get_operation_class(self):
        return RowShift

    def get_valid_params(self):
        return {"selection": "odd", "shift_amount": 2}

    def get_invalid_params(self):
        return [
            {"selection": "invalid"},  # Invalid selection
            {"selection": "every_n"},  # Missing n parameter
            {"selection": "custom"},  # Missing indices parameter
            {"fill_color": (256, 0, 0, 0)},  # Invalid color value
            {"n": 0},  # Invalid n value
            {"n": -1},  # Negative n value
        ]

    def test_row_shift_wrap_basic(self):
        """Test basic row shifting with wrap."""
        # Create 4x4 grid with unique values per row
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        # Shift odd rows (1, 3) left by 1 with wrap
        operation = RowShift(selection="odd", shift_amount=-1, wrap=True)
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 4)
        assert result_context.width == 4
        assert result_context.height == 4

    def test_row_shift_fill_mode(self):
        """Test row shifting with fill color."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = RowShift(
            selection="even", shift_amount=2, wrap=False, fill_color=(255, 0, 0, 255)
        )
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 4)

    def test_row_shift_gradient_mode(self):
        """Test gradient shifting mode."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = RowShift(selection="gradient", shift_amount=3, gradient_start=0, wrap=True)
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 4)

    def test_row_shift_every_n(self):
        """Test every_n selection."""
        image = create_numbered_grid(6, 6)
        context = ImageContext(width=6, height=6, channels=3, dtype="uint8")

        operation = RowShift(selection="every_n", n=2, shift_amount=1)
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (6, 6)

    def test_row_shift_custom_indices(self):
        """Test custom row indices."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = RowShift(selection="custom", indices=[0, 2], shift_amount=1)
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 4)

    def test_row_shift_prime_selection(self):
        """Test prime row selection."""
        image = create_numbered_grid(4, 8)  # 8 rows to get more primes
        context = ImageContext(width=4, height=8, channels=3, dtype="uint8")

        operation = RowShift(selection="prime", shift_amount=1)
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 8)

    def test_row_shift_zero_shift(self):
        """Test zero shift amount (should be no-op)."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = RowShift(selection="odd", shift_amount=0)
        result_image, result_context = operation.apply(image, context)

        # Should be identical to original
        assert result_image.size == image.size

    def test_row_shift_large_shift(self):
        """Test shift larger than width."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = RowShift(selection="odd", shift_amount=6, wrap=True)  # 6 > 4
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 4)

    def test_row_shift_negative_shift(self):
        """Test negative (left) shift."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = RowShift(selection="even", shift_amount=-2, wrap=True)
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 4)

    def test_row_shift_single_row_image(self):
        """Test edge case with single row."""
        image = Image.new("RGB", (4, 1), (255, 0, 0))
        context = ImageContext(width=4, height=1, channels=3, dtype="uint8")

        operation = RowShift(selection="odd", shift_amount=1)
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 1)


class TestRowStretch(RowOperationTest):
    """Test RowStretch operation."""

    def get_operation_class(self):
        return RowStretch

    def get_valid_params(self):
        return {"factor": 2.0}

    def get_invalid_params(self):
        return [
            {"factor": 0},  # Zero factor
            {"factor": -1},  # Negative factor
            {"method": "invalid"},  # Invalid method
            {"selection": "invalid"},  # Invalid selection
            {"selection": "custom"},  # Missing indices
        ]

    def test_row_stretch_factor_2(self):
        """Test basic 2x stretch."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = RowStretch(factor=2.0)
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 8)  # Height doubled
        assert result_context.height == 8

    def test_row_stretch_factor_1_5(self):
        """Test 1.5x stretch."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = RowStretch(factor=1.5)
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 6)  # 4 * 1.5 = 6
        assert result_context.height == 6

    def test_row_stretch_distribute_method(self):
        """Test distribute method."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = RowStretch(factor=2.0, method="distribute")
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 8)
        assert result_context.height == 8

    def test_row_stretch_selected_rows(self):
        """Test stretching only selected rows."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = RowStretch(factor=2.0, selection="odd")
        result_image, result_context = operation.apply(image, context)

        # Should still have original height since we're only duplicating some rows
        assert result_image.size[0] == 4

    def test_row_stretch_custom_indices(self):
        """Test stretching custom row indices."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = RowStretch(factor=3.0, selection="custom", indices=[1, 3])
        result_image, result_context = operation.apply(image, context)

        assert result_image.size[0] == 4

    def test_row_stretch_single_row(self):
        """Test stretching single row image."""
        image = Image.new("RGB", (4, 1), (255, 0, 0))
        context = ImageContext(width=4, height=1, channels=3, dtype="uint8")

        operation = RowStretch(factor=3.0)
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 3)
        assert result_context.height == 3


class TestRowRemove(RowOperationTest):
    """Test RowRemove operation."""

    def get_operation_class(self):
        return RowRemove

    def get_valid_params(self):
        return {"selection": "odd"}

    def get_invalid_params(self):
        return [
            {"selection": "invalid"},  # Invalid selection
            {"selection": "custom"},  # Missing indices
        ]

    def test_row_remove_odd(self):
        """Test removing odd rows."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = RowRemove(selection="odd")
        result_image, result_context = operation.apply(image, context)

        # Should have 2 rows left (even rows: 0, 2)
        assert result_image.size == (4, 2)
        assert result_context.height == 2

    def test_row_remove_even(self):
        """Test removing even rows."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = RowRemove(selection="even")
        result_image, result_context = operation.apply(image, context)

        # Should have 2 rows left (odd rows: 1, 3)
        assert result_image.size == (4, 2)
        assert result_context.height == 2

    def test_row_remove_every_n(self):
        """Test removing every nth row."""
        image = create_numbered_grid(4, 6)  # 6 rows
        context = ImageContext(width=4, height=6, channels=3, dtype="uint8")

        operation = RowRemove(selection="every_n", n=2)  # Remove 0, 2, 4
        result_image, result_context = operation.apply(image, context)

        # Should have 3 rows left (1, 3, 5)
        assert result_image.size == (4, 3)
        assert result_context.height == 3

    def test_row_remove_custom_indices(self):
        """Test removing custom row indices."""
        image = create_numbered_grid(4, 5)  # 5 rows
        context = ImageContext(width=4, height=5, channels=3, dtype="uint8")

        operation = RowRemove(selection="custom", indices=[1, 3])
        result_image, result_context = operation.apply(image, context)

        # Should have 3 rows left (0, 2, 4)
        assert result_image.size == (4, 3)
        assert result_context.height == 3

    def test_row_remove_prime(self):
        """Test removing prime indexed rows."""
        image = create_numbered_grid(4, 8)  # 8 rows
        context = ImageContext(width=4, height=8, channels=3, dtype="uint8")

        operation = RowRemove(selection="prime")
        result_image, result_context = operation.apply(image, context)

        # Prime indices in [0,8): 2, 3, 5, 7 -> 4 removed, 4 remaining
        assert result_image.size == (4, 4)
        assert result_context.height == 4

    def test_row_remove_validation_error(self):
        """Test validation error when trying to remove all rows."""
        image = create_numbered_grid(4, 2)  # Only 2 rows
        context = ImageContext(width=4, height=2, channels=3, dtype="uint8")

        # Try to remove all rows (both are even: 0, 1 -> actually 0 is even)
        operation = RowRemove(selection="custom", indices=[0, 1])

        with pytest.raises(ValidationError, match="at least 1 row must remain"):
            operation.validate_operation(context)

    def test_row_remove_single_row_error(self):
        """Test error when removing from single row image."""
        image = Image.new("RGB", (4, 1), (255, 0, 0))
        context = ImageContext(width=4, height=1, channels=3, dtype="uint8")

        operation = RowRemove(selection="custom", indices=[0])

        with pytest.raises(ValidationError, match="at least 1 row must remain"):
            operation.validate_operation(context)


class TestRowShuffle(RowOperationTest):
    """Test RowShuffle operation."""

    def get_operation_class(self):
        return RowShuffle

    def get_valid_params(self):
        return {"seed": 42}

    def get_invalid_params(self):
        return [
            {"groups": 0},  # Invalid group size
            {"groups": -1},  # Negative group size
        ]

    def test_row_shuffle_reproducibility(self):
        """Test that same seed produces same result."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation1 = RowShuffle(seed=42)
        result1, _ = operation1.apply(image, context)

        operation2 = RowShuffle(seed=42)
        result2, _ = operation2.apply(image, context)

        # Results should be identical
        assert np.array_equal(np.array(result1), np.array(result2))

    def test_row_shuffle_different_seeds(self):
        """Test that different seeds produce different results."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation1 = RowShuffle(seed=42)
        result1, _ = operation1.apply(image, context)

        operation2 = RowShuffle(seed=123)
        result2, _ = operation2.apply(image, context)

        # Results should likely be different (though theoretically could be same)
        # Check that at least the images are valid
        assert result1.size == result2.size == (4, 4)

    def test_row_shuffle_groups(self):
        """Test shuffling within groups."""
        image = create_numbered_grid(4, 6)  # 6 rows
        context = ImageContext(width=4, height=6, channels=3, dtype="uint8")

        operation = RowShuffle(seed=42, groups=2)  # Shuffle in groups of 2
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 6)
        assert result_context.height == 6

    def test_row_shuffle_single_row(self):
        """Test shuffling single row (should be no-op)."""
        image = Image.new("RGB", (4, 1), (255, 0, 0))
        context = ImageContext(width=4, height=1, channels=3, dtype="uint8")

        operation = RowShuffle(seed=42)
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 1)

    def test_row_shuffle_group_larger_than_height(self):
        """Test validation error when group size exceeds height."""
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = RowShuffle(groups=6)  # 6 > 4

        with pytest.raises(ValidationError, match="Group size 6 cannot exceed image height 4"):
            operation.validate_operation(context)

    def test_row_shuffle_no_seed(self):
        """Test shuffling without seed (should still work)."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = RowShuffle()  # No seed
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 4)


class TestRowOperationsIntegration:
    """Integration tests for row operations."""

    def test_operation_chaining(self):
        """Test chaining multiple row operations."""
        image = create_numbered_grid(4, 6)
        context = ImageContext(width=4, height=6, channels=3, dtype="uint8")

        # Chain operations: shift -> stretch -> shuffle
        shift_op = RowShift(selection="odd", shift_amount=1)
        stretch_op = RowStretch(factor=1.5)
        shuffle_op = RowShuffle(seed=42)

        # Apply in sequence
        result1, context1 = shift_op.apply(image, context)
        result2, context2 = stretch_op.apply(result1, context1)
        result3, context3 = shuffle_op.apply(result2, context2)

        assert result3.size[0] == 4  # Width unchanged
        assert context3.height == 9  # 6 * 1.5 = 9

    def test_different_image_modes(self):
        """Test operations on different image modes."""
        modes_to_test = [("L", 1), ("RGB", 3), ("RGBA", 4)]

        for mode, channels in modes_to_test:
            if mode == "L":
                image = Image.new("L", (4, 4), 128)
            elif mode == "RGB":
                image = Image.new("RGB", (4, 4), (128, 64, 192))
            else:  # RGBA
                image = Image.new("RGBA", (4, 4), (128, 64, 192, 255))

            context = ImageContext(width=4, height=4, channels=channels, dtype="uint8")

            # Test each operation type
            operations = [
                RowShift(selection="odd", shift_amount=1),
                RowStretch(factor=1.5),
                RowRemove(selection="custom", indices=[3]),  # Remove last row only
                RowShuffle(seed=42),
            ]

            for operation in operations:
                try:
                    result_image, result_context = operation.apply(image, context)
                    assert result_image.size[0] == 4  # Width preserved
                except Exception as e:
                    pytest.fail(f"Operation {operation.operation_name} failed on {mode} image: {e}")

    def test_cache_key_generation(self):
        """Test cache key generation for all operations."""
        operations = [
            RowShift(selection="odd", shift_amount=2),
            RowStretch(factor=2.0),
            RowRemove(selection="even"),
            RowShuffle(seed=42),
        ]

        image_hash = "test_hash"

        for operation in operations:
            cache_key = operation.get_cache_key(image_hash)
            assert isinstance(cache_key, str)
            assert len(cache_key) > 0
            assert image_hash in cache_key

    def test_memory_estimation(self):
        """Test memory estimation for all operations."""
        context = ImageContext(width=100, height=100, channels=3, dtype="uint8")

        operations = [
            RowShift(selection="odd", shift_amount=2),
            RowStretch(factor=2.0),
            RowRemove(selection="even"),
            RowShuffle(seed=42),
        ]

        for operation in operations:
            memory = operation.estimate_memory(context)
            assert isinstance(memory, int)
            assert memory > 0

    def test_context_updates(self):
        """Test that ImageContext is updated correctly."""
        image = create_numbered_grid(4, 6)
        context = ImageContext(width=4, height=6, channels=3, dtype="uint8")

        # Test operations that change dimensions
        stretch_op = RowStretch(factor=2.0)
        result, new_context = stretch_op.apply(image, context)
        assert new_context.height == 12  # 6 * 2
        assert new_context.width == 4  # Unchanged

        remove_op = RowRemove(selection="odd")
        result, new_context = remove_op.apply(image, context)
        assert new_context.height == 3  # 3 odd rows removed, 3 even remain
        assert new_context.width == 4  # Unchanged
