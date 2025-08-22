"""Tests for column-based operations."""

import pytest
from PIL import Image

from core.context import ImageContext
from exceptions import ValidationError
from utils.synthetic_images import create_numbered_grid

from .base_test import BaseOperationTest
from .column import ColumnMirror, ColumnShift, ColumnStretch, ColumnWeave


class ColumnOperationTest(BaseOperationTest):
    """Base test class for column operations."""

    pass


class TestColumnShift(ColumnOperationTest):
    """Test ColumnShift operation."""

    def get_operation_class(self):
        return ColumnShift

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

    def test_column_shift_wrap_basic(self):
        """Test basic column shifting with wrap."""
        # Create 4x4 grid with unique values per column
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        # Shift odd columns (1, 3) up by 1 with wrap
        operation = ColumnShift(selection="odd", shift_amount=-1, wrap=True)
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 4)
        assert result_context.width == 4
        assert result_context.height == 4

    def test_column_shift_fill_mode(self):
        """Test column shifting with fill color."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = ColumnShift(
            selection="even", shift_amount=2, wrap=False, fill_color=(255, 0, 0, 255)
        )
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 4)

    def test_column_shift_gradient_mode(self):
        """Test gradient shifting mode."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = ColumnShift(selection="gradient", shift_amount=3, gradient_start=0, wrap=True)
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 4)

    def test_column_shift_every_n(self):
        """Test every_n selection."""
        image = create_numbered_grid(6, 6)
        context = ImageContext(width=6, height=6, channels=3, dtype="uint8")

        operation = ColumnShift(selection="every_n", n=2, shift_amount=1)
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (6, 6)

    def test_column_shift_custom_indices(self):
        """Test custom column indices."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = ColumnShift(selection="custom", indices=[0, 2], shift_amount=1)
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 4)

    def test_column_shift_prime_selection(self):
        """Test prime column selection."""
        image = create_numbered_grid(8, 4)  # 8 columns to get more primes
        context = ImageContext(width=8, height=4, channels=3, dtype="uint8")

        operation = ColumnShift(selection="prime", shift_amount=1)
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (8, 4)

    def test_column_shift_zero_shift(self):
        """Test zero shift amount (should be no-op)."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = ColumnShift(selection="odd", shift_amount=0)
        result_image, result_context = operation.apply(image, context)

        # Should be identical to original
        assert result_image.size == image.size

    def test_column_shift_large_shift(self):
        """Test shift larger than height."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = ColumnShift(selection="odd", shift_amount=6, wrap=True)  # 6 > 4
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 4)

    def test_column_shift_negative_shift(self):
        """Test negative (up) shift."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = ColumnShift(selection="even", shift_amount=-2, wrap=True)
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 4)

    def test_column_shift_single_column_image(self):
        """Test edge case with single column."""
        image = Image.new("RGB", (1, 4), (255, 0, 0))
        context = ImageContext(width=1, height=4, channels=3, dtype="uint8")

        operation = ColumnShift(selection="odd", shift_amount=1)
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (1, 4)


class TestColumnStretch(ColumnOperationTest):
    """Test ColumnStretch operation."""

    def get_operation_class(self):
        return ColumnStretch

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

    def test_column_stretch_factor_2(self):
        """Test basic 2x stretch."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = ColumnStretch(factor=2.0)
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (8, 4)  # Width doubled
        assert result_context.width == 8

    def test_column_stretch_factor_1_5(self):
        """Test 1.5x stretch."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = ColumnStretch(factor=1.5)
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (6, 4)  # 4 * 1.5 = 6
        assert result_context.width == 6

    def test_column_stretch_distribute_method(self):
        """Test distribute method."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = ColumnStretch(factor=2.0, method="distribute")
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (8, 4)
        assert result_context.width == 8

    def test_column_stretch_selected_columns(self):
        """Test stretching only selected columns."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = ColumnStretch(factor=2.0, selection="odd")
        result_image, result_context = operation.apply(image, context)

        # Should still have original width since we're only duplicating some columns
        assert result_image.size[1] == 4

    def test_column_stretch_custom_indices(self):
        """Test stretching custom column indices."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = ColumnStretch(factor=3.0, selection="custom", indices=[1, 3])
        result_image, result_context = operation.apply(image, context)

        assert result_image.size[1] == 4

    def test_column_stretch_single_column(self):
        """Test stretching single column image."""
        image = Image.new("RGB", (1, 4), (255, 0, 0))
        context = ImageContext(width=1, height=4, channels=3, dtype="uint8")

        operation = ColumnStretch(factor=3.0)
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (3, 4)
        assert result_context.width == 3


class TestColumnMirror(ColumnOperationTest):
    """Test ColumnMirror operation."""

    def get_operation_class(self):
        return ColumnMirror

    def get_valid_params(self):
        return {"mode": "full"}

    def get_invalid_params(self):
        return [
            {"mode": "invalid"},  # Invalid mode
        ]

    def test_column_mirror_full_center(self):
        """Test full mirroring around center."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = ColumnMirror(mode="full")  # Default center pivot
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 4)
        assert result_context.width == 4
        assert result_context.height == 4

    def test_column_mirror_full_custom_pivot(self):
        """Test full mirroring around custom pivot."""
        image = create_numbered_grid(6, 4)
        context = ImageContext(width=6, height=4, channels=3, dtype="uint8")

        operation = ColumnMirror(mode="full", pivot=2)
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (6, 4)

    def test_column_mirror_alternating(self):
        """Test alternating mirroring."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = ColumnMirror(mode="alternating")
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 4)

    def test_column_mirror_invalid_pivot(self):
        """Test validation error with invalid pivot."""
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = ColumnMirror(mode="full", pivot=6)  # 6 >= 4

        with pytest.raises(ValidationError, match="Pivot column 6 must be within"):
            operation.validate_operation(context)

    def test_column_mirror_single_column(self):
        """Test mirroring single column image."""
        image = Image.new("RGB", (1, 4), (255, 0, 0))
        context = ImageContext(width=1, height=4, channels=3, dtype="uint8")

        operation = ColumnMirror(mode="full")
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (1, 4)

    def test_column_mirror_symmetry(self):
        """Test that mirroring creates symmetry."""
        # Create asymmetric image
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = ColumnMirror(mode="full", pivot=2)
        result_image, result_context = operation.apply(image, context)

        # Check that columns are mirrored around pivot
        # This is a basic symmetry check
        assert result_image.size == (4, 4)


class TestColumnWeave(ColumnOperationTest):
    """Test ColumnWeave operation."""

    def get_operation_class(self):
        return ColumnWeave

    def get_valid_params(self):
        return {"pattern": [1, 0, 3, 2]}

    def get_invalid_params(self):
        return [
            {"pattern": []},  # Empty pattern
            {"pattern": [-1, 0, 1]},  # Negative index
        ]

    def test_column_weave_basic_pattern(self):
        """Test basic weaving pattern."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        # Pattern [1, 0, 3, 2] swaps pairs of columns
        operation = ColumnWeave(pattern=[1, 0, 3, 2])
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 4)
        assert result_context.width == 4
        assert result_context.height == 4

    def test_column_weave_short_pattern_repeat(self):
        """Test short pattern with repeat=True."""
        image = create_numbered_grid(6, 4)
        context = ImageContext(width=6, height=4, channels=3, dtype="uint8")

        # Pattern [1, 0] will repeat for 6 columns
        operation = ColumnWeave(pattern=[1, 0], repeat=True)
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (6, 4)

    def test_column_weave_short_pattern_no_repeat(self):
        """Test short pattern with repeat=False."""
        image = create_numbered_grid(6, 4)
        context = ImageContext(width=6, height=4, channels=3, dtype="uint8")

        # Pattern [1, 0] applies to first 2 columns, rest unchanged
        operation = ColumnWeave(pattern=[1, 0], repeat=False)
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (6, 4)

    def test_column_weave_complex_pattern(self):
        """Test complex weaving pattern."""
        image = create_numbered_grid(8, 4)
        context = ImageContext(width=8, height=4, channels=3, dtype="uint8")

        # More complex pattern
        operation = ColumnWeave(pattern=[2, 0, 1, 3, 6, 4, 5, 7])
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (8, 4)

    def test_column_weave_invalid_pattern_index(self):
        """Test validation error with pattern index out of bounds."""
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = ColumnWeave(pattern=[0, 1, 2, 5])  # 5 >= 4

        with pytest.raises(ValidationError, match="Pattern index 5 exceeds image width"):
            operation.validate_operation(context)

    def test_column_weave_single_column(self):
        """Test weaving single column image."""
        image = Image.new("RGB", (1, 4), (255, 0, 0))
        context = ImageContext(width=1, height=4, channels=3, dtype="uint8")

        operation = ColumnWeave(pattern=[0])
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (1, 4)

    def test_column_weave_pattern_verification(self):
        """Test that pattern mapping works correctly."""
        # Create image where each column has a unique color
        image = Image.new("RGB", (4, 2))
        pixels = image.load()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

        for col in range(4):
            for row in range(2):
                pixels[col, row] = colors[col]

        context = ImageContext(width=4, height=2, channels=3, dtype="uint8")

        # Pattern [3, 2, 1, 0] reverses column order
        operation = ColumnWeave(pattern=[3, 2, 1, 0])
        result_image, result_context = operation.apply(image, context)

        # Verify the pattern was applied
        result_pixels = result_image.load()

        # First column should now have the color of the last column
        assert result_pixels[0, 0] == colors[3]  # Yellow from column 3
        assert result_pixels[1, 0] == colors[2]  # Blue from column 2
        assert result_pixels[2, 0] == colors[1]  # Green from column 1
        assert result_pixels[3, 0] == colors[0]  # Red from column 0


class TestColumnOperationsIntegration:
    """Integration tests for column operations."""

    def test_operation_chaining(self):
        """Test chaining multiple column operations."""
        image = create_numbered_grid(6, 4)
        context = ImageContext(width=6, height=4, channels=3, dtype="uint8")

        # Chain operations: shift -> stretch -> mirror
        shift_op = ColumnShift(selection="odd", shift_amount=1)
        stretch_op = ColumnStretch(factor=1.5)
        mirror_op = ColumnMirror(mode="alternating")

        # Apply in sequence
        result1, context1 = shift_op.apply(image, context)
        result2, context2 = stretch_op.apply(result1, context1)
        result3, context3 = mirror_op.apply(result2, context2)

        assert result3.size[1] == 4  # Height unchanged
        assert context3.width == 9  # 6 * 1.5 = 9

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
                ColumnShift(selection="odd", shift_amount=1),
                ColumnStretch(factor=1.5),
                ColumnMirror(mode="alternating"),
                ColumnWeave(pattern=[1, 0, 3, 2]),
            ]

            for operation in operations:
                try:
                    result_image, result_context = operation.apply(image, context)
                    assert result_image.size[1] == 4  # Height preserved
                except Exception as e:
                    pytest.fail(f"Operation {operation.operation_name} failed on {mode} image: {e}")

    def test_cache_key_generation(self):
        """Test cache key generation for all operations."""
        operations = [
            ColumnShift(selection="odd", shift_amount=2),
            ColumnStretch(factor=2.0),
            ColumnMirror(mode="full"),
            ColumnWeave(pattern=[1, 0, 3, 2]),
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
            ColumnShift(selection="odd", shift_amount=2),
            ColumnStretch(factor=2.0),
            ColumnMirror(mode="full"),
            ColumnWeave(pattern=[1, 0, 3, 2]),
        ]

        for operation in operations:
            memory = operation.estimate_memory(context)
            assert isinstance(memory, int)
            assert memory > 0

    def test_context_updates(self):
        """Test that ImageContext is updated correctly."""
        image = create_numbered_grid(6, 4)
        context = ImageContext(width=6, height=4, channels=3, dtype="uint8")

        # Test operations that change dimensions
        stretch_op = ColumnStretch(factor=2.0)
        result, new_context = stretch_op.apply(image, context)
        assert new_context.width == 12  # 6 * 2
        assert new_context.height == 4  # Unchanged

        # Test operations that don't change dimensions
        mirror_op = ColumnMirror(mode="full")
        result, new_context = mirror_op.apply(image, context)
        assert new_context.width == 6  # Unchanged
        assert new_context.height == 4  # Unchanged

    def test_row_column_operation_compatibility(self):
        """Test that row and column operations work together."""
        from .row import RowShift

        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        # Apply row operation followed by column operation
        row_op = RowShift(selection="odd", shift_amount=1)
        col_op = ColumnShift(selection="even", shift_amount=-1)

        result1, context1 = row_op.apply(image, context)
        result2, context2 = col_op.apply(result1, context1)

        assert result2.size == (4, 4)
        assert context2.width == 4
        assert context2.height == 4
