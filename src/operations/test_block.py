"""Tests for block-based operations."""

import numpy as np
import pytest
from PIL import Image

from core.context import ImageContext
from exceptions import ProcessingError, ValidationError
from utils.synthetic_images import create_numbered_grid

from .base_test import BaseOperationTest
from .block import BlockFilter, BlockRotate, BlockScramble, BlockShift


class BlockOperationTest(BaseOperationTest):
    """Base test class for block operations."""

    pass


class TestBlockFilter(BlockOperationTest):
    """Test BlockFilter operation."""

    def get_operation_class(self):
        return BlockFilter

    def get_valid_params(self):
        return {"block_width": 4, "block_height": 4, "condition": "checkerboard"}

    def get_invalid_params(self):
        return [
            {"block_width": 0},  # Invalid block width
            {"block_height": 0},  # Invalid block height
            {"condition": "invalid"},  # Invalid condition
            {"condition": "custom"},  # Missing keep_blocks
            {"fill_color": (256, 0, 0, 0)},  # Invalid color value
            {"padding_mode": "invalid"},  # Invalid padding mode
        ]

    def test_block_filter_division_test(self):
        """Test 10×10 image with 3×3 blocks (specific test case from SPEC.md)."""
        # Create 10x10 image
        image = create_numbered_grid(10, 10)
        context = ImageContext(width=10, height=10, channels=3, dtype="uint8")

        # Apply block filter with 3x3 blocks
        operation = BlockFilter(
            block_width=3, block_height=3, condition="checkerboard", padding_mode="crop"
        )
        result_image, result_context = operation.apply(image, context)

        # With crop mode, 10x10 image with 3x3 blocks should become 9x9 (3*3 blocks)
        assert result_image.size == (9, 9)
        assert result_context.width == 9
        assert result_context.height == 9

    def test_block_filter_checkerboard_pattern(self):
        """Test checkerboard pattern filtering."""
        image = create_numbered_grid(8, 8)
        context = ImageContext(width=8, height=8, channels=3, dtype="uint8")

        operation = BlockFilter(block_width=2, block_height=2, condition="checkerboard")
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (8, 8)

    def test_block_filter_diagonal_pattern(self):
        """Test diagonal pattern filtering."""
        image = create_numbered_grid(8, 8)
        context = ImageContext(width=8, height=8, channels=3, dtype="uint8")

        operation = BlockFilter(block_width=2, block_height=2, condition="diagonal")
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (8, 8)

    def test_block_filter_corners_pattern(self):
        """Test corners pattern filtering."""
        image = create_numbered_grid(8, 8)
        context = ImageContext(width=8, height=8, channels=3, dtype="uint8")

        operation = BlockFilter(block_width=2, block_height=2, condition="corners")
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (8, 8)

    def test_block_filter_custom_blocks(self):
        """Test custom block selection."""
        image = create_numbered_grid(8, 8)
        context = ImageContext(width=8, height=8, channels=3, dtype="uint8")

        # Keep blocks 0, 2, 5 in a 4x4 grid (2x2 blocks)
        operation = BlockFilter(
            block_width=2, block_height=2, condition="custom", keep_blocks=[0, 2, 5]
        )
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (8, 8)

    def test_block_filter_padding_extend(self):
        """Test extend padding mode."""
        image = create_numbered_grid(10, 10)
        context = ImageContext(width=10, height=10, channels=3, dtype="uint8")

        operation = BlockFilter(
            block_width=3, block_height=3, condition="checkerboard", padding_mode="extend"
        )
        result_image, result_context = operation.apply(image, context)

        # Should extend to 12x12 (4*3 blocks)
        assert result_image.size == (12, 12)
        assert result_context.width == 12
        assert result_context.height == 12

    def test_block_filter_padding_fill(self):
        """Test fill padding mode."""
        image = create_numbered_grid(10, 10)
        context = ImageContext(width=10, height=10, channels=3, dtype="uint8")

        operation = BlockFilter(
            block_width=3,
            block_height=3,
            condition="checkerboard",
            padding_mode="fill",
            fill_color=(255, 0, 0, 255),
        )
        result_image, result_context = operation.apply(image, context)

        # Should extend to 12x12 (4*3 blocks)
        assert result_image.size == (12, 12)

    def test_block_filter_oversized_blocks(self):
        """Test validation error with blocks larger than image."""
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = BlockFilter(block_width=8, block_height=8)

        with pytest.raises(ValidationError, match="Block size .* cannot exceed image size"):
            operation.validate_operation(context)

    def test_block_filter_single_block(self):
        """Test with image smaller than block size."""
        context = ImageContext(width=2, height=2, channels=3, dtype="uint8")

        operation = BlockFilter(block_width=4, block_height=4)

        with pytest.raises(ValidationError):
            operation.validate_operation(context)


class TestBlockShift(BlockOperationTest):
    """Test BlockShift operation."""

    def get_operation_class(self):
        return BlockShift

    def get_valid_params(self):
        return {"block_width": 4, "block_height": 4, "shift_map": {0: 1, 1: 0}}

    def get_invalid_params(self):
        return [
            {"block_width": 0},  # Invalid block width
            {"shift_map": {}},  # Empty shift map
        ]

    def test_block_shift_mapping_test(self):
        """Test that blocks move to correct positions."""
        # Create 8x8 image (4 blocks of 4x4)
        image = create_numbered_grid(8, 8)
        context = ImageContext(width=8, height=8, channels=3, dtype="uint8")

        # Swap blocks 0 and 3 (top-left and bottom-right)
        operation = BlockShift(
            block_width=4, block_height=4, shift_map={0: 3, 3: 0}, swap_mode="move"
        )
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (8, 8)
        assert result_context.width == 8
        assert result_context.height == 8

    def test_block_shift_swap_mode(self):
        """Test swap mode operation."""
        image = create_numbered_grid(8, 8)
        context = ImageContext(width=8, height=8, channels=3, dtype="uint8")

        operation = BlockShift(
            block_width=4, block_height=4, shift_map={0: 1, 1: 0}, swap_mode="swap"
        )
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (8, 8)

    def test_block_shift_complex_mapping(self):
        """Test complex block shifting."""
        image = create_numbered_grid(12, 8)  # 3x2 grid of 4x4 blocks
        context = ImageContext(width=12, height=8, channels=3, dtype="uint8")

        # Rearrange multiple blocks
        operation = BlockShift(
            block_width=4,
            block_height=4,
            shift_map={0: 2, 1: 0, 2: 1},  # Rotate first row
            swap_mode="move",
        )
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (12, 8)

    def test_block_shift_invalid_indices(self):
        """Test validation error with invalid shift map indices."""
        image = create_numbered_grid(8, 8)
        context = ImageContext(width=8, height=8, channels=3, dtype="uint8")

        # Invalid destination index (4 blocks total: 0,1,2,3)
        operation = BlockShift(
            block_width=4,
            block_height=4,
            shift_map={0: 5},  # 5 is out of range
        )

        with pytest.raises((ValidationError, ProcessingError)):
            operation.apply(image, context)

    def test_block_shift_with_padding(self):
        """Test block shifting with non-divisible dimensions."""
        image = create_numbered_grid(10, 10)
        context = ImageContext(width=10, height=10, channels=3, dtype="uint8")

        operation = BlockShift(
            block_width=4, block_height=4, shift_map={0: 1, 1: 0}, padding_mode="crop"
        )
        result_image, result_context = operation.apply(image, context)

        # Should crop to 8x8 (2x2 grid of 4x4 blocks)
        assert result_image.size == (8, 8)


class TestBlockRotate(BlockOperationTest):
    """Test BlockRotate operation."""

    def get_operation_class(self):
        return BlockRotate

    def get_valid_params(self):
        return {"block_width": 4, "block_height": 4, "rotation": 90}

    def get_invalid_params(self):
        return [
            {"block_width": 0},  # Invalid block width
            {"rotation": 45},  # Invalid rotation angle
            {"selection": "custom"},  # Missing indices
        ]

    def test_block_rotate_degrees_test(self):
        """Test 90°, 180°, 270° rotations work correctly."""
        image = create_numbered_grid(8, 8)
        context = ImageContext(width=8, height=8, channels=3, dtype="uint8")

        # Test each rotation angle
        for rotation in [90, 180, 270]:
            operation = BlockRotate(
                block_width=4, block_height=4, rotation=rotation, selection="all"
            )
            result_image, result_context = operation.apply(image, context)

            assert result_image.size == (8, 8)
            assert result_context.width == 8
            assert result_context.height == 8

    def test_block_rotate_selected_blocks(self):
        """Test rotating only selected blocks."""
        image = create_numbered_grid(8, 8)
        context = ImageContext(width=8, height=8, channels=3, dtype="uint8")

        # Rotate only checkerboard blocks
        operation = BlockRotate(
            block_width=4, block_height=4, rotation=90, selection="checkerboard"
        )
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (8, 8)

    def test_block_rotate_custom_selection(self):
        """Test rotating custom block indices."""
        image = create_numbered_grid(8, 8)
        context = ImageContext(width=8, height=8, channels=3, dtype="uint8")

        operation = BlockRotate(
            block_width=4,
            block_height=4,
            rotation=180,
            selection="custom",
            indices=[0, 3],  # Top-left and bottom-right blocks
        )
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (8, 8)

    def test_block_rotate_single_block(self):
        """Test rotating single block."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = BlockRotate(block_width=4, block_height=4, rotation=90)
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 4)

    def test_block_rotate_asymmetric_blocks(self):
        """Test rotating non-square blocks."""
        image = create_numbered_grid(8, 6)
        context = ImageContext(width=8, height=6, channels=3, dtype="uint8")

        operation = BlockRotate(block_width=4, block_height=3, rotation=90)
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (8, 6)


class TestBlockScramble(BlockOperationTest):
    """Test BlockScramble operation."""

    def get_operation_class(self):
        return BlockScramble

    def get_valid_params(self):
        return {"block_width": 4, "block_height": 4, "seed": 42}

    def get_invalid_params(self):
        return [
            {"block_width": 0},  # Invalid block width
        ]

    def test_block_scramble_reproducibility(self):
        """Test that same seed produces same result."""
        image = create_numbered_grid(8, 8)
        context = ImageContext(width=8, height=8, channels=3, dtype="uint8")

        operation1 = BlockScramble(block_width=4, block_height=4, seed=42)
        result1, _ = operation1.apply(image, context)

        operation2 = BlockScramble(block_width=4, block_height=4, seed=42)
        result2, _ = operation2.apply(image, context)

        # Results should be identical
        assert np.array_equal(np.array(result1), np.array(result2))

    def test_block_scramble_different_seeds(self):
        """Test that different seeds produce different results."""
        image = create_numbered_grid(8, 8)
        context = ImageContext(width=8, height=8, channels=3, dtype="uint8")

        operation1 = BlockScramble(block_width=4, block_height=4, seed=42)
        result1, _ = operation1.apply(image, context)

        operation2 = BlockScramble(block_width=4, block_height=4, seed=123)
        result2, _ = operation2.apply(image, context)

        # Results should likely be different
        assert result1.size == result2.size == (8, 8)

    def test_block_scramble_with_exclusions(self):
        """Test scrambling with excluded blocks."""
        image = create_numbered_grid(8, 8)
        context = ImageContext(width=8, height=8, channels=3, dtype="uint8")

        operation = BlockScramble(
            block_width=4,
            block_height=4,
            seed=42,
            exclude=[0, 3],  # Keep corner blocks in place
        )
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (8, 8)

    def test_block_scramble_single_block(self):
        """Test scrambling single block (should be no-op)."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = BlockScramble(block_width=4, block_height=4, seed=42)
        result_image, result_context = operation.apply(image, context)

        # Single block should remain unchanged
        assert result_image.size == (4, 4)

    def test_block_scramble_many_small_blocks(self):
        """Test scrambling many small blocks."""
        image = create_numbered_grid(8, 8)
        context = ImageContext(width=8, height=8, channels=3, dtype="uint8")

        operation = BlockScramble(block_width=2, block_height=2, seed=42)
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (8, 8)

    def test_block_scramble_no_seed(self):
        """Test scrambling without seed (should still work)."""
        image = create_numbered_grid(8, 8)
        context = ImageContext(width=8, height=8, channels=3, dtype="uint8")

        operation = BlockScramble(block_width=4, block_height=4)  # No seed
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (8, 8)


class TestBlockOperationsIntegration:
    """Integration tests for block operations."""

    def test_operation_chaining(self):
        """Test chaining multiple block operations."""
        image = create_numbered_grid(8, 8)
        context = ImageContext(width=8, height=8, channels=3, dtype="uint8")

        # Chain operations: filter -> rotate -> scramble
        filter_op = BlockFilter(block_width=4, block_height=4, condition="checkerboard")
        rotate_op = BlockRotate(block_width=4, block_height=4, rotation=90)
        scramble_op = BlockScramble(block_width=4, block_height=4, seed=42)

        # Apply in sequence
        result1, context1 = filter_op.apply(image, context)
        result2, context2 = rotate_op.apply(result1, context1)
        result3, context3 = scramble_op.apply(result2, context2)

        assert result3.size == (8, 8)
        assert context3.width == 8
        assert context3.height == 8

    def test_different_image_modes(self):
        """Test operations on different image modes."""
        modes_to_test = [("L", 1), ("RGB", 3), ("RGBA", 4)]

        for mode, channels in modes_to_test:
            if mode == "L":
                image = Image.new("L", (8, 8), 128)
            elif mode == "RGB":
                image = Image.new("RGB", (8, 8), (128, 64, 192))
            else:  # RGBA
                image = Image.new("RGBA", (8, 8), (128, 64, 192, 255))

            context = ImageContext(width=8, height=8, channels=channels, dtype="uint8")

            # Test each operation type
            operations = [
                BlockFilter(block_width=4, block_height=4, condition="checkerboard"),
                BlockShift(block_width=4, block_height=4, shift_map={0: 1, 1: 0}),
                BlockRotate(block_width=4, block_height=4, rotation=90),
                BlockScramble(block_width=4, block_height=4, seed=42),
            ]

            for operation in operations:
                try:
                    result_image, result_context = operation.apply(image, context)
                    assert result_image.size == (8, 8)
                except Exception as e:
                    pytest.fail(f"Operation {operation.operation_name} failed on {mode} image: {e}")

    def test_cache_key_generation(self):
        """Test cache key generation for all operations."""
        operations = [
            BlockFilter(block_width=4, block_height=4, condition="checkerboard"),
            BlockShift(block_width=4, block_height=4, shift_map={0: 1}),
            BlockRotate(block_width=4, block_height=4, rotation=90),
            BlockScramble(block_width=4, block_height=4, seed=42),
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
            BlockFilter(block_width=10, block_height=10, condition="checkerboard"),
            BlockShift(block_width=10, block_height=10, shift_map={0: 1}),
            BlockRotate(block_width=10, block_height=10, rotation=90),
            BlockScramble(block_width=10, block_height=10, seed=42),
        ]

        for operation in operations:
            memory = operation.estimate_memory(context)
            assert isinstance(memory, int)
            assert memory > 0

    def test_block_utilities(self):
        """Test block utility functions."""
        from .block import _calculate_grid_dimensions, _get_block_bounds

        # Test grid calculation
        grid_dims = _calculate_grid_dimensions((8, 6), (2, 3))
        assert grid_dims == (4, 2)  # 4 columns, 2 rows

        # Test block bounds
        bounds = _get_block_bounds(0, (4, 2), (2, 3))
        assert bounds == (0, 0, 2, 3)  # Top-left block

        bounds = _get_block_bounds(5, (4, 2), (2, 3))
        assert bounds == (2, 3, 4, 6)  # Second row, second column

    def test_padding_modes_comparison(self):
        """Test different padding modes produce different results."""
        image = create_numbered_grid(10, 10)
        context = ImageContext(width=10, height=10, channels=3, dtype="uint8")

        # Test all padding modes
        crop_op = BlockFilter(
            block_width=3, block_height=3, condition="checkerboard", padding_mode="crop"
        )
        extend_op = BlockFilter(
            block_width=3, block_height=3, condition="checkerboard", padding_mode="extend"
        )
        fill_op = BlockFilter(
            block_width=3, block_height=3, condition="checkerboard", padding_mode="fill"
        )

        crop_result, crop_context = crop_op.apply(image, context)
        extend_result, extend_context = extend_op.apply(image, context)
        fill_result, fill_context = fill_op.apply(image, context)

        # Different padding modes should produce different sizes
        assert crop_result.size == (9, 9)  # Cropped
        assert extend_result.size == (12, 12)  # Extended
        assert fill_result.size == (12, 12)  # Filled

        # But extend and fill should have different content (though same size)
        assert not np.array_equal(np.array(extend_result), np.array(fill_result))

    def test_cross_operation_compatibility(self):
        """Test that block operations work with row/column operations."""
        from .column import ColumnShift
        from .row import RowShift

        image = create_numbered_grid(8, 8)
        context = ImageContext(width=8, height=8, channels=3, dtype="uint8")

        # Apply block operation followed by row/column operations
        block_op = BlockScramble(block_width=4, block_height=4, seed=42)
        row_op = RowShift(selection="odd", shift_amount=1)
        col_op = ColumnShift(selection="even", shift_amount=-1)

        result1, context1 = block_op.apply(image, context)
        result2, context2 = row_op.apply(result1, context1)
        result3, context3 = col_op.apply(result2, context2)

        assert result3.size == (8, 8)
        assert context3.width == 8
        assert context3.height == 8
