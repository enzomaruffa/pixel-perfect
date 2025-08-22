"""Tests for pixel-level operations."""

import pytest
from PIL import Image

from core.context import ImageContext
from exceptions import ValidationError
from utils.synthetic_images import create_gradient, create_numbered_grid

from .base_test import PixelOperationTest
from .pixel import PixelFilter, PixelMath, PixelSort


class TestPixelFilter(PixelOperationTest):
    """Test PixelFilter operation."""

    def get_operation_class(self):
        return PixelFilter

    def get_valid_params(self):
        return {"condition": "prime"}

    def get_invalid_params(self):
        return [
            {"condition": "invalid"},  # Invalid condition
            {"condition": "custom"},  # Missing custom_expression
            {"condition": "custom", "custom_expression": "import os"},  # Unsafe expression
            {"fill_color": (256, 0, 0, 0)},  # Invalid color value
        ]

    def test_prime_filter_4x4_grid(self):
        """Test prime filter on 4x4 grid (specific test case from SPEC.md)."""
        # Create 4x4 grid with numbered pixels (0-15)
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        # Apply prime filter
        operation = PixelFilter(condition="prime", fill_color=(0, 0, 0, 0))
        result_image, result_context = operation.apply(image, context)

        # Convert to RGBA for testing
        result_rgba = result_image.convert("RGBA")

        # Prime indices in 4x4 grid: 2, 3, 5, 7, 11, 13
        # These should be preserved, others should be transparent black
        prime_indices = {2, 3, 5, 7, 11, 13}

        for i in range(16):
            row = i // 4
            col = i % 4
            pixel = result_rgba.getpixel((col, row))

            if i in prime_indices:
                # Prime pixels should be preserved (not transparent black)
                assert pixel[3] > 0, f"Prime pixel at index {i} should not be transparent"
            else:
                # Non-prime pixels should be transparent black
                assert pixel == (0, 0, 0, 0), (
                    f"Non-prime pixel at index {i} should be transparent black"
                )

    def test_odd_even_filters(self):
        """Test odd and even filters."""
        image = create_numbered_grid(3, 3)  # 9 pixels
        context = ImageContext(width=3, height=3, channels=3, dtype="uint8")

        # Test odd filter
        odd_op = PixelFilter(condition="odd")
        odd_result, _ = odd_op.apply(image, context)

        # Test even filter
        even_op = PixelFilter(condition="even")
        even_result, _ = even_op.apply(image, context)

        # Both should return valid images
        assert odd_result.size == (3, 3)
        assert even_result.size == (3, 3)

    def test_custom_expression_filter(self):
        """Test custom expression filter."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        # Filter pixels where index is divisible by 3
        operation = PixelFilter(condition="custom", custom_expression="i % 3 == 0")
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 4)
        assert result_context.width == 4
        assert result_context.height == 4

    def test_2d_index_mode(self):
        """Test 2D indexing mode."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = PixelFilter(condition="even", index_mode="2d")
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 4)

    def test_preserve_alpha_option(self):
        """Test alpha preservation option."""
        # Create RGBA image
        image = Image.new("RGBA", (4, 4), (255, 0, 0, 128))
        context = ImageContext(width=4, height=4, channels=4, dtype="uint8")

        # Test with preserve_alpha=True
        op_preserve = PixelFilter(condition="even", preserve_alpha=True)
        result_preserve, _ = op_preserve.apply(image, context)

        # Test with preserve_alpha=False
        op_no_preserve = PixelFilter(condition="even", preserve_alpha=False)
        result_no_preserve, _ = op_no_preserve.apply(image, context)

        assert result_preserve.mode == "RGBA"
        assert result_no_preserve.mode == "RGBA"

    def test_fibonacci_filter(self):
        """Test Fibonacci number filter."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = PixelFilter(condition="fibonacci")
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 4)

    def test_invalid_custom_expression(self):
        """Test that invalid custom expressions raise errors."""
        with pytest.raises(ValidationError):
            PixelFilter(condition="custom", custom_expression="import os")


class TestPixelMath(PixelOperationTest):
    """Test PixelMath operation."""

    def get_operation_class(self):
        return PixelMath

    def get_valid_params(self):
        return {"expression": "r * 0.5"}

    def get_invalid_params(self):
        return [
            {"expression": "import os"},  # Unsafe expression
            {"channels": ["invalid"]},  # Invalid channel
            {"channels": []},  # Empty channels
            {"expression": ""},  # Empty expression
        ]

    def test_brightness_adjustment(self):
        """Test simple brightness adjustment."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        # Increase brightness by 50
        operation = PixelMath(expression="r + 50", channels=["r", "g", "b"])
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 4)
        assert result_context.width == 4
        assert result_context.height == 4

    def test_channel_mixing(self):
        """Test channel mixing expression."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        # Mix red and green channels
        operation = PixelMath(expression="r * 0.5 + g * 0.5", channels=["r"])
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 4)

    def test_position_based_transformation(self):
        """Test position-based transformation using x,y coordinates."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        # Create horizontal gradient based on x position
        operation = PixelMath(expression="x * 255 / width", channels=["r"])
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 4)

    def test_clamping_behavior(self):
        """Test value clamping."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        # Expression that would exceed 255
        operation = PixelMath(expression="r + 500", channels=["r"], clamp=True)
        result_image, result_context = operation.apply(image, context)

        # Values should be clamped to [0, 255]
        pixels = list(result_image.getdata())
        for pixel in pixels:
            assert all(0 <= c <= 255 for c in pixel[:3])

    def test_no_clamping(self):
        """Test behavior without clamping."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        # Simple multiplication that stays in range
        operation = PixelMath(expression="r * 0.5", channels=["r"], clamp=False)
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 4)

    def test_math_functions(self):
        """Test mathematical functions in expressions."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        # Use mathematical functions
        operation = PixelMath(expression="abs(r - 128)", channels=["r"])
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 4)

    def test_alpha_channel_math(self):
        """Test math operations on alpha channel."""
        image = Image.new("RGBA", (4, 4), (255, 0, 0, 128))
        context = ImageContext(width=4, height=4, channels=4, dtype="uint8")

        operation = PixelMath(expression="a * 2", channels=["a"], clamp=True)
        result_image, result_context = operation.apply(image, context)

        assert result_image.mode == "RGBA"


class TestPixelSort(PixelOperationTest):
    """Test PixelSort operation."""

    def get_operation_class(self):
        return PixelSort

    def get_valid_params(self):
        return {"direction": "horizontal", "sort_by": "brightness"}

    def get_invalid_params(self):
        return [
            {"direction": "invalid"},  # Invalid direction
            {"sort_by": "invalid"},  # Invalid sort criteria
            {"threshold": -1},  # Invalid threshold
            {"threshold": 256},  # Invalid threshold
        ]

    def test_horizontal_sorting(self):
        """Test horizontal pixel sorting."""
        image = create_gradient(4, 4, "horizontal")
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = PixelSort(direction="horizontal", sort_by="brightness")
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 4)
        assert result_context.width == 4
        assert result_context.height == 4

    def test_vertical_sorting(self):
        """Test vertical pixel sorting."""
        image = create_gradient(4, 4, "vertical")
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = PixelSort(direction="vertical", sort_by="brightness")
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 4)

    def test_diagonal_sorting(self):
        """Test diagonal pixel sorting."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = PixelSort(direction="diagonal", sort_by="brightness")
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 4)

    def test_color_channel_sorting(self):
        """Test sorting by different color channels."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        # Test sorting by each color channel
        for sort_by in ["red", "green", "blue"]:
            operation = PixelSort(direction="horizontal", sort_by=sort_by)
            result_image, result_context = operation.apply(image, context)
            assert result_image.size == (4, 4)

    def test_hue_saturation_sorting(self):
        """Test sorting by hue and saturation."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        # Test HSV-based sorting
        for sort_by in ["hue", "saturation"]:
            operation = PixelSort(direction="horizontal", sort_by=sort_by)
            result_image, result_context = operation.apply(image, context)
            assert result_image.size == (4, 4)

    def test_reverse_sorting(self):
        """Test reverse sort order."""
        image = create_gradient(4, 4, "horizontal")
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        # Normal sort
        normal_op = PixelSort(direction="horizontal", sort_by="brightness", reverse=False)
        normal_result, _ = normal_op.apply(image, context)

        # Reverse sort
        reverse_op = PixelSort(direction="horizontal", sort_by="brightness", reverse=True)
        reverse_result, _ = reverse_op.apply(image, context)

        assert normal_result.size == reverse_result.size
        # Results should be different (unless image is uniform)

    def test_threshold_sorting(self):
        """Test sorting with threshold."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = PixelSort(direction="horizontal", sort_by="brightness", threshold=128)
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (4, 4)

    def test_edge_case_1x1_image(self):
        """Test sorting on 1x1 image."""
        image = Image.new("RGB", (1, 1), (255, 0, 0))
        context = ImageContext(width=1, height=1, channels=3, dtype="uint8")

        operation = PixelSort(direction="horizontal", sort_by="brightness")
        result_image, result_context = operation.apply(image, context)

        assert result_image.size == (1, 1)


class TestPixelOperationsIntegration:
    """Integration tests for pixel operations."""

    def test_operation_chaining(self):
        """Test chaining multiple pixel operations."""
        image = create_numbered_grid(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        # Chain operations: filter -> math -> sort
        filter_op = PixelFilter(condition="even")
        math_op = PixelMath(expression="r * 1.2", channels=["r"])
        sort_op = PixelSort(direction="horizontal", sort_by="red")

        # Apply in sequence
        result1, context1 = filter_op.apply(image, context)
        result2, context2 = math_op.apply(result1, context1)
        result3, context3 = sort_op.apply(result2, context2)

        assert result3.size == (4, 4)
        assert context3.width == 4
        assert context3.height == 4

    def test_different_image_modes(self):
        """Test operations on different image modes."""
        modes_to_test = ["L", "RGB", "RGBA"]

        for mode in modes_to_test:
            if mode == "L":
                image = Image.new("L", (4, 4), 128)
                channels = 1
            elif mode == "RGB":
                image = Image.new("RGB", (4, 4), (128, 64, 192))
                channels = 3
            else:  # RGBA
                image = Image.new("RGBA", (4, 4), (128, 64, 192, 255))
                channels = 4

            context = ImageContext(width=4, height=4, channels=channels, dtype="uint8")

            # Test each operation type
            operations = [
                PixelFilter(condition="even"),
                PixelMath(expression="r * 0.8", channels=["r"]),
                PixelSort(direction="horizontal", sort_by="brightness"),
            ]

            for operation in operations:
                try:
                    result_image, result_context = operation.apply(image, context)
                    assert result_image.size == (4, 4)
                except Exception as e:
                    pytest.fail(f"Operation {operation.operation_name} failed on {mode} image: {e}")

    def test_cache_key_generation(self):
        """Test cache key generation for all operations."""
        operations = [
            PixelFilter(condition="prime"),
            PixelMath(expression="r * 0.5"),
            PixelSort(direction="horizontal", sort_by="brightness"),
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
            PixelFilter(condition="prime"),
            PixelMath(expression="r * 0.5"),
            PixelSort(direction="horizontal", sort_by="brightness"),
        ]

        for operation in operations:
            memory = operation.estimate_memory(context)
            assert isinstance(memory, int)
            assert memory > 0
            # Should be at least the base image size
            assert memory >= context.memory_estimate
