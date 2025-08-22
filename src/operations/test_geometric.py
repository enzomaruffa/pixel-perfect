"""Tests for geometric transformation operations."""

import numpy as np
import pytest

from core.context import ImageContext
from exceptions import ValidationError
from utils.synthetic_images import create_numbered_grid

from .geometric import GridWarp, PerspectiveStretch, RadialStretch


class TestGridWarp:
    """Test GridWarp operation."""

    def create_test_image(self, width=8, height=8):
        """Create a simple test image for geometric operations."""
        return create_numbered_grid(width, height)

    def test_grid_warp_creation(self):
        """Test GridWarp can be created with valid parameters."""
        operation = GridWarp(axis="horizontal", frequency=1.0, amplitude=5.0)
        assert operation.axis == "horizontal"
        assert operation.frequency == 1.0
        assert operation.amplitude == 5.0

    def test_grid_warp_invalid_parameters(self):
        """Test parameter validation with invalid inputs."""
        with pytest.raises(ValueError):  # Pydantic validation error
            GridWarp(frequency=0)

        with pytest.raises(ValueError):
            GridWarp(frequency=-1)

    def test_grid_warp_horizontal_basic(self):
        """Test basic horizontal warp."""
        image = self.create_test_image(8, 8)
        context = ImageContext(width=8, height=8, channels=3, dtype="uint8")

        operation = GridWarp(axis="horizontal", frequency=1.0, amplitude=2.0)
        validated_context = operation.validate_operation(context)
        result_image, result_context = operation.apply(image, validated_context)

        # Should maintain same dimensions
        assert result_image.size == image.size

        # Should have different content due to warping
        original_array = np.array(image)
        result_array = np.array(result_image)
        assert not np.array_equal(original_array, result_array)

    def test_grid_warp_vertical_basic(self):
        """Test basic vertical warp."""
        image = self.create_test_image(8, 8)
        context = ImageContext(width=8, height=8, channels=3, dtype="uint8")

        operation = GridWarp(axis="vertical", frequency=1.0, amplitude=2.0)
        validated_context = operation.validate_operation(context)
        result_image, result_context = operation.apply(image, validated_context)

        # Should maintain same dimensions
        assert result_image.size == image.size

        # Should have different content due to warping
        original_array = np.array(image)
        result_array = np.array(result_image)
        assert not np.array_equal(original_array, result_array)

    def test_grid_warp_both_axes(self):
        """Test warping on both axes."""
        image = self.create_test_image(8, 8)
        context = ImageContext(width=8, height=8, channels=3, dtype="uint8")

        operation = GridWarp(axis="both", frequency=1.0, amplitude=1.0)
        validated_context = operation.validate_operation(context)
        result_image, result_context = operation.apply(image, validated_context)

        # Should maintain same dimensions
        assert result_image.size == image.size

        # Should have different content due to warping
        original_array = np.array(image)
        result_array = np.array(result_image)
        assert not np.array_equal(original_array, result_array)

    def test_grid_warp_zero_amplitude(self):
        """Test that zero amplitude produces no change."""
        image = self.create_test_image(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = GridWarp(amplitude=0.0)
        validated_context = operation.validate_operation(context)
        result_image, result_context = operation.apply(image, validated_context)

        # Should be identical to original
        original_array = np.array(image)
        result_array = np.array(result_image)
        np.testing.assert_array_equal(original_array, result_array)

    def test_grid_warp_interpolation_methods(self):
        """Test different interpolation methods."""
        image = self.create_test_image(8, 8)
        context = ImageContext(width=8, height=8, channels=3, dtype="uint8")

        # Test nearest neighbor
        operation_nearest = GridWarp(interpolation="nearest", amplitude=2.0)
        validated_context = operation_nearest.validate_operation(context)
        result_nearest, _ = operation_nearest.apply(image, validated_context)

        # Test bilinear
        operation_bilinear = GridWarp(interpolation="bilinear", amplitude=2.0)
        validated_context = operation_bilinear.validate_operation(context)
        result_bilinear, _ = operation_bilinear.apply(image, validated_context)

        # Results should be different due to different interpolation
        nearest_array = np.array(result_nearest)
        bilinear_array = np.array(result_bilinear)
        assert not np.array_equal(nearest_array, bilinear_array)

    def test_grid_warp_large_amplitude_warning(self):
        """Test that large amplitude generates warning."""
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = GridWarp(amplitude=10.0)  # Large amplitude for 4x4 image
        validated_context = operation.validate_operation(context)

        assert len(validated_context.warnings) > 0

    def test_grid_warp_cache_key(self):
        """Test cache key generation."""
        operation = GridWarp(axis="horizontal", frequency=1.0, amplitude=5.0)
        cache_key = operation.get_cache_key("test_hash")
        assert isinstance(cache_key, str)
        assert "test_hash" in cache_key
        assert "gridwarp" in cache_key

    def test_grid_warp_memory_estimation(self):
        """Test memory estimation."""
        operation = GridWarp()
        context = ImageContext(width=8, height=8, channels=3, dtype="uint8")
        memory_estimate = operation.estimate_memory(context)
        assert memory_estimate > 0
        # Should be at least as large as the coordinate arrays
        min_expected = context.width * context.height * 2 * 8  # x,y coordinates
        assert memory_estimate >= min_expected


class TestPerspectiveStretch:
    """Test PerspectiveStretch operation."""

    def create_test_image(self, width=8, height=8):
        """Create a simple test image for geometric operations."""
        return create_numbered_grid(width, height)

    def test_perspective_stretch_creation(self):
        """Test PerspectiveStretch can be created with valid parameters."""
        operation = PerspectiveStretch(top_factor=1.0, bottom_factor=2.0)
        assert operation.top_factor == 1.0
        assert operation.bottom_factor == 2.0

    def test_perspective_stretch_invalid_parameters(self):
        """Test parameter validation with invalid inputs."""
        with pytest.raises(ValueError):  # Pydantic validation error
            PerspectiveStretch(top_factor=0)

        with pytest.raises(ValueError):
            PerspectiveStretch(bottom_factor=-1)

    def test_perspective_stretch_basic(self):
        """Test basic perspective stretch."""
        image = self.create_test_image(8, 8)
        context = ImageContext(width=8, height=8, channels=3, dtype="uint8")

        operation = PerspectiveStretch(top_factor=1.0, bottom_factor=2.0)
        validated_context = operation.validate_operation(context)
        result_image, result_context = operation.apply(image, validated_context)

        # Width should increase due to max factor
        assert result_context.width >= context.width
        # Height should remain the same
        assert result_context.height == context.height

    def test_perspective_stretch_no_change(self):
        """Test perspective stretch with same top and bottom factors."""
        image = self.create_test_image(4, 4)
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = PerspectiveStretch(top_factor=1.0, bottom_factor=1.0)
        validated_context = operation.validate_operation(context)
        result_image, result_context = operation.apply(image, validated_context)

        # Should maintain dimensions
        assert result_context.width == context.width
        assert result_context.height == context.height

    def test_perspective_stretch_extreme_factors_warning(self):
        """Test that extreme factors generate warnings."""
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        # Test large factor warning
        operation_large = PerspectiveStretch(top_factor=5.0, bottom_factor=1.0)
        validated_context = operation_large.validate_operation(context)
        assert len(validated_context.warnings) > 0

        # Test small factor warning
        operation_small = PerspectiveStretch(top_factor=0.05, bottom_factor=1.0)
        validated_context = operation_small.validate_operation(context)
        assert len(validated_context.warnings) > 0


class TestRadialStretch:
    """Test RadialStretch operation."""

    def create_test_image(self, width=8, height=8):
        """Create a simple test image for geometric operations."""
        return create_numbered_grid(width, height)

    def test_radial_stretch_creation(self):
        """Test RadialStretch can be created with valid parameters."""
        operation = RadialStretch(factor=1.5, falloff="linear")
        assert operation.factor == 1.5
        assert operation.falloff == "linear"

    def test_radial_stretch_invalid_parameters(self):
        """Test parameter validation with invalid inputs."""
        with pytest.raises(ValueError):  # Pydantic validation error
            RadialStretch(factor=0)

        with pytest.raises(ValueError):
            RadialStretch(factor=-1)

    def test_radial_stretch_expansion(self):
        """Test radial expansion."""
        image = self.create_test_image(8, 8)
        context = ImageContext(width=8, height=8, channels=3, dtype="uint8")

        operation = RadialStretch(factor=2.0)
        validated_context = operation.validate_operation(context)
        result_image, result_context = operation.apply(image, validated_context)

        # Expansion should increase dimensions
        assert result_context.width >= context.width
        assert result_context.height >= context.height

    def test_radial_stretch_contraction(self):
        """Test radial contraction."""
        image = self.create_test_image(8, 8)
        context = ImageContext(width=8, height=8, channels=3, dtype="uint8")

        operation = RadialStretch(factor=0.5)
        validated_context = operation.validate_operation(context)
        result_image, result_context = operation.apply(image, validated_context)

        # Contraction maintains same output size
        assert result_context.width == context.width
        assert result_context.height == context.height

    def test_radial_stretch_auto_center(self):
        """Test automatic center detection."""
        image = self.create_test_image(8, 8)
        context = ImageContext(width=8, height=8, channels=3, dtype="uint8")

        operation = RadialStretch(center="auto", factor=1.5)
        validated_context = operation.validate_operation(context)
        result_image, result_context = operation.apply(image, validated_context)

        # Should process without error
        assert result_image.size[0] > 0
        assert result_image.size[1] > 0

    def test_radial_stretch_custom_center(self):
        """Test custom center point."""
        image = self.create_test_image(8, 8)
        context = ImageContext(width=8, height=8, channels=3, dtype="uint8")

        operation = RadialStretch(center=(2, 2), factor=1.5)
        validated_context = operation.validate_operation(context)
        result_image, result_context = operation.apply(image, validated_context)

        # Should process without error
        assert result_image.size[0] > 0
        assert result_image.size[1] > 0

    def test_radial_stretch_invalid_center(self):
        """Test validation error with invalid center."""
        context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        operation = RadialStretch(center=(10, 10))  # Outside image bounds

        with pytest.raises(ValidationError):
            operation.validate_operation(context)

    def test_radial_stretch_falloff_functions(self):
        """Test different falloff functions."""
        image = self.create_test_image(8, 8)
        context = ImageContext(width=8, height=8, channels=3, dtype="uint8")

        # Test linear falloff
        operation_linear = RadialStretch(factor=1.5, falloff="linear")
        validated_context = operation_linear.validate_operation(context)
        result_linear, _ = operation_linear.apply(image, validated_context)

        # Test quadratic falloff
        operation_quadratic = RadialStretch(factor=1.5, falloff="quadratic")
        validated_context = operation_quadratic.validate_operation(context)
        result_quadratic, _ = operation_quadratic.apply(image, validated_context)

        # Test exponential falloff
        operation_exponential = RadialStretch(factor=1.5, falloff="exponential")
        validated_context = operation_exponential.validate_operation(context)
        result_exponential, _ = operation_exponential.apply(image, validated_context)

        # Results should be different due to different falloff functions
        linear_array = np.array(result_linear)
        quadratic_array = np.array(result_quadratic)
        exponential_array = np.array(result_exponential)

        assert not np.array_equal(linear_array, quadratic_array)
        assert not np.array_equal(linear_array, exponential_array)
        assert not np.array_equal(quadratic_array, exponential_array)


class TestGeometricOperationsIntegration:
    """Integration tests for geometric operations."""

    def create_test_image(self, width=8, height=8):
        """Create a simple test image for geometric operations."""
        return create_numbered_grid(width, height)

    def test_operation_chaining(self):
        """Test chaining multiple geometric operations."""
        image = self.create_test_image(8, 8)
        context = ImageContext(width=8, height=8, channels=3, dtype="uint8")

        # Chain operations
        warp_op = GridWarp(amplitude=1.0)
        perspective_op = PerspectiveStretch(top_factor=1.0, bottom_factor=1.5)
        radial_op = RadialStretch(factor=1.2)

        # Apply operations in sequence
        validated_context1 = warp_op.validate_operation(context)
        result1, context1 = warp_op.apply(image, validated_context1)

        validated_context2 = perspective_op.validate_operation(context1)
        result2, context2 = perspective_op.apply(result1, validated_context2)

        validated_context3 = radial_op.validate_operation(context2)
        result3, context3 = radial_op.apply(result2, validated_context3)

        # Final result should be valid
        assert result3.size[0] > 0
        assert result3.size[1] > 0
        assert context3.memory_estimate > 0

    def test_different_image_modes(self):
        """Test operations on different image modes."""
        # Test RGB
        rgb_image = create_numbered_grid(4, 4, "RGB")
        rgb_context = ImageContext(width=4, height=4, channels=3, dtype="uint8")

        # Test RGBA
        rgba_image = create_numbered_grid(4, 4, "RGBA")
        rgba_context = ImageContext(width=4, height=4, channels=4, dtype="uint8")

        # Test L (grayscale)
        l_image = create_numbered_grid(4, 4, "L")
        l_context = ImageContext(width=4, height=4, channels=1, dtype="uint8")

        operation = GridWarp(amplitude=1.0)

        # All modes should work
        validated_rgb_context = operation.validate_operation(rgb_context)
        rgb_result, _ = operation.apply(rgb_image, validated_rgb_context)

        validated_rgba_context = operation.validate_operation(rgba_context)
        rgba_result, _ = operation.apply(rgba_image, validated_rgba_context)

        validated_l_context = operation.validate_operation(l_context)
        l_result, _ = operation.apply(l_image, validated_l_context)

        assert rgb_result.mode == "RGB"
        assert rgba_result.mode == "RGBA"
        assert l_result.mode == "L"

    def test_cache_key_generation(self):
        """Test cache key generation for all operations."""
        operations = [GridWarp(), PerspectiveStretch(), RadialStretch()]

        for operation in operations:
            cache_key = operation.get_cache_key("test_hash")
            assert isinstance(cache_key, str)
            assert len(cache_key) > 0
            assert "test_hash" in cache_key

    def test_memory_estimation(self):
        """Test memory estimation for all operations."""
        context = ImageContext(width=16, height=16, channels=3, dtype="uint8")

        operations = [GridWarp(), PerspectiveStretch(), RadialStretch()]

        for operation in operations:
            memory_estimate = operation.estimate_memory(context)
            assert memory_estimate > 0
            # Should be at least as large as the input image
            min_expected = context.width * context.height * context.channels
            assert memory_estimate >= min_expected
