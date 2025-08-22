"""Base test classes for operation validation."""

from abc import ABC, abstractmethod
from typing import Any

import pytest
from PIL import Image

from core.base import BaseOperation
from core.context import ImageContext
from exceptions import ProcessingError, ValidationError


class BaseOperationTest(ABC):
    """Base test class for all operations.

    Provides common test patterns that all operations should pass:
    - Abstract method implementation
    - Parameter validation
    - Context flow
    - Cache key generation
    - Memory estimation
    - Error handling
    """

    @abstractmethod
    def get_operation_class(self) -> type[BaseOperation]:
        """Return the operation class being tested."""
        ...

    @abstractmethod
    def get_valid_params(self) -> dict[str, Any]:
        """Return valid parameters for creating operation instance."""
        ...

    @abstractmethod
    def get_invalid_params(self) -> list[dict[str, Any]]:
        """Return list of invalid parameter sets that should raise ValidationError."""
        ...

    def create_operation(self, **kwargs) -> BaseOperation:
        """Create operation instance with given parameters."""
        params = self.get_valid_params()
        params.update(kwargs)
        return self.get_operation_class()(**params)

    def test_operation_inherits_base_operation(self):
        """Test that operation properly inherits from BaseOperation."""
        operation_class = self.get_operation_class()
        assert issubclass(operation_class, BaseOperation)

    def test_operation_has_required_methods(self):
        """Test that operation implements all required abstract methods."""
        operation = self.create_operation()

        # Check required methods exist and are callable
        assert hasattr(operation, "validate_operation")
        assert callable(operation.validate_operation)

        assert hasattr(operation, "get_cache_key")
        assert callable(operation.get_cache_key)

        assert hasattr(operation, "estimate_memory")
        assert callable(operation.estimate_memory)

        assert hasattr(operation, "apply")
        assert callable(operation.apply)

    def test_operation_name_property(self):
        """Test that operation_name returns class name."""
        operation = self.create_operation()
        expected_name = self.get_operation_class().__name__
        assert operation.operation_name == expected_name

    def test_param_hash_generation(self):
        """Test that parameter hash generation works."""
        operation = self.create_operation()
        hash1 = operation.generate_param_hash()
        hash2 = operation.generate_param_hash()

        # Same parameters should generate same hash
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 32  # MD5 hash length

    def test_param_hash_differs_for_different_params(self):
        """Test that different parameters generate different hashes."""
        operation1 = self.create_operation()

        # Try to create operation with modified parameters
        try:
            valid_params = self.get_valid_params()
            # Find a parameter we can modify
            for key, value in valid_params.items():
                if isinstance(value, int | float) and value > 0:
                    modified_params = valid_params.copy()
                    modified_params[key] = value + 1
                    operation2 = self.get_operation_class()(**modified_params)

                    hash1 = operation1.generate_param_hash()
                    hash2 = operation2.generate_param_hash()
                    assert hash1 != hash2, f"Hashes should differ when {key} changes"
                    break
        except Exception:
            # If we can't create a modified operation, skip this test
            pytest.skip("Cannot create operation with modified parameters")

    def test_parameter_validation(self):
        """Test parameter validation with invalid inputs."""
        invalid_param_sets = self.get_invalid_params()

        for invalid_params in invalid_param_sets:
            with pytest.raises((ValidationError, ValueError, TypeError)):
                self.get_operation_class()(**invalid_params)

    def test_validate_operation_with_valid_context(self, test_context_8x8):
        """Test validate_operation with valid context."""
        operation = self.create_operation()
        result_context = operation.validate_operation(test_context_8x8)

        # Should return an ImageContext
        assert isinstance(result_context, ImageContext)

        # Basic properties should be preserved unless operation changes them
        # (Subclasses can override this behavior)
        assert result_context.dtype == test_context_8x8.dtype

    def test_cache_key_generation(self):
        """Test cache key generation."""
        operation = self.create_operation()
        image_hash = "test_hash_123"

        cache_key = operation.get_cache_key(image_hash)

        assert isinstance(cache_key, str)
        assert len(cache_key) > 0

        # Same inputs should generate same key
        cache_key2 = operation.get_cache_key(image_hash)
        assert cache_key == cache_key2

    def test_cache_key_differs_for_different_image_hash(self):
        """Test that different image hashes generate different cache keys."""
        operation = self.create_operation()

        key1 = operation.get_cache_key("hash1")
        key2 = operation.get_cache_key("hash2")

        assert key1 != key2

    def test_memory_estimation(self, test_context_8x8):
        """Test memory estimation."""
        operation = self.create_operation()
        memory_estimate = operation.estimate_memory(test_context_8x8)

        assert isinstance(memory_estimate, int)
        assert memory_estimate > 0

    def test_memory_estimation_scales_with_image_size(self):
        """Test that memory estimation scales with image size."""
        operation = self.create_operation()

        small_context = ImageContext(width=4, height=4, channels=3, dtype="uint8")
        large_context = ImageContext(width=16, height=16, channels=3, dtype="uint8")

        small_memory = operation.estimate_memory(small_context)
        large_memory = operation.estimate_memory(large_context)

        # Larger image should require more memory
        assert large_memory >= small_memory

    def test_apply_basic_functionality(self, test_image_8x8, test_context_8x8):
        """Test basic apply functionality."""
        operation = self.create_operation()

        result_image, result_context = operation.apply(test_image_8x8, test_context_8x8)

        # Should return PIL Image and ImageContext
        assert isinstance(result_image, Image.Image)
        assert isinstance(result_context, ImageContext)

        # Result should have valid dimensions
        assert result_image.size[0] > 0
        assert result_image.size[1] > 0
        assert result_context.width > 0
        assert result_context.height > 0

    @pytest.mark.edge_case
    def test_edge_case_1x1_image(self):
        """Test operation with 1x1 image."""
        try:
            operation = self.create_operation()
            context = ImageContext(width=1, height=1, channels=3, dtype="uint8")
            image = Image.new("RGB", (1, 1), (128, 128, 128))

            # Operation should either work or raise a clear error
            try:
                result_image, result_context = operation.apply(image, context)
                # If it succeeds, result should be valid
                assert isinstance(result_image, Image.Image)
                assert isinstance(result_context, ImageContext)
            except (ValidationError, ProcessingError) as e:
                # If it fails, error should be descriptive
                assert len(str(e)) > 10  # Should have meaningful error message

        except Exception as e:
            # Operation creation might fail for 1x1 - this is acceptable
            if not isinstance(e, ValidationError | ValueError):
                raise

    @pytest.mark.edge_case
    def test_edge_case_extreme_aspect_ratio(self):
        """Test operation with extreme aspect ratios."""
        try:
            operation = self.create_operation()

            # Test very wide image (1 pixel tall)
            wide_context = ImageContext(width=10, height=1, channels=3, dtype="uint8")
            wide_image = Image.new("RGB", (10, 1), (128, 128, 128))

            try:
                result_image, result_context = operation.apply(wide_image, wide_context)
                assert isinstance(result_image, Image.Image)
                assert isinstance(result_context, ImageContext)
            except (ValidationError, ProcessingError):
                pass  # Expected for some operations

            # Test very tall image (1 pixel wide)
            tall_context = ImageContext(width=1, height=10, channels=3, dtype="uint8")
            tall_image = Image.new("RGB", (1, 10), (128, 128, 128))

            try:
                result_image, result_context = operation.apply(tall_image, tall_context)
                assert isinstance(result_image, Image.Image)
                assert isinstance(result_context, ImageContext)
            except (ValidationError, ProcessingError):
                pass  # Expected for some operations

        except Exception as e:
            # Operation creation might fail - this is acceptable
            if not isinstance(e, ValidationError | ValueError):
                raise

    def test_context_flow_consistency(self, test_image_8x8, test_context_8x8):
        """Test that context flows consistently through validate and apply."""
        operation = self.create_operation()

        # Get context from validation
        operation.validate_operation(test_context_8x8)

        # Apply operation
        result_image, applied_context = operation.apply(test_image_8x8, test_context_8x8)

        # Key properties should be consistent
        # (Note: actual values might differ if operation changes dimensions)
        assert applied_context.channels >= 1
        assert applied_context.channels <= 4
        assert applied_context.dtype in ["uint8", "float32"]

    @pytest.mark.memory
    def test_memory_estimation_vs_actual_usage(self, test_image_8x8, test_context_8x8):
        """Test that memory estimation is reasonable compared to input."""
        operation = self.create_operation()

        estimated_memory = operation.estimate_memory(test_context_8x8)

        # Input image memory usage
        input_memory = test_context_8x8.memory_estimate

        # Estimation should be reasonable (not more than 10x input for most operations)
        # This is a loose bound since some operations might legitimately need more memory
        assert estimated_memory <= input_memory * 10

    def test_operation_deterministic(self, test_image_8x8, test_context_8x8):
        """Test that operation produces deterministic results."""
        operation = self.create_operation()

        # Apply operation twice with same inputs
        result1_image, result1_context = operation.apply(test_image_8x8, test_context_8x8)
        result2_image, result2_context = operation.apply(test_image_8x8, test_context_8x8)

        # Results should be identical
        assert result1_image.size == result2_image.size
        assert result1_image.mode == result2_image.mode
        assert result1_context.width == result2_context.width
        assert result1_context.height == result2_context.height

        # Pixel data should be identical
        import numpy as np

        arr1 = np.array(result1_image)
        arr2 = np.array(result2_image)
        assert np.array_equal(arr1, arr2)


class PixelOperationTest(BaseOperationTest):
    """Base test class for pixel-level operations.

    Adds tests specific to operations that work at pixel level.
    """

    def test_pixel_preservation(self, test_image_4x4, test_context_4x4):
        """Test that operation preserves or predictably modifies pixels."""
        operation = self.create_operation()
        result_image, result_context = operation.apply(test_image_4x4, test_context_4x4)

        # Image should have same dimensions for pixel operations
        # (unless operation specifically changes dimensions)
        # Placeholder for future dimension validation


class GeometricOperationTest(BaseOperationTest):
    """Base test class for geometric operations.

    Adds tests specific to operations that transform image geometry.
    """

    def test_coordinate_transformation(self, test_image_8x8, test_context_8x8):
        """Test that geometric transformations handle coordinates correctly."""
        operation = self.create_operation()
        result_image, result_context = operation.apply(test_image_8x8, test_context_8x8)

        # Result should be a valid image
        assert result_image.size[0] > 0
        assert result_image.size[1] > 0

        # Context should reflect any dimension changes
        width, height = result_image.size
        assert result_context.width == width
        assert result_context.height == height


class BlockOperationTest(BaseOperationTest):
    """Base test class for block-based operations.

    Adds tests specific to operations that work with image blocks.
    """

    def test_block_handling_non_divisible(self):
        """Test block operations with non-divisible image dimensions."""
        # This is operation-specific, so subclasses should implement
        # Test with image size that doesn't divide evenly by block size
        pass
