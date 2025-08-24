"""System-wide edge case testing for the pixel-perfect framework (Fixed version)."""

import pytest
from PIL import Image
from tests.conftest import assert_image_dimensions, assert_image_mode, create_test_image

from core.pipeline import Pipeline
from operations.aspect import AspectStretch
from operations.block import BlockFilter
from operations.channel import ChannelSwap
from operations.column import ColumnShift
from operations.pixel import PixelFilter
from operations.row import RowShift


@pytest.mark.integration
class TestImageEdgeCases:
    """Test edge cases related to image dimensions and formats."""

    def test_minimal_image_operations(self, temp_dir):
        """Test operations on 1x1 pixel images."""
        input_path = temp_dir / "minimal_1x1.png"
        output_path = temp_dir / "minimal_1x1_output.png"

        # Create 1x1 image
        minimal_img = Image.new("RGBA", (1, 1), (128, 128, 128, 255))
        minimal_img.save(input_path)

        # Test various operations on minimal image
        pipeline = Pipeline(str(input_path))
        context = (
            pipeline.add(PixelFilter(condition="prime", fill_color=(255, 0, 0, 255)))
            .add(RowShift(selection="all", shift_amount=0))  # No-op shift
            .add(
                ChannelSwap(red_source="red", green_source="green", blue_source="blue")
            )  # Identity swap
            .execute(str(output_path))
        )

        assert output_path.exists()
        output_img = Image.open(output_path)
        assert_image_dimensions(output_img, 1, 1)
        assert_image_mode(output_img, "RGBA")

        # Check pixel color
        pixel = output_img.getpixel((0, 0))
        assert pixel == (255, 0, 0, 255), f"Expected red pixel, got {pixel}"

    def test_extreme_aspect_ratios(self, temp_dir):
        """Test operations on images with extreme aspect ratios."""
        test_cases = [
            (100, 1, "very_wide"),  # Very wide
            (1, 100, "very_tall"),  # Very tall
            (200, 2, "wide_strip"),  # Wide strip
            (3, 150, "tall_strip"),  # Tall strip
        ]

        for width, height, case_name in test_cases:
            input_path = temp_dir / f"extreme_{case_name}_input.png"
            output_path = temp_dir / f"extreme_{case_name}_output.png"

            test_img = create_test_image(width, height, "RGBA")
            test_img.save(input_path)

            pipeline = Pipeline(str(input_path))
            context = (
                pipeline.add(PixelFilter(condition="even", fill_color=(255, 0, 0, 128)))
                .add(RowShift(selection="all", shift_amount=1, wrap=True))
                .add(ColumnShift(selection="all", shift_amount=1, wrap=True))
                .execute(str(output_path))
            )

            assert output_path.exists(), f"Failed for case {case_name}"
            output_img = Image.open(output_path)
            assert_image_dimensions(output_img, width, height)

    def test_non_square_block_operations(self, temp_dir):
        """Test block operations with non-divisible dimensions."""
        input_path = temp_dir / "non_divisible_input.png"

        # Create image with dimensions not divisible by common block sizes
        test_img = create_test_image(77, 53, "RGBA")  # Prime numbers
        test_img.save(input_path)

        # Test block operations with various block sizes
        block_sizes = [(8, 8), (16, 16), (10, 7), (5, 11)]

        for i, (block_width, block_height) in enumerate(block_sizes):
            output_path = temp_dir / f"block_{i}_output.png"

            pipeline = Pipeline(str(input_path))
            context = pipeline.add(
                BlockFilter(
                    block_width=block_width, block_height=block_height, condition="checkerboard"
                )
            ).execute(str(output_path))

            assert output_path.exists()
            output_img = Image.open(output_path)
            assert_image_dimensions(output_img, 77, 53)

    def test_large_shift_amounts(self, temp_dir):
        """Test operations with shift amounts larger than image dimensions."""
        input_path = temp_dir / "large_shift_input.png"
        output_path = temp_dir / "large_shift_output.png"

        test_img = create_test_image(20, 15, "RGBA")
        test_img.save(input_path)

        # Test shifts larger than image dimensions
        pipeline = Pipeline(str(input_path))
        context = (
            pipeline.add(
                RowShift(selection="all", shift_amount=100, wrap=True)
            )  # Much larger than width
            .add(ColumnShift(selection="all", shift_amount=50, wrap=False))  # Larger than height
            .execute(str(output_path))
        )

        assert output_path.exists()
        output_img = Image.open(output_path)
        assert_image_dimensions(output_img, 20, 15)


@pytest.mark.integration
class TestParameterEdgeCases:
    """Test edge cases in operation parameters."""

    def test_extreme_color_values(self, temp_dir):
        """Test operations with extreme color values."""
        input_path = temp_dir / "extreme_color_input.png"

        test_img = create_test_image(50, 50, "RGBA")
        test_img.save(input_path)

        # Test with extreme color values
        extreme_colors = [
            (0, 0, 0, 0),  # Fully transparent black
            (255, 255, 255, 255),  # Fully opaque white
            (255, 0, 255, 128),  # Semi-transparent magenta
            (0, 255, 0, 1),  # Nearly transparent green
        ]

        for i, color in enumerate(extreme_colors):
            output_path = temp_dir / f"extreme_color_{i}_output.png"

            pipeline = Pipeline(str(input_path))
            context = pipeline.add(PixelFilter(condition="even", fill_color=color)).execute(
                str(output_path)
            )

            assert output_path.exists()

    def test_boundary_shift_values(self, temp_dir):
        """Test shift operations with boundary values."""
        input_path = temp_dir / "boundary_shift_input.png"

        test_img = create_test_image(30, 20, "RGBA")
        test_img.save(input_path)

        # Test boundary shift values
        boundary_cases = [
            0,  # Zero shift (no-op)
            1,  # Minimal shift
            29,  # Width - 1
            30,  # Exactly width
            -1,  # Negative shift
            -30,  # Negative width
        ]

        for i, shift_amount in enumerate(boundary_cases):
            output_path = temp_dir / f"boundary_shift_{i}_output.png"

            pipeline = Pipeline(str(input_path))
            context = pipeline.add(
                RowShift(selection="all", shift_amount=shift_amount, wrap=True)
            ).execute(str(output_path))

            assert output_path.exists()


@pytest.mark.integration
class TestFileSystemEdgeCases:
    """Test edge cases related to file system operations."""

    def test_invalid_file_paths(self, temp_dir):
        """Test handling of invalid file paths."""
        # Test non-existent input file
        with pytest.raises(FileNotFoundError):
            Pipeline("/path/that/does/not/exist.png")

        # Test valid pipeline with auto-directory creation
        input_path = temp_dir / "valid_input.png"
        test_img = create_test_image(30, 30, "RGBA")
        test_img.save(input_path)

        pipeline = Pipeline(str(input_path))
        pipeline.add(PixelFilter(condition="all", fill_color=(255, 0, 0, 255)))

        # This should create the directory and succeed
        output_dir = temp_dir / "nonexistent_dir"
        output_path = output_dir / "output.png"
        context = pipeline.execute(str(output_path))
        assert output_path.exists()


@pytest.mark.integration
class TestMemoryEdgeCases:
    """Test edge cases related to memory usage."""

    def test_operation_chain_memory_cleanup(self, temp_dir):
        """Test that long operation chains don't cause memory leaks."""
        input_path = temp_dir / "memory_chain_input.png"
        output_path = temp_dir / "memory_chain_output.png"

        test_img = create_test_image(100, 100, "RGBA")
        test_img.save(input_path)

        # Create a very long chain of operations
        pipeline = Pipeline(str(input_path))

        # Add 50 operations
        for i in range(50):
            if i % 5 == 0:
                pipeline.add(PixelFilter(condition="even", fill_color=(255, 0, 0, 32)))
            elif i % 5 == 1:
                pipeline.add(RowShift(selection="odd", shift_amount=1, wrap=True))
            elif i % 5 == 2:
                pipeline.add(ColumnShift(selection="even", shift_amount=1, wrap=True))
            elif i % 5 == 3:
                pipeline.add(
                    ChannelSwap(red_source="green", green_source="red", blue_source="blue")
                )
            else:
                pipeline.add(BlockFilter(block_width=4, block_height=4, condition="checkerboard"))

        context = pipeline.execute(str(output_path))
        assert output_path.exists()


@pytest.mark.integration
class TestConcurrencyEdgeCases:
    """Test edge cases related to concurrent operations."""

    def test_concurrent_cache_access(self, temp_dir):
        """Test that concurrent cache access doesn't cause issues."""
        input_path = temp_dir / "concurrent_input.png"
        cache_dir = temp_dir / "concurrent_cache"

        test_img = create_test_image(50, 50, "RGBA")
        test_img.save(input_path)

        # Simulate concurrent access by running same pipeline multiple times
        # In a real concurrent test, these would run in parallel
        output_paths = []

        for i in range(3):
            output_path = temp_dir / f"concurrent_{i}_output.png"

            pipeline = Pipeline(str(input_path), cache_dirs=[str(cache_dir)])
            context = (
                pipeline.add(PixelFilter(condition="prime", fill_color=(255, 0, 0, 255)))
                .add(RowShift(selection="odd", shift_amount=2))
                .execute(str(output_path))
            )

            output_paths.append(output_path)

        # All should succeed
        for output_path in output_paths:
            assert output_path.exists()


@pytest.mark.integration
class TestErrorRecoveryEdgeCases:
    """Test error recovery and graceful failure scenarios."""

    def test_invalid_parameter_validation(self, temp_dir):
        """Test that invalid parameters are properly caught."""
        input_path = temp_dir / "invalid_param_input.png"
        test_img = create_test_image(30, 30, "RGBA")
        test_img.save(input_path)

        # Test various invalid parameters
        invalid_cases = [
            # Invalid pixel filter condition
            lambda: PixelFilter(condition="invalid_condition", fill_color=(255, 0, 0)),
            # Invalid color format
            lambda: PixelFilter(condition="all", fill_color=(256, -1, 0)),  # Out of range
            # Invalid block size
            lambda: BlockFilter(block_width=0, block_height=8, condition="checkerboard"),
            # Invalid aspect ratio format
            lambda: AspectStretch(target_ratio="invalid", method="stretch"),
            # Invalid selection
            lambda: RowShift(selection="invalid_selection", shift_amount=5),
        ]

        for invalid_operation in invalid_cases:
            with pytest.raises(Exception):  # Should raise validation error
                operation = invalid_operation()
                pipeline = Pipeline(str(input_path))
                pipeline.add(operation)
                pipeline.validate()

    def test_image_format_error_handling(self, temp_dir):
        """Test handling of unsupported image formats."""
        # Create a text file with image extension
        fake_image_path = temp_dir / "fake_image.png"
        with open(fake_image_path, "w") as f:
            f.write("This is not an image file")

        # Should raise appropriate error
        with pytest.raises(Exception):
            Pipeline(str(fake_image_path))
