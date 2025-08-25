"""Simple integration test to verify test setup."""

import pytest
from PIL import Image

from operations.pixel import PixelFilter
from operations.row import RowShift
from tests.conftest import (
    assert_image_dimensions,
    assert_image_mode,
    create_test_image,
    create_test_pipeline,
)


@pytest.mark.integration
class TestSimpleIntegration:
    """Simple integration tests to verify test framework."""

    def test_basic_pipeline_execution(self, temp_dir):
        """Test basic pipeline execution works."""
        input_path = temp_dir / "simple_input.png"
        output_path = temp_dir / "simple_output.png"

        # Create test image
        test_img = create_test_image(50, 50, "RGBA")
        test_img.save(input_path)

        # Execute simple pipeline
        pipeline = create_test_pipeline(input_path)
        pipeline.add(PixelFilter(condition="even", fill_color=(255, 0, 0, 255))).execute(
            str(output_path)
        )

        # Verify output
        assert output_path.exists()
        output_img = Image.open(output_path)
        assert_image_dimensions(output_img, 50, 50)
        assert_image_mode(output_img, "RGBA")

    def test_two_operation_pipeline(self, temp_dir):
        """Test pipeline with two operations."""
        input_path = temp_dir / "two_op_input.png"
        output_path = temp_dir / "two_op_output.png"

        test_img = create_test_image(30, 30, "RGBA")
        test_img.save(input_path)

        pipeline = create_test_pipeline(input_path)
        (
            pipeline.add(PixelFilter(condition="odd", fill_color=(0, 255, 0, 255)))
            .add(RowShift(selection="even", shift_amount=2, wrap=True))
            .execute(str(output_path))
        )

        assert output_path.exists()
        output_img = Image.open(output_path)
        assert_image_dimensions(output_img, 30, 30)
        assert_image_mode(output_img, "RGBA")

    def test_test_utilities_work(self, temp_dir):
        """Test that test utilities are working."""
        # Test create_test_image
        img = create_test_image(25, 25, "RGBA")
        assert img.size == (25, 25)
        assert img.mode == "RGBA"

        # Test RGB mode
        img_rgb = create_test_image(10, 10, "RGB")
        assert img_rgb.size == (10, 10)
        assert img_rgb.mode == "RGB"

        # Test assertions work
        assert_image_dimensions(img, 25, 25)
        assert_image_mode(img, "RGBA")

    def test_temp_dir_fixture(self, temp_dir):
        """Test that temp_dir fixture works."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()

        # Create a file in temp dir
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        assert test_file.read_text() == "test content"
