"""End-to-end pipeline integration tests."""

import pytest
from PIL import Image
from pydantic import ValidationError

from core.pipeline import Pipeline
from operations.aspect import AspectStretch
from operations.block import BlockFilter
from operations.channel import ChannelSwap
from operations.column import ColumnShift
from operations.geometric import GridWarp
from operations.pattern import Dither, Mosaic
from operations.pixel import PixelFilter
from operations.row import RowShift
from presets import get_all_presets, get_preset
from tests.conftest import (
    assert_image_dimensions,
    assert_image_mode,
    assert_images_similar,
    create_test_image,
    create_test_pipeline,
)


@pytest.mark.integration
class TestEndToEndPipelines:
    """Test complete pipeline workflows."""

    def test_artistic_glitch_pipeline(self, temp_dir):
        """Test the artistic effect pipeline from SPEC.md."""
        # Create test input image
        input_path = temp_dir / "test_input.png"
        output_path = temp_dir / "test_output.png"

        test_img = create_test_image(100, 100, "RGBA")
        test_img.save(input_path)

        # Create and execute pipeline
        pipeline = create_test_pipeline(input_path)
        (
            pipeline.add(BlockFilter(block_width=8, block_height=8, condition="checkerboard"))
            .add(RowShift(selection="odd", shift_amount=3, wrap=True))
            .add(PixelFilter(condition="prime", fill_color=(255, 0, 0, 255)))
            .add(AspectStretch(target_ratio="1:1", method="segment"))
            .execute(str(output_path))
        )

        # Verify output
        assert output_path.exists()
        output_img = Image.open(output_path)
        assert_image_mode(output_img, "RGBA")
        assert output_img.size[0] == output_img.size[1]  # 1:1 aspect ratio

    def test_retro_effect_pipeline(self, temp_dir):
        """Test retro effect with channel manipulation and dithering."""
        input_path = temp_dir / "retro_input.png"
        output_path = temp_dir / "retro_output.png"

        # Create colorful test image
        test_img = create_test_image(80, 60, "RGBA")
        test_img.save(input_path)

        pipeline = create_test_pipeline(input_path)
        (
            pipeline.add(ChannelSwap(red_source="green", green_source="blue", blue_source="red"))
            .add(Dither(method="floyd_steinberg", levels=4))
            .execute(str(output_path))
        )

        assert output_path.exists()
        output_img = Image.open(output_path)
        assert_image_dimensions(output_img, 80, 60)
        assert_image_mode(output_img, "RGBA")

    def test_geometric_transformation_pipeline(self, temp_dir):
        """Test complex geometric transformations."""
        input_path = temp_dir / "geo_input.png"
        output_path = temp_dir / "geo_output.png"

        test_img = create_test_image(64, 64, "RGBA")
        test_img.save(input_path)

        pipeline = create_test_pipeline(input_path)
        (
            pipeline.add(RowShift(selection="even", shift_amount=2, wrap=True))
            .add(ColumnShift(selection="odd", shift_amount=3, wrap=False))
            .add(GridWarp(axis="both", frequency=1.0, amplitude=5.0))
            .execute(str(output_path))
        )

        assert output_path.exists()
        output_img = Image.open(output_path)
        assert_image_mode(output_img, "RGBA")

    def test_mosaic_art_pipeline(self, temp_dir):
        """Test mosaic creation pipeline."""
        input_path = temp_dir / "mosaic_input.png"
        output_path = temp_dir / "mosaic_output.png"

        test_img = create_test_image(96, 96, "RGBA")
        test_img.save(input_path)

        pipeline = create_test_pipeline(input_path)
        pipeline.add(Mosaic(tile_size=(8, 8), gap_size=1, mode="average")).execute(str(output_path))

        assert output_path.exists()
        output_img = Image.open(output_path)
        assert_image_dimensions(output_img, 96, 96)

    def test_preset_application(self, temp_dir):
        """Test applying built-in presets."""
        input_path = temp_dir / "preset_input.png"
        output_path = temp_dir / "preset_output.png"

        test_img = create_test_image(50, 50, "RGBA")
        test_img.save(input_path)

        # Test glitch-effect preset
        preset_config = get_preset("glitch-effect")

        pipeline = create_test_pipeline(input_path)
        for op_config in preset_config["operations"]:
            if op_config["type"] == "PixelFilter":
                pipeline.add(PixelFilter(**op_config["params"]))
            elif op_config["type"] == "RowShift":
                pipeline.add(RowShift(**op_config["params"]))
            elif op_config["type"] == "ColumnShift":
                pipeline.add(ColumnShift(**op_config["params"]))

        pipeline.execute(str(output_path))
        assert output_path.exists()

    def test_all_presets_executable(self, temp_dir):
        """Test that all built-in presets can be executed without errors."""
        input_path = temp_dir / "all_presets_input.png"
        test_img = create_test_image(32, 32, "RGBA")
        test_img.save(input_path)

        all_presets = get_all_presets()

        for preset_name, preset_config in all_presets.items():
            output_path = temp_dir / f"preset_{preset_name}_output.png"

            try:
                pipeline = create_test_pipeline(input_path)

                # Add operations based on preset configuration
                for op_config in preset_config["operations"]:
                    op_type = op_config["type"]
                    op_params = op_config["params"]

                    # Map operation types to classes
                    if op_type == "PixelFilter":
                        pipeline.add(PixelFilter(**op_params))
                    elif op_type == "RowShift":
                        pipeline.add(RowShift(**op_params))
                    elif op_type == "ColumnShift":
                        pipeline.add(ColumnShift(**op_params))
                    elif op_type == "BlockFilter":
                        pipeline.add(BlockFilter(**op_params))
                    elif op_type == "Mosaic":
                        pipeline.add(Mosaic(**op_params))
                    elif op_type == "ChannelSwap":
                        pipeline.add(ChannelSwap(**op_params))
                    elif op_type == "Dither":
                        pipeline.add(Dither(**op_params))
                    # Add more operation types as needed

                pipeline.execute(str(output_path))
                assert output_path.exists(), f"Preset {preset_name} failed to produce output"

            except Exception as e:
                pytest.fail(f"Preset {preset_name} failed with error: {e}")

    def test_pipeline_with_caching(self, temp_dir):
        """Test pipeline execution with caching enabled."""
        input_path = temp_dir / "cache_input.png"
        output_path1 = temp_dir / "cache_output1.png"
        output_path2 = temp_dir / "cache_output2.png"
        cache_dir = temp_dir / "cache"

        test_img = create_test_image(40, 40, "RGBA")
        test_img.save(input_path)

        # First execution (should populate cache)
        pipeline1 = Pipeline(str(input_path), cache_dirs=[str(cache_dir)])
        (
            pipeline1.add(PixelFilter(condition="prime", fill_color=(255, 0, 0, 255)))
            .add(RowShift(selection="odd", shift_amount=2))
            .execute(str(output_path1))
        )

        # Second execution (should use cache)
        pipeline2 = Pipeline(str(input_path), cache_dirs=[str(cache_dir)])
        (
            pipeline2.add(PixelFilter(condition="prime", fill_color=(255, 0, 0, 255)))
            .add(RowShift(selection="odd", shift_amount=2))
            .execute(str(output_path2))
        )

        # Both should produce identical results
        assert output_path1.exists() and output_path2.exists()

        img1 = Image.open(output_path1)
        img2 = Image.open(output_path2)
        assert_images_similar(img1, img2, ssim_threshold=1.0, mse_threshold=0.0)

    def test_pipeline_error_recovery(self, temp_dir):
        """Test pipeline behavior with invalid operations."""
        input_path = temp_dir / "error_input.png"
        test_img = create_test_image(20, 20, "RGBA")
        test_img.save(input_path)

        # Test with invalid pixel filter condition
        with pytest.raises(ValidationError):  # Should raise validation error
            pipeline = create_test_pipeline(input_path)
            pipeline.add(PixelFilter(condition="invalid_condition", fill_color=(255, 0, 0)))
            pipeline.validate()  # This should fail

    def test_large_pipeline_memory_efficiency(self, temp_dir):
        """Test pipeline with many operations for memory efficiency."""
        input_path = temp_dir / "large_pipeline_input.png"
        output_path = temp_dir / "large_pipeline_output.png"

        # Smaller image for this test to focus on operation count
        test_img = create_test_image(30, 30, "RGBA")
        test_img.save(input_path)

        # Create pipeline with many operations
        pipeline = create_test_pipeline(input_path)

        # Add multiple operations of different types
        for i in range(3):
            pipeline.add(
                RowShift(selection="odd" if i % 2 == 0 else "even", shift_amount=1, wrap=True)
            )
            pipeline.add(
                ColumnShift(selection="even" if i % 2 == 0 else "odd", shift_amount=1, wrap=True)
            )
            pipeline.add(
                PixelFilter(condition="even" if i % 2 == 0 else "odd", fill_color=(255, 0, 0, 128))
            )

        pipeline.execute(str(output_path))
        assert output_path.exists()

        # Verify output is still valid
        output_img = Image.open(output_path)
        assert_image_dimensions(output_img, 30, 30)
        assert_image_mode(output_img, "RGBA")


@pytest.mark.integration
@pytest.mark.slow
class TestLargeImagePipelines:
    """Test pipelines with larger images."""

    def test_high_resolution_pipeline(self, temp_dir):
        """Test pipeline with high-resolution image."""
        input_path = temp_dir / "hires_input.png"
        output_path = temp_dir / "hires_output.png"

        # Create larger test image (but not too large for CI)
        test_img = create_test_image(800, 600, "RGBA")
        test_img.save(input_path)

        pipeline = create_test_pipeline(input_path)
        (
            pipeline.add(PixelFilter(condition="prime", fill_color=(255, 0, 0, 255)))
            .add(RowShift(selection="odd", shift_amount=5))
            .execute(str(output_path))
        )

        assert output_path.exists()
        output_img = Image.open(output_path)
        assert_image_dimensions(output_img, 800, 600)

    def test_extreme_aspect_ratio_pipeline(self, temp_dir):
        """Test pipeline with extreme aspect ratios."""
        input_path = temp_dir / "extreme_aspect_input.png"
        output_path = temp_dir / "extreme_aspect_output.png"

        # Very wide image
        test_img = create_test_image(200, 10, "RGBA")
        test_img.save(input_path)

        pipeline = create_test_pipeline(input_path)
        (
            pipeline.add(RowShift(selection="all", shift_amount=10, wrap=True))
            .add(AspectStretch(target_ratio="2:1", method="simple"))
            .execute(str(output_path))
        )

        assert output_path.exists()
        output_img = Image.open(output_path)
        # Should be resized to 2:1 aspect ratio
        assert output_img.size[0] == output_img.size[1] * 2


@pytest.mark.integration
class TestPipelineEdgeCases:
    """Test edge cases in pipeline execution."""

    def test_minimal_image_pipeline(self, temp_dir):
        """Test pipeline with 1x1 image."""
        input_path = temp_dir / "minimal_input.png"
        output_path = temp_dir / "minimal_output.png"

        # 1x1 pixel image
        test_img = create_test_image(1, 1, "RGBA")
        test_img.save(input_path)

        pipeline = create_test_pipeline(input_path)
        pipeline.add(PixelFilter(condition="even", fill_color=(255, 0, 0, 255))).execute(
            str(output_path)
        )

        assert output_path.exists()
        output_img = Image.open(output_path)
        assert_image_dimensions(output_img, 1, 1)

    def test_empty_pipeline(self, temp_dir):
        """Test pipeline with no operations."""
        input_path = temp_dir / "empty_input.png"
        output_path = temp_dir / "empty_output.png"

        test_img = create_test_image(50, 50, "RGBA")
        test_img.save(input_path)

        pipeline = create_test_pipeline(input_path)
        pipeline.execute(str(output_path))

        assert output_path.exists()
        output_img = Image.open(output_path)

        # Should be identical to input
        input_img = Image.open(input_path)
        assert_images_similar(input_img, output_img, ssim_threshold=1.0, mse_threshold=0.0)
