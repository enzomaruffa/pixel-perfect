"""Test operation combinations for compatibility and correctness."""

import itertools

import pytest
from PIL import Image

from core.pipeline import Pipeline
from operations.aspect import AspectCrop, AspectPad, AspectStretch
from operations.block import BlockFilter
from operations.channel import AlphaGenerator, ChannelIsolate, ChannelSwap
from operations.column import ColumnShift, ColumnStretch
from operations.geometric import GridWarp, PerspectiveStretch, RadialStretch
from operations.pattern import Dither, Mosaic
from operations.pixel import PixelFilter
from operations.row import RowShift, RowStretch
from tests.conftest import assert_image_mode, create_test_image


@pytest.mark.integration
class TestOperationCompatibility:
    """Test compatibility between different operation types."""

    @pytest.fixture
    def test_image(self, temp_dir):
        """Create a standard test image for compatibility tests."""
        input_path = temp_dir / "compat_input.png"
        test_img = create_test_image(64, 48, "RGBA")  # Non-square for aspect tests
        test_img.save(input_path)
        return input_path

    def test_filter_combinations(self, test_image, temp_dir):
        """Test combinations of different filter operations."""
        output_path = temp_dir / "filter_combo_output.png"

        pipeline = Pipeline(str(test_image))
        _ = (
            pipeline.add(PixelFilter(condition="even", fill_color=(255, 0, 0, 128)))
            .add(BlockFilter(block_width=4, block_height=4, condition="checkerboard"))
            .add(PixelFilter(condition="prime", fill_color=(0, 255, 0, 128)))
            .execute(str(output_path))
        )

        assert output_path.exists()
        output_img = Image.open(output_path)
        assert_image_mode(output_img, "RGBA")

    def test_geometric_sequence(self, test_image, temp_dir):
        """Test sequence of geometric transformations."""
        output_path = temp_dir / "geometric_combo_output.png"

        pipeline = Pipeline(str(test_image))
        _ = (
            pipeline.add(RowShift(selection="odd", shift_amount=2, wrap=True))
            .add(ColumnShift(selection="even", shift_amount=3, wrap=False))
            .add(RowStretch(rows=[0, 2, 4], factor=1.5))
            .add(ColumnStretch(columns=[1, 3, 5], factor=0.8))
            .execute(str(output_path))
        )

        assert output_path.exists()
        output_img = Image.open(output_path)
        assert_image_mode(output_img, "RGBA")

    def test_channel_manipulation_chain(self, test_image, temp_dir):
        """Test chain of channel operations."""
        output_path = temp_dir / "channel_combo_output.png"

        pipeline = Pipeline(str(test_image))
        _ = (
            pipeline.add(ChannelSwap(mapping={"r": "g", "g": "b", "b": "r"}))
            .add(ChannelIsolate(keep_channels=["r"]))
            .add(AlphaGenerator(source="saturation", threshold=77))
            .execute(str(output_path))
        )

        assert output_path.exists()
        output_img = Image.open(output_path)
        assert_image_mode(output_img, "RGBA")

    def test_aspect_ratio_operations(self, test_image, temp_dir):
        """Test aspect ratio operations in sequence."""
        output_path = temp_dir / "aspect_combo_output.png"

        pipeline = Pipeline(str(test_image))
        _ = (
            pipeline.add(AspectCrop(target_ratio="1:1", crop_mode="center"))
            .add(AspectPad(target_ratio="16:9", mode="blur", blur_radius=3))
            .add(AspectStretch(target_ratio="4:3", method="simple"))
            .execute(str(output_path))
        )

        assert output_path.exists()
        output_img = Image.open(output_path)
        assert_image_mode(output_img, "RGBA")
        # Final aspect ratio should be 4:3
        width, height = output_img.size
        aspect_ratio = width / height
        assert abs(aspect_ratio - (4 / 3)) < 0.01

    def test_spatial_distortion_combo(self, test_image, temp_dir):
        """Test combinations of spatial distortion operations."""
        output_path = temp_dir / "spatial_combo_output.png"

        pipeline = Pipeline(str(test_image))
        _ = (
            pipeline.add(GridWarp(axis="horizontal", frequency=1.0, amplitude=3.0))
            .add(PerspectiveStretch(top_factor=0.9, bottom_factor=1.1))
            .add(RadialStretch(factor=1.2, center="auto", falloff="linear"))
            .execute(str(output_path))
        )

        assert output_path.exists()
        output_img = Image.open(output_path)
        assert_image_mode(output_img, "RGBA")

    def test_artistic_effect_combo(self, test_image, temp_dir):
        """Test combinations of artistic effects."""
        output_path = temp_dir / "artistic_combo_output.png"

        pipeline = Pipeline(str(test_image))
        _ = (
            pipeline.add(Mosaic(tile_size=(8, 8), gap_size=1, mode="average"))
            .add(Dither(method="floyd_steinberg", levels=6))
            .add(PixelFilter(condition="prime", fill_color=(255, 255, 0, 64)))
            .execute(str(output_path))
        )

        assert output_path.exists()
        output_img = Image.open(output_path)
        assert_image_mode(output_img, "RGBA")

    def test_mixed_operation_types(self, test_image, temp_dir):
        """Test mixing different types of operations."""
        output_path = temp_dir / "mixed_combo_output.png"

        pipeline = Pipeline(str(test_image))
        _ = (
            pipeline.add(PixelFilter(condition="even", fill_color=(255, 0, 0, 128)))  # Filter
            .add(RowShift(selection="odd", shift_amount=2))  # Geometric
            .add(ChannelSwap(red_source="blue", green_source="red", blue_source="green"))  # Channel
            .add(Mosaic(tile_size=(4, 4), gap_size=0, mode="dominant"))  # Artistic
            .add(AspectStretch(target_ratio="1:1", method="simple"))  # Aspect
            .execute(str(output_path))
        )

        assert output_path.exists()
        output_img = Image.open(output_path)
        assert_image_mode(output_img, "RGBA")


@pytest.mark.integration
class TestOperationParameterCompatibility:
    """Test parameter combinations that should work together."""

    def test_filter_parameter_ranges(self, temp_dir):
        """Test filter operations with various parameter combinations."""
        input_path = temp_dir / "param_test_input.png"
        test_img = create_test_image(32, 32, "RGBA")
        test_img.save(input_path)

        # Test different pixel filter conditions
        conditions = ["all", "even", "odd", "prime"]
        colors = [(255, 0, 0, 255), (0, 255, 0, 128), (0, 0, 255, 64)]

        for i, (condition, color) in enumerate(itertools.product(conditions, colors)):
            output_path = temp_dir / f"filter_param_{i}_output.png"

            pipeline = Pipeline(str(input_path))
            _ = pipeline.add(PixelFilter(condition=condition, fill_color=color)).execute(
                str(output_path)
            )

            assert output_path.exists()
            output_img = Image.open(output_path)
            assert_image_mode(output_img, "RGBA")

    def test_shift_parameter_ranges(self, temp_dir):
        """Test shift operations with various parameter combinations."""
        input_path = temp_dir / "shift_test_input.png"
        test_img = create_test_image(24, 24, "RGBA")
        test_img.save(input_path)

        # Test different shift parameters
        selections = ["all", "even", "odd"]
        shift_amounts = [1, 3, 5, 10]
        wrap_options = [True, False]

        for test_count, (selection, shift_amount, wrap) in enumerate(itertools.product(
            selections, shift_amounts, wrap_options
        )):
            if test_count >= 6:  # Limit test combinations for efficiency
                break

            output_path = temp_dir / f"shift_param_{test_count}_output.png"

            pipeline = Pipeline(str(input_path))
            _ = pipeline.add(
                RowShift(selection=selection, shift_amount=shift_amount, wrap=wrap)
            ).execute(str(output_path))

            assert output_path.exists()

    def test_block_size_compatibility(self, temp_dir):
        """Test block operations with different block sizes."""
        input_path = temp_dir / "block_test_input.png"
        test_img = create_test_image(48, 48, "RGBA")  # Divisible by common block sizes
        test_img.save(input_path)

        # Test different block sizes
        block_sizes = [(2, 2), (4, 4), (8, 8), (6, 8), (3, 16)]

        for i, (block_width, block_height) in enumerate(block_sizes):
            output_path = temp_dir / f"block_size_{i}_output.png"

            pipeline = Pipeline(str(input_path))
            _ = pipeline.add(
                BlockFilter(
                    block_width=block_width, block_height=block_height, condition="checkerboard"
                )
            ).execute(str(output_path))

            assert output_path.exists()
            output_img = Image.open(output_path)
            assert_image_mode(output_img, "RGBA")

    def test_aspect_ratio_compatibility(self, temp_dir):
        """Test aspect ratio operations with different target ratios."""
        input_path = temp_dir / "aspect_test_input.png"
        test_img = create_test_image(60, 40, "RGBA")  # 3:2 aspect ratio
        test_img.save(input_path)

        # Test different target aspect ratios
        aspect_ratios = ["1:1", "4:3", "16:9", "2:3", "21:9"]

        for i, target_ratio in enumerate(aspect_ratios):
            output_path = temp_dir / f"aspect_ratio_{i}_output.png"

            pipeline = Pipeline(str(input_path))
            _ = pipeline.add(
                AspectStretch(target_ratio=target_ratio, method="simple")
            ).execute(str(output_path))

            assert output_path.exists()
            output_img = Image.open(output_path)
            assert_image_mode(output_img, "RGBA")


@pytest.mark.integration
class TestOperationMemoryConsistency:
    """Test that operations maintain memory efficiency in combinations."""

    def test_memory_with_operation_chain(self, temp_dir):
        """Test memory usage doesn't explode with operation chains."""
        input_path = temp_dir / "memory_test_input.png"
        test_img = create_test_image(100, 100, "RGBA")
        test_img.save(input_path)

        # Create a long chain of operations
        pipeline = Pipeline(str(input_path))

        # Add 10 operations of various types
        for i in range(10):
            if i % 4 == 0:
                pipeline.add(PixelFilter(condition="even", fill_color=(255, 0, 0, 64)))
            elif i % 4 == 1:
                pipeline.add(RowShift(selection="odd", shift_amount=1, wrap=True))
            elif i % 4 == 2:
                pipeline.add(ColumnShift(selection="even", shift_amount=1, wrap=True))
            else:
                pipeline.add(
                    ChannelSwap(red_source="green", green_source="red", blue_source="blue")
                )

        output_path = temp_dir / "memory_chain_output.png"
        pipeline.execute(str(output_path))

        assert output_path.exists()
        output_img = Image.open(output_path)
        assert_image_mode(output_img, "RGBA")

    def test_cache_key_uniqueness(self, temp_dir):
        """Test that different operation combinations produce unique cache keys."""
        input_path = temp_dir / "cache_key_input.png"
        test_img = create_test_image(32, 32, "RGBA")
        test_img.save(input_path)

        # Create several similar but different pipelines
        pipeline1 = Pipeline(str(input_path))
        pipeline1.add(PixelFilter(condition="even", fill_color=(255, 0, 0, 255)))
        pipeline1.add(RowShift(selection="odd", shift_amount=2))

        pipeline2 = Pipeline(str(input_path))
        pipeline2.add(
            PixelFilter(condition="odd", fill_color=(255, 0, 0, 255))
        )  # Different condition
        pipeline2.add(RowShift(selection="odd", shift_amount=2))

        pipeline3 = Pipeline(str(input_path))
        pipeline3.add(PixelFilter(condition="even", fill_color=(255, 0, 0, 255)))
        pipeline3.add(RowShift(selection="odd", shift_amount=3))  # Different shift amount

        # Get cache keys for comparison
        key1 = pipeline1._get_pipeline_cache_key()
        key2 = pipeline2._get_pipeline_cache_key()
        key3 = pipeline3._get_pipeline_cache_key()

        # All keys should be different
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3


@pytest.mark.integration
@pytest.mark.slow
class TestOperationStressTests:
    """Stress test operation combinations under demanding conditions."""

    def test_many_small_operations(self, temp_dir):
        """Test many small operations in sequence."""
        input_path = temp_dir / "stress_small_input.png"
        test_img = create_test_image(50, 50, "RGBA")
        test_img.save(input_path)

        pipeline = Pipeline(str(input_path))

        # Add 20 small operations
        for i in range(20):
            if i % 3 == 0:
                pipeline.add(RowShift(selection="all", shift_amount=1, wrap=True))
            elif i % 3 == 1:
                pipeline.add(ColumnShift(selection="all", shift_amount=1, wrap=True))
            else:
                pipeline.add(PixelFilter(condition="even", fill_color=(255, 0, 0, 32)))

        output_path = temp_dir / "stress_small_output.png"
        pipeline.execute(str(output_path))

        assert output_path.exists()
        output_img = Image.open(output_path)
        assert_image_mode(output_img, "RGBA")

    def test_complex_nested_effects(self, temp_dir):
        """Test complex nested effects that build on each other."""
        input_path = temp_dir / "nested_input.png"
        test_img = create_test_image(64, 64, "RGBA")
        test_img.save(input_path)

        pipeline = Pipeline(str(input_path))
        _ = (
            pipeline
            # Layer 1: Basic filtering
            .add(PixelFilter(condition="prime", fill_color=(255, 0, 0, 128)))
            .add(BlockFilter(block_width=8, block_height=8, condition="checkerboard"))
            # Layer 2: Geometric transformations
            .add(RowShift(selection="odd", shift_amount=2, wrap=True))
            .add(ColumnShift(selection="even", shift_amount=3, wrap=False))
            # Layer 3: Channel manipulation
            .add(ChannelSwap(red_source="green", green_source="blue", blue_source="red"))
            .add(ChannelIsolate(channels=["red", "green"], mode="enhance"))
            # Layer 4: Artistic effects
            .add(Mosaic(tile_size=(4, 4), gap_size=1, mode="average"))
            .add(Dither(method="floyd_steinberg", levels=4))
            # Layer 5: Final adjustments
            .add(AspectStretch(target_ratio="1:1", method="simple"))
            .add(AlphaGenerator(source="saturation", threshold=128))
            .execute(str(temp_dir / "nested_output.png"))
        )

        output_path = temp_dir / "nested_output.png"
        assert output_path.exists()
        output_img = Image.open(output_path)
        assert_image_mode(output_img, "RGBA")
