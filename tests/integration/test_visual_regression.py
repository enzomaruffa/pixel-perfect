"""Visual regression testing for artistic operations."""

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from core.pipeline import Pipeline
from operations.aspect import AspectStretch
from operations.block import BlockFilter
from operations.channel import ChannelSwap
from operations.column import ColumnShift
from operations.geometric import GridWarp
from operations.pattern import Dither, Mosaic
from operations.pixel import PixelFilter
from operations.row import RowShift, RowStretch
from presets import get_preset
from tests.conftest import ImageComparison, create_test_image


class VisualRegressionManager:
    """Manage reference images and visual regression testing."""

    def __init__(self, expected_outputs_dir: Path):
        self.expected_dir = expected_outputs_dir
        self.expected_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.expected_dir / "reference_metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict[str, Any]:
        """Load reference image metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        return {}

    def _save_metadata(self):
        """Save reference image metadata."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def get_reference_path(self, test_name: str) -> Path:
        """Get path for reference image."""
        return self.expected_dir / f"{test_name}_reference.png"

    def has_reference(self, test_name: str) -> bool:
        """Check if reference image exists."""
        return self.get_reference_path(test_name).exists()

    def create_reference(self, test_name: str, image: Image.Image, metadata: dict | None = None):
        """Create a new reference image."""
        ref_path = self.get_reference_path(test_name)
        image.save(ref_path)

        # Store metadata
        self.metadata[test_name] = {
            "created": True,
            "image_size": image.size,
            "image_mode": image.mode,
            "metadata": metadata or {},
        }
        self._save_metadata()

    def compare_with_reference(
        self,
        test_name: str,
        current_image: Image.Image,
        ssim_threshold: float = 0.95,
        mse_threshold: float = 100.0,
    ) -> dict[str, Any]:
        """Compare current image with reference."""
        if not self.has_reference(test_name):
            return {"has_reference": False, "message": "No reference image found"}

        ref_path = self.get_reference_path(test_name)
        reference_image = Image.open(ref_path)

        comparison = ImageComparison()

        ssim_score = comparison.calculate_ssim(current_image, reference_image)
        mse_score = comparison.calculate_mse(current_image, reference_image)

        # Calculate perceptual hash distance
        current_hash = comparison.perceptual_hash(current_image)
        ref_hash = comparison.perceptual_hash(reference_image)
        hash_distance = comparison.hamming_distance(current_hash, ref_hash)

        return {
            "has_reference": True,
            "ssim_score": ssim_score,
            "mse_score": mse_score,
            "hash_distance": hash_distance,
            "ssim_pass": ssim_score >= ssim_threshold,
            "mse_pass": mse_score <= mse_threshold,
            "hash_pass": hash_distance <= 8,  # Threshold for perceptual hash
            "reference_size": reference_image.size,
            "current_size": current_image.size,
            "size_match": reference_image.size == current_image.size,
        }


@pytest.fixture
def visual_regression(expected_outputs_dir):
    """Provide visual regression testing utilities."""
    return VisualRegressionManager(expected_outputs_dir)


@pytest.mark.visual
class TestVisualRegression:
    """Test visual output consistency for artistic operations."""

    @pytest.fixture
    def standard_test_image(self, temp_dir):
        """Create a standard test image for visual regression tests."""
        input_path = temp_dir / "visual_test_input.png"
        # Create a more interesting test image with patterns
        test_img = Image.new("RGBA", (128, 96))
        pixels = []

        for y in range(96):
            for x in range(128):
                # Create a pattern with gradients and geometric shapes
                r = int((x / 128) * 255)
                g = int((y / 96) * 255)
                b = int(((x + y) % 64) / 64 * 255)

                # Add some geometric patterns
                if (x // 8 + y // 8) % 2 == 0:
                    r = min(255, r + 50)

                if x % 16 < 8 and y % 16 < 8:
                    g = min(255, g + 100)

                pixels.append((r, g, b, 255))

        test_img.putdata(pixels)
        test_img.save(input_path)
        return input_path

    def test_pixel_filter_visual_consistency(
        self, standard_test_image, temp_dir, visual_regression
    ):
        """Test PixelFilter visual output consistency."""
        test_name = "pixel_filter_prime"
        output_path = temp_dir / f"{test_name}_output.png"

        pipeline = Pipeline(str(standard_test_image))
        pipeline.add(PixelFilter(condition="prime", fill_color=(255, 0, 0, 255))).execute(
            str(output_path)
        )

        current_image = Image.open(output_path)

        # Create reference if it doesn't exist
        if not visual_regression.has_reference(test_name):
            visual_regression.create_reference(
                test_name,
                current_image,
                {
                    "operation": "PixelFilter",
                    "parameters": {"condition": "prime", "fill_color": [255, 0, 0, 255]},
                },
            )
            pytest.skip(f"Created reference image for {test_name}")

        # Compare with reference
        comparison = visual_regression.compare_with_reference(test_name, current_image)

        assert comparison["size_match"], (
            f"Size mismatch: {comparison['current_size']} vs {comparison['reference_size']}"
        )
        assert comparison["ssim_pass"], f"SSIM too low: {comparison['ssim_score']:.3f}"
        assert comparison["mse_pass"], f"MSE too high: {comparison['mse_score']:.1f}"

    def test_block_filter_visual_consistency(
        self, standard_test_image, temp_dir, visual_regression
    ):
        """Test BlockFilter visual output consistency."""
        test_name = "block_filter_checkerboard"
        output_path = temp_dir / f"{test_name}_output.png"

        pipeline = Pipeline(str(standard_test_image))
        pipeline.add(BlockFilter(block_width=8, block_height=8, condition="checkerboard")).execute(
            str(output_path)
        )

        current_image = Image.open(output_path)

        if not visual_regression.has_reference(test_name):
            visual_regression.create_reference(test_name, current_image)
            pytest.skip(f"Created reference image for {test_name}")

        comparison = visual_regression.compare_with_reference(test_name, current_image)
        assert comparison["ssim_pass"], f"SSIM too low: {comparison['ssim_score']:.3f}"
        assert comparison["mse_pass"], f"MSE too high: {comparison['mse_score']:.1f}"

    def test_row_shift_visual_consistency(self, standard_test_image, temp_dir, visual_regression):
        """Test RowShift visual output consistency."""
        test_name = "row_shift_odd"
        output_path = temp_dir / f"{test_name}_output.png"

        pipeline = Pipeline(str(standard_test_image))
        pipeline.add(RowShift(selection="odd", shift_amount=5, wrap=True)).execute(str(output_path))

        current_image = Image.open(output_path)

        if not visual_regression.has_reference(test_name):
            visual_regression.create_reference(test_name, current_image)
            pytest.skip(f"Created reference image for {test_name}")

        comparison = visual_regression.compare_with_reference(test_name, current_image)
        assert comparison["ssim_pass"], f"SSIM too low: {comparison['ssim_score']:.3f}"
        assert comparison["mse_pass"], f"MSE too high: {comparison['mse_score']:.1f}"

    def test_mosaic_visual_consistency(self, standard_test_image, temp_dir, visual_regression):
        """Test Mosaic visual output consistency."""
        test_name = "mosaic_average"
        output_path = temp_dir / f"{test_name}_output.png"

        pipeline = Pipeline(str(standard_test_image))
        pipeline.add(Mosaic(tile_size=(8, 8), gap_size=2, mode="average")).execute(str(output_path))

        current_image = Image.open(output_path)

        if not visual_regression.has_reference(test_name):
            visual_regression.create_reference(test_name, current_image)
            pytest.skip(f"Created reference image for {test_name}")

        comparison = visual_regression.compare_with_reference(test_name, current_image)
        assert comparison["ssim_pass"], f"SSIM too low: {comparison['ssim_score']:.3f}"
        assert comparison["mse_pass"], f"MSE too high: {comparison['mse_score']:.1f}"

    def test_channel_swap_visual_consistency(
        self, standard_test_image, temp_dir, visual_regression
    ):
        """Test ChannelSwap visual output consistency."""
        test_name = "channel_swap_rgb"
        output_path = temp_dir / f"{test_name}_output.png"

        pipeline = Pipeline(str(standard_test_image))
        pipeline.add(
            ChannelSwap(red_source="green", green_source="blue", blue_source="red")
        ).execute(str(output_path))

        current_image = Image.open(output_path)

        if not visual_regression.has_reference(test_name):
            visual_regression.create_reference(test_name, current_image)
            pytest.skip(f"Created reference image for {test_name}")

        comparison = visual_regression.compare_with_reference(test_name, current_image)
        assert comparison["ssim_pass"], f"SSIM too low: {comparison['ssim_score']:.3f}"
        assert comparison["mse_pass"], f"MSE too high: {comparison['mse_score']:.1f}"

    def test_dither_visual_consistency(self, standard_test_image, temp_dir, visual_regression):
        """Test Dither visual output consistency."""
        test_name = "dither_floyd_steinberg"
        output_path = temp_dir / f"{test_name}_output.png"

        pipeline = Pipeline(str(standard_test_image))
        pipeline.add(Dither(method="floyd_steinberg", levels=4)).execute(str(output_path))

        current_image = Image.open(output_path)

        if not visual_regression.has_reference(test_name):
            visual_regression.create_reference(test_name, current_image)
            pytest.skip(f"Created reference image for {test_name}")

        comparison = visual_regression.compare_with_reference(test_name, current_image)
        assert comparison["ssim_pass"], f"SSIM too low: {comparison['ssim_score']:.3f}"
        assert comparison["mse_pass"], f"MSE too high: {comparison['mse_score']:.1f}"

    def test_aspect_stretch_visual_consistency(
        self, standard_test_image, temp_dir, visual_regression
    ):
        """Test AspectStretch visual output consistency."""
        test_name = "aspect_stretch_square"
        output_path = temp_dir / f"{test_name}_output.png"

        pipeline = Pipeline(str(standard_test_image))
        pipeline.add(AspectStretch(target_ratio="1:1", method="simple")).execute(str(output_path))

        current_image = Image.open(output_path)

        if not visual_regression.has_reference(test_name):
            visual_regression.create_reference(test_name, current_image)
            pytest.skip(f"Created reference image for {test_name}")

        comparison = visual_regression.compare_with_reference(test_name, current_image)
        assert comparison["ssim_pass"], f"SSIM too low: {comparison['ssim_score']:.3f}"
        assert comparison["mse_pass"], f"MSE too high: {comparison['mse_score']:.1f}"

    def test_grid_warp_visual_consistency(self, standard_test_image, temp_dir, visual_regression):
        """Test GridWarp visual output consistency."""
        test_name = "grid_warp_horizontal"
        output_path = temp_dir / f"{test_name}_output.png"

        pipeline = Pipeline(str(standard_test_image))
        pipeline.add(GridWarp(axis="horizontal", frequency=2.0, amplitude=5.0)).execute(
            str(output_path)
        )

        current_image = Image.open(output_path)

        if not visual_regression.has_reference(test_name):
            visual_regression.create_reference(test_name, current_image)
            pytest.skip(f"Created reference image for {test_name}")

        comparison = visual_regression.compare_with_reference(test_name, current_image)
        assert comparison["ssim_pass"], f"SSIM too low: {comparison['ssim_score']:.3f}"
        assert comparison["mse_pass"], f"MSE too high: {comparison['mse_score']:.1f}"


@pytest.mark.visual
class TestPresetVisualRegression:
    """Test visual consistency of built-in presets."""

    @pytest.fixture
    def preset_test_image(self, temp_dir):
        """Create test image specifically for preset testing."""
        input_path = temp_dir / "preset_visual_input.png"
        test_img = create_test_image(80, 80, "RGBA")
        test_img.save(input_path)
        return input_path

    def test_glitch_effect_preset_visual(self, preset_test_image, temp_dir, visual_regression):
        """Test glitch-effect preset visual consistency."""
        test_name = "preset_glitch_effect"
        output_path = temp_dir / f"{test_name}_output.png"

        preset_config = get_preset("glitch-effect")

        pipeline = Pipeline(str(preset_test_image))
        for op_config in preset_config["operations"]:
            if op_config["type"] == "PixelFilter":
                pipeline.add(PixelFilter(**op_config["params"]))
            elif op_config["type"] == "RowShift":
                pipeline.add(RowShift(**op_config["params"]))
            elif op_config["type"] == "ColumnShift":
                pipeline.add(ColumnShift(**op_config["params"]))

        pipeline.execute(str(output_path))
        current_image = Image.open(output_path)

        if not visual_regression.has_reference(test_name):
            visual_regression.create_reference(
                test_name,
                current_image,
                {"preset": "glitch-effect", "description": preset_config.get("description", "")},
            )
            pytest.skip(f"Created reference image for {test_name}")

        comparison = visual_regression.compare_with_reference(test_name, current_image)
        assert comparison["ssim_pass"], f"SSIM too low: {comparison['ssim_score']:.3f}"
        assert comparison["mse_pass"], f"MSE too high: {comparison['mse_score']:.1f}"

    def test_mosaic_art_preset_visual(self, preset_test_image, temp_dir, visual_regression):
        """Test mosaic-art preset visual consistency."""
        test_name = "preset_mosaic_art"
        output_path = temp_dir / f"{test_name}_output.png"

        preset_config = get_preset("mosaic-art")

        pipeline = Pipeline(str(preset_test_image))
        for op_config in preset_config["operations"]:
            if op_config["type"] == "BlockFilter":
                pipeline.add(BlockFilter(**op_config["params"]))
            elif op_config["type"] == "Mosaic":
                pipeline.add(Mosaic(**op_config["params"]))

        pipeline.execute(str(output_path))
        current_image = Image.open(output_path)

        if not visual_regression.has_reference(test_name):
            visual_regression.create_reference(test_name, current_image)
            pytest.skip(f"Created reference image for {test_name}")

        comparison = visual_regression.compare_with_reference(test_name, current_image)
        assert comparison["ssim_pass"], f"SSIM too low: {comparison['ssim_score']:.3f}"
        assert comparison["mse_pass"], f"MSE too high: {comparison['mse_score']:.1f}"


@pytest.mark.visual
class TestComplexPipelineVisual:
    """Test visual consistency of complex multi-operation pipelines."""

    def test_complex_artistic_pipeline_visual(self, temp_dir, visual_regression):
        """Test complex artistic pipeline visual consistency."""
        test_name = "complex_artistic_pipeline"
        input_path = temp_dir / "complex_input.png"
        output_path = temp_dir / f"{test_name}_output.png"

        # Create distinctive test image
        test_img = create_test_image(96, 96, "RGBA")
        test_img.save(input_path)

        pipeline = Pipeline(str(input_path))
        (
            pipeline.add(PixelFilter(condition="prime", fill_color=(255, 0, 0, 128)))
            .add(BlockFilter(block_width=6, block_height=6, condition="checkerboard"))
            .add(RowShift(selection="odd", shift_amount=3, wrap=True))
            .add(ChannelSwap(red_source="green", green_source="blue", blue_source="red"))
            .add(Mosaic(tile_size=(8, 8), gap_size=1, mode="average"))
            .execute(str(output_path))
        )

        current_image = Image.open(output_path)

        if not visual_regression.has_reference(test_name):
            visual_regression.create_reference(
                test_name,
                current_image,
                {
                    "pipeline": "complex_artistic",
                    "operations": [
                        "PixelFilter",
                        "BlockFilter",
                        "RowShift",
                        "ChannelSwap",
                        "Mosaic",
                    ],
                },
            )
            pytest.skip(f"Created reference image for {test_name}")

        comparison = visual_regression.compare_with_reference(test_name, current_image)
        assert comparison["ssim_pass"], f"SSIM too low: {comparison['ssim_score']:.3f}"
        assert comparison["mse_pass"], f"MSE too high: {comparison['mse_score']:.1f}"

    def test_geometric_transformation_visual(self, temp_dir, visual_regression):
        """Test geometric transformation pipeline visual consistency."""
        test_name = "geometric_transformation"
        input_path = temp_dir / "geometric_input.png"
        output_path = temp_dir / f"{test_name}_output.png"

        test_img = create_test_image(64, 64, "RGBA")
        test_img.save(input_path)

        pipeline = Pipeline(str(input_path))
        (
            pipeline.add(RowShift(selection="even", shift_amount=2, wrap=True))
            .add(ColumnShift(selection="odd", shift_amount=3, wrap=False))
            .add(RowStretch(rows=[0, 2, 4, 6], factor=1.5))
            .add(AspectStretch(target_ratio="4:3", method="simple"))
            .execute(str(output_path))
        )

        current_image = Image.open(output_path)

        if not visual_regression.has_reference(test_name):
            visual_regression.create_reference(test_name, current_image)
            pytest.skip(f"Created reference image for {test_name}")

        comparison = visual_regression.compare_with_reference(test_name, current_image)
        assert comparison["ssim_pass"], f"SSIM too low: {comparison['ssim_score']:.3f}"
        assert comparison["mse_pass"], f"MSE too high: {comparison['mse_score']:.1f}"


@pytest.mark.visual
class TestVisualRegressionUtils:
    """Test visual regression testing utilities."""

    def test_reference_creation_and_comparison(self, temp_dir, visual_regression):
        """Test creating and comparing reference images."""
        test_name = "utils_test"

        # Create test image
        test_img = create_test_image(50, 50, "RGBA")

        # Test reference creation
        visual_regression.create_reference(test_name, test_img, {"test": "metadata"})
        assert visual_regression.has_reference(test_name)

        # Test identical image comparison
        comparison = visual_regression.compare_with_reference(test_name, test_img)
        assert comparison["ssim_pass"]
        assert comparison["mse_pass"]
        assert comparison["hash_pass"]
        assert comparison["size_match"]

        # Test slightly different image
        modified_img = test_img.copy()
        # Change a few pixels
        pixels = list(modified_img.getdata())
        pixels[0] = (255, 0, 0, 255)
        pixels[1] = (0, 255, 0, 255)
        modified_img.putdata(pixels)

        comparison = visual_regression.compare_with_reference(test_name, modified_img)
        # Should still pass with small changes
        assert comparison["ssim_pass"]
        assert comparison["size_match"]
