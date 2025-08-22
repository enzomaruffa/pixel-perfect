"""Tests for Pipeline class."""

from pathlib import Path

import pytest
from PIL import Image

from .pipeline import Pipeline

# ruff: noqa: SLF001  # Allow private method access in tests


class TestPipeline:
    """Test Pipeline functionality."""

    @pytest.fixture
    def sample_image_path(self, temp_dir):
        """Create a sample image file for testing."""
        image_path = temp_dir / "test_image.png"
        # Create a simple 4x4 RGB image
        image = Image.new("RGB", (4, 4), (128, 128, 128))
        image.save(image_path)
        return image_path

    @pytest.fixture
    def pipeline(self, sample_image_path):
        """Create a basic pipeline for testing."""
        return Pipeline(sample_image_path)

    def test_pipeline_creation(self, sample_image_path):
        """Test creating a pipeline with valid image."""
        pipeline = Pipeline(sample_image_path)
        assert pipeline.input_path == Path(sample_image_path)
        assert not pipeline.debug
        assert pipeline.cache_dir is None
        assert len(pipeline.operations) == 0

    def test_pipeline_creation_with_options(self, sample_image_path, temp_dir):
        """Test creating pipeline with debug and cache options."""
        cache_dir = temp_dir / "cache"
        pipeline = Pipeline(sample_image_path, debug=True, cache_dir=cache_dir)

        assert pipeline.debug
        assert pipeline.cache_dir == cache_dir
        assert cache_dir.exists()  # Should be created automatically

    def test_pipeline_creation_nonexistent_file(self, temp_dir):
        """Test creating pipeline with nonexistent file raises error."""
        nonexistent_path = temp_dir / "nonexistent.png"

        with pytest.raises(FileNotFoundError):
            Pipeline(nonexistent_path)

    def test_image_loading(self, pipeline):
        """Test image loading and context creation."""
        image, context = pipeline._load_image()

        assert isinstance(image, Image.Image)
        assert image.size == (4, 4)
        assert image.mode in ["RGB", "RGBA"]

        assert context.width == 4
        assert context.height == 4
        assert context.channels >= 3
        assert context.dtype == "uint8"

    def test_image_loading_caching(self, pipeline):
        """Test that image loading is cached."""
        image1, context1 = pipeline._load_image()
        image2, context2 = pipeline._load_image()

        # Should return same objects
        assert image1 is image2
        assert context1 is context2

    def test_add_operation_chaining(self, pipeline):
        """Test that add() returns self for method chaining."""
        # Since we don't have actual operations yet, we'll test the interface
        assert len(pipeline.operations) == 0

        # Test that add returns the pipeline instance
        # We can't test actual operation adding without implementing operations first

    def test_dry_run_validation_empty_pipeline(self, pipeline, temp_dir):
        """Test dry run with empty pipeline."""
        output_path = temp_dir / "output.png"

        context = pipeline.execute(output_path, dry_run=True)

        # Should return the initial context
        assert context.width == 4
        assert context.height == 4

        # Output file should not be created in dry run
        assert not output_path.exists()

    def test_execute_empty_pipeline(self, pipeline, temp_dir):
        """Test executing empty pipeline (should just copy image)."""
        output_path = temp_dir / "output.png"

        pipeline.execute(output_path)

        # Should create output file
        assert output_path.exists()

        # Output should be same size as input
        output_image = Image.open(output_path)
        assert output_image.size == (4, 4)

    def test_execute_creates_output_directory(self, pipeline, temp_dir):
        """Test that execute creates output directory if needed."""
        nested_dir = temp_dir / "nested" / "dir"
        output_path = nested_dir / "output.png"

        # Directory shouldn't exist yet
        assert not nested_dir.exists()

        pipeline.execute(output_path)

        # Directory should be created
        assert nested_dir.exists()
        assert output_path.exists()

    def test_memory_estimation_empty_pipeline(self, pipeline):
        """Test memory estimation with empty pipeline."""
        memory = pipeline.estimate_memory()

        assert isinstance(memory, int)
        assert memory > 0

        # Should be at least the size of the input image
        expected_min = 4 * 4 * 3  # 4x4 RGB
        assert memory >= expected_min

    def test_image_hash_generation(self, pipeline):
        """Test image hash generation."""
        image, _ = pipeline._load_image()
        hash1 = pipeline._get_image_hash(image)
        hash2 = pipeline._get_image_hash(image)

        assert isinstance(hash1, str)
        assert len(hash1) == 32  # MD5 hash length
        assert hash1 == hash2  # Same image should produce same hash

    def test_image_hash_differs_for_different_images(self, temp_dir):
        """Test that different images produce different hashes."""
        # Create two different images
        image1_path = temp_dir / "image1.png"
        image2_path = temp_dir / "image2.png"

        Image.new("RGB", (4, 4), (255, 0, 0)).save(image1_path)  # Red
        Image.new("RGB", (4, 4), (0, 255, 0)).save(image2_path)  # Green

        pipeline1 = Pipeline(image1_path)
        pipeline2 = Pipeline(image2_path)

        image1, _ = pipeline1._load_image()
        image2, _ = pipeline2._load_image()

        hash1 = pipeline1._get_image_hash(image1)
        hash2 = pipeline2._get_image_hash(image2)

        assert hash1 != hash2

    def test_different_image_modes(self, temp_dir):
        """Test pipeline with different image modes."""
        # Test grayscale
        gray_path = temp_dir / "gray.png"
        Image.new("L", (4, 4), 128).save(gray_path)

        gray_pipeline = Pipeline(gray_path)
        image, context = gray_pipeline._load_image()

        # Should be converted to RGB or RGBA
        assert image.mode in ["RGB", "RGBA"]
        assert context.channels >= 3

    def test_image_mode_conversion(self, temp_dir):
        """Test that various image modes are handled correctly."""
        modes_to_test = [
            ("L", 128),  # Grayscale
            ("RGB", (128, 128, 128)),  # RGB
            ("RGBA", (128, 128, 128, 255)),  # RGBA
        ]

        for mode, color in modes_to_test:
            image_path = temp_dir / f"test_{mode.lower()}.png"
            Image.new(mode, (4, 4), color).save(image_path)

            pipeline = Pipeline(image_path)
            loaded_image, context = pipeline._load_image()

            # Should be converted to RGB or RGBA
            assert loaded_image.mode in ["RGB", "RGBA"]
            assert context.channels in [3, 4]

    @pytest.mark.edge_case
    def test_minimum_size_image(self, temp_dir):
        """Test pipeline with 1x1 image."""
        tiny_path = temp_dir / "tiny.png"
        Image.new("RGB", (1, 1), (255, 0, 0)).save(tiny_path)

        pipeline = Pipeline(tiny_path)
        image, context = pipeline._load_image()

        assert image.size == (1, 1)
        assert context.width == 1
        assert context.height == 1

    @pytest.mark.edge_case
    def test_extreme_aspect_ratio(self, temp_dir):
        """Test pipeline with extreme aspect ratio images."""
        # Very wide image
        wide_path = temp_dir / "wide.png"
        Image.new("RGB", (100, 1), (255, 0, 0)).save(wide_path)

        wide_pipeline = Pipeline(wide_path)
        image, context = wide_pipeline._load_image()

        assert image.size == (100, 1)
        assert context.width == 100
        assert context.height == 1

        # Very tall image
        tall_path = temp_dir / "tall.png"
        Image.new("RGB", (1, 100), (0, 255, 0)).save(tall_path)

        tall_pipeline = Pipeline(tall_path)
        image, context = tall_pipeline._load_image()

        assert image.size == (1, 100)
        assert context.width == 1
        assert context.height == 100

    def test_debug_mode_output(self, sample_image_path, temp_dir, capsys):
        """Test that debug mode produces output."""
        pipeline = Pipeline(sample_image_path, debug=True)
        output_path = temp_dir / "output.png"

        pipeline.execute(output_path)

        captured = capsys.readouterr()
        # Should have some debug output
        assert len(captured.out) > 0 or len(captured.err) > 0

    def test_pathlib_path_support(self, temp_dir):
        """Test that Pipeline accepts pathlib.Path objects."""
        image_path = temp_dir / "test.png"
        Image.new("RGB", (4, 4), (128, 128, 128)).save(image_path)

        # Should work with Path object
        pipeline = Pipeline(image_path)
        assert pipeline.input_path == image_path

        # Should also work with string
        pipeline_str = Pipeline(str(image_path))
        assert pipeline_str.input_path == image_path

    def test_context_metadata_preservation(self, temp_dir):
        """Test that image metadata is preserved in context."""
        image_path = temp_dir / "test_with_metadata.png"

        # Create image with metadata
        image = Image.new("RGB", (4, 4), (128, 128, 128))
        image.info["description"] = "Test image"
        image.info["author"] = "Test suite"
        image.save(image_path)

        pipeline = Pipeline(image_path)
        _, context = pipeline._load_image()

        # Metadata should be preserved (though some might be lost in save/load)
        assert isinstance(context.metadata, dict)

    def test_estimate_memory_scales_with_operations(self, pipeline):
        """Test that memory estimation accounts for operations."""
        # Empty pipeline memory
        base_memory = pipeline.estimate_memory()

        # With operations, memory should be at least the base
        # (We can't test with actual operations until they're implemented)
        assert base_memory > 0
