"""Tests for utility functions."""

import pytest
from PIL import Image

from exceptions import DimensionError, ProcessingError, ValidationError

from .cache import (
    format_bytes,
    get_cache_size,
)
from .image import (
    calculate_memory_usage,
    create_context_from_image,
    create_filled_image,
    ensure_image_mode,
    extract_image_region,
    get_image_hash,
    get_pixel_at_index,
    paste_image_region,
    resize_image_proportional,
)
from .synthetic_images import (
    create_checkerboard,
    create_gradient,
    create_numbered_grid,
    create_test_suite,
)
from .validation import (
    validate_channel_list,
    validate_color_tuple,
    validate_expression_safe,
    validate_image_dimensions,
    validate_indices,
    validate_ratio_string,
)


class TestValidationUtils:
    """Test validation utility functions."""

    def test_validate_image_dimensions_valid(self):
        """Test validation with valid dimensions."""
        # Should not raise any exceptions
        validate_image_dimensions(100, 100)
        validate_image_dimensions(1, 1)  # Minimum valid
        validate_image_dimensions(1920, 1080)  # Common resolution

    def test_validate_image_dimensions_invalid(self):
        """Test validation with invalid dimensions."""
        with pytest.raises(DimensionError):
            validate_image_dimensions(0, 100)

        with pytest.raises(DimensionError):
            validate_image_dimensions(100, -5)

        with pytest.raises(DimensionError):
            validate_image_dimensions(100000, 100000)  # Too large

    def test_validate_indices_valid(self):
        """Test index validation with valid indices."""
        validate_indices([0, 1, 2, 3], 10)
        validate_indices([], 5)  # Empty list is valid
        validate_indices([9], 10)  # Maximum valid index

    def test_validate_indices_invalid(self):
        """Test index validation with invalid indices."""
        with pytest.raises(ValidationError):
            validate_indices([10], 10)  # Index too large

        with pytest.raises(ValidationError):
            validate_indices([-1], 10)  # Negative index

        with pytest.raises(ValidationError):
            validate_indices([0, 5, 15], 10)  # Mixed valid/invalid

    def test_validate_color_tuple_valid(self):
        """Test color validation with valid colors."""
        # RGB
        result = validate_color_tuple((255, 128, 0))
        assert result == (255, 128, 0, 255)  # Should add alpha

        # RGBA
        result = validate_color_tuple((255, 128, 0, 128))
        assert result == (255, 128, 0, 128)

        # Request specific channel count
        result = validate_color_tuple((255, 128, 0, 128), channels=3)
        assert result == (255, 128, 0)  # Should remove alpha

    def test_validate_color_tuple_invalid(self):
        """Test color validation with invalid colors."""
        with pytest.raises(ValidationError):
            validate_color_tuple((255, 128))  # Too few values

        with pytest.raises(ValidationError):
            validate_color_tuple((255, 128, 0, 255, 100))  # Too many values

        with pytest.raises(ValidationError):
            validate_color_tuple((256, 128, 0))  # Value too large

        with pytest.raises(ValidationError):
            validate_color_tuple((255, -1, 0))  # Negative value

    def test_validate_ratio_string_valid(self):
        """Test ratio string validation with valid inputs."""
        assert validate_ratio_string("16:9") == pytest.approx(16 / 9)
        assert validate_ratio_string("4:3") == pytest.approx(4 / 3)
        assert validate_ratio_string("1:1") == 1.0
        assert validate_ratio_string("1.5") == 1.5
        assert validate_ratio_string("2.0") == 2.0

    def test_validate_ratio_string_invalid(self):
        """Test ratio string validation with invalid inputs."""
        with pytest.raises(ValidationError):
            validate_ratio_string("16:0")  # Zero height

        with pytest.raises(ValidationError):
            validate_ratio_string("-1.5")  # Negative ratio

        with pytest.raises(ValidationError):
            validate_ratio_string("invalid")  # Not a number or ratio

    def test_validate_expression_safe_valid(self):
        """Test expression validation with safe expressions."""
        validate_expression_safe("r + g + b")
        validate_expression_safe("r * 0.5")
        validate_expression_safe("sin(x) + cos(y)")
        validate_expression_safe("(r + g) / 2")

    def test_validate_expression_safe_invalid(self):
        """Test expression validation with unsafe expressions."""
        with pytest.raises(ValidationError):
            validate_expression_safe("import os")

        with pytest.raises(ValidationError):
            validate_expression_safe("eval('malicious')")

        with pytest.raises(ValidationError):
            validate_expression_safe("__import__")

        with pytest.raises(ValidationError):
            validate_expression_safe("")  # Empty expression

    def test_validate_channel_list_valid(self):
        """Test channel list validation with valid inputs."""
        result = validate_channel_list(["r", "g", "b"])
        assert result == ["r", "g", "b"]

        result = validate_channel_list(["R", "G", "B", "A"])  # Case insensitive
        assert result == ["r", "g", "b", "a"]

    def test_validate_channel_list_invalid(self):
        """Test channel list validation with invalid inputs."""
        with pytest.raises(ValidationError):
            validate_channel_list([])  # Empty list

        with pytest.raises(ValidationError):
            validate_channel_list(["r", "g", "x"])  # Invalid channel

        with pytest.raises(ValidationError):
            validate_channel_list(["r", "r"])  # Duplicate channel


class TestImageUtils:
    """Test image utility functions."""

    def test_ensure_image_mode_no_conversion_needed(self):
        """Test mode conversion when no conversion is needed."""
        image = Image.new("RGB", (10, 10), (255, 0, 0))
        result = ensure_image_mode(image, "RGB")
        assert result.mode == "RGB"

    def test_ensure_image_mode_conversions(self):
        """Test various mode conversions."""
        # RGB to RGBA
        rgb_image = Image.new("RGB", (10, 10), (255, 0, 0))
        rgba_result = ensure_image_mode(rgb_image, "RGBA")
        assert rgba_result.mode == "RGBA"

        # L to RGB
        gray_image = Image.new("L", (10, 10), 128)
        rgb_result = ensure_image_mode(gray_image, "RGB")
        assert rgb_result.mode == "RGB"

        # RGBA to RGB (should composite against white)
        rgba_image = Image.new("RGBA", (10, 10), (255, 0, 0, 128))
        rgb_result = ensure_image_mode(rgba_image, "RGB")
        assert rgb_result.mode == "RGB"

    def test_get_pixel_at_index_linear(self):
        """Test linear pixel indexing."""
        image = create_numbered_grid(4, 4)

        # Test a few specific pixels
        pixel_0 = get_pixel_at_index(image, 0, "linear")  # Top-left
        pixel_5 = get_pixel_at_index(image, 5, "linear")  # Row 1, col 1

        assert isinstance(pixel_0, tuple)
        assert isinstance(pixel_5, tuple)

    def test_get_pixel_at_index_2d(self):
        """Test 2D pixel indexing."""
        image = create_numbered_grid(4, 4)

        pixel = get_pixel_at_index(image, (1, 2), "2d")  # type: ignore[arg-type] # Row 1, col 2
        assert isinstance(pixel, tuple)

    def test_get_pixel_at_index_bounds_checking(self):
        """Test pixel index bounds checking."""
        image = create_numbered_grid(4, 4)

        with pytest.raises(ProcessingError):
            get_pixel_at_index(image, 16, "linear")  # Out of bounds

        with pytest.raises(ProcessingError):
            get_pixel_at_index(image, (4, 0), "2d")  # type: ignore[arg-type] # Row out of bounds

    def test_create_filled_image(self):
        """Test filled image creation."""
        image = create_filled_image(10, 10, (255, 0, 0, 255), "RGBA")

        assert image.size == (10, 10)
        assert image.mode == "RGBA"

        # Check a pixel to verify color
        pixel = image.getpixel((5, 5))
        assert pixel == (255, 0, 0, 255)

    def test_calculate_memory_usage(self):
        """Test memory usage calculation."""
        # uint8 RGB: 100 * 100 * 3 * 1 = 30000 bytes
        memory = calculate_memory_usage(100, 100, 3, "uint8")
        assert memory == 30000

        # float32 RGBA: 50 * 50 * 4 * 4 = 40000 bytes
        memory = calculate_memory_usage(50, 50, 4, "float32")
        assert memory == 40000

    def test_get_image_hash_consistency(self):
        """Test image hash consistency."""
        image = create_numbered_grid(8, 8)

        hash1 = get_image_hash(image)
        hash2 = get_image_hash(image)

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 32  # MD5 hex length

    def test_get_image_hash_different_images(self):
        """Test that different images produce different hashes."""
        image1 = create_numbered_grid(8, 8)
        image2 = create_checkerboard(8, 8)

        hash1 = get_image_hash(image1)
        hash2 = get_image_hash(image2)

        assert hash1 != hash2

    def test_resize_image_proportional(self):
        """Test proportional image resizing."""
        image = create_numbered_grid(100, 50)  # 2:1 aspect ratio

        # Resize to fit in 60x60 (should become 60x30)
        resized = resize_image_proportional(image, 60, 60)
        assert resized.size == (60, 30)

        # No resize needed if already smaller
        small_image = create_numbered_grid(10, 10)
        no_resize = resize_image_proportional(small_image, 60, 60)
        assert no_resize.size == (10, 10)

    def test_extract_image_region(self):
        """Test image region extraction."""
        image = create_numbered_grid(10, 10)

        # Extract 4x4 region from (2, 2)
        region = extract_image_region(image, 2, 2, 4, 4)
        assert region.size == (4, 4)

    def test_extract_image_region_bounds_checking(self):
        """Test region extraction bounds checking."""
        image = create_numbered_grid(10, 10)

        with pytest.raises(ProcessingError):
            extract_image_region(image, 8, 8, 4, 4)  # Would exceed bounds

    def test_paste_image_region(self):
        """Test image region pasting."""
        target = create_filled_image(10, 10, (0, 0, 0), "RGB")
        source = create_filled_image(4, 4, (255, 255, 255), "RGB")

        result = paste_image_region(target, source, 3, 3)

        # Check that paste worked
        assert result.size == (10, 10)
        # Pixel at (4, 4) should be white (from source)
        assert result.getpixel((4, 4)) == (255, 255, 255)
        # Pixel at (0, 0) should be black (original target)
        assert result.getpixel((0, 0)) == (0, 0, 0)

    def test_create_context_from_image(self):
        """Test ImageContext creation from PIL Image."""
        image = Image.new("RGB", (100, 50), (255, 0, 0))

        context = create_context_from_image(image)

        assert context.width == 100
        assert context.height == 50
        assert context.channels == 3
        assert context.dtype == "uint8"


class TestCacheUtils:
    """Test cache utility functions."""

    def test_format_bytes(self):
        """Test byte formatting."""
        assert format_bytes(0) == "0 B"
        assert format_bytes(1024) == "1.0 KB"
        assert format_bytes(1024 * 1024) == "1.0 MB"
        assert format_bytes(1536) == "1.5 KB"
        assert format_bytes(123) == "123 B"

    def test_cache_size_empty_directory(self, temp_dir):
        """Test cache size calculation for empty directory."""
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()

        size_info = get_cache_size(cache_dir)
        assert size_info["total_bytes"] == 0
        assert size_info["entry_count"] == 0
        assert size_info["files"] == 0

    def test_cache_size_nonexistent_directory(self, temp_dir):
        """Test cache size calculation for nonexistent directory."""
        nonexistent = temp_dir / "nonexistent"

        size_info = get_cache_size(nonexistent)
        assert size_info["total_bytes"] == 0
        assert size_info["entry_count"] == 0


class TestTestImages:
    """Test synthetic test image generation."""

    def test_create_numbered_grid(self):
        """Test numbered grid creation."""
        image = create_numbered_grid(4, 4)

        assert image.size == (4, 4)
        assert image.mode == "RGB"

        # Test that pixels have predictable values
        # (Exact values depend on how overflow is handled)

    def test_create_gradient(self):
        """Test gradient creation."""
        # Horizontal gradient
        h_grad = create_gradient(10, 5, "horizontal")
        assert h_grad.size == (10, 5)
        assert h_grad.mode == "RGB"

        # Vertical gradient
        v_grad = create_gradient(5, 10, "vertical")
        assert v_grad.size == (5, 10)

        # Diagonal gradient
        d_grad = create_gradient(8, 8, "diagonal")
        assert d_grad.size == (8, 8)

    def test_create_gradient_invalid_direction(self):
        """Test gradient creation with invalid direction."""
        with pytest.raises(ProcessingError):
            create_gradient(10, 10, "invalid")  # type: ignore[arg-type]

    def test_create_checkerboard(self):
        """Test checkerboard creation."""
        image = create_checkerboard(8, 8, 2)

        assert image.size == (8, 8)
        assert image.mode == "RGB"

    def test_create_test_suite(self):
        """Test complete test suite generation."""
        test_suite = create_test_suite()

        assert isinstance(test_suite, dict)
        assert len(test_suite) > 0

        # Check that all values are PIL Images
        for name, image in test_suite.items():
            assert isinstance(image, Image.Image)
            assert isinstance(name, str)

    def test_test_suite_contains_expected_images(self):
        """Test that test suite contains expected image types."""
        test_suite = create_test_suite()

        # Should contain grids for different sizes
        grid_images = [name for name in test_suite if name.startswith("grid_")]
        assert len(grid_images) > 0

        # Should contain gradients
        grad_images = [name for name in test_suite if name.startswith("grad_")]
        assert len(grad_images) > 0

    def test_edge_case_images(self):
        """Test edge case image generation."""
        # 1x1 image
        tiny = create_numbered_grid(1, 1)
        assert tiny.size == (1, 1)

        # Extreme aspect ratios
        wide = create_numbered_grid(10, 1)
        assert wide.size == (10, 1)

        tall = create_numbered_grid(1, 10)
        assert tall.size == (1, 10)

    @pytest.mark.slow
    def test_large_test_images(self):
        """Test creating larger test images."""
        large_grid = create_numbered_grid(100, 100)
        assert large_grid.size == (100, 100)

        large_checker = create_checkerboard(64, 64, 8)
        assert large_checker.size == (64, 64)

    def test_different_image_modes(self):
        """Test creating images in different modes."""
        # RGB (default)
        rgb_image = create_numbered_grid(8, 8, "RGB")
        assert rgb_image.mode == "RGB"

        # Grayscale
        gray_image = create_numbered_grid(8, 8, "L")
        assert gray_image.mode == "L"

        # RGBA
        rgba_image = create_numbered_grid(8, 8, "RGBA")
        assert rgba_image.mode == "RGBA"
