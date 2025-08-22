"""Pytest configuration and fixtures for pixel-perfect tests."""

import tempfile
from pathlib import Path

import pytest
from PIL import Image

from .core.context import ImageContext
from .utils.synthetic_images import (
    create_aspect_stretch_test,
    create_block_filter_test,
    create_checkerboard,
    create_column_stretch_test,
    create_gradient,
    create_numbered_grid,
    create_pixel_filter_test,
    create_row_shift_test,
    create_test_suite,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def test_image_4x4():
    """Standard 4x4 test image with numbered pixels."""
    return create_numbered_grid(4, 4)


@pytest.fixture
def test_image_8x8():
    """Standard 8x8 test image with numbered pixels."""
    return create_numbered_grid(8, 8)


@pytest.fixture
def test_image_16x16():
    """Standard 16x16 test image with numbered pixels."""
    return create_numbered_grid(16, 16)


@pytest.fixture
def test_image_1x1():
    """Edge case: 1x1 test image."""
    return create_numbered_grid(1, 1)


@pytest.fixture
def test_image_extreme_wide():
    """Edge case: extremely wide image (1x10)."""
    return create_numbered_grid(1, 10)


@pytest.fixture
def test_image_extreme_tall():
    """Edge case: extremely tall image (10x1)."""
    return create_numbered_grid(10, 1)


@pytest.fixture
def test_context_4x4():
    """ImageContext for 4x4 RGB image."""
    return ImageContext(width=4, height=4, channels=3, dtype="uint8")


@pytest.fixture
def test_context_8x8():
    """ImageContext for 8x8 RGB image."""
    return ImageContext(width=8, height=8, channels=3, dtype="uint8")


@pytest.fixture
def test_context_1x1():
    """ImageContext for 1x1 RGB image."""
    return ImageContext(width=1, height=1, channels=3, dtype="uint8")


@pytest.fixture
def gradient_horizontal():
    """Horizontal gradient for stretch testing."""
    return create_gradient(8, 4, "horizontal")


@pytest.fixture
def gradient_vertical():
    """Vertical gradient for stretch testing."""
    return create_gradient(4, 8, "vertical")


@pytest.fixture
def checkerboard_8x8():
    """8x8 checkerboard pattern."""
    return create_checkerboard(8, 8, 2)


@pytest.fixture
def pixel_filter_test_image():
    """4x4 grid for PixelFilter prime test."""
    return create_pixel_filter_test()


@pytest.fixture
def row_shift_test_image():
    """4x4 image with unique row colors for RowShift test."""
    return create_row_shift_test()


@pytest.fixture
def block_filter_test_image():
    """10x10 image for BlockFilter division test."""
    return create_block_filter_test()


@pytest.fixture
def column_stretch_test_image():
    """4x4 pattern for ColumnStretch test."""
    return create_column_stretch_test()


@pytest.fixture
def aspect_stretch_test_image():
    """15x10 image for AspectStretch segment test."""
    return create_aspect_stretch_test()


@pytest.fixture
def test_suite():
    """Complete test image suite."""
    return create_test_suite()


@pytest.fixture
def grayscale_image():
    """Grayscale test image."""
    return create_numbered_grid(8, 8, "L")


@pytest.fixture
def rgba_image():
    """RGBA test image."""
    return create_numbered_grid(8, 8, "RGBA")


@pytest.fixture
def cache_dir(temp_dir):
    """Temporary cache directory."""
    cache_path = temp_dir / "cache"
    cache_path.mkdir()
    return cache_path


# Pytest markers for different test categories
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "edge_case: marks tests as edge case tests")
    config.addinivalue_line("markers", "visual: marks tests that produce visual output")
    config.addinivalue_line("markers", "memory: marks tests that check memory usage")


# Helper functions for tests
def assert_image_dimensions(image: Image.Image, expected_width: int, expected_height: int):
    """Assert that image has expected dimensions."""
    width, height = image.size
    assert width == expected_width, f"Expected width {expected_width}, got {width}"
    assert height == expected_height, f"Expected height {expected_height}, got {height}"


def assert_context_dimensions(context: ImageContext, expected_width: int, expected_height: int):
    """Assert that context has expected dimensions."""
    assert context.width == expected_width, f"Expected width {expected_width}, got {context.width}"
    assert context.height == expected_height, (
        f"Expected height {expected_height}, got {context.height}"
    )


def assert_pixel_value(image: Image.Image, x: int, y: int, expected_value):
    """Assert that pixel at (x, y) has expected value."""
    actual_value = image.getpixel((x, y))
    assert actual_value == expected_value, (
        f"Pixel at ({x}, {y}): expected {expected_value}, got {actual_value}"
    )


def assert_no_warnings(context: ImageContext):
    """Assert that context has no warnings."""
    assert not context.warnings, f"Unexpected warnings: {context.warnings}"


def get_pixel_linear_index(image: Image.Image, index: int) -> tuple[int, ...]:
    """Get pixel value at linear index (row-major order)."""
    width, height = image.size
    row = index // width
    col = index % width
    pixel = image.getpixel((col, row))
    # Convert to tuple if needed
    if isinstance(pixel, tuple):
        return tuple(int(v) if v is not None else 0 for v in pixel)
    else:
        return (int(pixel) if pixel is not None else 0,)


def count_pixels_with_value(image: Image.Image, value) -> int:
    """Count pixels with specific value."""
    width, height = image.size
    count = 0
    for y in range(height):
        for x in range(width):
            if image.getpixel((x, y)) == value:
                count += 1
    return count
