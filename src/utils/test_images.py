"""Synthetic test image generation for validating operations."""

from typing import Literal

import numpy as np
from PIL import Image

from exceptions import ProcessingError


def create_numbered_grid(width: int, height: int, mode: str = "RGB") -> Image.Image:
    """Create image where each pixel value equals row * width + col.

    This creates a predictable pattern where pixel values increase linearly
    from top-left to bottom-right, making it easy to verify operations
    that rearrange or filter pixels.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        mode: Image mode ("L", "RGB", "RGBA")

    Returns:
        PIL Image with numbered pixel values

    Raises:
        ProcessingError: If image creation fails
    """
    try:
        total_pixels = width * height

        # Create array with linear indices
        # Handle large images by using modulo from the start
        indices = np.arange(total_pixels) % 256
        indices = indices.astype(np.uint8)

        # Reshape to image dimensions
        if mode == "L":
            data = indices.reshape(height, width)
        elif mode == "RGB":
            # Replicate across RGB channels
            data = np.stack([indices.reshape(height, width)] * 3, axis=-1)
        elif mode == "RGBA":
            # RGB + opaque alpha
            rgb_data = np.stack([indices.reshape(height, width)] * 3, axis=-1)
            alpha_data = np.full((height, width, 1), 255, dtype=np.uint8)
            data = np.concatenate([rgb_data, alpha_data], axis=-1)
        else:
            raise ProcessingError(f"Unsupported mode: {mode}")

        image = Image.fromarray(data)
        if image.mode != mode:
            image = image.convert(mode)
        return image

    except Exception as e:
        raise ProcessingError(f"Failed to create numbered grid {width}×{height}: {e}") from e


def create_gradient(
    width: int,
    height: int,
    direction: Literal["horizontal", "vertical", "diagonal"] = "horizontal",
    mode: str = "RGB",
) -> Image.Image:
    """Create gradient image for testing stretch operations.

    Args:
        width: Image width
        height: Image height
        direction: Gradient direction
        mode: Image mode

    Returns:
        PIL Image with gradient pattern

    Raises:
        ProcessingError: If image creation fails
    """
    try:
        if direction == "horizontal":
            # Gradient from left (0) to right (255)
            gradient = np.linspace(0, 255, width, dtype=np.uint8)
            data = np.tile(gradient, (height, 1))
        elif direction == "vertical":
            # Gradient from top (0) to bottom (255)
            gradient = np.linspace(0, 255, height, dtype=np.uint8)
            data = np.tile(gradient.reshape(-1, 1), (1, width))
        elif direction == "diagonal":
            # Diagonal gradient from top-left to bottom-right
            x_grad = np.linspace(0, 127, width)
            y_grad = np.linspace(0, 127, height)
            xx, yy = np.meshgrid(x_grad, y_grad)
            data = np.clip(xx + yy, 0, 255).astype(np.uint8)
        else:
            raise ProcessingError(f"Invalid gradient direction: {direction}")

        # Convert to requested mode
        if mode == "RGB":
            data = np.stack([data] * 3, axis=-1)
        elif mode == "RGBA":
            rgb_data = np.stack([data] * 3, axis=-1)
            alpha_data = np.full((height, width, 1), 255, dtype=np.uint8)
            data = np.concatenate([rgb_data, alpha_data], axis=-1)
        elif mode != "L":
            raise ProcessingError(f"Unsupported mode: {mode}")

        image = Image.fromarray(data)
        if image.mode != mode:
            image = image.convert(mode)
        return image

    except Exception as e:
        raise ProcessingError(f"Failed to create gradient {direction}: {e}") from e


def create_checkerboard(
    width: int, height: int, square_size: int = 4, mode: str = "RGB"
) -> Image.Image:
    """Create checkerboard pattern for testing pattern operations.

    Args:
        width: Image width
        height: Image height
        square_size: Size of each checker square
        mode: Image mode

    Returns:
        PIL Image with checkerboard pattern

    Raises:
        ProcessingError: If image creation fails
    """
    try:
        # Create checkerboard pattern
        data = np.zeros((height, width), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                # Determine which square we're in
                square_x = x // square_size
                square_y = y // square_size
                # Checkerboard pattern: alternating 0 and 255
                if (square_x + square_y) % 2 == 0:
                    data[y, x] = 255

        # Convert to requested mode
        if mode == "RGB":
            data = np.stack([data] * 3, axis=-1)
        elif mode == "RGBA":
            rgb_data = np.stack([data] * 3, axis=-1)
            alpha_data = np.full((height, width, 1), 255, dtype=np.uint8)
            data = np.concatenate([rgb_data, alpha_data], axis=-1)
        elif mode != "L":
            raise ProcessingError(f"Unsupported mode: {mode}")

        image = Image.fromarray(data)
        if image.mode != mode:
            image = image.convert(mode)
        return image

    except Exception as e:
        raise ProcessingError(f"Failed to create checkerboard: {e}") from e


def create_channel_test(
    width: int, height: int, channels: Literal["r", "g", "b", "all"] = "r"
) -> Image.Image:
    """Create image with specific color channels for channel operation testing.

    Args:
        width: Image width
        height: Image height
        channels: Which channel(s) to activate

    Returns:
        PIL Image with specified channels active

    Raises:
        ProcessingError: If image creation fails
    """
    try:
        if channels == "r":
            # Pure red
            data = np.zeros((height, width, 3), dtype=np.uint8)
            data[:, :, 0] = 255  # Red channel
        elif channels == "g":
            # Pure green
            data = np.zeros((height, width, 3), dtype=np.uint8)
            data[:, :, 1] = 255  # Green channel
        elif channels == "b":
            # Pure blue
            data = np.zeros((height, width, 3), dtype=np.uint8)
            data[:, :, 2] = 255  # Blue channel
        elif channels == "all":
            # White (all channels)
            data = np.full((height, width, 3), 255, dtype=np.uint8)
        else:
            raise ProcessingError(f"Invalid channel specification: {channels}")

        return Image.fromarray(data)

    except Exception as e:
        raise ProcessingError(f"Failed to create channel test image: {e}") from e


def create_stripes(
    width: int,
    height: int,
    stripe_width: int = 1,
    orientation: Literal["horizontal", "vertical"] = "vertical",
    mode: str = "RGB",
) -> Image.Image:
    """Create high frequency stripe pattern for aliasing tests.

    Args:
        width: Image width
        height: Image height
        stripe_width: Width of each stripe in pixels
        orientation: Stripe orientation
        mode: Image mode

    Returns:
        PIL Image with stripe pattern

    Raises:
        ProcessingError: If image creation fails
    """
    try:
        data = np.zeros((height, width), dtype=np.uint8)

        if orientation == "vertical":
            # Vertical stripes
            for x in range(width):
                if (x // stripe_width) % 2 == 0:
                    data[:, x] = 255
        elif orientation == "horizontal":
            # Horizontal stripes
            for y in range(height):
                if (y // stripe_width) % 2 == 0:
                    data[y, :] = 255
        else:
            raise ProcessingError(f"Invalid orientation: {orientation}")

        # Convert to requested mode
        if mode == "RGB":
            data = np.stack([data] * 3, axis=-1)
        elif mode == "RGBA":
            rgb_data = np.stack([data] * 3, axis=-1)
            alpha_data = np.full((height, width, 1), 255, dtype=np.uint8)
            data = np.concatenate([rgb_data, alpha_data], axis=-1)
        elif mode != "L":
            raise ProcessingError(f"Unsupported mode: {mode}")

        image = Image.fromarray(data)
        if image.mode != mode:
            image = image.convert(mode)
        return image

    except Exception as e:
        raise ProcessingError(f"Failed to create stripes: {e}") from e


def create_circle(
    width: int,
    height: int,
    radius: int | None = None,
    center: tuple[int, int] | None = None,
    mode: str = "RGB",
) -> Image.Image:
    """Create circle for geometric operation testing.

    Args:
        width: Image width
        height: Image height
        radius: Circle radius (defaults to min(width, height) // 4)
        center: Circle center (defaults to image center)
        mode: Image mode

    Returns:
        PIL Image with circle

    Raises:
        ProcessingError: If image creation fails
    """
    try:
        if radius is None:
            radius = min(width, height) // 4

        if center is None:
            center = (width // 2, height // 2)

        cx, cy = center
        data = np.zeros((height, width), dtype=np.uint8)

        # Create coordinate grids
        y_coords, x_coords = np.ogrid[:height, :width]

        # Calculate distance from center
        distances = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)

        # Set pixels inside radius to white
        data[distances <= radius] = 255

        # Convert to requested mode
        if mode == "RGB":
            data = np.stack([data] * 3, axis=-1)
        elif mode == "RGBA":
            rgb_data = np.stack([data] * 3, axis=-1)
            alpha_data = np.full((height, width, 1), 255, dtype=np.uint8)
            data = np.concatenate([rgb_data, alpha_data], axis=-1)
        elif mode != "L":
            raise ProcessingError(f"Unsupported mode: {mode}")

        image = Image.fromarray(data)
        if image.mode != mode:
            image = image.convert(mode)
        return image

    except Exception as e:
        raise ProcessingError(f"Failed to create circle: {e}") from e


def create_noise(
    width: int,
    height: int,
    noise_type: Literal["uniform", "gaussian"] = "uniform",
    mode: str = "RGB",
    seed: int = 42,
) -> Image.Image:
    """Create noise pattern for testing robustness.

    Args:
        width: Image width
        height: Image height
        noise_type: Type of noise to generate
        mode: Image mode
        seed: Random seed for reproducibility

    Returns:
        PIL Image with noise pattern

    Raises:
        ProcessingError: If image creation fails
    """
    try:
        np.random.seed(seed)

        if noise_type == "uniform":
            data = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        elif noise_type == "gaussian":
            # Gaussian noise with mean=128, std=64
            noise = np.random.normal(128, 64, (height, width))
            data = np.clip(noise, 0, 255).astype(np.uint8)
        else:
            raise ProcessingError(f"Invalid noise type: {noise_type}")

        # Convert to requested mode
        if mode == "RGB":
            data = np.stack([data] * 3, axis=-1)
        elif mode == "RGBA":
            rgb_data = np.stack([data] * 3, axis=-1)
            alpha_data = np.full((height, width, 1), 255, dtype=np.uint8)
            data = np.concatenate([rgb_data, alpha_data], axis=-1)
        elif mode != "L":
            raise ProcessingError(f"Unsupported mode: {mode}")

        image = Image.fromarray(data)
        if image.mode != mode:
            image = image.convert(mode)
        return image

    except Exception as e:
        raise ProcessingError(f"Failed to create noise: {e}") from e


def create_test_suite() -> dict[str, Image.Image]:
    """Generate complete set of test images for comprehensive testing.

    Returns:
        Dictionary mapping test names to PIL Images
    """
    test_images = {}

    # Standard test sizes
    sizes = [
        (4, 4),  # Tiny for detailed verification
        (8, 8),  # Small for pattern testing
        (16, 16),  # Medium for block operations
        (1, 1),  # Edge case: minimum size
        (1, 10),  # Edge case: extreme aspect ratio
        (10, 1),  # Edge case: extreme aspect ratio
    ]

    for width, height in sizes:
        size_key = f"{width}x{height}"

        # Numbered grid for pixel-level verification
        test_images[f"grid_{size_key}"] = create_numbered_grid(width, height)

        # Only create other patterns for non-degenerate sizes
        if width > 1 and height > 1:
            # Gradients for stretch testing
            test_images[f"grad_h_{size_key}"] = create_gradient(width, height, "horizontal")
            test_images[f"grad_v_{size_key}"] = create_gradient(width, height, "vertical")
            test_images[f"grad_d_{size_key}"] = create_gradient(width, height, "diagonal")

            # Checkerboard for pattern operations
            square_size = max(1, min(width, height) // 4)
            test_images[f"checker_{size_key}"] = create_checkerboard(width, height, square_size)

            # Channel tests
            test_images[f"red_{size_key}"] = create_channel_test(width, height, "r")
            test_images[f"green_{size_key}"] = create_channel_test(width, height, "g")
            test_images[f"blue_{size_key}"] = create_channel_test(width, height, "b")

    # Special test images
    test_images["stripes_v_32x32"] = create_stripes(32, 32, 1, "vertical")
    test_images["stripes_h_32x32"] = create_stripes(32, 32, 1, "horizontal")
    test_images["circle_32x32"] = create_circle(32, 32)
    test_images["noise_16x16"] = create_noise(16, 16, "uniform")

    return test_images


def save_test_suite(output_dir: str = "test_images") -> None:
    """Save complete test suite to disk for visual inspection.

    Args:
        output_dir: Directory to save test images
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    test_images = create_test_suite()

    for name, image in test_images.items():
        file_path = output_path / f"{name}.png"
        image.save(file_path)

    print(f"Saved {len(test_images)} test images to {output_path}")


# Convenience functions for specific test cases from SPEC.md
def create_pixel_filter_test() -> Image.Image:
    """Create 4×4 grid for PixelFilter prime test (pixels 2,3,5,7,11,13 preserved)."""
    return create_numbered_grid(4, 4)


def create_row_shift_test() -> Image.Image:
    """Create 4×4 image with unique row values for RowShift wrap test."""
    data = np.zeros((4, 4, 3), dtype=np.uint8)
    for row in range(4):
        # Each row has a different color
        color_value = (row + 1) * 60  # 60, 120, 180, 240
        data[row, :, :] = color_value
    return Image.fromarray(data, "RGB")


def create_block_filter_test() -> Image.Image:
    """Create 10×10 image for BlockFilter division test with 3×3 blocks."""
    return create_numbered_grid(10, 10)


def create_column_stretch_test() -> Image.Image:
    """Create 4×4 pattern for ColumnStretch factor test."""
    return create_stripes(4, 4, 1, "vertical")


def create_aspect_stretch_test() -> Image.Image:
    """Create 15×10 image for AspectStretch segment test to 1:1 with 3 segments."""
    return create_gradient(15, 10, "horizontal")
