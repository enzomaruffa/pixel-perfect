"""Test utilities for the pixel-perfect integration tests."""

import tempfile
from pathlib import Path

from PIL import Image


def create_numbered_grid(width: int, height: int, cell_size: int = 8) -> Image.Image:
    """Create a numbered grid test image."""
    img = Image.new("RGBA", (width, height), (255, 255, 255, 255))

    # Create a simple grid pattern with numbers
    pixels = []
    for y in range(height):
        for x in range(width):
            # Create grid lines
            grid_x = (x // cell_size) % 2
            grid_y = (y // cell_size) % 2

            if (x % cell_size == 0) or (y % cell_size == 0):
                # Grid lines in black
                pixels.append((0, 0, 0, 255))
            elif (grid_x + grid_y) % 2 == 0:
                # Checkered pattern
                pixels.append((200, 200, 200, 255))
            else:
                pixels.append((128, 128, 128, 255))

    img.putdata(pixels)
    return img


def create_gradient_image(width: int, height: int, direction: str = "horizontal") -> Image.Image:
    """Create a gradient test image."""
    img = Image.new("RGBA", (width, height))
    pixels = []

    for y in range(height):
        for x in range(width):
            if direction == "horizontal":
                intensity = int((x / width) * 255)
                pixels.append((intensity, intensity, intensity, 255))
            elif direction == "vertical":
                intensity = int((y / height) * 255)
                pixels.append((intensity, intensity, intensity, 255))
            elif direction == "diagonal":
                intensity = int(((x + y) / (width + height)) * 255)
                pixels.append((intensity, intensity, intensity, 255))
            else:  # radial
                center_x, center_y = width // 2, height // 2
                distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                max_distance = ((center_x) ** 2 + (center_y) ** 2) ** 0.5
                intensity = int((distance / max_distance) * 255) if max_distance > 0 else 0
                intensity = min(255, intensity)
                pixels.append((intensity, intensity, intensity, 255))

    img.putdata(pixels)
    return img


def create_color_pattern_image(width: int, height: int, pattern: str = "rgb_bars") -> Image.Image:
    """Create a colorful pattern test image."""
    img = Image.new("RGBA", (width, height))
    pixels = []

    for y in range(height):
        for x in range(width):
            if pattern == "rgb_bars":
                # Vertical RGB color bars
                section = (x * 3) // width
                if section == 0:
                    pixels.append((255, 0, 0, 255))  # Red
                elif section == 1:
                    pixels.append((0, 255, 0, 255))  # Green
                else:
                    pixels.append((0, 0, 255, 255))  # Blue
            elif pattern == "rainbow":
                # Rainbow pattern
                hue = (x / width) * 360
                sat = 1.0
                val = 1.0
                # Simple HSV to RGB conversion
                c = val * sat
                x_mod = (hue / 60) % 2
                m = val - c
                if 0 <= hue < 60:
                    r, g, b = c, c * x_mod, 0
                elif 60 <= hue < 120:
                    r, g, b = c * (2 - x_mod), c, 0
                elif 120 <= hue < 180:
                    r, g, b = 0, c, c * x_mod
                elif 180 <= hue < 240:
                    r, g, b = 0, c * (2 - x_mod), c
                elif 240 <= hue < 300:
                    r, g, b = c * x_mod, 0, c
                else:
                    r, g, b = c, 0, c * (2 - x_mod)

                r, g, b = int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)
                pixels.append((r, g, b, 255))
            else:  # checkerboard
                # Checkered pattern
                cell_size = 8
                check_x = (x // cell_size) % 2
                check_y = (y // cell_size) % 2
                if (check_x + check_y) % 2 == 0:
                    pixels.append((255, 255, 255, 255))
                else:
                    pixels.append((0, 0, 0, 255))

    img.putdata(pixels)
    return img


def save_test_images(output_dir: Path) -> dict:
    """Create and save a suite of test images for integration tests."""
    output_dir.mkdir(parents=True, exist_ok=True)

    test_images = {}

    # Standard test images
    test_images["grid_64x64"] = create_numbered_grid(64, 64)
    test_images["grid_100x75"] = create_numbered_grid(100, 75)
    test_images["gradient_horizontal"] = create_gradient_image(80, 60, "horizontal")
    test_images["gradient_vertical"] = create_gradient_image(80, 60, "vertical")
    test_images["gradient_radial"] = create_gradient_image(80, 80, "radial")
    test_images["rgb_bars"] = create_color_pattern_image(90, 60, "rgb_bars")
    test_images["rainbow"] = create_color_pattern_image(120, 80, "rainbow")
    test_images["checkerboard"] = create_color_pattern_image(64, 64, "checkerboard")

    # Edge case images
    test_images["minimal_1x1"] = Image.new("RGBA", (1, 1), (128, 128, 128, 255))
    test_images["tiny_3x3"] = create_numbered_grid(3, 3, 1)
    test_images["wide_200x10"] = create_gradient_image(200, 10, "horizontal")
    test_images["tall_10x200"] = create_gradient_image(10, 200, "vertical")
    test_images["large_500x400"] = create_numbered_grid(500, 400, 16)

    # Save all images
    for name, image in test_images.items():
        image.save(output_dir / f"{name}.png")

    return test_images


def create_temp_test_suite() -> tuple[Path, dict]:
    """Create a temporary directory with test images."""
    temp_dir = Path(tempfile.mkdtemp())
    test_images = save_test_images(temp_dir)
    return temp_dir, test_images
