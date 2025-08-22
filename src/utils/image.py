"""Image processing utility functions."""

import hashlib
from typing import Literal

import numpy as np
from PIL import Image

from core.context import ImageContext
from exceptions import ProcessingError


def ensure_image_mode(image: Image.Image, required_mode: str) -> Image.Image:
    """Convert image to required mode if necessary.

    Args:
        image: Input PIL Image
        required_mode: Target mode ("L", "RGB", "RGBA")

    Returns:
        Image in the required mode

    Raises:
        ProcessingError: If conversion fails
    """
    if image.mode == required_mode:
        return image

    try:
        if required_mode == "RGBA":
            if image.mode == "L":
                # Grayscale to RGBA: expand to RGB first, then add alpha
                rgb_image = image.convert("RGB")
                return rgb_image.convert("RGBA")
            else:
                return image.convert("RGBA")
        elif required_mode == "RGB":
            if image.mode == "RGBA":
                # RGBA to RGB: composite against white background
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])  # Use alpha as mask
                return background
            else:
                return image.convert("RGB")
        elif required_mode == "L":
            return image.convert("L")
        else:
            raise ProcessingError(f"Unsupported target mode: {required_mode}")

    except Exception as e:
        raise ProcessingError(
            f"Failed to convert image from {image.mode} to {required_mode}: {e}"
        ) from e


def get_pixel_at_index(
    image: Image.Image, index: int, mode: Literal["linear", "2d"] = "linear"
) -> tuple[int, ...]:
    """Get pixel value at specified index.

    Args:
        image: PIL Image
        index: Pixel index
        mode: "linear" for row-major indexing, "2d" for (row, col) indexing

    Returns:
        Pixel value as tuple

    Raises:
        ProcessingError: If index is out of bounds
    """
    width, height = image.size
    total_pixels = width * height

    if mode == "linear":
        if index < 0 or index >= total_pixels:
            raise ProcessingError(f"Pixel index {index} out of bounds (0-{total_pixels - 1})")

        row = index // width
        col = index % width
    else:  # mode == "2d"
        if isinstance(index, tuple | list) and len(index) == 2:
            row, col = index
        else:
            raise ProcessingError(f"2D mode requires (row, col) tuple, got {index}")

        if row < 0 or row >= height or col < 0 or col >= width:
            raise ProcessingError(f"Pixel coordinates ({row}, {col}) out of bounds")

    try:
        pixel = image.getpixel((col, row))
        # Ensure tuple return type with proper int conversion
        if isinstance(pixel, int | float):
            return (int(pixel),)
        if isinstance(pixel, list | tuple):
            return tuple(
                int(p) if isinstance(p, float) else (p if p is not None else 0) for p in pixel
            )
        return (int(pixel) if pixel is not None else 0,)
    except Exception as e:
        raise ProcessingError(f"Failed to get pixel at ({row}, {col}): {e}") from e


def create_filled_image(
    width: int, height: int, color: tuple[int, ...], mode: str = "RGBA"
) -> Image.Image:
    """Create an image filled with specified color.

    Args:
        width: Image width
        height: Image height
        color: Fill color tuple
        mode: Image mode

    Returns:
        New PIL Image filled with color

    Raises:
        ProcessingError: If image creation fails
    """
    try:
        # Ensure color tuple matches mode
        if mode == "L" and len(color) > 1:
            # Convert to grayscale using luminance formula
            r, g, b = color[:3]
            gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
            color = (gray_value,)
        elif mode == "RGB" and len(color) > 3:
            color = color[:3]
        elif mode == "RGBA" and len(color) == 3:
            color = color + (255,)

        return Image.new(mode, (width, height), color)

    except Exception as e:
        raise ProcessingError(f"Failed to create {width}×{height} image: {e}") from e


def copy_image_with_context(image: Image.Image, context: ImageContext) -> Image.Image:
    """Create a deep copy of image preserving metadata.

    Args:
        image: Source PIL Image
        context: Image context with metadata

    Returns:
        Deep copy of the image

    Raises:
        ProcessingError: If copying fails
    """
    try:
        # Create copy
        copied = image.copy()

        # Preserve metadata if available
        if context.metadata:
            for key, value in context.metadata.items():
                if isinstance(value, str | int | float):
                    from contextlib import suppress

                    with suppress(TypeError, ValueError):
                        copied.info[key] = value

        return copied

    except Exception as e:
        raise ProcessingError(f"Failed to copy image: {e}") from e


def calculate_memory_usage(
    width: int, height: int, channels: int, dtype: Literal["uint8", "float32"]
) -> int:
    """Calculate memory usage for image with given parameters.

    Args:
        width: Image width
        height: Image height
        channels: Number of channels
        dtype: Data type

    Returns:
        Memory usage in bytes
    """
    bytes_per_pixel = 1 if dtype == "uint8" else 4
    return width * height * channels * bytes_per_pixel


def get_image_hash(image: Image.Image) -> str:
    """Generate MD5 hash of image data.

    Args:
        image: PIL Image

    Returns:
        Hex string of image hash

    Raises:
        ProcessingError: If hash generation fails
    """
    try:
        # Convert to numpy array for consistent hashing
        arr = np.array(image)
        return hashlib.md5(arr.tobytes()).hexdigest()

    except Exception as e:
        raise ProcessingError(f"Failed to generate image hash: {e}") from e


def resize_image_proportional(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    """Resize image proportionally to fit within max dimensions.

    Args:
        image: Source PIL Image
        max_width: Maximum width
        max_height: Maximum height

    Returns:
        Resized image

    Raises:
        ProcessingError: If resizing fails
    """
    try:
        width, height = image.size

        # Calculate scale factor to fit within max dimensions
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h)

        if scale >= 1.0:
            return image  # No need to resize

        new_width = int(width * scale)
        new_height = int(height * scale)

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    except Exception as e:
        raise ProcessingError(f"Failed to resize image: {e}") from e


def extract_image_region(
    image: Image.Image, x: int, y: int, width: int, height: int
) -> Image.Image:
    """Extract rectangular region from image.

    Args:
        image: Source PIL Image
        x: Left coordinate
        y: Top coordinate
        width: Region width
        height: Region height

    Returns:
        Extracted region as new image

    Raises:
        ProcessingError: If extraction fails or coordinates are invalid
    """
    img_width, img_height = image.size

    # Validate coordinates
    if x < 0 or y < 0 or x + width > img_width or y + height > img_height:
        raise ProcessingError(
            f"Region ({x}, {y}, {width}, {height}) exceeds image bounds ({img_width}, {img_height})"
        )

    try:
        return image.crop((x, y, x + width, y + height))

    except Exception as e:
        raise ProcessingError(f"Failed to extract region: {e}") from e


def paste_image_region(target: Image.Image, source: Image.Image, x: int, y: int) -> Image.Image:
    """Paste source image into target at specified position.

    Args:
        target: Target PIL Image (will be modified)
        source: Source PIL Image to paste
        x: Left position in target
        y: Top position in target

    Returns:
        Modified target image

    Raises:
        ProcessingError: If pasting fails
    """
    try:
        # Check if source fits in target at given position
        target_width, target_height = target.size
        source_width, source_height = source.size

        if x + source_width > target_width or y + source_height > target_height:
            raise ProcessingError(
                f"Source image {source_width}×{source_height} at ({x}, {y}) exceeds target bounds {target_width}×{target_height}"
            )

        # Handle alpha channel for proper compositing
        if source.mode == "RGBA":
            target.paste(source, (x, y), source)  # Use alpha as mask
        else:
            target.paste(source, (x, y))

        return target

    except Exception as e:
        raise ProcessingError(f"Failed to paste image region: {e}") from e


def create_context_from_image(image: Image.Image) -> ImageContext:
    """Create ImageContext from PIL Image.

    Args:
        image: PIL Image

    Returns:
        ImageContext with image properties
    """
    channels_map = {"L": 1, "RGB": 3, "RGBA": 4}
    channels = channels_map.get(image.mode, 4)

    # Extract metadata
    metadata = {}
    if hasattr(image, "info") and image.info:
        for key, value in image.info.items():
            if isinstance(value, str | int | float | bool):
                metadata[key] = value

    return ImageContext(
        width=image.width,
        height=image.height,
        channels=channels,
        dtype="uint8",  # PIL images are typically uint8
        metadata=metadata,
    )
