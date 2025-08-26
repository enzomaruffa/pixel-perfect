"""Artistic pattern effect operations."""

from typing import Literal

import numpy as np
from PIL import Image
from pydantic import Field, field_validator

from core.base import BaseOperation
from core.context import ImageContext
from exceptions import ProcessingError, ValidationError
from utils.validation import validate_color_tuple


def _generate_bayer_matrix(size: int) -> np.ndarray:
    """Generate Bayer matrix for ordered dithering.

    Args:
        size: Matrix size (must be power of 2)

    Returns:
        Bayer matrix normalized to 0-1 range
    """
    if size <= 0 or (size & (size - 1)) != 0:
        raise ValidationError(f"Matrix size must be power of 2, got {size}")

    if size == 1:
        return np.array([[0]], dtype=np.float32)
    elif size == 2:
        return np.array([[0, 2], [3, 1]], dtype=np.float32) / 4.0
    else:
        # Recursive generation for larger matrices
        half_size = size // 2
        half_matrix = _generate_bayer_matrix(half_size)

        # Construct full matrix
        full_matrix = np.zeros((size, size), dtype=np.float32)
        full_matrix[0:half_size, 0:half_size] = half_matrix
        full_matrix[0:half_size, half_size:size] = half_matrix + 2
        full_matrix[half_size:size, 0:half_size] = half_matrix + 3
        full_matrix[half_size:size, half_size:size] = half_matrix + 1

        return full_matrix / (size * size)


def _quantize_value(value: float, levels: int) -> int:
    """Quantize value to specified number of levels.

    Args:
        value: Input value (0-255)
        levels: Number of quantization levels

    Returns:
        Quantized value (0-255)
    """
    if levels <= 1:
        return 0

    # Calculate quantization step
    step = 255.0 / (levels - 1)

    # Find nearest quantization level
    level_index = round(value / step)
    level_index = max(0, min(levels - 1, level_index))

    return int(level_index * step)


def _sample_tile_color(
    tile_pixels: np.ndarray, mode: str, random_seed: int | None = None
) -> tuple[int, ...]:
    """Extract representative color from tile region.

    Args:
        tile_pixels: Tile pixel array (H, W, C)
        mode: Sampling mode ("average", "center", "random")
        random_seed: Random seed for reproducible sampling

    Returns:
        Representative color tuple
    """
    if tile_pixels.size == 0:
        return (0, 0, 0)

    if mode == "average":
        # Calculate mean color
        mean_color = np.mean(tile_pixels.reshape(-1, tile_pixels.shape[-1]), axis=0)
        return tuple(int(c) for c in mean_color)

    elif mode == "center":
        # Use center pixel
        center_y = tile_pixels.shape[0] // 2
        center_x = tile_pixels.shape[1] // 2
        center_pixel = tile_pixels[center_y, center_x]
        return tuple(int(c) for c in center_pixel)

    elif mode == "random":
        # Use random pixel
        if random_seed is not None:
            np.random.seed(random_seed)

        flat_pixels = tile_pixels.reshape(-1, tile_pixels.shape[-1])
        random_index = np.random.randint(0, len(flat_pixels))
        random_pixel = flat_pixels[random_index]
        return tuple(int(c) for c in random_pixel)

    else:
        raise ValidationError(f"Invalid sampling mode: {mode}")


def _draw_tile_with_gaps(
    canvas: np.ndarray,
    position: tuple[int, int],
    color: tuple[int, ...],
    tile_size: tuple[int, int],
    gap_size: int,
) -> None:
    """Draw tile on canvas with gaps.

    Args:
        canvas: Output canvas array
        position: Tile position (x, y)
        color: Tile color
        tile_size: Tile dimensions
        gap_size: Gap size in pixels
    """
    x, y = position
    tile_width, tile_height = tile_size

    # Calculate actual tile area (excluding gaps)
    actual_width = tile_width - gap_size
    actual_height = tile_height - gap_size

    if actual_width <= 0 or actual_height <= 0:
        return

    # Draw tile (leaving gaps around edges)
    end_x = min(x + actual_width, canvas.shape[1])
    end_y = min(y + actual_height, canvas.shape[0])

    if len(canvas.shape) == 3:
        # Color image
        canvas[y:end_y, x:end_x] = color[: canvas.shape[2]]
    else:
        # Grayscale image
        canvas[y:end_y, x:end_x] = color[0] if color else 0


def _diffuse_error(error_buffer: np.ndarray, x: int, y: int, error: np.ndarray) -> None:
    """Diffuse quantization error using Floyd-Steinberg weights.

    Args:
        error_buffer: Error accumulation buffer
        x, y: Current pixel position
        error: Error to diffuse
    """
    height, width = error_buffer.shape[:2]

    # Floyd-Steinberg error diffusion weights
    # Current pixel: 0
    # Right: 7/16, Below: 5/16, Below-left: 3/16, Below-right: 1/16

    if x + 1 < width:
        error_buffer[y, x + 1] += error * (7.0 / 16.0)

    if y + 1 < height:
        if x - 1 >= 0:
            error_buffer[y + 1, x - 1] += error * (3.0 / 16.0)

        error_buffer[y + 1, x] += error * (5.0 / 16.0)

        if x + 1 < width:
            error_buffer[y + 1, x + 1] += error * (1.0 / 16.0)


class Mosaic(BaseOperation):
    """Create mosaic/tile effect with configurable sampling."""

    tile_width: int = Field(8, ge=1, le=256, description="Width of mosaic tiles in pixels")
    tile_height: int = Field(8, ge=1, le=256, description="Height of mosaic tiles in pixels")
    gap_size: int = Field(1, ge=0, description="Spacing between tiles in pixels")
    gap_color: tuple[int, int, int, int] = Field((0, 0, 0, 255), description="RGBA color for gaps")
    sample_mode: Literal["average", "center", "random"] = Field(
        "average", description="Tile color sampling method"
    )
    random_seed: int | None = Field(None, description="Random seed for reproducible sampling")

    @field_validator("gap_color")
    @classmethod
    def validate_gap_color(cls, v):
        """Validate gap color."""
        return validate_color_tuple(v, channels=4)

    def validate_operation(self, context: ImageContext) -> ImageContext:
        """Validate operation against image context."""
        # Mosaic doesn't change image dimensions
        return context

    def get_cache_key(self, image_hash: str) -> str:
        """Generate cache key for this operation."""
        config_str = f"{self.tile_width}_{self.tile_height}_{self.gap_size}_{self.gap_color}_{self.sample_mode}_{self.random_seed}"
        return f"mosaic_{image_hash}_{hash(config_str)}"

    def estimate_memory(self, context: ImageContext) -> int:
        """Estimate memory usage for this operation."""
        # Same dimensions as input
        return context.width * context.height * context.channels

    def apply(self, image: Image.Image, context: ImageContext) -> tuple[Image.Image, ImageContext]:
        """Apply mosaic effect."""
        try:
            image_array = np.array(image)
            height, width = image_array.shape[:2]
            channels = image_array.shape[2] if len(image_array.shape) == 3 else 1

            # Create output canvas
            if len(image_array.shape) == 3:
                canvas = np.full(
                    (height, width, channels), self.gap_color[:channels], dtype=np.uint8
                )
            else:
                canvas = np.full((height, width), self.gap_color[0], dtype=np.uint8)

            tile_width, tile_height = self.tile_width, self.tile_height

            # Process tiles
            for y in range(0, height, tile_height):
                for x in range(0, width, tile_width):
                    # Extract tile region
                    end_x = min(x + tile_width, width)
                    end_y = min(y + tile_height, height)

                    tile_pixels = image_array[y:end_y, x:end_x]

                    if tile_pixels.size == 0:
                        continue

                    # Sample tile color
                    seed = None
                    if self.random_seed is not None:
                        # Create unique seed for each tile
                        seed = self.random_seed + y * width + x

                    tile_color = _sample_tile_color(tile_pixels, self.sample_mode, seed)

                    # Draw tile with gaps
                    _draw_tile_with_gaps(
                        canvas,
                        (x, y),
                        tile_color,
                        (self.tile_width, self.tile_height),
                        self.gap_size,
                    )

            # Convert back to PIL Image
            result_image = Image.fromarray(canvas, image.mode)

            # Update context
            updated_context = context.copy_with_updates(
                memory_estimate=self.estimate_memory(context)
            )

            return result_image, updated_context

        except Exception as e:
            raise ProcessingError(f"Mosaic failed: {e}") from e


class Dither(BaseOperation):
    """Apply various dithering patterns for artistic effects."""

    method: Literal["floyd_steinberg", "ordered", "random"] = Field(
        "floyd_steinberg", description="Dithering algorithm"
    )
    levels: int = Field(2, ge=2, le=256, description="Number of color levels per channel")
    pattern_size: int = Field(4, description="Size of ordered dithering matrix (power of 2)")
    random_seed: int | None = Field(None, description="Random seed for reproducible dithering")

    @field_validator("pattern_size")
    @classmethod
    def validate_pattern_size(cls, v):
        """Validate pattern size is power of 2."""
        if v <= 0 or (v & (v - 1)) != 0:
            raise ValueError("Pattern size must be power of 2")
        if v > 16:
            raise ValueError("Pattern size too large (max 16)")
        return v

    def validate_operation(self, context: ImageContext) -> ImageContext:
        """Validate operation against image context."""
        # Dithering doesn't change image dimensions
        return context

    def get_cache_key(self, image_hash: str) -> str:
        """Generate cache key for this operation."""
        config_str = f"{self.method}_{self.levels}_{self.pattern_size}_{self.random_seed}"
        return f"dither_{image_hash}_{hash(config_str)}"

    def estimate_memory(self, context: ImageContext) -> int:
        """Estimate memory usage for this operation."""
        base_memory = context.width * context.height * context.channels

        if self.method == "floyd_steinberg":
            # Need error buffer
            return base_memory * 2
        elif self.method == "ordered":
            # Need threshold matrix
            return base_memory + (self.pattern_size * self.pattern_size * 4)
        else:  # random
            return base_memory

    def apply(self, image: Image.Image, context: ImageContext) -> tuple[Image.Image, ImageContext]:
        """Apply dithering effect."""
        try:
            image_array = np.array(image).astype(np.float32)
            height, width = image_array.shape[:2]

            if self.method == "floyd_steinberg":
                result_array = self._apply_floyd_steinberg(image_array)
            elif self.method == "ordered":
                result_array = self._apply_ordered_dithering(image_array)
            elif self.method == "random":
                result_array = self._apply_random_dithering(image_array)
            else:
                raise ValidationError(f"Unknown dithering method: {self.method}")

            # Convert back to uint8 and PIL Image
            result_array = np.clip(result_array, 0, 255).astype(np.uint8)
            result_image = Image.fromarray(result_array, image.mode)

            # Update context
            updated_context = context.copy_with_updates(
                memory_estimate=self.estimate_memory(context)
            )

            return result_image, updated_context

        except Exception as e:
            raise ProcessingError(f"Dither failed: {e}") from e

    def _apply_floyd_steinberg(self, image_array: np.ndarray) -> np.ndarray:
        """Apply Floyd-Steinberg error diffusion dithering."""
        height, width = image_array.shape[:2]

        if len(image_array.shape) == 3:
            channels = image_array.shape[2]
            result = image_array.copy()

            for c in range(channels):
                channel = result[:, :, c].copy()

                for y in range(height):
                    for x in range(width):
                        old_pixel = channel[y, x]
                        new_pixel = _quantize_value(old_pixel, self.levels)
                        channel[y, x] = new_pixel

                        error = old_pixel - new_pixel

                        # Diffuse error to neighboring pixels
                        if x + 1 < width:
                            channel[y, x + 1] += error * (7.0 / 16.0)

                        if y + 1 < height:
                            if x - 1 >= 0:
                                channel[y + 1, x - 1] += error * (3.0 / 16.0)

                            channel[y + 1, x] += error * (5.0 / 16.0)

                            if x + 1 < width:
                                channel[y + 1, x + 1] += error * (1.0 / 16.0)

                result[:, :, c] = channel
        else:
            # Grayscale
            result = image_array.copy()

            for y in range(height):
                for x in range(width):
                    old_pixel = result[y, x]
                    new_pixel = _quantize_value(old_pixel, self.levels)
                    result[y, x] = new_pixel

                    error = old_pixel - new_pixel

                    # Diffuse error
                    if x + 1 < width:
                        result[y, x + 1] += error * (7.0 / 16.0)

                    if y + 1 < height:
                        if x - 1 >= 0:
                            result[y + 1, x - 1] += error * (3.0 / 16.0)

                        result[y + 1, x] += error * (5.0 / 16.0)

                        if x + 1 < width:
                            result[y + 1, x + 1] += error * (1.0 / 16.0)

        return result

    def _apply_ordered_dithering(self, image_array: np.ndarray) -> np.ndarray:
        """Apply ordered (Bayer matrix) dithering."""
        bayer_matrix = _generate_bayer_matrix(self.pattern_size)
        height, width = image_array.shape[:2]

        result = image_array.copy()

        if len(image_array.shape) == 3:
            channels = image_array.shape[2]

            for y in range(height):
                for x in range(width):
                    threshold = bayer_matrix[y % self.pattern_size, x % self.pattern_size]

                    for c in range(channels):
                        old_pixel = result[y, x, c]

                        # Add threshold offset
                        offset_pixel = old_pixel + (threshold - 0.5) * (255.0 / self.levels)
                        new_pixel = _quantize_value(offset_pixel, self.levels)

                        result[y, x, c] = new_pixel
        else:
            # Grayscale
            for y in range(height):
                for x in range(width):
                    threshold = bayer_matrix[y % self.pattern_size, x % self.pattern_size]
                    old_pixel = result[y, x]

                    # Add threshold offset
                    offset_pixel = old_pixel + (threshold - 0.5) * (255.0 / self.levels)
                    new_pixel = _quantize_value(offset_pixel, self.levels)

                    result[y, x] = new_pixel

        return result

    def _apply_random_dithering(self, image_array: np.ndarray) -> np.ndarray:
        """Apply random threshold dithering."""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        height, width = image_array.shape[:2]
        result = image_array.copy()

        # Generate random noise
        noise_strength = 255.0 / (self.levels * 2)

        if len(image_array.shape) == 3:
            channels = image_array.shape[2]
            noise = np.random.uniform(-noise_strength, noise_strength, (height, width, channels))

            for y in range(height):
                for x in range(width):
                    for c in range(channels):
                        old_pixel = result[y, x, c]
                        noisy_pixel = old_pixel + noise[y, x, c]
                        new_pixel = _quantize_value(noisy_pixel, self.levels)
                        result[y, x, c] = new_pixel
        else:
            # Grayscale
            noise = np.random.uniform(-noise_strength, noise_strength, (height, width))

            for y in range(height):
                for x in range(width):
                    old_pixel = result[y, x]
                    noisy_pixel = old_pixel + noise[y, x]
                    new_pixel = _quantize_value(noisy_pixel, self.levels)
                    result[y, x] = new_pixel

        return result
