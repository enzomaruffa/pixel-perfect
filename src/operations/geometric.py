"""Geometric transformation operations."""

import math
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import Field

from core.base import BaseOperation
from core.context import ImageContext
from exceptions import ProcessingError, ValidationError


def _nearest_neighbor(image_array: np.ndarray, x: float, y: float) -> np.ndarray:
    """Sample pixel using nearest neighbor interpolation.

    Args:
        image_array: Image data as numpy array (H, W, C)
        x: X coordinate (may be fractional)
        y: Y coordinate (may be fractional)

    Returns:
        Pixel value(s) as numpy array
    """
    height, width = image_array.shape[:2]

    # Round to nearest integer coordinates
    xi = int(round(x))
    yi = int(round(y))

    # Clamp to image bounds
    xi = max(0, min(width - 1, xi))
    yi = max(0, min(height - 1, yi))

    return image_array[yi, xi]


def _bilinear_interpolate(image_array: np.ndarray, x: float, y: float) -> np.ndarray:
    """Sample pixel using bilinear interpolation.

    Args:
        image_array: Image data as numpy array (H, W, C)
        x: X coordinate (may be fractional)
        y: Y coordinate (may be fractional)

    Returns:
        Interpolated pixel value(s) as numpy array
    """
    height, width = image_array.shape[:2]

    # Get integer coordinates and fractional parts
    x1 = int(math.floor(x))
    y1 = int(math.floor(y))
    x2 = x1 + 1
    y2 = y1 + 1

    # Fractional parts
    fx = x - x1
    fy = y - y1

    # Clamp coordinates to image bounds
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))

    # Get four corner pixels
    p11 = image_array[y1, x1].astype(np.float64)
    p21 = image_array[y1, x2].astype(np.float64)
    p12 = image_array[y2, x1].astype(np.float64)
    p22 = image_array[y2, x2].astype(np.float64)

    # Bilinear interpolation
    top = p11 * (1 - fx) + p21 * fx
    bottom = p12 * (1 - fx) + p22 * fx
    result = top * (1 - fy) + bottom * fy

    return result.astype(image_array.dtype)


def _sample_pixel_safe(
    image_array: np.ndarray, x: float, y: float, method: str = "bilinear"
) -> np.ndarray:
    """Sample pixel with bounds checking and specified interpolation method.

    Args:
        image_array: Image data as numpy array (H, W, C)
        x: X coordinate
        y: Y coordinate
        method: Interpolation method ("nearest", "bilinear")

    Returns:
        Pixel value(s) as numpy array
    """
    height, width = image_array.shape[:2]

    # Check if coordinates are out of bounds
    if x < 0 or x >= width or y < 0 or y >= height:
        # Return transparent black for out-of-bounds
        if len(image_array.shape) == 3:
            return np.zeros(image_array.shape[2], dtype=image_array.dtype)
        else:
            return np.array(0, dtype=image_array.dtype)

    if method == "nearest":
        return _nearest_neighbor(image_array, x, y)
    elif method == "bilinear":
        return _bilinear_interpolate(image_array, x, y)
    else:
        raise ValueError(f"Unsupported interpolation method: {method}")


class GridWarp(BaseOperation):
    """Apply wave-like distortions to create grid warp effects."""

    axis: Literal["horizontal", "vertical", "both"] = Field(
        "horizontal", description="Warp direction"
    )
    frequency: float = Field(1.0, gt=0, description="Wave frequency (cycles per image dimension)")
    amplitude: float = Field(10.0, description="Displacement amount in pixels")
    phase: float = Field(0.0, ge=0, lt=2 * math.pi, description="Wave phase offset (0-2π)")
    interpolation: Literal["nearest", "bilinear"] = Field(
        "bilinear", description="Interpolation method"
    )

    def validate_operation(self, context: ImageContext) -> ImageContext:
        """Validate parameters against image context."""
        # Check if amplitude is reasonable relative to image size
        max_dimension = max(context.width, context.height)
        if abs(self.amplitude) > max_dimension * 0.5:
            context.add_warning(
                f"Large amplitude {self.amplitude} may cause significant distortion"
            )

        return context

    def get_cache_key(self, image_hash: str) -> str:
        """Generate cache key for this operation."""
        config_str = (
            f"{self.axis}_{self.frequency}_{self.amplitude}_{self.phase}_{self.interpolation}"
        )
        return f"gridwarp_{image_hash}_{hash(config_str)}"

    def estimate_memory(self, context: ImageContext) -> int:
        """Estimate memory usage for this operation."""
        # Need coordinate arrays and output image
        coordinate_memory = context.width * context.height * 2 * 8  # x,y coordinates (float64)
        output_memory = context.width * context.height * context.channels
        return coordinate_memory + output_memory

    def apply(self, image: Image.Image, context: ImageContext) -> tuple[Image.Image, ImageContext]:
        """Apply grid warp transformation."""
        try:
            # Convert to numpy array
            image_array = np.array(image)
            height, width = image_array.shape[:2]

            # Create output array
            output_array = np.zeros_like(image_array)

            # Generate coordinate grids
            x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
            x_coords = x_coords.astype(np.float64)
            y_coords = y_coords.astype(np.float64)

            # Apply warp transformation based on axis
            if self.axis in ["horizontal", "both"]:
                # Horizontal warp: x displacement based on y position
                x_displacement = self.amplitude * np.sin(
                    2 * math.pi * self.frequency * y_coords / height + self.phase
                )
                x_coords = x_coords + x_displacement

            if self.axis in ["vertical", "both"]:
                # Vertical warp: y displacement based on x position
                y_displacement = self.amplitude * np.sin(
                    2 * math.pi * self.frequency * x_coords / width + self.phase
                )
                y_coords = y_coords + y_displacement

            # Sample pixels at warped coordinates
            for i in range(height):
                for j in range(width):
                    source_x = x_coords[i, j]
                    source_y = y_coords[i, j]

                    output_array[i, j] = _sample_pixel_safe(
                        image_array, source_x, source_y, self.interpolation
                    )

            # Convert back to PIL Image
            result_image = Image.fromarray(output_array, image.mode)

            # Update context with memory usage
            updated_context = context.copy_with_updates(
                memory_estimate=self.estimate_memory(context)
            )

            return result_image, updated_context

        except Exception as e:
            raise ProcessingError(f"GridWarp failed: {e}") from e


class PerspectiveStretch(BaseOperation):
    """Simulate perspective distortion with linear scale interpolation."""

    top_factor: float = Field(1.0, gt=0, description="Scale factor for top edge")
    bottom_factor: float = Field(0.5, gt=0, description="Scale factor for bottom edge")
    interpolation: Literal["nearest", "bilinear"] = Field(
        "bilinear", description="Interpolation method"
    )

    def validate_operation(self, context: ImageContext) -> ImageContext:
        """Validate parameters against image context."""
        # Check for extreme scale factors
        if self.top_factor > 3.0 or self.bottom_factor > 3.0:
            context.add_warning("Large scale factors may cause significant distortion")

        if self.top_factor < 0.1 or self.bottom_factor < 0.1:
            context.add_warning("Small scale factors may cause significant compression")

        # Calculate output dimensions
        max_factor = max(self.top_factor, self.bottom_factor)
        new_width = int(context.width * max_factor)

        return context.copy_with_updates(width=new_width)

    def get_cache_key(self, image_hash: str) -> str:
        """Generate cache key for this operation."""
        config_str = f"{self.top_factor}_{self.bottom_factor}_{self.interpolation}"
        return f"perspective_{image_hash}_{hash(config_str)}"

    def estimate_memory(self, context: ImageContext) -> int:
        """Estimate memory usage for this operation."""
        # Calculate output dimensions
        max_factor = max(self.top_factor, self.bottom_factor)
        output_width = int(context.width * max_factor)
        output_height = context.height

        return output_width * output_height * context.channels

    def apply(self, image: Image.Image, context: ImageContext) -> tuple[Image.Image, ImageContext]:
        """Apply perspective stretch transformation."""
        try:
            # Convert to numpy array
            image_array = np.array(image)
            height, width = image_array.shape[:2]

            # Calculate output dimensions
            max_factor = max(self.top_factor, self.bottom_factor)
            output_width = int(width * max_factor)
            output_height = height

            # Create output array
            if len(image_array.shape) == 3:
                output_array = np.zeros(
                    (output_height, output_width, image_array.shape[2]), dtype=image_array.dtype
                )
            else:
                output_array = np.zeros((output_height, output_width), dtype=image_array.dtype)

            # Apply perspective transformation
            for i in range(output_height):
                # Linear interpolation of scale factor from top to bottom
                t = i / (height - 1) if height > 1 else 0
                current_scale = self.top_factor + t * (self.bottom_factor - self.top_factor)

                # Calculate source and destination widths for this row
                source_width = int(width * current_scale)
                center_offset = (output_width - source_width) // 2

                for j in range(source_width):
                    # Map output coordinate to source coordinate
                    source_x = j / current_scale if current_scale > 0 else 0
                    source_y = i

                    output_x = center_offset + j
                    if 0 <= output_x < output_width:
                        output_array[i, output_x] = _sample_pixel_safe(
                            image_array, source_x, source_y, self.interpolation
                        )

            # Convert back to PIL Image
            result_image = Image.fromarray(output_array, image.mode)

            # Update context
            updated_context = context.copy_with_updates(
                width=output_width,
                height=output_height,
                memory_estimate=self.estimate_memory(context),
            )

            return result_image, updated_context

        except Exception as e:
            raise ProcessingError(f"PerspectiveStretch failed: {e}") from e


class RadialStretch(BaseOperation):
    """Stretch image radially from center point outward."""

    center: tuple[int, int] | Literal["auto"] = Field(
        "auto", description="Center point for radial stretch"
    )
    factor: float = Field(1.5, gt=0, description="Stretch factor (>1 expand, <1 contract)")
    falloff: Literal["linear", "quadratic", "exponential"] = Field(
        "linear", description="Falloff function for stretch effect"
    )
    interpolation: Literal["nearest", "bilinear"] = Field(
        "bilinear", description="Interpolation method"
    )

    def validate_operation(self, context: ImageContext) -> ImageContext:
        """Validate parameters against image context."""
        # Check center coordinates if specified
        if isinstance(self.center, tuple):
            cx, cy = self.center
            if cx < 0 or cx >= context.width or cy < 0 or cy >= context.height:
                raise ValidationError(
                    f"Center point ({cx}, {cy}) is outside image bounds "
                    f"({context.width}x{context.height})"
                )

        # Check for extreme stretch factors
        if self.factor > 3.0:
            context.add_warning("Large stretch factor may cause significant expansion")
        elif self.factor < 0.3:
            context.add_warning("Small stretch factor may cause significant contraction")

        # Calculate output dimensions
        if self.factor > 1:
            new_width = int(context.width * self.factor)
            new_height = int(context.height * self.factor)
        else:
            new_width = context.width
            new_height = context.height

        return context.copy_with_updates(width=new_width, height=new_height)

    def get_cache_key(self, image_hash: str) -> str:
        """Generate cache key for this operation."""
        config_str = f"{self.center}_{self.factor}_{self.falloff}_{self.interpolation}"
        return f"radial_{image_hash}_{hash(config_str)}"

    def estimate_memory(self, context: ImageContext) -> int:
        """Estimate memory usage for this operation."""
        # Estimate output size based on stretch factor
        if self.factor > 1:
            # Expansion: output may be larger
            scale = self.factor
            output_width = int(context.width * scale)
            output_height = int(context.height * scale)
        else:
            # Contraction: output same size as input
            output_width = context.width
            output_height = context.height

        return output_width * output_height * context.channels

    def apply(self, image: Image.Image, context: ImageContext) -> tuple[Image.Image, ImageContext]:
        """Apply radial stretch transformation."""
        try:
            # Convert to numpy array
            image_array = np.array(image)
            height, width = image_array.shape[:2]

            # Determine center point
            if self.center == "auto":
                cx, cy = width / 2, height / 2
            else:
                cx, cy = self.center

            # Calculate maximum radius for normalization
            max_radius = math.sqrt(
                max(
                    cx**2 + cy**2,
                    (width - cx) ** 2 + cy**2,
                    cx**2 + (height - cy) ** 2,
                    (width - cx) ** 2 + (height - cy) ** 2,
                )
            )

            # For expansion, create larger output image
            if self.factor > 1:
                output_width = int(width * self.factor)
                output_height = int(height * self.factor)
                output_cx = output_width / 2
                output_cy = output_height / 2
            else:
                output_width = width
                output_height = height
                output_cx = cx
                output_cy = cy

            # Create output array
            if len(image_array.shape) == 3:
                output_array = np.zeros(
                    (output_height, output_width, image_array.shape[2]), dtype=image_array.dtype
                )
            else:
                output_array = np.zeros((output_height, output_width), dtype=image_array.dtype)

            # Apply radial transformation
            for i in range(output_height):
                for j in range(output_width):
                    # Calculate distance from center in output space
                    dx = j - output_cx
                    dy = i - output_cy
                    distance = math.sqrt(dx**2 + dy**2)

                    if distance == 0:
                        # Center pixel maps to itself
                        source_x = cx
                        source_y = cy
                    else:
                        # Calculate normalized distance for falloff
                        normalized_distance = distance / max_radius if max_radius > 0 else 0

                        # Apply falloff function
                        if self.falloff == "linear":
                            falloff_factor = 1.0
                        elif self.falloff == "quadratic":
                            falloff_factor = normalized_distance
                        else:  # exponential
                            falloff_factor = math.exp(-2 * normalized_distance)

                        # Calculate inverse transformation
                        scale = 1 / (self.factor * falloff_factor) if falloff_factor > 0 else 1

                        # Map to source coordinates
                        source_distance = distance * scale
                        angle = math.atan2(dy, dx)
                        source_x = cx + source_distance * math.cos(angle)
                        source_y = cy + source_distance * math.sin(angle)

                    # Sample source pixel
                    output_array[i, j] = _sample_pixel_safe(
                        image_array, source_x, source_y, self.interpolation
                    )

            # Convert back to PIL Image
            result_image = Image.fromarray(output_array, image.mode)

            # Update context
            updated_context = context.copy_with_updates(
                width=output_width,
                height=output_height,
                memory_estimate=self.estimate_memory(context),
            )

            return result_image, updated_context

        except Exception as e:
            raise ProcessingError(f"RadialStretch failed: {e}") from e
