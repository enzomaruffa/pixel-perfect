"""Pixel-level image processing operations."""

from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, field_validator, model_validator

from core.base import BaseOperation
from core.context import ImageContext
from exceptions import ProcessingError, ValidationError
from operations._constants import get_standardized_description, get_standardized_error
from utils.validation import validate_channel_list, validate_color_tuple, validate_expression_safe


class PixelFilterConfig(BaseModel):
    """Configuration for PixelFilter operation."""

    condition: Literal["prime", "odd", "even", "fibonacci", "custom"] = Field(
        "prime",
        description="Mathematical condition for filtering pixels (prime numbers, odd/even indices, fibonacci sequence, or custom expression using variables 'x', 'y', 'i')",
    )
    custom_expression: str | None = Field(
        None,
        description="Custom expression using variables: 'x' (column), 'y' (row), 'i' (linear index), 'width', 'height'. Example: 'x + y > 10'",
    )
    fill_color: tuple[int, int, int, int] = Field(
        (0, 0, 0, 0), description=get_standardized_description("fill_color", "color")
    )
    preserve_alpha: bool = Field(
        True, description=get_standardized_description("preserve_alpha", "boolean")
    )
    index_mode: Literal["linear", "2d"] = Field(
        "linear",
        description="Index calculation mode: 'linear' uses sequential pixel index, '2d' uses row/column coordinates",
    )

    @field_validator("custom_expression")
    @classmethod
    def validate_custom_expression(cls, v: str | None, values) -> str | None:
        """Validate custom expression if provided."""
        # In Pydantic v2, we need to use model_validator for cross-field validation
        if v is not None:
            validate_expression_safe(v)
        return v

    @model_validator(mode="after")
    def validate_custom_expression_required(self) -> "PixelFilterConfig":
        """Validate model after all fields are set."""
        if self.condition == "custom" and not self.custom_expression:
            raise ValidationError(
                get_standardized_error(
                    "required_param", param="custom_expression", condition="condition='custom'"
                )
            )
        elif self.condition != "custom" and self.custom_expression is not None:
            raise ValidationError(
                "custom_expression should only be provided when condition='custom'"
            )

        return self

    @field_validator("fill_color")
    @classmethod
    def validate_fill_color(cls, v: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """Validate fill color values."""
        validated = validate_color_tuple(v, channels=4)
        # Type assert since we know validate_color_tuple with channels=4 returns 4-tuple
        return validated  # type: ignore[return-value]


class PixelMathConfig(BaseModel):
    """Configuration for PixelMath operation."""

    expression: str = Field(
        ...,
        description="Mathematical expression using variables: 'r', 'g', 'b', 'a' (pixel channels), 'x' (column), 'y' (row), 'width', 'height'. Example: 'r * 0.8 + g * 0.2'",
    )
    channels: list[str] = Field(
        ["r", "g", "b"],
        description="Color channels to apply the expression to (subset of ['r', 'g', 'b', 'a'])",
    )
    clamp: bool = Field(
        True,
        description="Whether to clamp result values to valid range [0-255] (recommended to prevent overflow)",
    )

    @field_validator("expression")
    @classmethod
    def validate_expression(cls, v: str) -> str:
        """Validate mathematical expression."""
        validate_expression_safe(v)
        return v

    @field_validator("channels")
    @classmethod
    def validate_channels(cls, v: list[str]) -> list[str]:
        """Validate channel list."""
        return validate_channel_list(v)


class PixelSortConfig(BaseModel):
    """Configuration for PixelSort operation."""

    direction: Literal["horizontal", "vertical", "diagonal"] = "horizontal"
    sort_by: Literal["brightness", "hue", "saturation", "red", "green", "blue"] = "brightness"
    threshold: float | None = Field(None, description="Only sort pixels meeting threshold")
    reverse: bool = Field(False, description="Reverse sort order")

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: float | None) -> float | None:
        """Validate threshold value."""
        if v is not None and not (0 <= v <= 255):
            raise ValueError("Threshold must be between 0 and 255")
        return v


class PixelFilter(BaseOperation):
    """Filter pixels based on index conditions."""

    condition: Literal["prime", "odd", "even", "fibonacci", "custom"] = Field(
        "prime",
        description="Mathematical condition for filtering pixels (prime numbers, odd/even indices, fibonacci sequence, or custom expression using variables 'x', 'y', 'i')",
    )
    custom_expression: str | None = Field(None, description="Custom expression using 'i' for index")
    fill_color: tuple[int, int, int, int] = Field(
        (0, 0, 0, 0), description="RGBA color values (red, green, blue, alpha) in range [0-255]"
    )
    preserve_alpha: bool = Field(True, description="Keep original alpha channel")
    index_mode: Literal["linear", "2d"] = Field("linear", description="Index calculation mode")

    @field_validator("custom_expression")
    @classmethod
    def validate_custom_expression(cls, v: str | None, values) -> str | None:
        """Validate custom expression if provided."""
        if v is not None:
            validate_expression_safe(v)
        return v

    @model_validator(mode="after")
    def validate_custom_expression_required(self) -> "PixelFilter":
        """Validate model after all fields are set."""
        if self.condition == "custom" and not self.custom_expression:
            raise ValueError("custom_expression required when condition='custom'")
        elif self.condition != "custom" and self.custom_expression is not None:
            raise ValueError("custom_expression only valid when condition='custom'")
        return self

    @field_validator("fill_color")
    @classmethod
    def validate_fill_color(cls, v: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """Validate fill color values."""
        validated = validate_color_tuple(v, channels=4)
        # Type assert since we know validate_color_tuple with channels=4 returns 4-tuple
        return validated  # type: ignore[return-value]

    def validate_operation(self, context: ImageContext) -> ImageContext:
        """Validate operation against image context."""
        # PixelFilter preserves dimensions
        return context.copy_with_updates()

    def get_cache_key(self, image_hash: str) -> str:
        """Generate cache key for this operation."""
        config_str = f"{self.condition}_{self.custom_expression}_{self.fill_color}"
        config_str += f"_{self.preserve_alpha}_{self.index_mode}"
        return f"pixelfilter_{image_hash}_{hash(config_str)}"

    def estimate_memory(self, context: ImageContext) -> int:
        """Estimate memory usage for this operation."""
        # PixelFilter creates one copy of the image
        return context.memory_estimate * 2

    def apply(self, image: Image.Image, context: ImageContext) -> tuple[Image.Image, ImageContext]:
        """Apply pixel filter to image."""
        try:
            # Convert to RGBA for consistent processing
            rgba_image = image.convert("RGBA")
            pixels = np.array(rgba_image)
            height, width = pixels.shape[:2]

            # Create mask of pixels to keep
            mask = self._create_mask(width, height)

            # Apply filter
            filtered_pixels = pixels.copy()
            fill_r, fill_g, fill_b, fill_a = self.fill_color

            # Apply mask - set filtered pixels to fill color
            filtered_pixels[~mask] = [fill_r, fill_g, fill_b, fill_a]

            # Preserve alpha only affects how we handle the alpha of filtered pixels
            # If preserve_alpha is True, keep original alpha for non-filtered pixels
            # The test expects fill_color alpha to be applied to filtered pixels

            # Create result image
            result_image = Image.fromarray(filtered_pixels)
            if result_image.mode != "RGBA":
                result_image = result_image.convert("RGBA")

            # Only convert back to original mode if fill_color doesn't use transparency
            # If fill_color has alpha < 255, we need to keep RGBA mode
            if image.mode != "RGBA" and self.fill_color[3] == 255:
                result_image = result_image.convert(image.mode)

            return result_image, context.copy_with_updates()

        except Exception as e:
            raise ProcessingError(f"PixelFilter failed: {e}") from e

    def _create_mask(self, width: int, height: int) -> np.ndarray:
        """Create boolean mask for pixels to keep."""
        if self.index_mode == "linear":
            return self._create_linear_mask(width, height)
        else:  # 2d mode
            return self._create_2d_mask(width, height)

    def _create_linear_mask(self, width: int, height: int) -> np.ndarray:
        """Create mask using linear indexing."""
        total_pixels = width * height
        indices = np.arange(total_pixels)

        if self.condition == "prime":
            mask_1d = self._is_prime_vectorized(indices)
        elif self.condition == "odd":
            mask_1d = indices % 2 == 1
        elif self.condition == "even":
            mask_1d = indices % 2 == 0
        elif self.condition == "fibonacci":
            mask_1d = self._is_fibonacci_vectorized(indices)
        elif self.condition == "custom":
            mask_1d = self._evaluate_custom_expression(indices)
        else:
            raise ProcessingError(f"Unknown condition: {self.condition}")

        return mask_1d.reshape(height, width)

    def _create_2d_mask(self, width: int, height: int) -> np.ndarray:
        """Create mask using 2D indexing (separate row/col conditions)."""
        # For 2D mode, apply condition to both row and column indices
        row_indices = np.arange(height)[:, np.newaxis]
        col_indices = np.arange(width)[np.newaxis, :]

        if self.condition == "prime":
            row_mask = self._is_prime_vectorized(row_indices)
            col_mask = self._is_prime_vectorized(col_indices)
            return row_mask & col_mask
        elif self.condition == "odd":
            return (row_indices % 2 == 1) & (col_indices % 2 == 1)
        elif self.condition == "even":
            return (row_indices % 2 == 0) & (col_indices % 2 == 0)
        else:
            # For other conditions, fall back to linear mode
            return self._create_linear_mask(width, height)

    def _is_prime_vectorized(self, numbers: np.ndarray) -> np.ndarray:
        """Vectorized prime number check using sieve for efficiency."""
        result = np.zeros_like(numbers, dtype=bool)

        # Handle edge cases
        if numbers.size == 0:
            return result

        # Get unique values to avoid redundant checks
        unique_nums = np.unique(numbers.flatten())
        unique_nums = unique_nums[unique_nums > 0]  # Only positive numbers

        if len(unique_nums) == 0:
            return result

        max_val = int(np.max(unique_nums))

        if max_val < 2:
            return result

        # Use sieve of Eratosthenes for efficiency
        sieve = np.ones(max_val + 1, dtype=bool)
        sieve[0:2] = False  # 0 and 1 are not prime

        # Sieve algorithm - much faster than trial division
        for i in range(2, int(max_val**0.5) + 1):
            if sieve[i]:
                sieve[i * i : max_val + 1 : i] = False

        # Map back to original array
        flat_numbers = numbers.flatten().astype(int)
        valid_mask = (flat_numbers >= 0) & (flat_numbers <= max_val)
        flat_result = np.zeros_like(flat_numbers, dtype=bool)
        flat_result[valid_mask] = sieve[flat_numbers[valid_mask]]

        return flat_result.reshape(numbers.shape)

    def _is_fibonacci_vectorized(self, numbers: np.ndarray) -> np.ndarray:
        """Vectorized Fibonacci number check."""
        # Generate Fibonacci numbers up to the maximum value
        max_val = np.max(numbers) if len(numbers) > 0 else 0
        fib_set = set()
        a, b = 0, 1
        while a <= max_val:
            fib_set.add(a)
            a, b = b, a + b

        # Create mask
        return np.array([num in fib_set for num in numbers.flat]).reshape(numbers.shape)

    def _evaluate_custom_expression(self, indices: np.ndarray) -> np.ndarray:
        """Evaluate custom expression with safety checks."""
        try:
            # Create safe evaluation environment
            safe_dict = {
                "__builtins__": {},
                "i": indices,
                # Add safe math functions if needed
            }

            # Evaluate expression
            if self.custom_expression is None:
                raise ValidationError("Parameter is required for the selected configuration")
            result = eval(self.custom_expression, safe_dict)
            return np.array(result, dtype=bool)

        except Exception as e:
            raise ProcessingError(f"Custom expression evaluation failed: {e}") from e


class PixelMath(BaseOperation):
    """Apply mathematical transformations to pixel values."""

    expression: str = Field(..., description="Math expression using r,g,b,a,x,y variables")
    channels: list[str] = Field(["r", "g", "b"], description="Channels to affect")
    clamp: bool = Field(True, description="Clamp results to valid range [0, 255]")

    @field_validator("expression")
    @classmethod
    def validate_expression(cls, v: str) -> str:
        """Validate mathematical expression."""
        validate_expression_safe(v)
        return v

    @field_validator("channels")
    @classmethod
    def validate_channels(cls, v: list[str]) -> list[str]:
        """Validate channel list."""
        return validate_channel_list(v)

    def validate_operation(self, context: ImageContext) -> ImageContext:
        """Validate operation against image context."""
        # PixelMath preserves dimensions
        return context.copy_with_updates()

    def get_cache_key(self, image_hash: str) -> str:
        """Generate cache key for this operation."""
        config_str = f"{self.expression}_{self.channels}_{self.clamp}"
        return f"pixelmath_{image_hash}_{hash(config_str)}"

    def estimate_memory(self, context: ImageContext) -> int:
        """Estimate memory usage for this operation."""
        # PixelMath creates one copy plus working arrays
        return context.memory_estimate * 3

    def apply(self, image: Image.Image, context: ImageContext) -> tuple[Image.Image, ImageContext]:
        """Apply mathematical transformation to pixels."""
        try:
            # Convert to RGBA for consistent processing
            rgba_image = image.convert("RGBA")
            pixels = np.array(rgba_image, dtype=np.float32)
            height, width = pixels.shape[:2]

            # Create coordinate arrays
            y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

            # Extract channel arrays
            r, g, b, a = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2], pixels[:, :, 3]

            # Create safe evaluation environment
            safe_dict = {
                "__builtins__": {},
                "r": r,
                "g": g,
                "b": b,
                "a": a,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                # Add safe math functions
                "abs": np.abs,
                "min": np.minimum,
                "max": np.maximum,
                "sin": np.sin,
                "cos": np.cos,
                "tan": np.tan,
                "sqrt": np.sqrt,
                "exp": np.exp,
                "log": np.log,
            }

            # Evaluate expression
            result = eval(self.expression, safe_dict)

            # Apply to specified channels
            for channel in self.channels:
                if channel == "r":
                    pixels[:, :, 0] = result
                elif channel == "g":
                    pixels[:, :, 1] = result
                elif channel == "b":
                    pixels[:, :, 2] = result
                elif channel == "a":
                    pixels[:, :, 3] = result

            # Clamp values if requested
            if self.clamp:
                pixels = np.clip(pixels, 0, 255)

            # Convert back to uint8
            pixels = pixels.astype(np.uint8)
            result_image = Image.fromarray(pixels)
            if result_image.mode != "RGBA":
                result_image = result_image.convert("RGBA")

            # Convert back to original mode if needed
            if image.mode != "RGBA":
                result_image = result_image.convert(image.mode)

            return result_image, context.copy_with_updates()

        except Exception as e:
            raise ProcessingError(f"PixelMath failed: {e}") from e


class PixelSort(BaseOperation):
    """Sort pixels within regions based on criteria."""

    direction: Literal["horizontal", "vertical", "diagonal"] = "horizontal"
    sort_by: Literal["brightness", "hue", "saturation", "red", "green", "blue"] = "brightness"
    threshold: float | None = Field(None, description="Only sort pixels meeting threshold")
    reverse: bool = Field(False, description="Reverse sort order")

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: float | None) -> float | None:
        """Validate threshold value."""
        if v is not None and not (0 <= v <= 255):
            raise ValueError("Threshold must be between 0 and 255")
        return v

    def validate_operation(self, context: ImageContext) -> ImageContext:
        """Validate operation against image context."""
        # PixelSort preserves dimensions
        return context.copy_with_updates()

    def get_cache_key(self, image_hash: str) -> str:
        """Generate cache key for this operation."""
        config_str = f"{self.direction}_{self.sort_by}_{self.threshold}_{self.reverse}"
        return f"pixelsort_{image_hash}_{hash(config_str)}"

    def estimate_memory(self, context: ImageContext) -> int:
        """Estimate memory usage for this operation."""
        # PixelSort creates copies for sorting
        return context.memory_estimate * 3

    def apply(self, image: Image.Image, context: ImageContext) -> tuple[Image.Image, ImageContext]:
        """Apply pixel sorting to image."""
        try:
            # Convert to RGBA for consistent processing
            rgba_image = image.convert("RGBA")
            pixels = np.array(rgba_image)

            if self.direction == "horizontal":
                pixels = self._sort_horizontal(pixels)
            elif self.direction == "vertical":
                pixels = self._sort_vertical(pixels)
            elif self.direction == "diagonal":
                pixels = self._sort_diagonal(pixels)

            result_image = Image.fromarray(pixels)
            if result_image.mode != "RGBA":
                result_image = result_image.convert("RGBA")

            # Convert back to original mode if needed
            if image.mode != "RGBA":
                result_image = result_image.convert(image.mode)

            return result_image, context.copy_with_updates()

        except Exception as e:
            raise ProcessingError(f"PixelSort failed: {e}") from e

    def _sort_horizontal(self, pixels: np.ndarray) -> np.ndarray:
        """Sort pixels horizontally (within each row)."""
        result = pixels.copy()
        height = pixels.shape[0]

        for row in range(height):
            row_pixels = pixels[row]
            sort_values = self._get_sort_values(row_pixels)

            # Apply threshold if specified
            if self.threshold is not None:
                mask = sort_values >= self.threshold
                if np.any(mask):
                    indices = np.argsort(sort_values[mask])
                    if self.reverse:
                        indices = indices[::-1]
                    result[row][mask] = row_pixels[mask][indices]
            else:
                indices = np.argsort(sort_values)
                if self.reverse:
                    indices = indices[::-1]
                result[row] = row_pixels[indices]

        return result

    def _sort_vertical(self, pixels: np.ndarray) -> np.ndarray:
        """Sort pixels vertically (within each column)."""
        result = pixels.copy()
        width = pixels.shape[1]

        for col in range(width):
            col_pixels = pixels[:, col]
            sort_values = self._get_sort_values(col_pixels)

            # Apply threshold if specified
            if self.threshold is not None:
                mask = sort_values >= self.threshold
                if np.any(mask):
                    indices = np.argsort(sort_values[mask])
                    if self.reverse:
                        indices = indices[::-1]
                    result[:, col][mask] = col_pixels[mask][indices]
            else:
                indices = np.argsort(sort_values)
                if self.reverse:
                    indices = indices[::-1]
                result[:, col] = col_pixels[indices]

        return result

    def _sort_diagonal(self, pixels: np.ndarray) -> np.ndarray:
        """Sort pixels diagonally."""
        # For simplicity, sort along main diagonal
        height, width = pixels.shape[:2]
        result = pixels.copy()

        # Main diagonal (top-left to bottom-right)
        diagonal_length = min(height, width)
        diagonal_pixels = []

        for i in range(diagonal_length):
            diagonal_pixels.append(pixels[i, i])

        diagonal_pixels = np.array(diagonal_pixels)
        sort_values = self._get_sort_values(diagonal_pixels)

        indices = np.argsort(sort_values)
        if self.reverse:
            indices = indices[::-1]

        sorted_diagonal = diagonal_pixels[indices]

        for i, pixel in enumerate(sorted_diagonal):
            result[i, i] = pixel

        return result

    def _get_sort_values(self, pixels: np.ndarray) -> np.ndarray:
        """Get values to sort by based on sort criteria."""
        if self.sort_by == "brightness":
            # Calculate brightness as weighted average
            return 0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
        elif self.sort_by == "red":
            return pixels[:, 0]
        elif self.sort_by == "green":
            return pixels[:, 1]
        elif self.sort_by == "blue":
            return pixels[:, 2]
        elif self.sort_by == "hue":
            # Vectorized RGB to HSV conversion for hue
            rgb_normalized = pixels[:, :3] / 255.0
            v_max = np.max(rgb_normalized, axis=1)
            v_min = np.min(rgb_normalized, axis=1)
            delta = v_max - v_min

            # Calculate hue (vectorized)
            hue = np.zeros(len(pixels))

            # Where delta is 0, hue is undefined (set to 0)
            non_zero_delta = delta != 0

            # Red is max
            red_max = (v_max == rgb_normalized[:, 0]) & non_zero_delta
            hue[red_max] = (
                (rgb_normalized[red_max, 1] - rgb_normalized[red_max, 2]) / delta[red_max]
            ) % 6

            # Green is max
            green_max = (v_max == rgb_normalized[:, 1]) & non_zero_delta
            hue[green_max] = (
                2.0
                + (rgb_normalized[green_max, 2] - rgb_normalized[green_max, 0]) / delta[green_max]
            )

            # Blue is max
            blue_max = (v_max == rgb_normalized[:, 2]) & non_zero_delta
            hue[blue_max] = (
                4.0 + (rgb_normalized[blue_max, 0] - rgb_normalized[blue_max, 1]) / delta[blue_max]
            )

            return hue / 6.0  # Normalize to [0, 1]

        elif self.sort_by == "saturation":
            # Vectorized RGB to HSV conversion for saturation
            rgb_normalized = pixels[:, :3] / 255.0
            v_max = np.max(rgb_normalized, axis=1)
            v_min = np.min(rgb_normalized, axis=1)
            delta = v_max - v_min

            # Calculate saturation (vectorized)
            saturation = np.zeros(len(pixels))
            non_zero_max = v_max != 0
            saturation[non_zero_max] = delta[non_zero_max] / v_max[non_zero_max]

            return saturation
        else:
            raise ProcessingError(f"Unknown sort criteria: {self.sort_by}")
