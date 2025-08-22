"""Row-based image processing operations."""

import random
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import Field, field_validator, model_validator

from core.base import BaseOperation
from core.context import ImageContext
from exceptions import ProcessingError, ValidationError
from utils.validation import validate_color_tuple


def _select_rows(selection: str, height: int, **kwargs) -> np.ndarray:
    """Select row indices based on selection criteria.

    Args:
        selection: Selection method ("odd", "even", "prime", "every_n", "custom", "gradient")
        height: Total number of rows in image
        **kwargs: Additional parameters (n, indices, etc.)

    Returns:
        Array of selected row indices
    """
    if selection == "odd":
        return np.arange(1, height, 2)
    elif selection == "even":
        return np.arange(0, height, 2)
    elif selection == "prime":
        return _get_prime_indices(height)
    elif selection == "every_n":
        n = kwargs.get("n", 1)
        if n <= 0:
            raise ValidationError("Parameter 'n' must be positive")
        return np.arange(0, height, n)
    elif selection == "custom":
        indices = kwargs.get("indices", [])
        if not indices:
            raise ValidationError("Custom selection requires 'indices' parameter")
        indices_array = np.array(indices)
        if np.any(indices_array < 0) or np.any(indices_array >= height):
            raise ValidationError(f"Row indices must be in range [0, {height})")
        return indices_array
    elif selection == "gradient":
        # For gradient mode, return all rows (shift calculation happens elsewhere)
        return np.arange(height)
    else:
        raise ValidationError(f"Unknown selection method: {selection}")


def _get_prime_indices(max_value: int) -> np.ndarray:
    """Get prime numbers up to max_value (exclusive)."""
    if max_value <= 2:
        return np.array([])

    # Sieve of Eratosthenes
    is_prime = np.ones(max_value, dtype=bool)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(max_value**0.5) + 1):
        if is_prime[i]:
            is_prime[i * i : max_value : i] = False

    return np.where(is_prime)[0]


class RowShift(BaseOperation):
    """Translate entire rows horizontally."""

    selection: Literal["odd", "even", "prime", "every_n", "custom", "gradient"] = "odd"
    n: int | None = Field(None, ge=1, description="For every_n selection")
    indices: list[int] | None = Field(None, description="For custom selection")
    shift_amount: int = Field(0, description="Pixels to shift (negative=left, positive=right)")
    wrap: bool = Field(True, description="Wrap around vs fill with color")
    fill_color: tuple[int, int, int, int] = Field((0, 0, 0, 0), description="RGBA fill color")
    gradient_start: int = Field(0, description="Starting shift for gradient mode")

    @field_validator("fill_color")
    @classmethod
    def validate_fill_color(cls, v):
        return validate_color_tuple(v)

    @field_validator("indices")
    @classmethod
    def validate_indices(cls, v):
        if v is not None and len(v) == 0:
            raise ValidationError("Custom indices cannot be empty")
        return v

    @model_validator(mode="after")
    def validate_selection_parameters(self) -> "RowShift":
        """Validate that required parameters are provided for each selection type."""
        if self.selection == "every_n" and self.n is None:
            raise ValueError("Parameter 'n' is required when selection='every_n'")
        elif self.selection == "custom" and self.indices is None:
            raise ValueError("Parameter 'indices' is required when selection='custom'")
        return self

    def validate_operation(self, context: ImageContext) -> ImageContext:
        """Validate operation against image context."""
        # Validate row selection
        _select_rows(self.selection, context.height, n=self.n, indices=self.indices)
        return context.copy_with_updates()

    def get_cache_key(self, image_hash: str) -> str:
        """Generate cache key for this operation."""
        config_str = f"{self.selection}_{self.n}_{self.indices}_{self.shift_amount}"
        config_str += f"_{self.wrap}_{self.fill_color}_{self.gradient_start}"
        return f"rowshift_{image_hash}_{hash(config_str)}"

    def estimate_memory(self, context: ImageContext) -> int:
        """Estimate memory usage in bytes."""
        return context.memory_estimate * 2  # Input + output

    def apply(self, image: Image.Image, context: ImageContext) -> tuple[Image.Image, ImageContext]:
        """Apply row shift to image."""
        try:
            # Convert to RGBA for consistent processing
            rgba_image = image.convert("RGBA")
            pixels = np.array(rgba_image)
            height, width = pixels.shape[:2]

            # Get rows to shift
            if self.selection == "gradient":
                selected_rows = np.arange(height)
            else:
                selected_rows = _select_rows(self.selection, height, n=self.n, indices=self.indices)

            # Create output array
            result_pixels = pixels.copy()

            for row_idx in selected_rows:
                if self.selection == "gradient":
                    # Calculate shift amount based on position in gradient
                    shift = int(
                        self.gradient_start
                        + (self.shift_amount - self.gradient_start) * row_idx / (height - 1)
                    )
                else:
                    shift = self.shift_amount

                if shift == 0:
                    continue

                row = pixels[row_idx]

                if self.wrap:
                    # Wrap around
                    shift = shift % width  # Handle shifts larger than width
                    if shift > 0:
                        result_pixels[row_idx] = np.concatenate([row[-shift:], row[:-shift]])
                    else:
                        shift = abs(shift)
                        result_pixels[row_idx] = np.concatenate([row[shift:], row[:shift]])
                else:
                    # Fill with color
                    shifted_row = np.full_like(row, self.fill_color)
                    if shift > 0:
                        # Shift right
                        if shift < width:
                            shifted_row[shift:] = row[:-shift]
                    else:
                        # Shift left
                        shift = abs(shift)
                        if shift < width:
                            shifted_row[:-shift] = row[shift:]
                    result_pixels[row_idx] = shifted_row

            # Create result image
            result_image = Image.fromarray(result_pixels)
            if result_image.mode != "RGBA":
                result_image = result_image.convert("RGBA")

            # Convert back to original mode if fill_color is opaque
            if image.mode != "RGBA" and self.fill_color[3] == 255:
                result_image = result_image.convert(image.mode)

            return result_image, context.copy_with_updates()

        except Exception as e:
            raise ProcessingError(f"RowShift failed: {e}") from e


class RowStretch(BaseOperation):
    """Duplicate rows to stretch image vertically."""

    factor: float = Field(2.0, gt=0, description="Stretch multiplier")
    method: Literal["duplicate", "distribute"] = "duplicate"
    selection: Literal["all", "odd", "even", "prime", "every_n", "custom"] = "all"
    n: int | None = Field(None, ge=1, description="For every_n selection")
    indices: list[int] | None = Field(None, description="For custom selection")

    @field_validator("indices")
    @classmethod
    def validate_indices(cls, v):
        if v is not None and len(v) == 0:
            raise ValidationError("Custom indices cannot be empty")
        return v

    @model_validator(mode="after")
    def validate_selection_parameters(self) -> "RowStretch":
        """Validate that required parameters are provided for each selection type."""
        if self.selection == "every_n" and self.n is None:
            raise ValueError("Parameter 'n' is required when selection='every_n'")
        elif self.selection == "custom" and self.indices is None:
            raise ValueError("Parameter 'indices' is required when selection='custom'")
        return self

    def validate_operation(self, context: ImageContext) -> ImageContext:
        """Validate operation against image context."""
        if self.selection != "all":
            _select_rows(self.selection, context.height, n=self.n, indices=self.indices)

        new_height = int(context.height * self.factor)
        return context.copy_with_updates(height=new_height)

    def get_cache_key(self, image_hash: str) -> str:
        """Generate cache key for this operation."""
        config_str = f"{self.factor}_{self.method}_{self.selection}_{self.n}_{self.indices}"
        return f"rowstretch_{image_hash}_{hash(config_str)}"

    def estimate_memory(self, context: ImageContext) -> int:
        """Estimate memory usage in bytes."""
        new_height = int(context.height * self.factor)
        bytes_per_pixel = 1 if context.dtype == "uint8" else 4
        return context.width * new_height * context.channels * bytes_per_pixel * 2

    def apply(self, image: Image.Image, context: ImageContext) -> tuple[Image.Image, ImageContext]:
        """Apply row stretch to image."""
        try:
            # Convert to RGBA for consistent processing
            rgba_image = image.convert("RGBA")
            pixels = np.array(rgba_image)
            height, width = pixels.shape[:2]

            new_height = int(height * self.factor)

            if self.method == "duplicate":
                if self.selection == "all":
                    # For "all", use the calculated new_height approach
                    repeat_factor = new_height / height
                    result_pixels = np.zeros((new_height, width, 4), dtype=np.uint8)

                    for i in range(new_height):
                        # Map new row index to original row
                        original_row = int(i / repeat_factor)
                        if original_row >= height:
                            original_row = height - 1
                        result_pixels[i] = pixels[original_row]
                else:
                    # For selective stretching, use repeat counts
                    selected_rows = _select_rows(
                        self.selection, height, n=self.n, indices=self.indices
                    )
                    repeat_counts = np.ones(height, dtype=int)
                    for row_idx in selected_rows:
                        repeat_counts[row_idx] = int(self.factor)

                    # Create stretched image
                    result_pixels = np.repeat(pixels, repeat_counts, axis=0)

            else:  # distribute
                # Distribute rows evenly across new height
                result_pixels = np.zeros((new_height, width, 4), dtype=np.uint8)

                for i in range(new_height):
                    # Map new row index to original row
                    original_row = int(i * height / new_height)
                    result_pixels[i] = pixels[original_row]

            # Create result image
            result_image = Image.fromarray(result_pixels)
            if result_image.mode != "RGBA":
                result_image = result_image.convert("RGBA")

            # Convert back to original mode if appropriate
            if image.mode != "RGBA":
                result_image = result_image.convert(image.mode)

            updated_context = context.copy_with_updates(height=new_height)
            return result_image, updated_context

        except Exception as e:
            raise ProcessingError(f"RowStretch failed: {e}") from e


class RowRemove(BaseOperation):
    """Delete specific rows from image."""

    selection: Literal["odd", "even", "prime", "every_n", "custom"] = "odd"
    n: int | None = Field(None, ge=1, description="For every_n selection")
    indices: list[int] | None = Field(None, description="For custom selection")

    @field_validator("indices")
    @classmethod
    def validate_indices(cls, v):
        if v is not None and len(v) == 0:
            raise ValidationError("Custom indices cannot be empty")
        return v

    @model_validator(mode="after")
    def validate_selection_parameters(self) -> "RowRemove":
        """Validate that required parameters are provided for each selection type."""
        if self.selection == "every_n" and self.n is None:
            raise ValueError("Parameter 'n' is required when selection='every_n'")
        elif self.selection == "custom" and self.indices is None:
            raise ValueError("Parameter 'indices' is required when selection='custom'")
        return self

    def validate_operation(self, context: ImageContext) -> ImageContext:
        """Validate operation against image context."""
        rows_to_remove = _select_rows(
            self.selection, context.height, n=self.n, indices=self.indices
        )

        remaining_rows = context.height - len(rows_to_remove)
        if remaining_rows <= 0:
            raise ValidationError("Cannot remove all rows - at least 1 row must remain")

        return context.copy_with_updates(height=remaining_rows)

    def get_cache_key(self, image_hash: str) -> str:
        """Generate cache key for this operation."""
        config_str = f"{self.selection}_{self.n}_{self.indices}"
        return f"rowremove_{image_hash}_{hash(config_str)}"

    def estimate_memory(self, context: ImageContext) -> int:
        """Estimate memory usage in bytes."""
        return context.memory_estimate * 2  # Input + output

    def apply(self, image: Image.Image, context: ImageContext) -> tuple[Image.Image, ImageContext]:
        """Apply row removal to image."""
        try:
            # Convert to RGBA for consistent processing
            rgba_image = image.convert("RGBA")
            pixels = np.array(rgba_image)
            height, width = pixels.shape[:2]

            # Get rows to remove
            rows_to_remove = _select_rows(self.selection, height, n=self.n, indices=self.indices)

            # Create mask of rows to keep
            keep_mask = np.ones(height, dtype=bool)
            keep_mask[rows_to_remove] = False

            # Remove rows
            result_pixels = pixels[keep_mask]

            # Create result image
            result_image = Image.fromarray(result_pixels)
            if result_image.mode != "RGBA":
                result_image = result_image.convert("RGBA")

            # Convert back to original mode if appropriate
            if image.mode != "RGBA":
                result_image = result_image.convert(image.mode)

            new_height = len(result_pixels)
            updated_context = context.copy_with_updates(height=new_height)
            return result_image, updated_context

        except Exception as e:
            raise ProcessingError(f"RowRemove failed: {e}") from e


class RowShuffle(BaseOperation):
    """Randomly reorder rows."""

    seed: int | None = Field(None, description="Random seed for reproducibility")
    groups: int = Field(1, ge=1, description="Shuffle within groups of N rows")

    def validate_operation(self, context: ImageContext) -> ImageContext:
        """Validate operation against image context."""
        if self.groups > context.height:
            raise ValidationError(
                f"Group size {self.groups} cannot exceed image height {context.height}"
            )
        return context.copy_with_updates()

    def get_cache_key(self, image_hash: str) -> str:
        """Generate cache key for this operation."""
        config_str = f"{self.seed}_{self.groups}"
        return f"rowshuffle_{image_hash}_{hash(config_str)}"

    def estimate_memory(self, context: ImageContext) -> int:
        """Estimate memory usage in bytes."""
        return context.memory_estimate * 2  # Input + output

    def apply(self, image: Image.Image, context: ImageContext) -> tuple[Image.Image, ImageContext]:
        """Apply row shuffle to image."""
        try:
            # Convert to RGBA for consistent processing
            rgba_image = image.convert("RGBA")
            pixels = np.array(rgba_image)
            height, width = pixels.shape[:2]

            # Set random seed if provided
            if self.seed is not None:
                random.seed(self.seed)
                np.random.seed(self.seed)

            result_pixels = pixels.copy()

            # Shuffle in groups
            for group_start in range(0, height, self.groups):
                group_end = min(group_start + self.groups, height)
                group_indices = list(range(group_start, group_end))

                # Shuffle the indices within this group
                random.shuffle(group_indices)

                # Apply the shuffle
                group_pixels = pixels[group_start:group_end].copy()
                for i, shuffled_idx in enumerate(group_indices):
                    result_pixels[group_start + i] = group_pixels[shuffled_idx - group_start]

            # Create result image
            result_image = Image.fromarray(result_pixels)
            if result_image.mode != "RGBA":
                result_image = result_image.convert("RGBA")

            # Convert back to original mode if appropriate
            if image.mode != "RGBA":
                result_image = result_image.convert(image.mode)

            return result_image, context.copy_with_updates()

        except Exception as e:
            raise ProcessingError(f"RowShuffle failed: {e}") from e
