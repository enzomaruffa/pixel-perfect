"""Column-based image processing operations."""

from typing import Literal

import numpy as np
from PIL import Image
from pydantic import Field, field_validator, model_validator

from core.base import BaseOperation
from core.context import ImageContext
from exceptions import ProcessingError, ValidationError
from utils.validation import validate_color_tuple, validate_expression_safe


def _select_columns(selection: str, width: int, **kwargs) -> np.ndarray:
    """Select column indices based on selection criteria.

    Args:
        selection: Selection method ("all", "odd", "even", "prime", "every_n", "custom", "gradient", "formula")
        width: Total number of columns in image
        **kwargs: Additional parameters (n, indices, formula, etc.)

    Returns:
        Array of selected column indices
    """
    if selection == "all":
        return np.arange(width)
    elif selection == "odd":
        return np.arange(1, width, 2)
    elif selection == "even":
        return np.arange(0, width, 2)
    elif selection == "prime":
        return _get_prime_indices(width)
    elif selection == "every_n":
        n = kwargs.get("n", 1)
        if n <= 0:
            raise ValidationError("Parameter 'n' must be positive")
        return np.arange(0, width, n)
    elif selection == "custom":
        indices = kwargs.get("indices", [])
        if not indices:
            raise ValidationError("Custom selection requires 'indices' parameter")
        indices_array = np.array(indices)
        if np.any(indices_array < 0) or np.any(indices_array >= width):
            raise ValidationError(f"Column indices must be in range [0, {width})")
        return indices_array
    elif selection == "gradient":
        # For gradient mode, return all columns (shift calculation happens elsewhere)
        return np.arange(width)
    elif selection == "formula":
        # For formula mode, return all columns (shift calculation happens elsewhere)
        return np.arange(width)
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


def _calculate_formula_shifts(formula: str, width: int) -> np.ndarray:
    """Calculate shift amounts for each column using a formula.

    Args:
        formula: Mathematical expression using 'j' for column index (0-based)
        width: Number of columns

    Returns:
        Array of shift amounts for each column
    """
    shifts = np.zeros(width, dtype=int)

    # Safe evaluation with limited scope
    allowed_names = {
        # Basic math functions
        "abs": abs,
        "min": min,
        "max": max,
        "round": round,
        "int": int,
        "float": float,
        # Math module functions
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "sqrt": np.sqrt,
        "pow": pow,
        "pi": np.pi,
        "e": np.e,
        # Numpy functions
        "floor": np.floor,
        "ceil": np.ceil,
    }

    for j in range(width):
        try:
            # Create evaluation context with column coordinate
            context = allowed_names.copy()
            context["x"] = j  # NEW STANDARD: x for column coordinate
            context["j"] = j  # DEPRECATED: keep for backward compatibility
            context["width"] = width

            # Evaluate formula and convert to int
            result = eval(formula, {"__builtins__": {}}, context)
            shifts[j] = int(result)
        except Exception as e:
            raise ProcessingError(f"Error evaluating formula '{formula}' at column {j}: {e}") from e

    return shifts


class ColumnShift(BaseOperation):
    """Translate entire columns vertically."""

    selection: Literal[
        "all", "odd", "even", "prime", "every_n", "custom", "gradient", "formula"
    ] = "odd"
    n: int | None = Field(None, ge=1, description="For every_n selection")
    indices: list[int] | None = Field(None, description="For custom selection")
    shift_amount: int = Field(0, description="Pixels to shift (negative=up, positive=down)")
    wrap: bool = Field(True, description="Wrap around vs fill with color")
    fill_color: tuple[int, int, int, int] = Field((0, 0, 0, 0), description="RGBA fill color")
    gradient_start: int = Field(0, description="Starting shift for gradient mode")
    formula: str | None = Field(
        None,
        description="Mathematical formula for formula mode (use 'x' for column coordinate, 0 to width-1)",
    )

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

    @field_validator("formula")
    @classmethod
    def validate_formula(cls, v):
        if v is not None:
            # Validate the formula is safe and syntactically correct
            validate_expression_safe(v)
        return v

    @model_validator(mode="after")
    def validate_selection_parameters(self) -> "ColumnShift":
        """Validate that required parameters are provided for each selection type."""
        if self.selection == "every_n" and self.n is None:
            raise ValueError("Parameter 'n' is required when selection='every_n'")
        elif self.selection == "custom" and self.indices is None:
            raise ValueError("Parameter 'indices' is required when selection='custom'")
        elif self.selection == "formula" and self.formula is None:
            raise ValueError("Parameter 'formula' is required when selection='formula'")
        return self

    def validate_operation(self, context: ImageContext) -> ImageContext:
        """Validate operation against image context."""
        # Validate column selection
        _select_columns(self.selection, context.width, n=self.n, indices=self.indices)
        return context.copy_with_updates()

    def get_cache_key(self, image_hash: str) -> str:
        """Generate cache key for this operation."""
        config_str = f"{self.selection}_{self.n}_{self.indices}_{self.shift_amount}"
        config_str += f"_{self.wrap}_{self.fill_color}_{self.gradient_start}_{self.formula}"
        return f"columnshift_{image_hash}_{hash(config_str)}"

    def estimate_memory(self, context: ImageContext) -> int:
        """Estimate memory usage in bytes."""
        return context.memory_estimate * 2  # Input + output

    def apply(self, image: Image.Image, context: ImageContext) -> tuple[Image.Image, ImageContext]:
        """Apply column shift to image."""
        try:
            # Convert to RGBA for consistent processing
            rgba_image = image.convert("RGBA")
            pixels = np.array(rgba_image)
            height, width = pixels.shape[:2]

            # Get columns to shift and calculate shift amounts
            if self.selection == "gradient":
                selected_columns = np.arange(width)
                # Pre-calculate all shift amounts for gradient
                column_shifts = np.array(
                    [
                        int(
                            self.gradient_start
                            + (self.shift_amount - self.gradient_start) * j / (width - 1)
                        )
                        for j in range(width)
                    ]
                )
            elif self.selection == "formula":
                if self.formula is None:
                    raise ValidationError("Formula is required when selection is 'formula'")
                selected_columns = np.arange(width)
                # Calculate shift amounts using formula
                column_shifts = _calculate_formula_shifts(self.formula, width)
            else:
                selected_columns = _select_columns(
                    self.selection, width, n=self.n, indices=self.indices
                )
                # Use constant shift amount for other modes
                column_shifts = np.full(width, self.shift_amount)

            # Create output array
            result_pixels = pixels.copy()

            for col_idx in selected_columns:
                if self.selection in ["gradient", "formula"]:
                    shift = column_shifts[col_idx]
                else:
                    shift = self.shift_amount

                if shift == 0:
                    continue

                column = pixels[:, col_idx]

                if self.wrap:
                    # Wrap around
                    shift = shift % height  # Handle shifts larger than height
                    if shift > 0:
                        result_pixels[:, col_idx] = np.concatenate(
                            [column[-shift:], column[:-shift]]
                        )
                    else:
                        shift = abs(shift)
                        result_pixels[:, col_idx] = np.concatenate([column[shift:], column[:shift]])
                else:
                    # Fill with color
                    shifted_column = np.full_like(column, self.fill_color)
                    if shift > 0:
                        # Shift down
                        if shift < height:
                            shifted_column[shift:] = column[:-shift]
                    else:
                        # Shift up
                        shift = abs(shift)
                        if shift < height:
                            shifted_column[:-shift] = column[shift:]
                    result_pixels[:, col_idx] = shifted_column

            # Create result image
            result_image = Image.fromarray(result_pixels)
            if result_image.mode != "RGBA":
                result_image = result_image.convert("RGBA")

            # Convert back to original mode if fill_color is opaque
            if image.mode != "RGBA" and self.fill_color[3] == 255:
                result_image = result_image.convert(image.mode)

            return result_image, context.copy_with_updates()

        except Exception as e:
            raise ProcessingError(f"ColumnShift failed: {e}") from e


class ColumnStretch(BaseOperation):
    """Duplicate columns to stretch image horizontally."""

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
    def validate_selection_parameters(self) -> "ColumnStretch":
        """Validate that required parameters are provided for each selection type."""
        if self.selection == "every_n" and self.n is None:
            raise ValueError("Parameter 'n' is required when selection='every_n'")
        elif self.selection == "custom" and self.indices is None:
            raise ValueError("Parameter 'indices' is required when selection='custom'")
        return self

    def validate_operation(self, context: ImageContext) -> ImageContext:
        """Validate operation against image context."""
        if self.selection != "all":
            _select_columns(self.selection, context.width, n=self.n, indices=self.indices)

        new_width = int(context.width * self.factor)
        return context.copy_with_updates(width=new_width)

    def get_cache_key(self, image_hash: str) -> str:
        """Generate cache key for this operation."""
        config_str = f"{self.factor}_{self.method}_{self.selection}_{self.n}_{self.indices}"
        return f"columnstretch_{image_hash}_{hash(config_str)}"

    def estimate_memory(self, context: ImageContext) -> int:
        """Estimate memory usage in bytes."""
        new_width = int(context.width * self.factor)
        bytes_per_pixel = 1 if context.dtype == "uint8" else 4
        return new_width * context.height * context.channels * bytes_per_pixel * 2

    def apply(self, image: Image.Image, context: ImageContext) -> tuple[Image.Image, ImageContext]:
        """Apply column stretch to image."""
        try:
            # Convert to RGBA for consistent processing
            rgba_image = image.convert("RGBA")
            pixels = np.array(rgba_image)
            height, width = pixels.shape[:2]

            new_width = int(width * self.factor)

            if self.method == "duplicate":
                if self.selection == "all":
                    # For "all", use the calculated new_width approach
                    repeat_factor = new_width / width
                    result_pixels = np.zeros((height, new_width, 4), dtype=np.uint8)

                    for i in range(new_width):
                        # Map new column index to original column
                        original_col = int(i / repeat_factor)
                        if original_col >= width:
                            original_col = width - 1
                        result_pixels[:, i] = pixels[:, original_col]
                else:
                    # For selective stretching, use repeat counts
                    selected_columns = _select_columns(
                        self.selection, width, n=self.n, indices=self.indices
                    )
                    repeat_counts = np.ones(width, dtype=int)
                    for col_idx in selected_columns:
                        repeat_counts[col_idx] = int(self.factor)

                    # Create stretched image
                    result_pixels = np.repeat(pixels, repeat_counts, axis=1)

            else:  # distribute
                # Distribute columns evenly across new width
                result_pixels = np.zeros((height, new_width, 4), dtype=np.uint8)

                for i in range(new_width):
                    # Map new column index to original column
                    original_col = int(i * width / new_width)
                    result_pixels[:, i] = pixels[:, original_col]

            # Create result image
            result_image = Image.fromarray(result_pixels)
            if result_image.mode != "RGBA":
                result_image = result_image.convert("RGBA")

            # Convert back to original mode if appropriate
            if image.mode != "RGBA":
                result_image = result_image.convert(image.mode)

            updated_context = context.copy_with_updates(width=new_width)
            return result_image, updated_context

        except Exception as e:
            raise ProcessingError(f"ColumnStretch failed: {e}") from e


class ColumnMirror(BaseOperation):
    """Reflect columns around vertical axis."""

    mode: Literal["full", "alternating"] = "full"
    pivot: int | None = Field(None, description="Column index to mirror around")

    def validate_operation(self, context: ImageContext) -> ImageContext:
        """Validate operation against image context."""
        if self.pivot is not None and (self.pivot < 0 or self.pivot >= context.width):
            raise ValidationError(f"Pivot column {self.pivot} must be within [0, {context.width})")
        return context.copy_with_updates()

    def get_cache_key(self, image_hash: str) -> str:
        """Generate cache key for this operation."""
        config_str = f"{self.mode}_{self.pivot}"
        return f"columnmirror_{image_hash}_{hash(config_str)}"

    def estimate_memory(self, context: ImageContext) -> int:
        """Estimate memory usage in bytes."""
        return context.memory_estimate * 2  # Input + output

    def apply(self, image: Image.Image, context: ImageContext) -> tuple[Image.Image, ImageContext]:
        """Apply column mirroring to image."""
        try:
            # Convert to RGBA for consistent processing
            rgba_image = image.convert("RGBA")
            pixels = np.array(rgba_image)
            height, width = pixels.shape[:2]

            result_pixels = pixels.copy()

            if self.mode == "full":
                # Mirror entire image around pivot or center
                pivot = width // 2 if self.pivot is None else self.pivot

                # Reflect columns around pivot
                for col in range(width):
                    mirror_col = 2 * pivot - col
                    if 0 <= mirror_col < width:
                        result_pixels[:, col] = pixels[:, mirror_col]

            else:  # alternating
                # Mirror every other column
                for col in range(1, width, 2):  # Odd columns
                    if col > 0:
                        result_pixels[:, col] = pixels[:, col - 1]  # Mirror from previous column

            # Create result image
            result_image = Image.fromarray(result_pixels)
            if result_image.mode != "RGBA":
                result_image = result_image.convert("RGBA")

            # Convert back to original mode if appropriate
            if image.mode != "RGBA":
                result_image = result_image.convert(image.mode)

            return result_image, context.copy_with_updates()

        except Exception as e:
            raise ProcessingError(f"ColumnMirror failed: {e}") from e


class ColumnWeave(BaseOperation):
    """Interlace columns from different parts of image."""

    pattern: list[int] = Field(..., description="List defining source indices mapping")
    repeat: bool = Field(True, description="Cycle pattern if shorter than width")

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, v):
        if not v:
            raise ValidationError("Pattern cannot be empty")
        if any(idx < 0 for idx in v):
            raise ValidationError("Pattern indices cannot be negative")
        return v

    def validate_operation(self, context: ImageContext) -> ImageContext:
        """Validate operation against image context."""
        max_pattern_idx = max(self.pattern)
        if max_pattern_idx >= context.width:
            raise ValidationError(
                f"Pattern index {max_pattern_idx} exceeds image width {context.width}"
            )
        return context.copy_with_updates()

    def get_cache_key(self, image_hash: str) -> str:
        """Generate cache key for this operation."""
        config_str = f"{self.pattern}_{self.repeat}"
        return f"columnweave_{image_hash}_{hash(config_str)}"

    def estimate_memory(self, context: ImageContext) -> int:
        """Estimate memory usage in bytes."""
        return context.memory_estimate * 2  # Input + output

    def apply(self, image: Image.Image, context: ImageContext) -> tuple[Image.Image, ImageContext]:
        """Apply column weaving to image."""
        try:
            # Convert to RGBA for consistent processing
            rgba_image = image.convert("RGBA")
            pixels = np.array(rgba_image)
            height, width = pixels.shape[:2]

            result_pixels = pixels.copy()

            # Apply pattern mapping
            for dest_col in range(width):
                if self.repeat:
                    # Cycle through pattern
                    pattern_idx = dest_col % len(self.pattern)
                    source_col = self.pattern[pattern_idx] % width
                else:
                    # Use pattern as-is, original columns for beyond pattern length
                    if dest_col < len(self.pattern):
                        source_col = self.pattern[dest_col] % width
                    else:
                        source_col = dest_col  # Keep original column

                result_pixels[:, dest_col] = pixels[:, source_col]

            # Create result image
            result_image = Image.fromarray(result_pixels)
            if result_image.mode != "RGBA":
                result_image = result_image.convert("RGBA")

            # Convert back to original mode if appropriate
            if image.mode != "RGBA":
                result_image = result_image.convert(image.mode)

            return result_image, context.copy_with_updates()

        except Exception as e:
            raise ProcessingError(f"ColumnWeave failed: {e}") from e
