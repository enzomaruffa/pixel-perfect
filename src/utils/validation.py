"""Parameter validation utilities for operations."""

import re
from typing import Any

from exceptions import DimensionError, ValidationError


def validate_image_dimensions(width: int, height: int) -> None:
    """Check that image dimensions are valid.

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Raises:
        DimensionError: If dimensions are invalid
    """
    if width <= 0 or height <= 0:
        raise DimensionError(f"Image dimensions must be positive, got {width}×{height}")

    if width > 65535 or height > 65535:
        raise DimensionError(f"Image dimensions too large, got {width}×{height}")


def validate_indices(indices: list[int], max_value: int, name: str = "index") -> None:
    """Validate that all indices are within valid range.

    Args:
        indices: List of indices to validate
        max_value: Maximum allowed index (exclusive)
        name: Name of the indices for error messages

    Raises:
        ValidationError: If any index is out of bounds
    """
    if not indices:
        return

    invalid_indices = [i for i in indices if i < 0 or i >= max_value]
    if invalid_indices:
        raise ValidationError(
            f"Invalid {name} indices {invalid_indices}, must be in range [0, {max_value})"
        )


def validate_color_tuple(color: tuple[int, ...], channels: int = 4) -> tuple[int, ...]:
    """Validate and normalize color tuple.

    Args:
        color: Color tuple (R, G, B) or (R, G, B, A)
        channels: Expected number of channels

    Returns:
        Normalized color tuple with correct number of channels

    Raises:
        ValidationError: If color format is invalid
    """
    if not isinstance(color, tuple | list):
        raise ValidationError(f"Color must be a tuple or list, got {type(color)}")

    if len(color) < 3:
        raise ValidationError(f"Color must have at least 3 values (RGB), got {len(color)}")

    if len(color) > 4:
        raise ValidationError(f"Color cannot have more than 4 values (RGBA), got {len(color)}")

    # Validate color values are in range
    for i, value in enumerate(color):
        if not isinstance(value, int) or value < 0 or value > 255:
            channel_name = ["R", "G", "B", "A"][i] if i < 4 else str(i)
            raise ValidationError(f"Color {channel_name} value must be 0-255, got {value}")

    # Normalize to requested number of channels
    color_list = list(color)
    if len(color_list) == 3 and channels == 4:
        color_list.append(255)  # Add opaque alpha
    elif len(color_list) == 4 and channels == 3:
        color_list = color_list[:3]  # Remove alpha

    return tuple(color_list)


def validate_ratio_string(ratio: str) -> float:
    """Parse and validate aspect ratio string.

    Args:
        ratio: Ratio string like "16:9" or "1.5"

    Returns:
        Ratio as decimal value

    Raises:
        ValidationError: If ratio format is invalid
    """
    # Handle decimal format
    try:
        decimal_ratio = float(ratio)
        if decimal_ratio <= 0:
            raise ValidationError(f"Aspect ratio must be positive, got {decimal_ratio}")
        return decimal_ratio
    except ValueError:
        pass

    # Handle W:H format
    if ":" in ratio:
        match = re.match(r"^(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)$", ratio.strip())
        if not match:
            raise ValidationError(f"Invalid ratio format '{ratio}', expected 'W:H' or decimal")

        width_str, height_str = match.groups()
        try:
            width = float(width_str)
            height = float(height_str)
            if width <= 0 or height <= 0:
                raise ValidationError(f"Ratio components must be positive, got {width}:{height}")
            return width / height
        except ValueError:
            raise ValidationError(f"Invalid numbers in ratio '{ratio}'") from None

    raise ValidationError(f"Invalid ratio format '{ratio}', expected 'W:H' or decimal")


def validate_selection_criteria(
    selection: str, context: Any, valid_selections: list[str] | None = None
) -> None:
    """Validate row/column selection criteria.

    Args:
        selection: Selection type ("odd", "even", "prime", etc.)
        context: ImageContext for bounds checking
        valid_selections: List of valid selection types

    Raises:
        ValidationError: If selection criteria is invalid
    """
    default_selections = ["odd", "even", "prime", "every_n", "custom", "gradient"]
    allowed = valid_selections or default_selections

    if selection not in allowed:
        raise ValidationError(f"Invalid selection '{selection}', must be one of {allowed}")

    # Additional validation based on selection type
    if selection == "gradient" and not hasattr(context, "height"):
        raise ValidationError("Gradient selection requires image height context")


def validate_expression_safe(expression: str) -> None:
    """Validate that a mathematical expression is safe to evaluate.

    Args:
        expression: Mathematical expression string

    Raises:
        ValidationError: If expression contains unsafe operations
    """
    if not expression.strip():
        raise ValidationError("Expression cannot be empty")

    # List of allowed characters and functions
    allowed_chars = set("0123456789+-*/().%abcdefghijklmnopqrstuvwxyz_=!<>")
    allowed_functions = {
        "sin",
        "cos",
        "tan",
        "sqrt",
        "abs",
        "min",
        "max",
        "floor",
        "ceil",
        "round",
        "log",
        "exp",
        "pow",
        "int",
        "float",
    }
    # STANDARD VARIABLES (preferred)
    allowed_vars = {
        "r", "g", "b", "a",           # Color channels (0-255)
        "x", "y",                     # Spatial coordinates (x=column, y=row)
        "width", "height",            # Image dimensions
        "i"                           # Linear pixel index (0 to width*height-1)
    }
    
    # DEPRECATED VARIABLES (for backward compatibility)
    deprecated_vars = {"j"}  # Use 'x' for column coordinate instead

    # Check for unsafe characters
    expr_lower = expression.lower()
    for char in expr_lower:
        if char.isspace():
            continue
        if char not in allowed_chars:
            raise ValidationError(f"Unsafe character '{char}' in expression")

    # Check for unsafe keywords
    unsafe_keywords = [
        "import",
        "exec",
        "eval",
        "open",
        "file",
        "input",
        "raw_input",
        "__",
        "lambda",
        "def",
        "class",
        "while",
        "for",
        "if",
        "try",
    ]

    for keyword in unsafe_keywords:
        if keyword in expr_lower:
            raise ValidationError(f"Unsafe keyword '{keyword}' in expression")

    # Validate function names
    import re

    function_pattern = r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
    functions_used = re.findall(function_pattern, expression)

    for func in functions_used:
        if func.lower() not in allowed_functions and func.lower() not in allowed_vars and func.lower() not in deprecated_vars:
            raise ValidationError(f"Unknown or unsafe function '{func}' in expression")
    
    # Check for deprecated variables and warn
    import re
    import warnings
    variable_pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b"
    variables_used = re.findall(variable_pattern, expression)
    
    for var in variables_used:
        if var in deprecated_vars:
            if var == "j":
                warnings.warn(
                    f"Variable 'j' is deprecated. Use 'x' for column coordinate instead. "
                    f"Example: '{expression.replace('j', 'x')}'",
                    DeprecationWarning,
                    stacklevel=3
                )


def validate_positive_number(value: float, name: str = "value") -> None:
    """Validate that a number is positive.

    Args:
        value: Number to validate
        name: Name for error messages

    Raises:
        ValidationError: If value is not positive
    """
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")


def validate_range(value: float, min_val: float, max_val: float, name: str = "value") -> None:
    """Validate that a value is within specified range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        name: Name for error messages

    Raises:
        ValidationError: If value is out of range
    """
    if value < min_val or value > max_val:
        raise ValidationError(f"{name} must be in range [{min_val}, {max_val}], got {value}")


def validate_channel_list(channels: list[str]) -> list[str]:
    """Validate and normalize channel list.

    Args:
        channels: List of channel names

    Returns:
        Validated channel list

    Raises:
        ValidationError: If channel names are invalid
    """
    valid_channels = {"r", "g", "b", "a"}

    if not channels:
        raise ValidationError("Channel list cannot be empty")

    normalized = []
    for channel in channels:
        if not isinstance(channel, str):
            raise ValidationError(f"Channel name must be string, got {type(channel)}")

        channel_lower = channel.lower().strip()
        if channel_lower not in valid_channels:
            raise ValidationError(f"Invalid channel '{channel}', must be one of {valid_channels}")

        if channel_lower in normalized:
            raise ValidationError(f"Duplicate channel '{channel}' in list")

        normalized.append(channel_lower)

    return normalized
