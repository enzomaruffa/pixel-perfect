"""Standardized constants and templates for operation consistency."""

# Standardized field descriptions for common parameter types

# Coordinate and dimension descriptions
COORDINATE_DESCRIPTIONS = {
    "shift_amount": "Number of pixels to shift (positive = right/down, negative = left/up)",
    "block_width": "Width of each block in pixels (must be positive)",
    "block_height": "Height of each block in pixels (must be positive)",
    "factor": "Scaling factor (>1 for stretching, <1 for compressing)",
    "amplitude": "Wave amplitude in pixels (controls distortion strength)",
    "frequency": "Wave frequency (higher values create more oscillations)",
}

# Color descriptions
COLOR_DESCRIPTIONS = {
    "fill_color": "RGBA color values (red, green, blue, alpha) in range [0-255]",
    "background_color": "RGBA background color (red, green, blue, alpha) in range [0-255]",
    "color": "RGB color values (red, green, blue) in range [0-255]",
}

# Selection descriptions
SELECTION_DESCRIPTIONS = {
    "selection": "Which items to affect (all, odd, even, specific indices, or custom condition)",
    "indices": "Specific zero-based indices to affect (e.g., [0, 2, 4] for first, third, fifth items)",
    "condition": "Mathematical or logical condition for selection",
}

# Common boolean descriptions
BOOLEAN_DESCRIPTIONS = {
    "wrap": "Whether to wrap content that extends beyond boundaries",
    "preserve_alpha": "Whether to keep the original alpha channel values",
    "optimize": "Whether to apply optimization during processing",
    "enabled": "Whether this operation is active in the pipeline",
}

# Method/mode descriptions
METHOD_DESCRIPTIONS = {
    "method": "Algorithm or technique to use for this operation",
    "mode": "Processing mode that controls operation behavior",
    "interpolation": "Interpolation method for smooth transitions (linear, cubic, etc.)",
}

# Standardized error messages
ERROR_MESSAGES = {
    "empty_indices": "Custom indices list cannot be empty when selection='indices'",
    "invalid_color": "Color values must be integers in range [0-255]",
    "invalid_dimension": "Dimension must be a positive integer",
    "required_param": "Parameter '{param}' is required when {condition}",
    "out_of_range": "{param} value {value} is outside valid range [{min_val}-{max_val}]",
}


def get_standardized_description(param_name: str, param_type: str = "generic") -> str:
    """Get standardized description for a parameter.

    Args:
        param_name: Name of the parameter
        param_type: Type category (coordinate, color, selection, boolean, method)

    Returns:
        Standardized description string
    """
    type_maps = {
        "coordinate": COORDINATE_DESCRIPTIONS,
        "color": COLOR_DESCRIPTIONS,
        "selection": SELECTION_DESCRIPTIONS,
        "boolean": BOOLEAN_DESCRIPTIONS,
        "method": METHOD_DESCRIPTIONS,
    }

    if param_type in type_maps and param_name in type_maps[param_type]:
        return type_maps[param_type][param_name]

    # Fallback to generic descriptions
    return f"Parameter for {param_name.replace('_', ' ')}"


def get_standardized_error(error_type: str, **kwargs) -> str:
    """Get standardized error message.

    Args:
        error_type: Type of error from ERROR_MESSAGES keys
        **kwargs: Template variables for the error message

    Returns:
        Formatted error message
    """
    if error_type in ERROR_MESSAGES:
        return ERROR_MESSAGES[error_type].format(**kwargs)
    return f"Validation error: {error_type}"
