"""Error translation utilities for user-friendly error messages."""

import re
from difflib import SequenceMatcher
from typing import Any

from pydantic import ValidationError


def translate_error(error: Exception, context: str | None = None) -> dict[str, Any]:
    """Convert technical errors to user-friendly messages.

    Args:
        error: Exception to translate
        context: Additional context for error interpretation

    Returns:
        Dictionary with 'message', 'suggestion', and 'details' keys
    """
    if isinstance(error, ValidationError):
        return translate_validation_error(error, context)
    elif "ProcessingError" in str(type(error)):
        return translate_processing_error(error, context)
    elif isinstance(error, FileNotFoundError):
        return translate_file_error(error, context)
    elif isinstance(error, ValueError):
        return translate_value_error(error, context)
    elif isinstance(error, TypeError):
        return translate_type_error(error, context)
    else:
        return {
            "message": "Unexpected error occurred",
            "details": str(error),
            "suggestion": "Please try again or check your input",
            "error_type": "unknown",
        }


def translate_validation_error(
    error: ValidationError, context: str | None = None
) -> dict[str, Any]:
    """Translate Pydantic validation errors to user-friendly messages."""
    errors = error.errors()

    if not errors:
        return {
            "message": "Validation error occurred",
            "suggestion": "Please check your input values",
            "details": str(error),
            "error_type": "validation",
        }

    # Handle the first error (most relevant)
    first_error = errors[0]
    field = first_error.get("loc", ["parameter"])[0] if first_error.get("loc") else "parameter"
    error_type = first_error.get("type", "")
    input_value = first_error.get("input", "unknown")

    if error_type == "literal_error":
        # Handle enum/literal choice errors
        expected = first_error.get("ctx", {}).get("expected", [])
        if isinstance(expected, str):
            # Parse expected string like "Literal['prime']"
            expected_match = re.findall(r"'([^']+)'", expected)
            expected = expected_match if expected_match else [expected]

        suggestion = suggest_closest_match(str(input_value), expected)

        return {
            "message": f'Invalid {field} "{input_value}". Choose one of: {", ".join(map(str, expected))}',
            "suggestion": f'Did you mean "{suggestion}"?'
            if suggestion
            else "Please select a valid option",
            "field": field,
            "valid_options": expected,
            "error_type": "invalid_choice",
        }

    elif error_type == "missing":
        return {
            "message": f'Required field "{field}" is missing',
            "suggestion": f"Please provide a value for {field}",
            "field": field,
            "error_type": "missing_required",
        }

    elif error_type in ["greater_than", "greater_than_equal", "less_than", "less_than_equal"]:
        limit = first_error.get("ctx", {}).get("limit_value", "limit")
        return {
            "message": f"{str(field).title()} value {input_value} is outside valid range",
            "suggestion": f"Use a value {error_type.replace('_', ' ')} {limit}",
            "field": field,
            "error_type": "range_error",
        }

    else:
        return {
            "message": f"Invalid {field}: {first_error.get('msg', 'validation failed')}",
            "suggestion": "Please check the input format and try again",
            "field": field,
            "error_type": "validation",
        }


def translate_processing_error(error: Exception, context: str | None = None) -> dict[str, Any]:
    """Translate processing/execution errors to user-friendly messages."""
    error_str = str(error)

    # Formula evaluation errors
    if "name" in error_str and "is not defined" in error_str:
        var_match = re.search(r"name '(\w+)' is not defined", error_str)
        if var_match:
            undefined_var = var_match.group(1)
            suggestion = suggest_variable_replacement(undefined_var)

            return {
                "message": f"Variable '{undefined_var}' not recognized in formula",
                "suggestion": suggestion,
                "variable": undefined_var,
                "error_type": "undefined_variable",
            }

    # Division by zero
    if "division by zero" in error_str.lower():
        return {
            "message": "Division by zero in formula",
            "suggestion": "Check your formula for divisions that might result in zero denominators",
            "error_type": "division_by_zero",
        }

    # Invalid syntax
    if "invalid syntax" in error_str.lower():
        return {
            "message": "Formula syntax error",
            "suggestion": "Check parentheses, operators, and variable names in your formula",
            "error_type": "syntax_error",
        }

    # Image format errors
    if "image" in error_str.lower() and (
        "format" in error_str.lower() or "mode" in error_str.lower()
    ):
        return {
            "message": "Image format error",
            "suggestion": "Ensure image is in RGB or RGBA mode. Try converting the image format first",
            "error_type": "image_format",
        }

    # Memory errors
    if "memory" in error_str.lower() or "out of memory" in error_str.lower():
        return {
            "message": "Not enough memory to process image",
            "suggestion": "Try using a smaller image or reducing the number of operations",
            "error_type": "memory_error",
        }

    return {
        "message": f"Processing failed: {error_str}",
        "suggestion": "Please check your operation parameters and try again",
        "error_type": "processing",
    }


def translate_file_error(error: FileNotFoundError, context: str | None = None) -> dict[str, Any]:
    """Translate file-related errors."""
    return {
        "message": "File not found",
        "suggestion": "Please check the file path and ensure the file exists",
        "details": str(error),
        "error_type": "file_not_found",
    }


def translate_value_error(error: ValueError, context: str | None = None) -> dict[str, Any]:
    """Translate ValueError exceptions."""
    error_str = str(error)

    # Color value errors
    if "color" in error_str.lower() and ("range" in error_str.lower() or "255" in error_str):
        return {
            "message": "Invalid color value",
            "suggestion": "Color values must be between 0 and 255",
            "error_type": "color_range",
        }

    # Custom expression errors
    if "custom_expression" in error_str:
        return {
            "message": "Custom expression is required",
            "suggestion": "When using 'custom' condition, you must provide a custom expression",
            "error_type": "missing_expression",
        }

    return {
        "message": f"Invalid value: {error_str}",
        "suggestion": "Please check your input values and ranges",
        "error_type": "value_error",
    }


def translate_type_error(error: TypeError, context: str | None = None) -> dict[str, Any]:
    """Translate TypeError exceptions."""
    return {
        "message": "Type error occurred",
        "suggestion": "Please check that all parameters have the correct data types",
        "details": str(error),
        "error_type": "type_error",
    }


def suggest_closest_match(value: str, options: list) -> str | None:
    """Suggest the closest matching option using fuzzy matching."""
    if not value or not options:
        return None

    value_lower = value.lower()
    best_match = None
    best_ratio = 0

    for option in options:
        option_str = str(option).lower()

        # Try exact substring match first
        if value_lower in option_str or option_str in value_lower:
            return str(option)

        # Use sequence matching for similarity
        ratio = SequenceMatcher(None, value_lower, option_str).ratio()
        if ratio > best_ratio and ratio > 0.3:  # Minimum similarity threshold
            best_match = str(option)
            best_ratio = ratio

    return best_match


def suggest_variable_replacement(undefined_var: str) -> str:
    """Suggest correct variable name for common mistakes."""
    variable_suggestions = {
        # Common mistakes
        "j": "Use 'x' for column coordinate (0 to width-1)",
        "col": "Use 'x' for column coordinate",
        "column": "Use 'x' for column coordinate",
        "row": "Use 'y' for row coordinate",
        "idx": "Use 'i' for pixel index",
        "index": "Use 'i' for pixel index",
        # Less common
        "red": "Use 'r' for red channel",
        "green": "Use 'g' for green channel",
        "blue": "Use 'b' for blue channel",
        "alpha": "Use 'a' for alpha channel",
        "w": "Use 'width' for image width",
        "h": "Use 'height' for image height",
    }

    # Direct suggestion
    if undefined_var in variable_suggestions:
        return variable_suggestions[undefined_var]

    # Fuzzy matching for typos
    standard_vars = ["x", "y", "r", "g", "b", "a", "i", "width", "height"]
    closest = suggest_closest_match(undefined_var, standard_vars)

    if closest:
        var_descriptions = {
            "x": "column coordinate (0 to width-1)",
            "y": "row coordinate (0 to height-1)",
            "r": "red channel (0-255)",
            "g": "green channel (0-255)",
            "b": "blue channel (0-255)",
            "a": "alpha channel (0-255)",
            "i": "linear pixel index",
            "width": "image width",
            "height": "image height",
        }
        description = var_descriptions.get(closest, closest)
        return f"Use '{closest}' for {description}"

    return "Available variables: x, y, r, g, b, a, i, width, height"


def format_error_message(error_info: dict[str, Any]) -> str:
    """Format error information into a clean display message."""
    message = error_info.get("message", "An error occurred")
    suggestion = error_info.get("suggestion")

    if suggestion:
        return f"{message}\n\n💡 {suggestion}"

    return message
