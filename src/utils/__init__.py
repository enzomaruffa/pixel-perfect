"""Utility functions for pixel-perfect."""

from utils.cache import (
    cleanup_old_cache,
    generate_operation_hash,
    get_cache_size,
    invalidate_cache_pattern,
    is_cache_entry_valid,
    load_cached_result,
    save_cached_result,
    verify_cache_integrity,
)
from utils.image import (
    calculate_memory_usage,
    copy_image_with_context,
    create_context_from_image,
    create_filled_image,
    ensure_image_mode,
    extract_image_region,
    get_image_hash,
    get_pixel_at_index,
    paste_image_region,
    resize_image_proportional,
)
from utils.validation import (
    validate_channel_list,
    validate_color_tuple,
    validate_expression_safe,
    validate_image_dimensions,
    validate_indices,
    validate_positive_number,
    validate_range,
    validate_ratio_string,
    validate_selection_criteria,
)

__all__ = [
    # Cache utilities
    "cleanup_old_cache",
    "generate_operation_hash",
    "get_cache_size",
    "invalidate_cache_pattern",
    "is_cache_entry_valid",
    "load_cached_result",
    "save_cached_result",
    "verify_cache_integrity",
    # Image utilities
    "calculate_memory_usage",
    "copy_image_with_context",
    "create_context_from_image",
    "create_filled_image",
    "ensure_image_mode",
    "extract_image_region",
    "get_image_hash",
    "get_pixel_at_index",
    "paste_image_region",
    "resize_image_proportional",
    # Validation utilities
    "validate_channel_list",
    "validate_color_tuple",
    "validate_expression_safe",
    "validate_image_dimensions",
    "validate_indices",
    "validate_positive_number",
    "validate_range",
    "validate_ratio_string",
    "validate_selection_criteria",
]
