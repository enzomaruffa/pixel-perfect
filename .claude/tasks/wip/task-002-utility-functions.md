# Task 002: Core Utility Functions

## Objective
Implement essential utility functions for validation, image processing, and caching that will be used across all operations.

## Requirements
1. Create validation utilities for common parameter validation
2. Implement image helper functions for loading, conversion, and basic manipulation
3. Add caching utilities for operation result storage and retrieval
4. Create hash generation functions for cache keys

## File Structure to Create
```
src/utils/
├── validation.py    # Parameter validation helpers
├── image.py         # Image processing utilities
└── cache.py         # Caching system utilities
```

## Implementation Details

### validation.py
- `validate_image_dimensions(width, height)` - Check valid dimensions
- `validate_indices(indices, max_value)` - Validate index lists
- `validate_color_tuple(color)` - Check RGBA color format
- `validate_ratio_string(ratio)` - Parse aspect ratio strings
- `validate_selection_criteria(selection, context)` - Check row/column selection

### image.py
- `ensure_image_mode(image, required_mode)` - Convert image modes
- `get_pixel_at_index(image, index, mode='linear')` - Index-based pixel access
- `create_filled_image(width, height, color)` - Generate solid color images
- `copy_image_with_context(image, context)` - Deep copy with metadata
- `calculate_memory_usage(width, height, channels, dtype)` - Memory estimation

### cache.py
- `generate_operation_hash(operation, image_hash)` - Create cache keys
- `save_cached_result(cache_dir, key, image, context)` - Store results
- `load_cached_result(cache_dir, key)` - Retrieve cached results
- `cleanup_old_cache(cache_dir, max_age_days)` - Cache maintenance

## Success Criteria
- [ ] All utility functions implemented and tested
- [ ] Functions handle edge cases (1x1 images, extreme values)
- [ ] Proper error handling with custom exceptions
- [ ] Type hints and documentation for all functions
- [ ] Integration with existing ImageContext and BaseOperation
- [ ] All utilities pass unit tests

## Dependencies
- Builds on: Task 001 (Basic Project Structure)
- Required for: All subsequent operation tasks
