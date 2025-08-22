# Task 003: Test Infrastructure and Synthetic Image Generation

## Objective
Create comprehensive testing infrastructure with synthetic image generation for validating all operations systematically.

## Requirements
1. Implement synthetic test image generators as specified in SPEC.md
2. Create base test classes for operation validation
3. Set up test patterns for different operation categories
4. Add automated edge case testing framework

## File Structure to Create
```
src/utils/
└── test_images.py       # Synthetic image generation

tests/
├── __init__.py
├── conftest.py          # Pytest configuration and fixtures
├── test_context.py      # ImageContext tests
├── test_pipeline.py     # Pipeline tests
└── operations/
    ├── __init__.py
    └── base_test.py     # Base classes for operation tests
```

## Implementation Details

### test_images.py - Synthetic Image Generators
- `create_numbered_grid(width, height)` - Each pixel value = row * width + col
- `create_gradient(width, height, direction)` - Gradient for stretch testing
- `create_checkerboard(width, height, square_size)` - Pattern operations testing
- `create_channel_test(width, height, channels)` - Single color channels
- `create_stripes(width, height, width, orientation)` - High frequency patterns
- `create_test_suite()` - Generate complete test image set

### base_test.py - Operation Test Framework
- `BaseOperationTest` class with common test patterns
- `validate_operation_contract(operation, context)` - Test abstract methods
- `test_edge_cases(operation)` - 1x1 images, extreme aspect ratios
- `test_memory_estimation(operation, context)` - Memory calculation accuracy
- `test_cache_key_generation(operation)` - Cache key uniqueness

### Test Patterns from SPEC.md
- **PixelFilter Prime Test**: 4×4 grid, verify pixels 2,3,5,7,11,13 preserved
- **RowShift Wrap Test**: 4×4 unique rows, shift odd rows left by 1
- **BlockFilter Division Test**: 10×10 image with 3×3 blocks
- **ColumnStretch Factor Test**: 4×4 pattern stretched 2× horizontal
- **AspectStretch Segment Test**: 15×10 image to 1:1 with 3 segments

### Edge Cases Framework
- 1×1 images (minimum valid size)
- Extreme aspect ratios (1×1000, 1000×1)
- Single channel images (grayscale)
- Images with existing alpha channel
- Non-divisible dimensions for block operations
- Out-of-bounds indices for row/column operations

## Success Criteria
- [ ] All synthetic image generators working correctly
- [ ] Base test classes support all operation types
- [ ] Edge case testing framework covers all scenarios
- [ ] Test patterns match SPEC.md specifications exactly
- [ ] All tests pass with proper assertion messages
- [ ] Test coverage setup for operation validation
- [ ] Performance benchmarking capabilities

## Dependencies
- Builds on: Task 001 (Basic Project Structure), Task 002 (Utility Functions)
- Required for: All operation implementation tasks
