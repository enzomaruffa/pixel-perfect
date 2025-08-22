# Task 004: Pixel-Level Operations

## Objective
Implement the three core pixel-level operations: PixelFilter, PixelMath, and PixelSort as specified in SPEC.md.

## Requirements
1. Implement PixelFilter for showing/hiding pixels based on index conditions
2. Create PixelMath for mathematical transformations on pixel values
3. Add PixelSort for sorting pixels within regions based on criteria
4. Ensure all operations handle RGBA channels correctly

## File Structure to Create
```
src/operations/
└── pixel.py             # All pixel-level operations
```

## Implementation Details

### PixelFilter
**Purpose**: Show/hide individual pixels based on their index position

**Parameters**:
- `condition`: "prime", "odd", "even", "fibonacci", "custom"
- `custom_expression`: String expression using 'i' for index (e.g., "i % 3 == 0")
- `fill_color`: RGBA tuple for filtered pixels (default: transparent)
- `preserve_alpha`: Keep original alpha channel
- `index_mode`: "linear" (row-major) or "2d" (separate row/col conditions)

**Behavior**:
- Linear mode: pixel at (row, col) has index = row * width + col
- 2D mode: separate conditions for row and column indices
- Prime indexing starts at 2 (0 and 1 are not prime)

### PixelMath
**Purpose**: Apply mathematical transformations to pixel values

**Parameters**:
- `expression`: Math expression using r, g, b, a, x, y variables
- `channels`: List of channels to affect ["r", "g", "b", "a"]
- `clamp`: Boolean to clamp results to valid range

**Examples**:
- `"r * 0.5 + g * 0.3"` - Custom channel mixing
- `"r + 50"` - Brightness adjustment
- `"r * (x / width)"` - Position-based gradient

### PixelSort
**Purpose**: Sort pixels within regions based on criteria

**Parameters**:
- `direction`: "horizontal", "vertical", "diagonal"
- `sort_by`: "brightness", "hue", "saturation", "red", "green", "blue"
- `threshold`: Only sort pixels meeting threshold condition
- `reverse`: Boolean for sort order

## Validation Requirements
- PixelFilter: Validate condition types and custom expressions
- PixelMath: Parse and validate mathematical expressions safely
- PixelSort: Ensure direction and sort criteria are valid
- All: Handle edge cases (1×1 images, invalid indices)

## Test Cases to Implement
- **PixelFilter Prime Test**: 4×4 grid (pixels 0-15), verify pixels 2,3,5,7,11,13 preserved
- **PixelMath Expression Test**: Various mathematical operations with bounds checking
- **PixelSort Direction Test**: Verify sorting works in all directions correctly

## Success Criteria
- [ ] All three operations inherit from BaseOperation correctly
- [ ] Parameter validation using Pydantic models
- [ ] Proper error handling for invalid expressions/parameters
- [ ] All operations update ImageContext appropriately
- [ ] Cache key generation includes all parameters
- [ ] Memory estimation accounts for operation complexity
- [ ] Comprehensive test coverage including edge cases
- [ ] Operations work with all image modes (L, RGB, RGBA)

## Dependencies
- Builds on: Task 001 (Basic Project Structure), Task 002 (Utility Functions), Task 003 (Test Infrastructure)
- Blocks: Task 005 (Row Operations) and subsequent operation tasks
