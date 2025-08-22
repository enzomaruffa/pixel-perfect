# Task 007: Block Operations

## Objective
Implement block-based operations that treat images as lower-resolution virtual grids: BlockFilter, BlockShift, BlockRotate, and BlockScramble as specified in SPEC.md.

## Requirements
1. Implement BlockFilter for showing/hiding blocks in virtual grid
2. Create BlockShift for rearranging blocks within grid
3. Add BlockRotate for rotating individual blocks
4. Implement BlockScramble for randomly shuffling blocks
5. Handle non-divisible dimensions with padding modes

## File Structure to Create
```
src/operations/
└── block.py             # All block-based operations
```

## Implementation Details

### BlockFilter
**Purpose**: Treat image as lower-resolution grid, show/hide blocks

**Parameters**:
- `block_width`: Width of each block in pixels
- `block_height`: Height of each block in pixels
- `keep_blocks`: List of block indices to preserve (row-major indexing)
- `condition`: "checkerboard", "diagonal", "corners", "custom"
- `fill_color`: RGBA for hidden blocks
- `padding_mode`: "crop", "extend", "fill" for non-divisible dimensions

**Validation**: Handles images not evenly divisible by block size

### BlockShift
**Purpose**: Rearrange blocks within virtual grid

**Parameters**:
- `block_size`: Tuple (width, height) for blocks
- `shift_map`: Dict mapping source index → destination index
- `swap_mode`: "move" or "swap" blocks

**Behavior**: Reorganizes image content at block level

### BlockRotate
**Purpose**: Rotate individual blocks

**Parameters**:
- `block_size`: Size of blocks
- `rotation`: 90, 180, or 270 degrees
- `selection`: Which blocks to rotate (uses block selection logic)

### BlockScramble
**Purpose**: Randomly shuffle blocks

**Parameters**:
- `block_size`: Size of blocks
- `seed`: Random seed for reproducibility
- `exclude`: List of block indices to keep in place

## Block Grid Logic
Implement shared block utilities:
- `_calculate_grid_dimensions(image_size, block_size)` - Get grid rows/cols
- `_get_block_bounds(block_index, grid_dims, block_size)` - Get pixel coordinates
- `_extract_block(image, bounds)` - Extract block as separate image
- `_place_block(image, block, bounds)` - Place block back into image
- `_handle_padding(image, block_size, padding_mode)` - Handle non-divisible dimensions

## Padding Modes for Non-Divisible Dimensions
- **crop**: Remove partial blocks at edges
- **extend**: Extend image to next block boundary (fill with edge pixels)
- **fill**: Extend image to next block boundary (fill with specified color)

## Block Selection Logic
Reuse selection patterns from row/column operations:
- "odd": blocks at odd indices
- "even": blocks at even indices
- "checkerboard": alternating pattern
- "diagonal": blocks along diagonal
- "corners": corner blocks only
- "custom": explicit list of block indices

## Validation Requirements
- Block size must be > 0 and <= image dimensions
- Block indices must be within calculated grid
- Rotation must be 90, 180, or 270 degrees
- Shift map indices must be valid block positions
- Handle edge cases of blocks larger than image

## Test Cases to Implement
- **BlockFilter Division Test**: 10×10 image with 3×3 blocks, verify padding handling
- **BlockShift Mapping Test**: Verify blocks move to correct positions
- **BlockRotate Degrees Test**: Ensure 90°, 180°, 270° rotations work correctly
- **BlockScramble Reproducibility**: Same seed produces same result

## Success Criteria
- [ ] All four operations inherit from BaseOperation correctly
- [ ] Block grid calculation handles non-divisible dimensions
- [ ] Padding modes work correctly for edge cases
- [ ] Block extraction/placement preserves pixel data
- [ ] Parameter validation using Pydantic models
- [ ] Proper error handling for invalid block sizes
- [ ] Cache key generation includes all parameters
- [ ] Memory estimation accounts for block operations
- [ ] Comprehensive test coverage including edge cases

## Dependencies
- Builds on: Task 001-006 (All previous tasks)
- Blocks: Task 008 (Geometric Operations)
