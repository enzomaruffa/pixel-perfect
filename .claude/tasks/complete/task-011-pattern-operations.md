# Task 011: Pattern Operations

## Objective
Implement artistic pattern effects: Mosaic and Dither as specified in SPEC.md.

## Requirements
1. Implement Mosaic for creating mosaic/tile effects
2. Create Dither for applying various dithering patterns
3. Handle different sampling and dithering algorithms
4. Ensure high-quality artistic output with configurable parameters

## File Structure to Create
```
src/operations/
└── pattern.py          # All pattern-based operations
```

## Implementation Details

### Mosaic
**Purpose**: Create mosaic/tile effect

**Parameters**:
- `tile_size`: Size of mosaic tiles (width, height) tuple
- `gap_size`: Spacing between tiles in pixels
- `gap_color`: Color for gaps between tiles
- `sample_mode`: "average", "center", "random"

**Sample Modes**:
- **Average**: Use average color of pixels in tile area
- **Center**: Use color of center pixel in tile
- **Random**: Use color of random pixel within tile

**Behavior**: Creates discrete tile effect with optional gaps between tiles

### Dither
**Purpose**: Apply dithering patterns

**Parameters**:
- `method`: "floyd_steinberg", "ordered", "random"
- `levels`: Number of color levels per channel (e.g., 2 for black/white)
- `pattern_size`: For ordered dithering, size of dither matrix

**Dithering Methods**:
- **Floyd-Steinberg**: Error diffusion dithering with quality results
- **Ordered**: Matrix-based dithering using Bayer matrices
- **Random**: Threshold dithering with random noise

## Mosaic Algorithm
```python
def apply_mosaic(image, tile_size, gap_size, gap_color, sample_mode):
    # Calculate grid of tiles
    # For each tile position:
    #   Extract tile region
    #   Sample color based on mode
    #   Draw tile with gaps
    # Combine into final image
```

## Dithering Algorithms

### Floyd-Steinberg Error Diffusion
```python
def floyd_steinberg_dither(image, levels):
    # For each pixel:
    #   Quantize to nearest level
    #   Calculate error
    #   Distribute error to neighboring pixels:
    #     right: 7/16, below: 5/16, below-left: 3/16, below-right: 1/16
```

### Ordered Dithering
```python
def ordered_dither(image, levels, matrix_size):
    # Generate Bayer matrix of specified size
    # For each pixel:
    #   Compare to threshold from matrix
    #   Quantize based on comparison
```

### Random Dithering
```python
def random_dither(image, levels, seed):
    # For each pixel:
    #   Add random noise
    #   Quantize to levels
```

## Pattern Utilities
- `_generate_bayer_matrix(size)` - Create ordered dithering matrix
- `_quantize_value(value, levels)` - Round to nearest quantization level
- `_sample_tile_color(tile_pixels, mode)` - Extract representative color
- `_draw_tile_with_gaps(canvas, position, color, tile_size, gap_size)` - Render tile
- `_diffuse_error(error_buffer, position, error, weights)` - Error diffusion

## Color Quantization
- Support different bit depths (1-bit, 2-bit, 4-bit, etc.)
- Handle per-channel quantization
- Preserve alpha channel during dithering
- Efficient quantization lookup tables

## Quality Considerations
- **Mosaic**: Smooth color sampling, clean tile edges
- **Floyd-Steinberg**: Proper error accumulation, avoid artifacts
- **Ordered**: High-quality Bayer matrices, smooth patterns
- **Random**: Good pseudo-random distribution, seed reproducibility

## Test Cases to Implement
- **Mosaic Average Test**: Verify average color calculation accuracy
- **Floyd-Steinberg Quality Test**: Check error distribution correctness
- **Ordered Pattern Test**: Ensure Bayer matrix generates correct patterns
- **Quantization Test**: Verify proper rounding to levels

## Performance Optimizations
- Use NumPy for efficient array operations
- Pre-calculate quantization lookup tables
- Optimize error diffusion memory access patterns
- Parallel processing for independent tiles

## Success Criteria
- [ ] Both operations inherit from BaseOperation correctly
- [ ] Mosaic produces clean tile effects with proper gaps
- [ ] All dithering methods produce high-quality results
- [ ] Color quantization works correctly for different bit depths
- [ ] Error diffusion algorithms avoid visual artifacts
- [ ] Parameter validation using Pydantic models
- [ ] Proper ImageContext handling (no dimension changes)
- [ ] Cache key generation includes all parameters
- [ ] Memory estimation accounts for algorithm complexity
- [ ] Comprehensive test coverage with visual quality validation

## Dependencies
- Builds on: Task 001-010 (All previous tasks)
- Blocks: Task 012 (Enhanced Caching System)
- Additional: NumPy for efficient array operations
