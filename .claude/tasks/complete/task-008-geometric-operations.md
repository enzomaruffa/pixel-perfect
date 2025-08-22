# Task 008: Geometric Operations

## Objective
Implement geometric transformations: GridWarp, PerspectiveStretch, and RadialStretch as specified in SPEC.md.

## Requirements
1. Implement GridWarp for wave-like distortions
2. Create PerspectiveStretch for simulating perspective distortion
3. Add RadialStretch for center-outward stretching effects
4. Handle interpolation methods for smooth transformations
5. Manage coordinate transformations and bounds checking

## File Structure to Create
```
src/operations/
└── geometric.py         # All geometric transformations
```

## Implementation Details

### GridWarp
**Purpose**: Apply wave-like distortions

**Parameters**:
- `axis`: "horizontal", "vertical", "both"
- `frequency`: Wave frequency (cycles per image width/height)
- `amplitude`: Displacement amount in pixels
- `phase`: Wave offset (0-2π)

**Mathematical Formula**:
- Horizontal: `new_x = x + amplitude * sin(2π * frequency * y / height + phase)`
- Vertical: `new_y = y + amplitude * sin(2π * frequency * x / width + phase)`

### PerspectiveStretch
**Purpose**: Simulate perspective distortion

**Parameters**:
- `top_factor`: Scale factor for top edge (1.0 = no change)
- `bottom_factor`: Scale factor for bottom edge
- `interpolation`: "nearest", "bilinear", "bicubic"

**Behavior**: Creates trapezoid effect, linearly interpolating scale from top to bottom

### RadialStretch
**Purpose**: Stretch from center point outward

**Parameters**:
- `center`: (x, y) coordinates or "auto" for image center
- `factor`: Stretch amount (>1 = expand, <1 = contract)
- `falloff`: "linear", "quadratic", "exponential"

**Mathematical Formula**:
- Distance from center: `d = sqrt((x-cx)² + (y-cy)²)`
- New distance: `new_d = d * factor * falloff_function(d/max_radius)`

## Interpolation Methods
Implement interpolation utilities:
- `_nearest_neighbor(image, x, y)` - Simple pixel lookup
- `_bilinear_interpolate(image, x, y)` - Linear interpolation between 4 pixels
- `_bicubic_interpolate(image, x, y)` - Cubic interpolation using 16 pixels
- `_sample_pixel_safe(image, x, y, method)` - Bounds-checked sampling

## Coordinate Transformation Utilities
- `_apply_warp_transform(coordinates, warp_params)` - Grid warp coordinate mapping
- `_apply_perspective_transform(coordinates, perspective_params)` - Perspective mapping
- `_apply_radial_transform(coordinates, radial_params)` - Radial mapping
- `_inverse_transform_bounds(image_size, transform_func)` - Calculate output bounds

## Validation Requirements
- Frequency must be > 0 for GridWarp
- Amplitude should be reasonable relative to image size
- Scale factors for PerspectiveStretch must be > 0
- Radial stretch factor must be > 0
- Center coordinates must be within image bounds
- Handle edge cases gracefully

## Test Cases to Implement
- **GridWarp Wave Test**: Verify sinusoidal displacement with known frequency
- **PerspectiveStretch Scale Test**: Check linear interpolation between top/bottom
- **RadialStretch Center Test**: Ensure center point remains fixed
- **Interpolation Quality Test**: Compare nearest vs bilinear vs bicubic results

## Performance Considerations
- Use NumPy for efficient coordinate array operations
- Pre-calculate transformation matrices where possible
- Optimize for common cases (axis-aligned, simple scales)
- Memory-efficient processing for large images

## Success Criteria
- [ ] All three operations inherit from BaseOperation correctly
- [ ] Mathematical transformations implemented accurately
- [ ] Multiple interpolation methods working correctly
- [ ] Proper bounds checking and edge handling
- [ ] Parameter validation using Pydantic models
- [ ] Efficient coordinate transformation algorithms
- [ ] Cache key generation includes all parameters
- [ ] Memory estimation accounts for coordinate arrays
- [ ] Comprehensive test coverage with known expected results

## Dependencies
- Builds on: Task 001-007 (All previous tasks)
- Blocks: Task 009 (Aspect Ratio Operations)
- Additional: NumPy for efficient array operations
