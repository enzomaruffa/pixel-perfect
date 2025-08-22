# Task 009: Aspect Ratio Operations

## Objective
Implement aspect ratio transformations: AspectStretch, AspectCrop, and AspectPad as specified in SPEC.md.

## Requirements
1. Implement AspectStretch for forcing specific aspect ratios via non-uniform scaling
2. Create AspectCrop for achieving aspect ratios through intelligent cropping
3. Add AspectPad for achieving aspect ratios by adding borders
4. Handle common aspect ratios and custom ratio parsing
5. Implement smart cropping algorithms for content preservation

## File Structure to Create
```
src/operations/
└── aspect.py           # All aspect ratio operations
```

## Implementation Details

### AspectStretch
**Purpose**: Force image to specific aspect ratio via non-uniform scaling

**Parameters**:
- `target_ratio`: "1:1", "4:5", "16:9", "9:16", or custom "W:H"
- `method`: "simple" (direct stretch) or "segment" (stretch in segments)
- `segment_count`: For segment method, number of vertical slices
- `preserve_center`: Boolean to minimize center distortion

**Methods**:
- **Simple**: Direct non-uniform scaling to target ratio
- **Segment**: Divide image into vertical segments, stretch each proportionally to reduce distortion

### AspectCrop
**Purpose**: Crop to achieve aspect ratio

**Parameters**:
- `target_ratio`: Desired ratio
- `anchor`: "center", "top", "bottom", "left", "right"
- `smart_crop`: Attempt to preserve high-detail areas

**Smart Crop Algorithm**:
- Calculate gradient/edge density in different regions
- Choose crop area that preserves maximum detail
- Fallback to anchor-based cropping if analysis fails

### AspectPad
**Purpose**: Add borders for aspect ratio

**Parameters**:
- `target_ratio`: Desired ratio
- `pad_color`: RGBA for padding
- `pad_mode`: "solid", "gradient", "mirror", "blur"

**Pad Modes**:
- **Solid**: Fill with specified color
- **Gradient**: Fade from edge colors to pad color
- **Mirror**: Reflect edge pixels outward
- **Blur**: Blur and extend edge regions

## Aspect Ratio Utilities
- `_parse_ratio_string(ratio)` - Convert "16:9" to decimal ratio
- `_calculate_target_dimensions(current_size, target_ratio)` - Get new dimensions
- `_get_crop_bounds(image_size, target_size, anchor)` - Calculate crop rectangle
- `_analyze_image_detail(image, regions)` - Smart crop detail analysis
- `_generate_padding(pad_mode, edge_pixels, pad_size)` - Create padding content

## Common Aspect Ratios
Pre-define common ratios:
- Square: 1:1 (1.0)
- Portrait: 4:5 (0.8), 9:16 (0.5625)
- Landscape: 16:9 (1.7778), 21:9 (2.3333)
- Photo: 3:2 (1.5), 4:3 (1.3333)

## Validation Requirements
- Target ratio must be > 0
- Segment count must be >= 1 for segment method
- Anchor must be valid position
- Pad color must be valid RGBA tuple
- Handle edge cases (already correct ratio, extreme ratios)

## Test Cases to Implement
- **AspectStretch Segment Test**: 15×10 image to 1:1 with 3 segments
- **AspectCrop Smart Test**: Verify detail preservation in smart crop
- **AspectPad Modes Test**: Ensure all padding modes work correctly
- **Ratio Parsing Test**: Verify "16:9" converts to correct decimal

## Algorithm Details

### Segment Stretching
```python
def apply_segment_stretch(image, target_ratio, segment_count):
    # Divide image into vertical segments
    # Calculate stretch factor for each segment
    # Apply non-uniform scaling per segment
    # Blend segment boundaries
```

### Smart Crop Analysis
```python
def analyze_detail_regions(image, crop_candidates):
    # Calculate gradient magnitude for each region
    # Weight by position (center bias)
    # Return region with highest detail score
```

## Success Criteria
- [ ] All three operations inherit from BaseOperation correctly
- [ ] Aspect ratio parsing handles common and custom ratios
- [ ] Segment stretching reduces distortion compared to simple stretch
- [ ] Smart crop preserves more detail than anchor-based crop
- [ ] All padding modes produce expected visual results
- [ ] Parameter validation using Pydantic models
- [ ] Proper ImageContext updates for dimension changes
- [ ] Cache key generation includes all parameters
- [ ] Memory estimation accounts for dimension changes
- [ ] Comprehensive test coverage including edge cases

## Dependencies
- Builds on: Task 001-008 (All previous tasks including Geometric Operations)
- Blocks: Task 010 (Channel Operations)
