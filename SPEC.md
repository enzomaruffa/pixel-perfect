Image Processing Pipeline - Enhanced Design Specification
Executive Overview
A Python-based image processing framework that applies sophisticated pixel-level transformations through a composable pipeline architecture. The system treats images as RGBA data, uses Pydantic for strict parameter validation, and provides comprehensive early validation against actual image dimensions before any processing begins. Operations work at multiple granularities (pixel, row, column, block) and can be chained together to create complex artistic and practical effects.
Core Design Philosophy

Fail Fast, Fail Clear: Complete pipeline validation after image load, before any processing
Composable Operations: Simple operations combine into complex effects
Channel Aware: Full RGBA support with operations declaring channel requirements
Cache Smart: Automatic per-operation caching for iterative development
Developer First: Clear APIs, comprehensive validation, helpful error messages

Key Features

Multi-Granularity Operations: Work at pixel, row, column, or virtual block levels
Virtual Resolution: Treat high-res images as lower-res grids while preserving actual pixels
Mathematical Filters: Apply conditions based on indices (prime, odd, even, custom expressions)
Dimensional Transforms: Stretch, shift, and manipulate image geometry
Dry Run Mode: Validate entire pipeline without processing
Intelligent Caching: Hash-based caching of operation results for fast iteration

Core Architecture
ImageContext
Flows through the pipeline carrying accumulated state:
pythonclass ImageContext:
    width: int          # Current dimensions
    height: int
    channels: int       # 1 (grayscale), 3 (RGB), or 4 (RGBA)
    dtype: str          # uint8, float32, etc.
    warnings: List[str] # Non-fatal issues
    metadata: dict      # Optional EXIF, color profile, etc.
Operation Base Class
pythonclass BaseOperation:
    def validate(self, context: ImageContext) -> ImageContext:
        """Validate operation against current image state, return updated context"""

    def get_cache_key(self, image_hash: str) -> str:
        """Generate unique cache key for this operation"""

    def estimate_memory(self, context: ImageContext) -> int:
        """Estimate memory usage in bytes"""

    def apply(self, image: Image, context: ImageContext) -> Tuple[Image, ImageContext]:
        """Transform image and update context"""
Pipeline Class
Main orchestrator that:

Loads image and creates initial ImageContext
Runs validation pass through all operations (dry run capability)
Checks cache for each operation before executing
Executes operations with progress tracking
Manages debug output and final save

Operation Specifications
1. Pixel-Level Operations
PixelFilter
Purpose: Show/hide individual pixels based on their index position
Parameters:

condition: "prime", "odd", "even", "fibonacci", "custom"
custom_expression: String expression using 'i' for index (e.g., "i % 3 == 0")
fill_color: RGBA tuple for filtered pixels (default: transparent)
preserve_alpha: Keep original alpha channel
index_mode: "linear" (row-major) or "2d" (separate row/col conditions)

Behavior:

Linear mode: Treats image as 1D array, pixel at (row, col) has index = row * width + col
2D mode: Can use separate conditions for row and column indices
Prime indexing starts at 2 (0 and 1 are not prime)

PixelMath
Purpose: Apply mathematical transformations to pixel values
Parameters:

expression: Math expression using r, g, b, a, x, y variables
channels: List of channels to affect ["r", "g", "b", "a"]
clamp: Boolean to clamp results to valid range

Example: expression="r * 0.5 + g * 0.3" creates custom channel mixing
PixelSort
Purpose: Sort pixels within regions based on criteria
Parameters:

direction: "horizontal", "vertical", "diagonal"
sort_by: "brightness", "hue", "saturation", "red", "green", "blue"
threshold: Only sort pixels meeting threshold condition
reverse: Boolean for sort order

2. Row Operations
RowShift
Purpose: Translate entire rows horizontally
Parameters:

selection: "odd", "even", "prime", "every_n", "custom", "gradient"
n: For "every_n" selection (e.g., every 3rd row)
indices: List of specific row indices for "custom"
shift_amount: Pixels to shift (negative = left, positive = right)
wrap: Boolean for wraparound vs fill
fill_color: RGBA for non-wrap mode
gradient_start: For gradient mode, shift increases from 0 to max

Validation: Ensures row indices within [0, height)
RowStretch
Purpose: Duplicate rows to stretch image vertically
Parameters:

factor: Stretch multiplier (2.0 = double height)
method: "duplicate" (repeat rows) or "distribute" (spread evenly)
selection: Which rows to duplicate

RowRemove
Purpose: Delete specific rows from image
Parameters:

selection: Row selection criteria
indices: Specific rows to remove
Validation: Ensures at least 1 row remains

RowShuffle
Purpose: Randomly reorder rows
Parameters:

seed: Random seed for reproducibility
groups: Shuffle within groups of N rows

3. Column Operations
ColumnShift
Purpose: Translate columns vertically
Parameters: Mirror of RowShift but vertical
ColumnStretch
Purpose: Duplicate columns to stretch horizontally
Parameters: Mirror of RowStretch but horizontal
ColumnMirror
Purpose: Reflect columns around vertical axis
Parameters:

mode: "full" (swap all), "alternating" (every other)
pivot: Column index to mirror around

ColumnWeave
Purpose: Interlace columns from different parts of image
Parameters:

pattern: List defining source indices [0, 2, 1, 3]
repeat: Boolean to cycle pattern

4. Block Operations
BlockFilter
Purpose: Treat image as lower-resolution grid, show/hide blocks
Parameters:

block_width: Width of each block in pixels
block_height: Height of each block in pixels
keep_blocks: List of block indices to preserve (row-major indexing)
condition: "checkerboard", "diagonal", "corners", "custom"
fill_color: RGBA for hidden blocks
padding_mode: "crop", "extend", "fill" for non-divisible dimensions

Validation: Handles images not evenly divisible by block size
BlockShift
Purpose: Rearrange blocks within virtual grid
Parameters:

block_size: Tuple (width, height) for blocks
shift_map: Dict mapping source index → destination index
swap_mode: "move" or "swap" blocks

BlockRotate
Purpose: Rotate individual blocks
Parameters:

block_size: Size of blocks
rotation: 90, 180, or 270 degrees
selection: Which blocks to rotate

BlockScramble
Purpose: Randomly shuffle blocks
Parameters:

block_size: Size of blocks
seed: Random seed
exclude: List of block indices to keep in place

5. Geometric Operations
GridWarp
Purpose: Apply wave-like distortions
Parameters:

axis: "horizontal", "vertical", "both"
frequency: Wave frequency
amplitude: Displacement amount
phase: Wave offset

PerspectiveStretch
Purpose: Simulate perspective distortion
Parameters:

top_factor: Scale factor for top edge
bottom_factor: Scale factor for bottom edge
interpolation: "nearest", "bilinear", "bicubic"

RadialStretch
Purpose: Stretch from center point outward
Parameters:

center: (x, y) coordinates or "auto"
factor: Stretch amount
falloff: "linear", "quadratic", "exponential"

6. Aspect Ratio Operations
AspectStretch
Purpose: Force image to specific aspect ratio via non-uniform scaling
Parameters:

target_ratio: "1:1", "4:5", "16:9", "9:16", or custom "W:H"
method: "simple" (direct stretch) or "segment" (stretch in segments)
segment_count: For segment method, number of vertical slices
preserve_center: Boolean to minimize center distortion

AspectCrop
Purpose: Crop to achieve aspect ratio
Parameters:

target_ratio: Desired ratio
anchor: "center", "top", "bottom", "left", "right"
smart_crop: Attempt to preserve high-detail areas

AspectPad
Purpose: Add borders for aspect ratio
Parameters:

target_ratio: Desired ratio
pad_color: RGBA for padding
pad_mode: "solid", "gradient", "mirror", "blur"

7. Channel Operations
ChannelSwap
Purpose: Rearrange color channels
Parameters:

mapping: Dict like {"r": "g", "g": "b", "b": "r"}

ChannelIsolate
Purpose: Keep only specific channels
Parameters:

keep_channels: List of channels to preserve
fill_value: Value for removed channels

AlphaGenerator
Purpose: Create alpha channel from image content
Parameters:

source: "luminance", "saturation", "specific_color"
threshold: Value for alpha generation
invert: Boolean to invert alpha

8. Pattern Operations
Mosaic
Purpose: Create mosaic/tile effect
Parameters:

tile_size: Size of mosaic tiles
gap_size: Spacing between tiles
gap_color: Color for gaps
sample_mode: "average", "center", "random"

Dither
Purpose: Apply dithering patterns
Parameters:

method: "floyd_steinberg", "ordered", "random"
levels: Number of color levels
pattern_size: For ordered dithering

Sample Pipeline Usage
python# Basic artistic effect combining multiple operations
pipeline = Pipeline("portrait.jpg", debug=True)
result = (pipeline
    # First, create a grid effect
    .add(BlockFilter(
        block_width=8,
        block_height=8,
        condition="checkerboard",
        fill_color=(0, 0, 0, 128)
    ))
    # Shift odd rows for glitch effect
    .add(RowShift(
        selection="odd",
        shift_amount=3,
        wrap=True
    ))
    # Keep only prime-indexed pixels in affected areas
    .add(PixelFilter(
        condition="prime",
        fill_color=(255, 0, 0, 255),
        index_mode="linear"
    ))
    # Stretch to Instagram square
    .add(AspectStretch(
        target_ratio="1:1",
        method="segment",
        segment_count=5,
        preserve_center=True
    ))
    .execute("output.png"))

# Complex geometric transformation
pipeline = Pipeline("input.jpg")
pipeline.add(ColumnStretch(factor=1.5, method="duplicate"))
pipeline.add(RowShift(selection="gradient", gradient_start=0, shift_amount=10))
pipeline.add(GridWarp(axis="horizontal", frequency=3, amplitude=20))

# Validation will catch errors early
try:
    result = pipeline.execute("warped.png", dry_run=True)  # Validation only
    if result.warnings:
        print(f"Warnings: {result.warnings}")
    result = pipeline.execute("warped.png")  # Actually process
except ValidationError as e:
    print(f"Pipeline invalid: {e}")

# Using cache for iterative development
pipeline = Pipeline("test.jpg", cache_dir="./cache")
pipeline.add(ExpensiveOperation())  # Will cache result
pipeline.add(RowShift(shift_amount=2))  # Tweak this parameter
# Re-running will use cached result from ExpensiveOperation
result = pipeline.execute("output.png")
Testing Strategy
Synthetic Test Image Generation
pythondef create_test_images():
    # Numbered grid: each pixel value = row * width + col
    grid_8x8 = create_numbered_grid(8, 8)

    # Gradient for stretch testing
    gradient_h = create_gradient(8, 4, direction="horizontal")

    # Checkerboard for pattern operations
    checker = create_checkerboard(16, 16, square_size=4)

    # Single color channels for channel operations
    red_only = create_channel_test(8, 8, channels="r")

    # High frequency pattern for aliasing tests
    stripes = create_stripes(32, 32, width=1, orientation="vertical")
Operation Test Patterns
PixelFilter Prime Test:

Input: 4×4 grid (pixels 0-15)
Expected: Pixels 2, 3, 5, 7, 11, 13 preserved
Verify: Exact preservation, correct fill color

RowShift Wrap Test:

Input: 4×4 with unique row values
Shift odd rows left by 1 with wrap
Verify: Circular shift on rows 1, 3 only

BlockFilter Division Test:

Input: 10×10 image, 3×3 blocks
Expected: Handles padding mode correctly
Verify: Edge blocks use specified padding

ColumnStretch Factor Test:

Input: 4×4 pattern
Stretch 2× horizontal
Verify: Each column duplicated exactly once

AspectStretch Segment Test:

Input: 15×10 image
Target: 1:1 with 3 segments
Verify: Each segment stretched uniformly

Validation Test Cases

Dimension Validation: Row operations with out-of-bounds indices
Channel Requirements: Grayscale operation on RGBA image
Memory Limits: Extreme stretch factors
Block Divisibility: Non-even block sizes with crop/pad modes
Pipeline Consistency: Operations that would create 0×0 result
Cache Integrity: Modified parameters trigger recalculation

Edge Cases to Test

1×1 images (minimum valid)
Extreme aspect ratios (1×1000)
Single channel images
Images with existing alpha
Operations resulting in >4GB memory usage
Prime numbers beyond image size
Negative shift amounts
Non-integer block positions
