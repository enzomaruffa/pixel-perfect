# Task 010: Channel Operations

## Objective
Implement color channel manipulations: ChannelSwap, ChannelIsolate, and AlphaGenerator as specified in SPEC.md.

## Requirements
1. Implement ChannelSwap for rearranging color channels
2. Create ChannelIsolate for keeping only specific channels
3. Add AlphaGenerator for creating alpha channels from image content
4. Handle different image modes (L, RGB, RGBA) correctly
5. Ensure proper channel validation and conversion

## File Structure to Create
```
src/operations/
└── channel.py          # All channel manipulation operations
```

## Implementation Details

### ChannelSwap
**Purpose**: Rearrange color channels

**Parameters**:
- `mapping`: Dict like {"r": "g", "g": "b", "b": "r"} - maps destination to source

**Behavior**:
- Accepts mappings for "r", "g", "b", "a" channels
- Automatically handles image mode conversion if needed
- Preserves channels not mentioned in mapping

**Examples**:
- `{"r": "b", "b": "r"}` - Swap red and blue channels
- `{"r": "g", "g": "r"}` - Swap red and green channels

### ChannelIsolate
**Purpose**: Keep only specific channels

**Parameters**:
- `keep_channels`: List of channels to preserve ["r", "g", "b", "a"]
- `fill_value`: Value for removed channels (0-255 for uint8)

**Behavior**:
- Sets non-preserved channels to fill_value
- Automatically converts to RGBA if alpha isolation requested
- Updates ImageContext.channels if needed

**Examples**:
- `keep_channels=["r"]` - Keep only red channel, zero out others
- `keep_channels=["a"]` - Keep only alpha, zero RGB

### AlphaGenerator
**Purpose**: Create alpha channel from image content

**Parameters**:
- `source`: "luminance", "saturation", "specific_color"
- `threshold`: Value for alpha generation (0-255)
- `invert`: Boolean to invert alpha values
- `color_target`: For "specific_color" mode, RGB tuple to target

**Source Modes**:
- **Luminance**: Use brightness as alpha (bright = opaque)
- **Saturation**: Use color saturation as alpha (saturated = opaque)
- **Specific Color**: Distance from target color determines alpha

## Channel Utilities
- `_ensure_image_mode(image, required_mode)` - Convert image mode safely
- `_extract_channel(image, channel_name)` - Get single channel as array
- `_combine_channels(channels_dict)` - Merge channels into image
- `_calculate_luminance(r, g, b)` - Standard luminance formula
- `_calculate_saturation(r, g, b)` - HSV saturation calculation
- `_color_distance(pixel, target)` - Euclidean distance in RGB space

## Channel Validation
- Validate channel names ("r", "g", "b", "a")
- Ensure mapping doesn't create circular references
- Check fill values are in valid range for dtype
- Handle grayscale to RGB conversion properly
- Verify color targets are valid RGB tuples

## Image Mode Handling
- **L (Grayscale)**: Convert to RGB for most operations
- **RGB**: Standard 3-channel handling
- **RGBA**: Full 4-channel support
- Auto-conversion when operations require specific modes

## Test Cases to Implement
- **ChannelSwap RGB Test**: Verify R↔B swap produces expected colors
- **ChannelIsolate Red Test**: Ensure only red channel preserved
- **AlphaGenerator Luminance Test**: Bright areas become opaque
- **Mode Conversion Test**: L→RGB→RGBA conversions work correctly

## Algorithm Details

### Luminance Calculation
```python
def calculate_luminance(r, g, b):
    # Standard ITU-R BT.709 formula
    return 0.299 * r + 0.587 * g + 0.114 * b
```

### Saturation Calculation
```python
def calculate_saturation(r, g, b):
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    if max_val == 0:
        return 0
    return (max_val - min_val) / max_val
```

### Color Distance
```python
def color_distance(pixel, target):
    return sqrt(sum((p - t) ** 2 for p, t in zip(pixel, target)))
```

## Success Criteria
- [ ] All three operations inherit from BaseOperation correctly
- [ ] Channel mapping works correctly for all combinations
- [ ] Image mode conversions handled automatically
- [ ] Alpha generation produces expected results for all source modes
- [ ] Parameter validation using Pydantic models
- [ ] Proper ImageContext updates for channel changes
- [ ] Cache key generation includes all parameters
- [ ] Memory estimation accounts for mode conversions
- [ ] Comprehensive test coverage including mode edge cases
- [ ] Operations preserve image quality during conversions

## Dependencies
- Builds on: Task 001-009 (All previous tasks)
- Blocks: Task 011 (Pattern Operations)
