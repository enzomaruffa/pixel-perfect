# Task 026: Remove Difficulty Indicators

**Status:** Todo
**Priority:** High (Week 1)
**Category:** UI Cleanup

## Problem

The 🟢🟡🔴 difficulty indicators for operations are:
- Confusing and subjective
- Not helpful for users
- Add visual clutter
- Don't provide actionable information

## Current Implementation

Located in `src/ui/components/operation_browser.py`:
```python
difficulty_colors = {"Beginner": "🟢", "Intermediate": "🟡", "Advanced": "🔴"}
```

Operations are categorized with arbitrary difficulty levels that don't help users understand what the operations actually do.

## Solution

### 1. Remove Difficulty Indicators
- Remove all difficulty color dots from operation browser
- Remove difficulty classification from operation registry
- Clean up visual presentation

### 2. Replace with Clear Categories
- Group operations by functional category instead:
  - **Pixel Effects** - PixelFilter, PixelMath, PixelSort
  - **Spatial** - RowShift, ColumnShift, GridWarp, etc.
  - **Color** - ChannelSwap, ChannelIsolate, AlphaGenerator
  - **Geometric** - AspectStretch, AspectCrop, AspectPad
  - **Artistic** - Mosaic, Dither, BlockFilter

### 3. Better Operation Descriptions
- Focus on what the operation does, not how "difficult" it is
- Clear, concise descriptions of functionality
- Practical use cases instead of difficulty ratings

## Implementation

### Files to Modify:
- `src/ui/components/operation_browser.py` - remove difficulty display
- `src/ui/components/operation_registry.py` - update categorization
- Any other files referencing difficulty levels

### Key Changes:

1. **Remove Difficulty Display:**
   ```python
   # Remove this line:
   # st.write(f"{difficulty_colors.get(op_info.get('difficulty', 'Intermediate'), '⚪')} **{op_info['name']}**")

   # Replace with:
   st.write(f"**{op_info['name']}**")
   ```

2. **Update Operation Categories:**
   ```python
   OPERATION_CATEGORIES = {
       "Pixel Effects": ["PixelFilter", "PixelMath", "PixelSort"],
       "Spatial": ["RowShift", "ColumnShift", "GridWarp", "PerspectiveStretch", "RadialStretch"],
       "Color": ["ChannelSwap", "ChannelIsolate", "AlphaGenerator"],
       "Geometric": ["AspectStretch", "AspectCrop", "AspectPad"],
       "Artistic": ["Mosaic", "Dither", "BlockFilter"]
   }
   ```

3. **Clean Operation Display:**
   - Remove difficulty references
   - Group by functional category
   - Focus on clear naming and descriptions

## Acceptance Criteria

- [ ] No difficulty color dots visible anywhere
- [ ] Operations grouped by functional category
- [ ] Clear, helpful operation descriptions
- [ ] Cleaner visual presentation
- [ ] All difficulty references removed from code

## Testing

- Browse all operation categories
- Verify no difficulty indicators appear
- Check operation descriptions are clear and helpful
- Ensure categorization makes logical sense
