# Task 006: Column Operations

## Objective
Implement column-based image transformations: ColumnShift, ColumnStretch, ColumnMirror, and ColumnWeave as specified in SPEC.md.

## Requirements
1. Implement ColumnShift for translating columns vertically (mirror of RowShift)
2. Create ColumnStretch for duplicating columns to stretch horizontally (mirror of RowStretch)
3. Add ColumnMirror for reflecting columns around vertical axis
4. Implement ColumnWeave for interlacing columns from different parts of image
5. Reuse selection logic patterns from row operations

## File Structure to Create
```
src/operations/
└── column.py            # All column-based operations
```

## Implementation Details

### ColumnShift
**Purpose**: Translate columns vertically

**Parameters**: Mirror of RowShift but vertical
- `selection`: "odd", "even", "prime", "every_n", "custom", "gradient"
- `n`: For "every_n" selection
- `indices`: List of specific column indices for "custom"
- `shift_amount`: Pixels to shift (negative = up, positive = down)
- `wrap`: Boolean for wraparound vs fill
- `fill_color`: RGBA for non-wrap mode
- `gradient_start`: For gradient mode

**Validation**: Ensures column indices within [0, width)

### ColumnStretch
**Purpose**: Duplicate columns to stretch horizontally

**Parameters**: Mirror of RowStretch but horizontal
- `factor`: Stretch multiplier (2.0 = double width)
- `method`: "duplicate" (repeat columns) or "distribute" (spread evenly)
- `selection`: Which columns to duplicate

**Behavior**: Updates ImageContext width appropriately

### ColumnMirror
**Purpose**: Reflect columns around vertical axis

**Parameters**:
- `mode`: "full" (swap all), "alternating" (every other)
- `pivot`: Column index to mirror around

**Behavior**: Creates symmetric patterns without changing dimensions

### ColumnWeave
**Purpose**: Interlace columns from different parts of image

**Parameters**:
- `pattern`: List defining source indices [0, 2, 1, 3] - maps destination to source
- `repeat`: Boolean to cycle pattern if shorter than width

**Example**: pattern=[2, 0, 1] takes column 2 → position 0, column 0 → position 1, etc.

## Common Column Selection Logic
Implement shared `_select_columns(selection, context, **kwargs)` function:
- Reuse patterns from row operations but for columns
- "odd": columns 1, 3, 5, ...
- "even": columns 0, 2, 4, ...
- "prime": columns at prime indices
- "every_n": every nth column
- "custom": explicit list of column indices

## Validation Requirements
- Column indices must be within [0, width)
- Selection criteria must be valid
- ColumnMirror pivot must be valid column index
- ColumnWeave pattern indices must be within [0, width)
- Handle edge case of 1-column images

## Test Cases to Implement
- **ColumnShift Wrap Test**: Mirror of row shift test but vertical
- **ColumnStretch Factor Test**: Verify columns duplicated correctly
- **ColumnMirror Symmetry Test**: Ensure proper reflection around pivot
- **ColumnWeave Pattern Test**: Verify pattern mapping works correctly

## Success Criteria
- [ ] All four operations inherit from BaseOperation correctly
- [ ] Shared column selection logic mirrors row selection patterns
- [ ] Parameter validation using Pydantic models
- [ ] Proper ImageContext updates for dimension changes
- [ ] Cache key generation includes all parameters
- [ ] Memory estimation accounts for column transformations
- [ ] Comprehensive test coverage including edge cases
- [ ] Operations handle 1-column images gracefully
- [ ] Mirror/weave operations produce expected visual results

## Dependencies
- Builds on: Task 001-005 (All previous tasks including Row Operations)
- Blocks: Task 007 (Block Operations)
