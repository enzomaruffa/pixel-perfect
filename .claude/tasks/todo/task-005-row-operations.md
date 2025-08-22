# Task 005: Row Operations

## Objective
Implement row-based image transformations: RowShift, RowStretch, RowRemove, and RowShuffle as specified in SPEC.md.

## Requirements
1. Implement RowShift for translating entire rows horizontally
2. Create RowStretch for duplicating rows to stretch image vertically
3. Add RowRemove for deleting specific rows from images
4. Implement RowShuffle for randomly reordering rows
5. Handle all row selection criteria consistently across operations

## File Structure to Create
```
src/operations/
└── row.py               # All row-based operations
```

## Implementation Details

### RowShift
**Purpose**: Translate entire rows horizontally

**Parameters**:
- `selection`: "odd", "even", "prime", "every_n", "custom", "gradient"
- `n`: For "every_n" selection (e.g., every 3rd row)
- `indices`: List of specific row indices for "custom"
- `shift_amount`: Pixels to shift (negative = left, positive = right)
- `wrap`: Boolean for wraparound vs fill
- `fill_color`: RGBA for non-wrap mode
- `gradient_start`: For gradient mode, shift increases from 0 to max

**Validation**: Ensures row indices within [0, height)

### RowStretch
**Purpose**: Duplicate rows to stretch image vertically

**Parameters**:
- `factor`: Stretch multiplier (2.0 = double height)
- `method`: "duplicate" (repeat rows) or "distribute" (spread evenly)
- `selection`: Which rows to duplicate

**Behavior**: Updates ImageContext height appropriately

### RowRemove
**Purpose**: Delete specific rows from image

**Parameters**:
- `selection`: Row selection criteria
- `indices`: Specific rows to remove

**Validation**: Ensures at least 1 row remains after removal

### RowShuffle
**Purpose**: Randomly reorder rows

**Parameters**:
- `seed`: Random seed for reproducibility
- `groups`: Shuffle within groups of N rows

## Common Row Selection Logic
Implement shared `_select_rows(selection, context, **kwargs)` function:
- "odd": rows 1, 3, 5, ...
- "even": rows 0, 2, 4, ...
- "prime": rows at prime indices (2, 3, 5, 7, ...)
- "every_n": every nth row starting from 0
- "custom": explicit list of row indices
- "gradient": for RowShift, applies increasing shift amounts

## Validation Requirements
- Row indices must be within [0, height)
- Selection criteria must be valid
- RowRemove must leave at least 1 row
- RowStretch factor must be > 0
- Handle edge case of 1-row images

## Test Cases to Implement
- **RowShift Wrap Test**: 4×4 with unique row values, shift odd rows left by 1 with wrap
- **RowStretch Factor Test**: Verify rows duplicated correctly with different factors
- **RowRemove Edge Test**: Ensure at least 1 row always remains
- **RowShuffle Reproducibility**: Same seed produces same result

## Success Criteria
- [ ] All four operations inherit from BaseOperation correctly
- [ ] Shared row selection logic used consistently
- [ ] Parameter validation using Pydantic models
- [ ] Proper ImageContext updates for dimension changes
- [ ] Cache key generation includes all parameters
- [ ] Memory estimation accounts for row transformations
- [ ] Comprehensive test coverage including edge cases
- [ ] Operations handle 1-row images gracefully

## Dependencies
- Builds on: Task 001-004 (Basic Structure, Utils, Tests, Pixel Operations)
- Blocks: Task 006 (Column Operations)
