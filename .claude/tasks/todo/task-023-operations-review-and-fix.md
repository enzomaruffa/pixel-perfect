# Task 023: Operations Review and Enhancement

## Objective
Comprehensively review all operations for completeness, consistency, and missing functionality. Fix parameter mismatches, add missing options, and ensure all operations follow established patterns.

## Current Issues Discovered

### 🔴 Critical Issues
1. **RowShift/RowFilter Selection Options**
   - Test expects `selection="all"` but valid options are: 'odd', 'even', 'prime', 'every_n', 'custom', 'gradient'
   - Missing common "all" option across row/column operations

2. **Parameter Signature Mismatches**
   - PixelFilter constructor doesn't match expected parameters in tests/presets
   - Operation registry may have outdated parameter definitions
   - Pydantic models may not match actual operation constructors

### 🟡 Consistency Issues
3. **Missing Parameter Options**
   - Some operations lack comprehensive selection modes
   - Inconsistent parameter naming across similar operations
   - Missing convenience options (like "all", "none", "random")

4. **Operation Categories Gaps**
   - Some operation types may be missing common variants
   - Incomplete parameter ranges or validation rules
   - Missing advanced options for power users

## Implementation Plan

### Phase 1: Operations Audit
1. **Complete Operations Inventory**
   - List all implemented operations by category
   - Document current parameter signatures
   - Identify missing common options

2. **Parameter Consistency Review**
   - Compare similar operations (Row vs Column variants)
   - Standardize naming conventions
   - Ensure selection modes are consistent

3. **Test Coverage Analysis**
   - Identify operations with insufficient tests
   - Find parameter combinations not tested
   - Document edge cases not covered

### Phase 2: Missing Functionality
1. **Add Missing Selection Options**
   ```python
   # Add "all" option to selection-based operations
   SelectionLiteral = Literal["all", "odd", "even", "prime", "every_n", "custom", "gradient"]
   ```

2. **Enhance Operation Parameters**
   - Add convenience options and shortcuts
   - Implement missing advanced parameters
   - Add parameter validation and helpful defaults

3. **Operation Variants**
   - Identify missing operation variants
   - Add commonly requested functionality
   - Ensure feature parity across categories

### Phase 3: Implementation Fixes
1. **Parameter Signature Fixes**
   - Fix constructor signatures to match Pydantic models
   - Update operation registry with correct defaults
   - Ensure UI form generators work with all operations

2. **Enhanced Validation**
   - Add comprehensive parameter validation
   - Better error messages for invalid parameters
   - Range checking and dependency validation

3. **Documentation Updates**
   - Update docstrings with current parameters
   - Add usage examples for complex operations
   - Document parameter interactions and constraints

## Operations Review Checklist

### Pixel Operations
- [ ] **PixelFilter**: Fix parameter signature, add "all" selection
- [ ] **PixelMath**: Validate expression parsing, add safety checks
- [ ] **PixelSort**: Ensure all sort modes work correctly

### Row Operations
- [ ] **RowShift**: Add "all" selection option, validate patterns
- [ ] **RowStretch**: Check stretch algorithms and edge cases
- [ ] **RowRemove**: Validate selection and removal logic
- [ ] **RowShuffle**: Ensure randomization options work

### Column Operations
- [ ] **ColumnShift**: Mirror RowShift functionality exactly
- [ ] **ColumnStretch**: Ensure parity with RowStretch
- [ ] **ColumnMirror**: Test all mirror modes
- [ ] **ColumnWeave**: Validate weaving patterns

### Block Operations
- [ ] **BlockFilter**: Fix type issues, validate grid logic
- [ ] **BlockShift**: Test movement patterns
- [ ] **BlockRotate**: Fix rotation logic and edge cases
- [ ] **BlockScramble**: Validate scrambling algorithms

### Geometric Operations
- [ ] **GridWarp**: Fix string concatenation, test warp modes
- [ ] **PerspectiveStretch**: Validate perspective calculations
- [ ] **RadialStretch**: Test radial transformations

### Aspect Operations
- [ ] **AspectStretch**: Test stretch modes and ratios
- [ ] **AspectCrop**: Validate crop positioning
- [ ] **AspectPad**: Test padding options and colors

### Channel Operations
- [ ] **ChannelSwap**: Fix color tuple handling
- [ ] **ChannelIsolate**: Test isolation modes
- [ ] **AlphaGenerator**: Validate alpha generation methods

### Pattern Operations
- [ ] **Mosaic**: Test mosaic generation
- [ ] **Dither**: Validate dithering algorithms

## Specific Fixes Needed

### 1. Add "all" Selection Option
```python
class SelectionConfig(BaseModel):
    selection: Literal["all", "odd", "even", "prime", "every_n", "custom", "gradient"]

    def get_indices(self, total_count: int) -> list[int]:
        if self.selection == "all":
            return list(range(total_count))
        # ... existing logic
```

### 2. Fix Parameter Signatures
```python
# Ensure constructor matches Pydantic model
class PixelFilter(BaseOperation):
    def __init__(self, condition: str, fill_color: tuple[int, int, int, int],
                 preserve_alpha: bool = True, index_mode: str = "linear",
                 custom_expression: str | None = None):
```

### 3. Enhance Validation
```python
class RowShiftConfig(BaseModel):
    pattern: Literal["linear", "wave", "random", "custom"]
    amplitude: float = Field(ge=0, le=1000, description="Shift amount in pixels")
    frequency: float = Field(ge=0, le=1, description="Pattern frequency")

    @model_validator(mode='after')
    def validate_custom_pattern(self):
        if self.pattern == "custom" and not hasattr(self, 'custom_function'):
            raise ValueError("Custom pattern requires custom_function parameter")
```

## Success Criteria
- [ ] All operations have consistent parameter signatures
- [ ] Common selection options ("all", "none", "random") available where applicable
- [ ] All operation constructors match their Pydantic models
- [ ] Operation registry matches actual implementations
- [ ] Comprehensive parameter validation with helpful error messages
- [ ] All operations support expected parameter ranges
- [ ] Missing operation variants identified and implemented
- [ ] Full test coverage for all parameter combinations
- [ ] Updated documentation for all operations

## Testing Requirements
1. **Parameter Validation Tests**: Every parameter combination
2. **Edge Case Tests**: Minimal inputs, extreme values, invalid parameters
3. **Consistency Tests**: Similar operations behave consistently
4. **Integration Tests**: Operations work correctly in pipelines
5. **Performance Tests**: No regression in processing speed

This comprehensive review ensures all operations are complete, consistent, and user-friendly.
