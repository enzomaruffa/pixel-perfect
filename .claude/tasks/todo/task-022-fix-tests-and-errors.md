# Task 022: Fix All Tests and Errors

## Objective
Comprehensively fix all failing tests, linting errors, and type checking issues across the entire codebase to achieve a clean, fully passing test suite.

## Current Issues Identified

### 🔴 Test Failures
1. **Edge Case Test**: `tests/integration/test_edge_cases.py::TestImageEdgeCases::test_minimal_image_operations`
   - Issue: `RowShift(selection="all", shift_amount=0)` - "all" is not a valid selection option
   - Current valid options: 'odd', 'even', 'prime', 'every_n', 'custom', 'gradient'

### 🔴 Type Checking Errors (basedpyright)

**src/operations/block.py:**
- Line 281, 502: `list[int] | None` cannot be assigned to `Iterable[int]`
- Line 526: `rotated_block` possibly unbound variable

**src/operations/channel.py:**
- Line 428: `tuple[int, int, int] | None` cannot be assigned to required `tuple[int, int, int]`

**src/operations/geometric.py:**
- Line 322: Implicit string concatenation not allowed

**src/operations/pixel.py:**
- Line 49: `tuple[int, ...]` not assignable to `tuple[int, int, int, int]`
- Lines 105-109: Multiple "No parameter named" errors for PixelFilter
- Line 266: `str | None` cannot be assigned to `eval()` parameter
- Lines 285, 387-390: Multiple "No parameter named" errors for operations

**src/presets/built_in.py:**
- Line 267: Implicit string concatenation not allowed

**src/ui/components/pipeline_executor.py:**
- Line 200: `tmp_output` possibly unbound variable

### 🟡 Pre-existing Operation Issues
Based on the test failure, there are likely inconsistencies in operation parameter definitions across the codebase.

## Implementation Plan

### Phase 1: Fix Test Infrastructure
1. **Update failing test cases** to use correct parameter values
2. **Review all test assertions** for current operation interfaces
3. **Fix test utilities** and helper functions
4. **Ensure test data integrity** with proper image fixtures

### Phase 2: Fix Type Issues
1. **Block Operations**: Fix None handling and variable binding
2. **Channel Operations**: Fix color tuple type handling
3. **Pixel Operations**: Fix tuple return types and parameter signatures
4. **Geometric Operations**: Fix string concatenation issues
5. **Presets**: Fix string formatting issues
6. **UI Components**: Fix variable scope issues

### Phase 3: Operation Parameter Consistency
1. **Audit all operations** for parameter consistency
2. **Fix missing or incorrect parameters** (e.g., PixelFilter signature mismatch)
3. **Update operation registry** to match actual implementations
4. **Validate Pydantic models** match operation constructors

### Phase 4: Test Coverage Enhancement
1. **Add missing test cases** for edge conditions
2. **Test all operation parameter combinations**
3. **Validate error handling** and edge cases
4. **Performance regression tests** for critical paths

## Specific Fixes Required

### Test Fixes
```python
# Replace invalid test cases like:
.add(RowShift(selection="all", shift_amount=0))

# With valid parameters:
.add(RowShift(selection="even", shift_amount=0))
```

### Type Fixes
```python
# Fix None handling in block operations
if indices is not None:
    excluded_indices = set(indices)
else:
    excluded_indices = set()

# Fix variable binding
rotated_block = original_block  # Ensure always assigned

# Fix tuple return types
def get_rgba_tuple(color) -> tuple[int, int, int, int]:
    if len(color) == 3:
        return (*color, 255)
    return color[:4]
```

### Operation Signature Fixes
```python
# Ensure operation constructors match Pydantic models
class PixelFilter(BaseOperation):
    def __init__(self, condition: str, fill_color: tuple[int, int, int, int],
                 preserve_alpha: bool = True, index_mode: str = "linear",
                 custom_expression: str | None = None):
        # Implementation
```

## Success Criteria
- [ ] All pytest tests pass without errors
- [ ] All ruff linting passes without warnings
- [ ] All basedpyright type checking passes
- [ ] No pre-commit hook failures
- [ ] 95%+ test coverage maintained
- [ ] All operations have consistent parameter signatures
- [ ] All edge cases properly handled
- [ ] Performance benchmarks remain stable

## Testing Strategy
1. **Unit Tests**: Every operation, utility, and component
2. **Integration Tests**: Full pipeline execution paths
3. **Edge Case Tests**: Minimal images, extreme parameters, error conditions
4. **Performance Tests**: Memory usage, execution time benchmarks
5. **UI Tests**: Streamlit component functionality
6. **Type Tests**: Static analysis validation

## Validation Steps
1. Run full test suite: `uv run pytest tests/ -v`
2. Check type safety: `uv run basedpyright src/`
3. Lint codebase: `uv run ruff check src/ tests/`
4. Run pre-commit: `pre-commit run --all-files`
5. Manual smoke tests of core functionality
6. Performance regression validation

This task ensures the codebase is production-ready with comprehensive test coverage, type safety, and consistent operation interfaces.
