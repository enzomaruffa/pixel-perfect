# Task 030: Variable Standardization Across Operations

**Status:** Todo
**Priority:** Critical (Week 2)
**Category:** Consistency

## Problem

**CRITICAL INCONSISTENCY** - Formula variables are inconsistent across operations:

### Current Mess:
- **RowShift formula**: uses `'i'` for row index
- **ColumnShift formula**: uses `'j'` for column index
- **PixelMath expression**: uses `'x', 'y'` for coordinates + `'r', 'g', 'b', 'a'` for colors
- **PixelFilter custom_expression**: uses `'i'` for pixel index
- **Validation allows**: `{"r", "g", "b", "a", "x", "y", "width", "height", "i", "j"}`

This is confusing and unprofessional for production use.

## NEW STANDARD

Establish consistent variable naming across ALL operations:

```
SPATIAL COORDINATES:
  x = column coordinate (0 to width-1)
  y = row coordinate (0 to height-1)

COLOR CHANNELS:
  r, g, b, a = red, green, blue, alpha values (0-255)

INDEX VALUES:
  i = linear pixel index (0 to width*height-1)

DIMENSIONS:
  width, height = image dimensions
```

## Migration Strategy

### Phase 1: Update Operations
1. **RowShift**: Change from `'i'` to `'y'` for row coordinate
2. **ColumnShift**: Change from `'j'` to `'x'` for column coordinate
3. **Keep PixelMath**: Already uses correct `x, y, r, g, b, a` variables
4. **Keep PixelFilter**: `'i'` is correct for linear pixel index

### Phase 2: Backward Compatibility
- Support both old and new variables during transition
- Add deprecation warnings for old variables
- Provide migration suggestions in error messages

### Phase 3: Documentation Update
- Update all formula examples
- Update validation allowed variables
- Update tooltip documentation

## Implementation

### Files to Modify:
- `src/operations/row.py` - change `'i'` to `'y'`
- `src/operations/column.py` - change `'j'` to `'x'`
- `src/utils/validation.py` - update allowed variables and add deprecation warnings
- All formula documentation and examples

### Key Changes:

1. **RowShift Formula Update:**
   ```python
   # OLD:
   context["i"] = i  # row index

   # NEW:
   context["y"] = i  # row coordinate (y-axis)
   context["i"] = i  # DEPRECATED - keep for backward compatibility
   ```

2. **ColumnShift Formula Update:**
   ```python
   # OLD:
   context["j"] = j  # column index

   # NEW:
   context["x"] = j  # column coordinate (x-axis)
   context["j"] = j  # DEPRECATED - keep for backward compatibility
   ```

3. **Enhanced Validation with Deprecation:**
   ```python
   def validate_expression_safe(expression: str, context_type: str = "generic") -> None:
       # Check for deprecated variables
       deprecated_vars = {
           "i": "y" if context_type == "row" else None,
           "j": "x" if context_type == "column" else None
       }

       for old_var, new_var in deprecated_vars.items():
           if old_var in expression and new_var:
               warnings.warn(
                   f"Variable '{old_var}' is deprecated. Use '{new_var}' instead. "
                   f"Example: '{expression.replace(old_var, new_var)}'",
                   DeprecationWarning
               )
   ```

4. **Updated Documentation:**
   ```python
   FORMULA_HELP = {
       "row": "Row formula using 'y' for row coordinate (0 to height-1). Example: 'y * 2', 'sin(y/10) * 20'",
       "column": "Column formula using 'x' for column coordinate (0 to width-1). Example: 'x * 2', 'cos(x/20) * 10'",
       "pixel": "Pixel expression using 'r,g,b,a' for colors, 'x,y' for position, 'i' for index",
   }
   ```

### Standard Variable Reference:
```python
STANDARD_VARIABLES = {
    "x": "Column coordinate (0 to width-1)",
    "y": "Row coordinate (0 to height-1)",
    "r": "Red channel value (0-255)",
    "g": "Green channel value (0-255)",
    "b": "Blue channel value (0-255)",
    "a": "Alpha channel value (0-255)",
    "i": "Linear pixel index (0 to width*height-1)",
    "width": "Image width in pixels",
    "height": "Image height in pixels"
}

DEPRECATED_VARIABLES = {
    "j": "Use 'x' for column coordinate instead"
}
```

## Acceptance Criteria

- [ ] All operations use consistent variable naming
- [ ] Row operations use 'y' for row coordinate
- [ ] Column operations use 'x' for column coordinate
- [ ] Pixel operations use 'r,g,b,a' for colors, 'x,y' for position, 'i' for index
- [ ] Backward compatibility maintained with deprecation warnings
- [ ] All documentation updated with new standards
- [ ] Formula examples use consistent variables
- [ ] Validation provides helpful migration suggestions

## Testing

- Test all formula-based operations with new variables
- Verify backward compatibility with old formulas
- Check deprecation warnings appear for old variables
- Validate all operation types work with standardized variables
- Test complex formulas mixing different variable types

## Documentation Updates

Update all formula examples:
- Row formulas: `'i * 2'` → `'y * 2'`
- Column formulas: `'j / 3'` → `'x / 3'`
- Pixel expressions: maintain `'r * 1.5'`, `'x + y'`, etc.

This standardization is CRITICAL for production-grade consistency and user experience.
