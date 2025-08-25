# Task 031: Enhanced Error Handling

**Status:** Todo
**Priority:** High (Week 2)
**Category:** Power Features

## Problem

Current error messages are cryptic and unhelpful:
- Pydantic validation errors are too technical
- No specific guidance on how to fix problems
- Generic error messages don't explain root cause
- Users left guessing what went wrong

## Current Issues

Examples of poor error messages:
- "ValidationError: 1 validation error for PixelFilter condition Input should be 'prime', 'odd', 'even', 'fibonacci' or 'custom'"
- "ProcessingError: Custom expression evaluation failed: name 'j' is not defined" 
- "AttributeError: 'NoneType' object has no attribute 'width'"

## Solution

### 1. User-Friendly Error Translation
Convert technical errors to clear, actionable messages:

**Before:** `ValidationError: 1 validation error for PixelFilter condition Input should be 'prime', 'odd', 'even', 'fibonacci' or 'custom'`

**After:** `Invalid condition "all" for PixelFilter. Choose one of: prime, odd, even, fibonacci, or custom. Did you mean "even"?`

### 2. Specific Fix Suggestions
For each error type, provide concrete solutions:

- **Parameter errors:** Show valid values and suggest closest match
- **Formula errors:** Show correct variable names and syntax
- **Image errors:** Explain format requirements and size limits
- **Pipeline errors:** Identify problematic operations

### 3. Contextual Error Handling
Different error handling based on context:
- **Parameter validation:** Real-time feedback in forms
- **Pipeline execution:** Clear operation-level error reporting
- **File operations:** Specific file format and content guidance

## Implementation

### Files to Create:
- `src/ui/utils/error_translator.py` - error message translation
- `src/ui/utils/error_suggestions.py` - fix suggestions

### Files to Modify:
- `src/ui/components/parameter_forms.py` - form validation errors
- `src/ui/components/layout.py` - pipeline execution errors
- `src/ui/components/pipeline_executor.py` - operation errors

### Key Implementation:

1. **Error Translation System:**
   ```python
   def translate_error(error: Exception, context: str = None) -> dict:
       """Convert technical errors to user-friendly messages."""
       
       if isinstance(error, ValidationError):
           return translate_validation_error(error, context)
       elif isinstance(error, ProcessingError):
           return translate_processing_error(error, context)
       elif isinstance(error, FileNotFoundError):
           return translate_file_error(error, context)
       else:
           return {
               "message": "Unexpected error occurred",
               "details": str(error),
               "suggestion": "Please try again or check your input"
           }
   
   def translate_validation_error(error: ValidationError, context: str) -> dict:
       field_errors = []
       for err in error.errors():
           field = err["loc"][0] if err["loc"] else "parameter"
           value = err.get("input", "unknown")
           
           if err["type"] == "literal_error":
               expected = err.get("ctx", {}).get("expected", [])
               suggestion = suggest_closest_match(value, expected)
               
               return {
                   "message": f'Invalid {field} "{value}". Choose one of: {", ".join(expected)}.',
                   "suggestion": f'Did you mean "{suggestion}"?' if suggestion else None,
                   "field": field
               }
   ```

2. **Smart Suggestions:**
   ```python
   def suggest_closest_match(value: str, options: list[str]) -> str | None:
       """Suggest closest valid option using fuzzy matching."""
       if not value or not options:
           return None
       
       # Simple similarity scoring
       best_match = None
       best_score = 0
       
       for option in options:
           # Calculate similarity (simplified)
           score = len(set(value.lower()) & set(option.lower())) / len(set(value.lower()) | set(option.lower()))
           if score > best_score and score > 0.3:
               best_match = option
               best_score = score
       
       return best_match
   ```

3. **Formula Error Enhancement:**
   ```python
   def enhance_formula_error(error: str, formula: str, context: str) -> dict:
       """Provide specific help for formula errors."""
       
       if "not defined" in error:
           # Extract undefined variable
           var_match = re.search(r"name '(\w+)' is not defined", error)
           if var_match:
               undefined_var = var_match.group(1)
               
               # Suggest correct variable
               suggestions = {
                   "j": "x (for column coordinate)",
                   "i": "y (for row coordinate)" if context == "row" else "i (for pixel index)",
                   "col": "x", "row": "y", "idx": "i"
               }
               
               suggestion = suggestions.get(undefined_var)
               if suggestion:
                   return {
                       "message": f"Variable '{undefined_var}' not recognized in formula",
                       "suggestion": f"Use '{suggestion}' instead",
                       "example": formula.replace(undefined_var, suggestion.split()[0])
                   }
   ```

4. **Real-time Parameter Validation:**
   ```python
   def render_parameter_with_validation(field_name, field_info, current_value, key_prefix):
       """Render parameter with real-time validation feedback."""
       
       try:
           # Create widget
           widget_value = create_widget_for_field(field_name, field_info, current_value, key_prefix)
           
           # Validate in real-time
           if widget_value != current_value:
               try:
                   # Test validation
                   validate_parameter(field_name, widget_value, field_info)
                   # Show success
                   st.success("✓ Valid", icon="✅")
               except Exception as e:
                   # Show friendly error
                   error_info = translate_error(e, context=field_name)
                   st.error(error_info["message"])
                   if error_info.get("suggestion"):
                       st.info(f"💡 {error_info['suggestion']}")
           
           return widget_value
           
       except Exception as e:
           error_info = translate_error(e, context=field_name)
           st.error(f"Parameter error: {error_info['message']}")
           return current_value
   ```

## Error Message Standards

### Parameter Errors:
- **Format:** `Invalid [parameter] "[value]". [Explanation]. [Suggestion]`
- **Example:** `Invalid condition "all". Choose: prime, odd, even, fibonacci, custom. Did you mean "even"?`

### Formula Errors:
- **Format:** `Formula error: [problem]. Use [solution]. Example: [corrected_formula]`
- **Example:** `Formula error: Variable 'j' not recognized. Use 'x' for column coordinate. Example: 'x * 2'`

### Pipeline Errors:
- **Format:** `[Operation] failed: [reason]. [Fix_suggestion]`
- **Example:** `PixelFilter failed: Invalid image format. Ensure image is RGB or RGBA mode.`

## Acceptance Criteria

- [ ] All error messages are user-friendly and actionable
- [ ] Parameter validation provides real-time feedback
- [ ] Formula errors suggest correct variable names
- [ ] Pipeline errors identify specific operation problems
- [ ] Error messages include fix suggestions when possible
- [ ] No technical jargon in user-facing errors
- [ ] Consistent error message format throughout app

## Testing

- Trigger various parameter validation errors
- Test formula errors with undefined variables
- Verify pipeline execution error reporting
- Check error message clarity and helpfulness
- Test error recovery and continuation