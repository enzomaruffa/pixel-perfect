# Task 027: Essential Tooltips System

**Status:** Todo
**Priority:** High (Week 1)
**Category:** UI Cleanup

## Problem

Parameters lack helpful documentation:
- No explanations of what parameters do
- No guidance on valid ranges or values
- Formula syntax is unclear
- Users must guess parameter meanings

## Goal

Add concise, informative tooltips to every parameter - focused on power users who want facts, not hand-holding.

## Solution

### 1. Parameter Tooltips
Every form field gets a `help` parameter with:
- **What it does** - Brief, clear explanation
- **Valid range** - Min/max values or options
- **Common values** - Typical/recommended settings
- **Formula syntax** - For expression fields

### 2. Tooltip Content Strategy
**Style:** Concise, technical, informative
**Format:** 
```
Brief description. Range: X-Y. Default: Z.
```

**Examples:**
- `shift_amount`: "Pixels to shift (negative=up, positive=down). Range: -1000 to 1000. Default: 0."
- `formula`: "Mathematical expression using 'i' for row index. Example: 'i * 2', 'sin(i/10) * 20'. Variables: i, width, height."
- `fill_color`: "RGBA color tuple (Red, Green, Blue, Alpha). Range: 0-255 each. Default: (0,0,0,0) transparent."

### 3. Formula Documentation
Special focus on formula fields:
- Available variables clearly listed
- Syntax examples provided
- Common mathematical functions listed
- Error-prone patterns explained

## Implementation

### Files to Modify:
- `src/ui/utils/form_generators.py` - tooltip generation
- `src/ui/components/parameter_forms.py` - preset tooltips
- All operation files - add help text to field definitions

### Key Changes:

1. **Enhanced Field Creation:**
   ```python
   def create_widget_for_field(field_name, field_info, current_value=None, key_prefix=""):
       # Get tooltip text
       help_text = get_parameter_help(field_name, field_info)
       
       # Apply to widget
       widget = st.number_input(
           label,
           value=current_value,
           help=help_text,  # This is the key addition
           key=f"{key_prefix}_param_{field_name}"
       )
   ```

2. **Parameter Help Database:**
   ```python
   PARAMETER_HELP = {
       "shift_amount": "Pixels to shift (negative=up, positive=down). Range: -1000 to 1000.",
       "selection": "Which rows/columns to affect. 'all'=every one, 'odd'=1,3,5..., 'even'=2,4,6...",
       "formula": "Math expression using 'i' for index. Examples: 'i*2', 'sin(i/10)*20'. Variables: i, width, height.",
       "fill_color": "RGBA color (Red, Green, Blue, Alpha). Range: 0-255 each. (255,0,0,255) = opaque red.",
       "condition": "Pixel selection criteria. 'prime'=prime indices, 'fibonacci'=fibonacci sequence.",
       "expression": "Math formula using r,g,b,a for colors, x,y for position. Example: 'r * 1.5', 'r + g + b'.",
       # ... more parameters
   }
   ```

3. **Formula Field Special Treatment:**
   ```python
   if "formula" in field_name or "expression" in field_name:
       # Enhanced help with variable reference
       help_text += f"\n\nAvailable variables: {', '.join(get_available_variables())}"
       help_text += f"\nFunctions: sin, cos, sqrt, abs, min, max, pow, floor, ceil"
   ```

## Acceptance Criteria

- [ ] Every parameter has helpful tooltip
- [ ] Formula fields include variable reference
- [ ] Tooltips are concise and informative
- [ ] No patronizing language - power user focused
- [ ] Consistent formatting across all parameters
- [ ] Special formula documentation included

## Testing

- Check every operation parameter has tooltip
- Verify formula fields have variable documentation
- Ensure tooltips are helpful without being verbose
- Test tooltip content accuracy