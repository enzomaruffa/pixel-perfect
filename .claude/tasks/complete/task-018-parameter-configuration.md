# Task 018: Dynamic Parameter Configuration Interface

## Objective
Create dynamic form interfaces that allow users to configure operation parameters in real-time, with validation, smart defaults, and immediate visual feedback.

## Requirements
1. Generate parameter forms dynamically from Pydantic schemas
2. Implement parameter validation with user-friendly error messages
3. Create parameter input widgets (sliders, dropdowns, color pickers)
4. Add parameter presets and common combinations
5. Enable real-time parameter preview

## File Structure to Extend
```
src/ui/components/
├── parameter_forms.py       # Main parameter form generation
├── parameter_widgets.py     # Custom input widgets
├── parameter_presets.py     # Parameter preset management
└── parameter_validation.py  # Validation and error handling

src/ui/utils/
├── form_generators.py      # Pydantic to Streamlit form conversion
├── widget_factory.py       # Widget creation based on parameter types
└── validation_helpers.py   # Parameter validation utilities
```

## Core Features

### Dynamic Form Generation
Generate Streamlit input widgets automatically from operation Pydantic schemas:
- Detect parameter types (int, float, str, bool, tuple, list, enum)
- Create appropriate input widgets for each type
- Handle complex types (color tuples, coordinate pairs)
- Support parameter dependencies and constraints

### Parameter Input Widgets
- **Numeric**: Sliders with min/max bounds, number inputs
- **Text**: Text inputs with validation patterns
- **Boolean**: Toggles and checkboxes
- **Enums**: Dropdowns and radio buttons
- **Colors**: Color pickers for RGBA tuples
- **Lists**: Multi-input fields and dynamic lists
- **Coordinates**: X/Y coordinate inputs with image overlay

### Real-time Validation
- Live parameter validation as user types
- Visual feedback for invalid parameters
- Helpful error messages and suggestions
- Parameter dependency validation
- Cross-parameter consistency checks

### Parameter Presets
- Common parameter combinations for each operation
- Save/load custom parameter sets
- Quick preset application buttons
- Preset sharing and import/export

## Technical Implementation

### Form Generator Core
```python
def generate_parameter_form(operation_class, current_params=None):
    """Generate Streamlit form inputs from Pydantic model"""
    model = operation_class.__pydantic_model__
    form_inputs = {}

    for field_name, field_info in model.model_fields.items():
        widget = create_widget_for_field(field_name, field_info, current_params)
        form_inputs[field_name] = widget

    return form_inputs

def create_widget_for_field(field_name, field_info, current_value):
    """Create appropriate Streamlit widget based on field type"""
    field_type = field_info.annotation

    if field_type == int:
        return st.number_input(field_name, value=current_value or 0)
    elif field_type == float:
        return st.slider(field_name, 0.0, 10.0, current_value or 1.0)
    elif field_type == bool:
        return st.checkbox(field_name, value=current_value or False)
    # ... more widget types
```

### Widget Factory
```python
class WidgetFactory:
    @staticmethod
    def create_color_picker(field_name, default_value):
        """Create RGBA color picker widget"""
        cols = st.columns(4)
        with cols[0]:
            r = st.slider(f"{field_name} Red", 0, 255, default_value[0])
        with cols[1]:
            g = st.slider(f"{field_name} Green", 0, 255, default_value[1])
        with cols[2]:
            b = st.slider(f"{field_name} Blue", 0, 255, default_value[2])
        with cols[3]:
            a = st.slider(f"{field_name} Alpha", 0, 255, default_value[3])
        return (r, g, b, a)

    @staticmethod
    def create_coordinate_picker(field_name, image_dimensions):
        """Create coordinate picker with image overlay"""
        # Interactive coordinate selection on image
        pass
```

### Parameter Validation UI
```python
def validate_parameters(operation_class, parameters):
    """Validate parameters and show user-friendly errors"""
    try:
        # Use Pydantic validation
        validated = operation_class(**parameters)
        return validated, None
    except ValidationError as e:
        error_messages = []
        for error in e.errors():
            field = error['loc'][0]
            message = error['msg']
            error_messages.append(f"**{field}**: {message}")
        return None, error_messages

def show_validation_errors(errors):
    """Display validation errors to user"""
    if errors:
        st.error("Please fix the following parameter errors:")
        for error in errors:
            st.write(f"• {error}")
```

## Parameter Widget Types

### Numeric Parameters
- **Integer**: Number input with validation bounds
- **Float**: Slider with step control
- **Range**: Dual-handle range sliders
- **Percentage**: 0-100% sliders with % display

### Text Parameters
- **String**: Text input with length validation
- **Expression**: Code input with syntax highlighting
- **File Path**: File picker with validation
- **Regex**: Pattern input with validation

### Choice Parameters
- **Enum**: Dropdown with descriptions
- **Boolean**: Toggle switch with labels
- **Multi-select**: Checkbox groups
- **Radio**: Radio button groups

### Complex Parameters
- **Color**: RGBA color picker with preview
- **Coordinate**: X/Y inputs with image overlay
- **Size**: Width/Height input pairs
- **List**: Dynamic add/remove list inputs

### Advanced Widgets
- **Conditional**: Show/hide based on other parameters
- **Dependent**: Update options based on other selections
- **Preview**: Real-time mini-preview of parameter effects
- **Range**: Min/max value pairs with validation

## Parameter Presets System

### Built-in Presets
```python
OPERATION_PRESETS = {
    "PixelFilter": {
        "Red Highlights": {
            "condition": "prime",
            "fill_color": [255, 0, 0, 128],
            "preserve_alpha": True
        },
        "Chess Pattern": {
            "condition": "custom",
            "custom_expression": "(i // 8 + (i % 8)) % 2 == 0"
        }
    }
}
```

### Preset Management
- Dropdown preset selector for each operation
- "Apply Preset" button to load preset values
- "Save Current" button to create custom presets
- Preset description and preview

### Custom Preset Storage
- Store in session state for current session
- Local storage for persistent presets
- Export/import preset JSON files
- Share presets via URL parameters

## Real-time Parameter Preview

### Mini-Preview System
- Small preview images showing parameter effects
- Update preview as parameters change
- Debounced updates for performance
- Preview on subset of image for speed

### Parameter Impact Visualization
- Show which areas of image are affected
- Highlight changed regions
- Before/after comparison sliders
- Parameter sensitivity indicators

## Integration Points

### With Operation Selector (Task 017)
- Receive selected operation and load parameter form
- Update operation in pipeline when parameters change
- Validate parameters before allowing pipeline execution

### With Pipeline Execution (Task 019)
- Pass validated parameters to pipeline operations
- Handle parameter errors gracefully
- Cache parameter combinations for performance

### With Real-time Preview (Task 020)
- Trigger preview updates when parameters change
- Optimize parameter changes for real-time feedback
- Handle preview cancellation and debouncing

## Success Criteria
- [ ] Parameter forms generate automatically from Pydantic schemas
- [ ] All parameter types have appropriate input widgets
- [ ] Parameter validation works with clear error messages
- [ ] Parameter presets load and save correctly
- [ ] Complex parameters (colors, coordinates) have intuitive interfaces
- [ ] Real-time validation provides immediate feedback
- [ ] Parameter changes update operation in pipeline
- [ ] UI handles parameter dependencies correctly

## Test Cases to Implement
- **Form Generation Test**: Forms generate for all operation types
- **Parameter Validation Test**: Invalid parameters show appropriate errors
- **Widget Functionality Test**: All widget types work correctly
- **Preset System Test**: Presets load, save, and apply properly
- **Real-time Updates Test**: Parameter changes update pipeline immediately
- **Complex Parameter Test**: Color pickers and coordinate selectors work
- **Dependency Test**: Parameter dependencies update correctly

## Dependencies
- Builds on: Task 017 (Operation Selector Interface)
- Required: Complete understanding of all operation Pydantic schemas
- Blocks: Task 019 (Real-time Pipeline Execution)

## Notes
- Focus on intuitive parameter discovery and configuration
- Make parameter validation helpful, not frustrating
- Ensure parameter changes feel responsive
- Consider parameter learning curve for new users
- Optimize for both quick experimentation and precise control
