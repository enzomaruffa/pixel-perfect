# Task 017: Operation Selector Interface

## Objective
Create an intuitive operation selection interface that allows users to browse, search, and add operations to their image processing pipeline through the Streamlit UI.

## Requirements
1. Design categorized operation browser
2. Implement operation search and filtering
3. Create operation addition interface
4. Display operation descriptions and examples
5. Integrate with pipeline builder

## File Structure to Extend
```
src/ui/components/
├── operation_selector.py    # Main operation selection interface
├── operation_browser.py     # Categorized operation browsing
├── operation_info.py        # Operation details and help
└── pipeline_builder.py      # Pipeline management interface
```

## Core Features

### Operation Categories Display
Display operations organized by category (matching existing structure):
- 🎯 Pixel Operations (PixelFilter, PixelMath, PixelSort)
- ↔️ Row Operations (RowShift, RowStretch, RowRemove, RowShuffle)
- ↕️ Column Operations (ColumnShift, ColumnStretch, ColumnMirror, ColumnWeave)
- 🔳 Block Operations (BlockFilter, BlockShift, BlockRotate, BlockScramble)
- 🌊 Geometric Operations (GridWarp, PerspectiveStretch, RadialStretch)
- 📐 Aspect Operations (AspectStretch, AspectCrop, AspectPad)
- 🎨 Channel Operations (ChannelSwap, ChannelIsolate, AlphaGenerator)
- 🎭 Pattern Operations (Mosaic, Dither)

### Operation Browser
- Expandable category sections in sidebar
- Operation cards with icons and descriptions
- Quick preview of operation effects
- Difficulty indicators (Beginner, Intermediate, Advanced)

### Search and Filter
- Text search across operation names and descriptions
- Filter by category
- Filter by complexity level
- Filter by compatibility with current pipeline

### Operation Information Panel
- Detailed parameter descriptions
- Example use cases and effects
- Parameter validation rules
- Preview of typical results

### Pipeline Builder Integration
- "Add to Pipeline" buttons for each operation
- Drag-and-drop operation reordering (future enhancement)
- Pipeline step preview and editing
- Remove operations from pipeline

## Technical Implementation

### Operation Registry
```python
from operations import *

OPERATION_REGISTRY = {
    "pixel": {
        "icon": "🎯",
        "name": "Pixel Operations",
        "operations": {
            "PixelFilter": {
                "class": PixelFilter,
                "description": "Filter pixels based on mathematical conditions",
                "difficulty": "Beginner",
                "example_params": {"condition": "prime", "fill_color": [255, 0, 0, 128]}
            },
            # ... more operations
        }
    }
    # ... more categories
}
```

### Operation Selection UI
```python
def render_operation_selector():
    with st.sidebar:
        st.header("Operations")

        # Search box
        search_term = st.text_input("Search operations...")

        # Category filter
        selected_categories = st.multiselect("Categories", list(OPERATION_CATEGORIES.keys()))

        # Operation browser
        for category, ops in OPERATION_REGISTRY.items():
            if st.expander(f"{ops['icon']} {ops['name']}"):
                for op_name, op_info in ops['operations'].items():
                    if st.button(f"➕ {op_name}"):
                        add_operation_to_pipeline(op_name, op_info['class'])
```

### Pipeline State Management
```python
def add_operation_to_pipeline(operation_name, operation_class):
    """Add operation to pipeline with default parameters"""
    if 'pipeline_operations' not in st.session_state:
        st.session_state.pipeline_operations = []

    # Create operation with default parameters
    default_params = get_default_parameters(operation_class)
    operation_config = {
        'name': operation_name,
        'class': operation_class,
        'params': default_params,
        'enabled': True
    }

    st.session_state.pipeline_operations.append(operation_config)
    st.rerun()
```

## UI Components

### Operation Card Component
- Operation name and category
- Brief description
- Difficulty indicator
- "Add to Pipeline" button
- "Learn More" link to detailed info

### Category Expander
- Category icon and name
- Operation count
- Collapsible list of operations
- Category-specific filters

### Pipeline Summary Panel
- List of added operations in order
- Enable/disable toggles for each operation
- Remove operation buttons
- Reorder controls (drag handles)

### Operation Info Modal
- Detailed parameter documentation
- Usage examples and tips
- Expected input/output information
- Links to related operations

## Search and Filter Logic

### Search Implementation
- Search across operation names
- Search operation descriptions
- Search parameter names
- Highlight matching terms

### Filter Categories
- By operation type/category
- By difficulty level
- By image compatibility (RGB/RGBA)
- By performance characteristics

### Smart Suggestions
- Suggest operations based on current pipeline
- Recommend complementary operations
- Show popular operation combinations

## Integration Points

### With Parameter Forms (Task 018)
- Pass selected operation to parameter configuration
- Validate parameter combinations
- Update operation in pipeline when parameters change

### With Pipeline Execution (Task 019)
- Convert UI operation list to Pipeline objects
- Handle operation ordering and dependencies
- Manage pipeline state and caching

### With Preset System
- Load preset operation sequences
- Save current pipeline as preset
- Share presets between users

## Success Criteria
- [ ] All operation categories display correctly
- [ ] Search functionality works across operations
- [ ] Operations can be added to pipeline successfully
- [ ] Operation information is clearly displayed
- [ ] Pipeline summary shows added operations
- [ ] Operations can be removed from pipeline
- [ ] UI is intuitive and responsive
- [ ] Integration with existing operation classes works seamlessly

## Test Cases to Implement
- **Category Display Test**: All categories render with correct operations
- **Search Functionality Test**: Search finds operations by name and description
- **Add Operation Test**: Operations added to pipeline correctly
- **Remove Operation Test**: Operations removed from pipeline properly
- **Pipeline State Test**: Pipeline state persists across interactions
- **Operation Info Test**: Operation details display accurately

## Dependencies
- Builds on: Task 016 (Streamlit Foundation)
- Required: Complete operation registry mapping
- Blocks: Task 018 (Parameter Configuration Interface)

## Notes
- Focus on intuitive operation discovery
- Make it easy for beginners to find suitable operations
- Provide enough information for advanced users
- Ensure smooth integration with parameter configuration
