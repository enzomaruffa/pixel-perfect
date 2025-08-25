# Task 028: Pipeline Save/Load System

**Status:** Todo
**Priority:** High (Week 2)
**Category:** Power Features

## Problem

No way to save and load complete pipelines:
- Users lose work when refreshing browser
- No persistence of complex pipeline configurations
- Can't share pipelines with others
- Only presets exist, but they're limited and static

## Current State

- Export manager has preset functionality
- Session state stores pipeline_operations temporarily
- No pipeline-specific save/load functionality

## Solution

### 1. Pipeline Serialization
Clean JSON format for pipeline storage:
```json
{
  "version": "1.0",
  "name": "My Pipeline",
  "description": "Custom image processing pipeline",
  "created": "2025-01-15T10:30:00Z",
  "operations": [
    {
      "id": "uuid-1234",
      "type": "PixelFilter",
      "name": "PixelFilter",
      "enabled": true,
      "params": {
        "condition": "prime",
        "fill_color": [255, 0, 0, 128]
      }
    }
  ]
}
```

### 2. Save Pipeline UI
Simple, clean interface:
- **Save Pipeline** button in pipeline section
- File name input with auto-suggestion
- Optional description field
- Download as .json file

### 3. Load Pipeline UI
- **Load Pipeline** file uploader
- Validation of pipeline format
- Option to replace current pipeline or merge
- Preview of operations before loading

## Implementation

### Files to Create:
- `src/ui/components/pipeline_manager.py` - main save/load logic
- `src/ui/utils/pipeline_serialization.py` - JSON handling

### Files to Modify:
- `src/ui/components/layout.py` - add save/load buttons
- `src/ui/components/session.py` - pipeline state management

### Key Implementation:

1. **Pipeline Serialization:**
   ```python
   def serialize_pipeline(operations, name=None, description=None):
       return {
           "version": "1.0",
           "name": name or f"Pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
           "description": description or "",
           "created": datetime.now().isoformat(),
           "operations": [
               {
                   "id": op["id"],
                   "type": op["class"].__name__,
                   "name": op["name"],
                   "enabled": op.get("enabled", True),
                   "params": op["params"]
               }
               for op in operations
           ]
       }
   ```

2. **Pipeline Loading:**
   ```python
   def load_pipeline_from_json(json_data):
       # Validate format
       validate_pipeline_format(json_data)

       # Convert to operation configs
       operations = []
       for op_data in json_data["operations"]:
           operation_class = get_operation_class(op_data["type"])
           operations.append({
               "id": op_data["id"],
               "name": op_data["name"],
               "class": operation_class,
               "params": op_data["params"],
               "enabled": op_data.get("enabled", True)
           })

       return operations
   ```

3. **UI Integration:**
   ```python
   def render_pipeline_save_load():
       col1, col2 = st.columns(2)

       with col1:
           if st.button("💾 Save Pipeline"):
               pipeline_data = serialize_pipeline(
                   st.session_state.pipeline_operations,
                   st.session_state.get("pipeline_name", ""),
                   st.session_state.get("pipeline_description", "")
               )
               st.download_button(
                   "Download Pipeline",
                   json.dumps(pipeline_data, indent=2),
                   file_name=f"{pipeline_data['name']}.json",
                   mime="application/json"
               )

       with col2:
           uploaded_file = st.file_uploader("📁 Load Pipeline", type="json")
           if uploaded_file:
               try:
                   pipeline_data = json.loads(uploaded_file.read())
                   operations = load_pipeline_from_json(pipeline_data)
                   st.session_state.pipeline_operations = operations
                   st.success(f"Loaded pipeline: {pipeline_data['name']}")
                   st.rerun()
               except Exception as e:
                   st.error(f"Failed to load pipeline: {e}")
   ```

## Acceptance Criteria

- [ ] Can save current pipeline as JSON file
- [ ] Can load pipeline from JSON file
- [ ] Pipeline validation prevents corrupted files
- [ ] Operations maintain their configuration after load
- [ ] Clear error messages for invalid files
- [ ] Option to replace or merge with current pipeline
- [ ] Metadata preserved (name, description, creation date)

## Testing

- Save pipeline with multiple operations
- Load saved pipeline and verify all operations work
- Test with various operation types and parameters
- Verify error handling with invalid JSON files
- Test pipeline merging functionality
