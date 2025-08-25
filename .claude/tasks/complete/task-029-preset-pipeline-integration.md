# Task 029: Preset to Pipeline Integration

**Status:** Todo
**Priority:** High (Week 2)
**Category:** Power Features

## Problem

Presets exist but can't be loaded into the current pipeline:
- Presets are isolated from pipeline workflow
- No way to use presets as starting points
- Users can't modify or extend existing presets
- Disconnect between preset system and pipeline builder

## Current State

- Presets defined in `src/presets/built_in.py`
- Export manager has preset functionality
- Presets can be exported but not loaded into pipeline
- No integration between preset browser and pipeline builder

## Solution

### 1. Preset Browser Enhancement
Add "Load to Pipeline" functionality:
- Browse available presets
- Preview preset operations
- One-click load into current pipeline
- Option to replace or append to current pipeline

### 2. Preset to Pipeline Conversion
Convert preset format to pipeline operations:
- Transform preset operation configs to pipeline format
- Generate unique IDs for loaded operations
- Apply proper operation class mapping
- Maintain parameter compatibility

### 3. Integration Points
- Add preset browser to pipeline builder tab
- Quick preset loading buttons
- Preset preview with operation list
- Clear workflow from preset → customize → execute

## Implementation

### Files to Create:
- `src/ui/components/preset_browser.py` - dedicated preset interface

### Files to Modify:
- `src/ui/components/layout.py` - add preset browser to pipeline tab
- `src/ui/components/export_manager.py` - extract preset logic
- `src/ui/utils/preset_converter.py` - preset to pipeline conversion

### Key Implementation:

1. **Preset to Pipeline Conversion:**
   ```python
   def convert_preset_to_pipeline(preset_config):
       operations = []
       
       for i, op_config in enumerate(preset_config["operations"]):
           # Get operation class
           operation_class = get_operation_class(op_config["type"])
           
           # Generate unique ID and name
           operation_id = str(uuid.uuid4())
           operation_name = f"{op_config['type']}"
           
           # Create pipeline operation config
           operations.append({
               "id": operation_id,
               "name": operation_name,
               "class": operation_class,
               "params": op_config["params"],
               "enabled": True
           })
       
       return operations
   ```

2. **Preset Browser UI:**
   ```python
   def render_preset_browser():
       st.subheader("📋 Load Preset")
       
       presets = get_all_presets()
       
       for preset_name, preset_config in presets.items():
           with st.expander(f"**{preset_name}**"):
               st.write(preset_config["description"])
               
               # Show operations preview
               st.write("**Operations:**")
               for op in preset_config["operations"]:
                   st.write(f"• {op['type']}")
               
               col1, col2 = st.columns(2)
               with col1:
                   if st.button("📥 Replace Pipeline", key=f"replace_{preset_name}"):
                       load_preset_to_pipeline(preset_config, replace=True)
               
               with col2:
                   if st.button("➕ Add to Pipeline", key=f"append_{preset_name}"):
                       load_preset_to_pipeline(preset_config, replace=False)
   ```

3. **Pipeline Integration:**
   ```python
   def load_preset_to_pipeline(preset_config, replace=True):
       new_operations = convert_preset_to_pipeline(preset_config)
       
       if replace:
           st.session_state.pipeline_operations = new_operations
           st.success(f"Pipeline replaced with {len(new_operations)} operations")
       else:
           st.session_state.pipeline_operations.extend(new_operations)
           st.success(f"Added {len(new_operations)} operations to pipeline")
       
       st.rerun()
   ```

4. **UI Layout Integration:**
   Add to pipeline builder tab:
   ```python
   # In layout.py - Pipeline Builder tab
   col1, col2, col3 = st.columns([1, 1, 1])
   
   with col1:
       render_operation_browser()  # Existing
   
   with col2:
       render_pipeline_summary()  # Existing
   
   with col3:
       render_preset_browser()     # New
   ```

## Acceptance Criteria

- [ ] Can browse all available presets
- [ ] Preview preset operations before loading
- [ ] One-click load preset into pipeline
- [ ] Option to replace or append to current pipeline
- [ ] Preset operations work correctly after loading
- [ ] Clear feedback when preset is loaded
- [ ] Integration with existing pipeline workflow

## Testing

- Load each built-in preset into pipeline
- Verify all operations load with correct parameters
- Test both replace and append modes
- Execute pipelines created from presets
- Verify parameter customization after preset loading