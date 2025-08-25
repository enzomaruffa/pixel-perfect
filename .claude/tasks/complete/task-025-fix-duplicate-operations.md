# Task 025: Fix Duplicate Operation Handling

**Status:** Todo
**Priority:** Critical (Week 1)
**Category:** Core Functionality

## Problem

Adding the same operation type multiple times breaks the app due to:
- Duplicate titles causing confusion
- Widget key conflicts in Streamlit
- No unique identification for operation instances

## Current Issues

- Operations only identified by name/type
- No instance IDs or unique identifiers
- Streamlit widget keys clash when same operation added twice
- Users can't distinguish between multiple instances of same operation

## Solution

### 1. UUID Generation
- Generate unique UUID for each operation instance
- Store UUID in operation config for permanent identification
- Use UUID in all widget keys to prevent conflicts

### 2. Smart Operation Titles
- Auto-increment operation titles: "PixelFilter", "PixelFilter (2)", "PixelFilter (3)"
- Allow user customization of operation names
- Display meaningful titles in pipeline view

### 3. Proper Instance Management
- Track operation instances in session state
- Unique identification for each operation in pipeline
- Proper cleanup when operations are removed

## Implementation

### Files to Modify:
- `src/ui/components/operation_browser.py` - operation addition logic
- `src/ui/components/parameter_forms.py` - widget key generation
- `src/ui/components/session.py` - session state management

### Key Changes:

1. **Operation Addition:**
   ```python
   import uuid

   def add_operation_to_pipeline(operation_class, params):
       operation_id = str(uuid.uuid4())

       # Generate smart title
       existing_count = len([op for op in pipeline_operations
                           if op['name'].startswith(operation_class.__name__)])
       title = operation_class.__name__
       if existing_count > 0:
           title += f" ({existing_count + 1})"

       operation_config = {
           'id': operation_id,
           'name': title,
           'class': operation_class,
           'params': params,
           'enabled': True
       }
   ```

2. **Widget Key Generation:**
   ```python
   def generate_widget_key(operation_id, param_name):
       return f"{operation_id}_{param_name}"
   ```

3. **Pipeline Management:**
   - Use operation IDs for all pipeline operations
   - Update all references to use IDs instead of names
   - Proper operation removal by ID

## Acceptance Criteria

- [ ] Can add same operation type multiple times without errors
- [ ] Each operation instance has unique, meaningful title
- [ ] No Streamlit widget key conflicts
- [ ] Operations can be configured independently
- [ ] Pipeline displays clear operation names
- [ ] Operations can be removed individually by ID

## Testing

- Add same operation (e.g., PixelFilter) 3 times
- Verify unique titles and no conflicts
- Configure different parameters for each instance
- Remove operations and verify no orphaned state
- Test with all operation types
