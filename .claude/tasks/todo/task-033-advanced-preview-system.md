# Task 033: Advanced Preview System

**Status:** Todo
**Priority:** Medium (Week 3)
**Category:** Power Features

## Problem

Current preview system is basic:
- Only shows final result in sidebar
- No step-by-step visualization
- No zoom or detailed inspection
- No before/after comparisons
- Limited to small thumbnail view

## Goal

Power user preview system with multiple view modes, detailed inspection capabilities, and step-by-step visualization.

## Solution

### 1. Multiple Preview Modes
- **Original Only** - Show source image
- **Final Result** - Show processed result
- **Side by Side** - Original vs processed
- **Step by Step** - Each operation's result
- **Overlay Compare** - Transparent overlay comparison
- **Zoom View** - Detailed pixel-level inspection

### 2. Step-by-Step Visualization
Show intermediate results after each operation:
- Visual pipeline with thumbnail previews
- Click any step to see full result
- Compare any two steps
- Identify which operations make biggest impact

### 3. Advanced Inspection Tools
- **Zoom/Pan** - Detailed pixel inspection
- **Pixel Info** - Click to see RGB values
- **Region Selection** - Focus on specific areas
- **Measurement Tools** - Distance and area tools

## Implementation

### Files to Create:
- `src/ui/components/advanced_preview.py` - main preview system
- `src/ui/components/step_visualizer.py` - step-by-step display
- `src/ui/utils/image_inspector.py` - zoom and inspection tools

### Files to Modify:
- `src/ui/components/layout.py` - integrate advanced preview
- `src/ui/components/pipeline_executor.py` - capture step results

### Key Implementation:

1. **Preview Mode Selector:**
   ```python
   def render_advanced_preview():
       st.subheader("🖼️ Advanced Preview")

       preview_mode = st.selectbox(
           "Preview Mode",
           ["original", "final", "side_by_side", "step_by_step", "overlay", "zoom"],
           format_func=lambda x: {
               "original": "📷 Original Only",
               "final": "✨ Final Result",
               "side_by_side": "📱 Side by Side",
               "step_by_step": "👁️ Step by Step",
               "overlay": "🔀 Overlay Compare",
               "zoom": "🔍 Zoom Inspector"
           }[x]
       )

       if preview_mode == "step_by_step":
           render_step_visualizer()
       elif preview_mode == "zoom":
           render_zoom_inspector()
       else:
           render_standard_preview(preview_mode)
   ```

2. **Step-by-Step Visualizer:**
   ```python
   def render_step_visualizer():
       if not st.session_state.get("last_execution_result"):
           st.info("Execute pipeline to see step-by-step results")
           return

       result = st.session_state.last_execution_result

       # Step selector
       step_options = ["Original"] + [f"Step {i+1}: {step['operation']}"
                                     for i, step in enumerate(result.steps)]

       selected_step = st.selectbox("Select Step", step_options)

       # Show selected step result
       if selected_step == "Original":
           st.image(st.session_state.original_image, caption="Original Image")
       else:
           step_index = int(selected_step.split()[1]) - 1
           step_result = result.steps[step_index]

           if "result_image" in step_result:
               st.image(step_result["result_image"], caption=f"After {step_result['operation']}")

               # Show step info
               st.info(f"⏱️ Execution time: {step_result['execution_time']:.3f}s")
               if step_result.get("cached"):
                   st.info("💾 Result from cache")
   ```

3. **Zoom Inspector:**
   ```python
   def render_zoom_inspector():
       if not st.session_state.get("processed_image"):
           st.info("Execute pipeline to enable zoom inspector")
           return

       image = st.session_state.processed_image

       # Zoom controls
       col1, col2 = st.columns(2)
       with col1:
           zoom_level = st.slider("Zoom Level", 1, 10, 2)
       with col2:
           show_grid = st.checkbox("Show Pixel Grid", value=True)

       # Image display with zoom
       display_zoomed_image(image, zoom_level, show_grid)

       # Pixel inspector
       if st.checkbox("Pixel Inspector"):
           render_pixel_inspector(image)
   ```

4. **Side-by-Side Comparison:**
   ```python
   def render_side_by_side():
       if not (st.session_state.get("original_image") and
               st.session_state.get("processed_image")):
           st.info("Need both original and processed images for comparison")
           return

       col1, col2 = st.columns(2)

       with col1:
           st.write("**Original**")
           st.image(st.session_state.original_image, use_container_width=True)
           original_info = get_image_info(st.session_state.original_image)
           st.caption(f"{original_info['width']}×{original_info['height']} | {original_info['mode']}")

       with col2:
           st.write("**Processed**")
           st.image(st.session_state.processed_image, use_container_width=True)
           processed_info = get_image_info(st.session_state.processed_image)
           st.caption(f"{processed_info['width']}×{processed_info['height']} | {processed_info['mode']}")

       # Show difference metrics
       render_comparison_metrics()
   ```

5. **Pipeline Step Capture:**
   ```python
   # Modify pipeline executor to capture intermediate results
   def execute_pipeline_with_step_capture(pipeline, operations):
       step_results = []
       current_image = pipeline.original_image

       for i, operation in enumerate(operations):
           if not operation.get("enabled", True):
               continue

           # Execute operation
           start_time = time.perf_counter()
           result_image, context = execute_single_operation(current_image, operation)
           execution_time = time.perf_counter() - start_time

           # Capture step result
           step_results.append({
               "step": i,
               "operation": operation["name"],
               "result_image": result_image.copy(),  # Store copy for visualization
               "execution_time": execution_time,
               "context": context
           })

           current_image = result_image

       return current_image, step_results
   ```

## Preview Features

### Standard Modes:
- **Original**: Source image with metadata
- **Final**: Final processed result
- **Side by Side**: Original vs processed with metrics

### Advanced Modes:
- **Step by Step**: Interactive step navigation
- **Overlay**: Transparent overlay with opacity slider
- **Zoom**: Pixel-level inspection with grid

### Inspector Tools:
- **Pixel Info**: Click to see RGB values at cursor
- **Region Stats**: Select area to see statistics
- **Measurements**: Distance and area measurement tools
- **Histogram**: Color distribution analysis

## Acceptance Criteria

- [ ] Multiple preview modes available
- [ ] Step-by-step visualization works correctly
- [ ] Zoom inspector shows pixel details
- [ ] Side-by-side comparison with metrics
- [ ] Smooth switching between preview modes
- [ ] Intermediate results captured during execution
- [ ] Performance optimized for large images
- [ ] Responsive design for different screen sizes

## Testing

- Test all preview modes with various images
- Verify step-by-step visualization accuracy
- Test zoom functionality with different image sizes
- Check performance with complex pipelines
- Validate comparison metrics accuracy
