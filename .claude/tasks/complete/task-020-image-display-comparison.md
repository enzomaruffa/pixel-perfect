# Task 020: Real-time Image Display and Comparison

## Objective
Create advanced image display and comparison interfaces that show original, processed, and intermediate results with interactive comparison tools, zoom controls, and real-time updates.

## Requirements
1. Implement side-by-side image comparison interface
2. Create interactive zoom and pan controls
3. Add before/after comparison tools (slider, toggle, split-screen)
4. Display intermediate pipeline step results
5. Support different image formats and sizes gracefully

## File Structure to Extend
```
src/ui/components/
├── image_display.py         # Main image display components
├── image_comparison.py      # Before/after comparison tools
├── image_viewer.py          # Interactive image viewer with zoom/pan
├── step_viewer.py           # Pipeline step result viewer
└── image_controls.py        # Image manipulation controls

src/ui/utils/
├── image_scaling.py         # Image scaling and resizing utilities
├── comparison_tools.py      # Image comparison algorithms
└── display_optimization.py  # Display performance optimizations
```

## Core Features

### Side-by-Side Comparison
- Original and processed image display
- Synchronized zoom and pan between images
- Configurable layout (horizontal, vertical, grid)
- Responsive sizing based on screen dimensions

### Interactive Comparison Tools
- **Slider Comparison**: Drag slider to reveal before/after
- **Toggle Comparison**: Click to switch between versions
- **Split Screen**: Vertical/horizontal divider with drag control
- **Overlay Mode**: Semi-transparent overlay with opacity control

### Advanced Image Viewer
- Zoom in/out with mouse wheel or controls
- Pan by dragging or keyboard arrows
- Fit to window/actual size/custom zoom levels
- Pixel-level inspection with coordinate display
- Image information overlay (dimensions, format, size)

### Pipeline Step Visualization
- Display result of each pipeline operation
- Step-by-step progression view
- Before/after for each individual operation
- Interactive step selection and navigation

## Technical Implementation

### Main Image Display Component
```python
def render_image_comparison():
    """Main image comparison interface"""
    if st.session_state.original_image and st.session_state.processed_image:

        # Comparison mode selector
        comparison_mode = st.selectbox(
            "Comparison Mode",
            ["Side by Side", "Slider", "Toggle", "Overlay"],
            key="comparison_mode"
        )

        if comparison_mode == "Side by Side":
            render_side_by_side_comparison()
        elif comparison_mode == "Slider":
            render_slider_comparison()
        elif comparison_mode == "Toggle":
            render_toggle_comparison()
        elif comparison_mode == "Overlay":
            render_overlay_comparison()

def render_side_by_side_comparison():
    """Side-by-side image comparison"""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        display_image_with_controls(
            st.session_state.original_image,
            key="original"
        )

    with col2:
        st.subheader("Processed")
        display_image_with_controls(
            st.session_state.processed_image,
            key="processed"
        )
```

### Interactive Image Viewer
```python
class InteractiveImageViewer:
    def __init__(self, image, key):
        self.image = image
        self.key = key
        self.zoom_level = st.session_state.get(f"{key}_zoom", 1.0)
        self.pan_x = st.session_state.get(f"{key}_pan_x", 0)
        self.pan_y = st.session_state.get(f"{key}_pan_y", 0)

    def render(self):
        """Render interactive image viewer"""
        # Image controls
        self.render_controls()

        # Calculate display size based on zoom and pan
        display_width = int(self.image.width * self.zoom_level)
        display_height = int(self.image.height * self.zoom_level)

        # Create cropped/scaled image for display
        display_image = self.prepare_display_image(
            display_width, display_height
        )

        # Display image with click handling
        st.image(
            display_image,
            caption=self.get_image_caption(),
            use_column_width=False
        )

        # Handle user interactions
        self.handle_interactions()

    def render_controls(self):
        """Render zoom and pan controls"""
        cols = st.columns(5)

        with cols[0]:
            if st.button("🔍+", key=f"{self.key}_zoom_in"):
                self.zoom_level *= 1.2
                self.update_session_state()

        with cols[1]:
            if st.button("🔍-", key=f"{self.key}_zoom_out"):
                self.zoom_level /= 1.2
                self.update_session_state()

        with cols[2]:
            if st.button("📐", key=f"{self.key}_fit"):
                self.fit_to_window()

        with cols[3]:
            if st.button("1:1", key=f"{self.key}_actual"):
                self.zoom_level = 1.0
                self.update_session_state()

        with cols[4]:
            zoom_percent = int(self.zoom_level * 100)
            st.write(f"{zoom_percent}%")
```

### Before/After Comparison Tools
```python
def render_slider_comparison():
    """Slider-based before/after comparison"""
    slider_position = st.slider(
        "Comparison Position",
        0.0, 1.0, 0.5,
        key="comparison_slider"
    )

    # Create composite image based on slider position
    composite_image = create_slider_composite(
        st.session_state.original_image,
        st.session_state.processed_image,
        slider_position
    )

    st.image(composite_image, use_column_width=True)

def create_slider_composite(original, processed, position):
    """Create composite image for slider comparison"""
    width = min(original.width, processed.width)
    height = min(original.height, processed.height)

    # Resize images to match if needed
    original_resized = original.resize((width, height))
    processed_resized = processed.resize((width, height))

    # Create composite based on slider position
    split_point = int(width * position)

    composite = Image.new('RGB', (width, height))

    # Left side: original
    if split_point > 0:
        left_region = original_resized.crop((0, 0, split_point, height))
        composite.paste(left_region, (0, 0))

    # Right side: processed
    if split_point < width:
        right_region = processed_resized.crop((split_point, 0, width, height))
        composite.paste(right_region, (split_point, 0))

    # Add divider line
    draw = ImageDraw.Draw(composite)
    draw.line([(split_point, 0), (split_point, height)], fill='red', width=2)

    return composite
```

### Pipeline Step Viewer
```python
def render_pipeline_step_viewer():
    """Display results from each pipeline step"""
    if not st.session_state.execution_results:
        st.info("Run pipeline to see step results")
        return

    st.subheader("Pipeline Steps")

    # Step selector
    step_names = [
        f"Step {r['step']+1}: {r['operation']}"
        for r in st.session_state.execution_results
    ]

    selected_step = st.selectbox(
        "Select Step",
        range(len(step_names)),
        format_func=lambda x: step_names[x],
        key="selected_step"
    )

    if selected_step is not None:
        render_step_comparison(selected_step)

def render_step_comparison(step_index):
    """Show before/after for selected step"""
    results = st.session_state.execution_results

    if step_index == 0:
        before_image = st.session_state.original_image
    else:
        before_image = results[step_index - 1]['image']

    after_image = results[step_index]['image']
    operation_name = results[step_index]['operation']

    st.subheader(f"Step {step_index + 1}: {operation_name}")

    # Performance info
    if results[step_index].get('execution_time'):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Execution Time", f"{results[step_index]['execution_time']:.2f}s")
        with col2:
            cached = "Yes" if results[step_index].get('cached') else "No"
            st.metric("Cached", cached)
        with col3:
            memory_mb = results[step_index].get('memory_usage', 0) / (1024*1024)
            st.metric("Memory", f"{memory_mb:.1f} MB")

    # Image comparison
    render_step_image_comparison(before_image, after_image)
```

### Image Display Optimization
```python
def optimize_image_for_display(image, max_width=800, max_height=600):
    """Optimize image for web display without losing aspect ratio"""
    original_width, original_height = image.size

    # Calculate scaling factor
    width_scale = max_width / original_width
    height_scale = max_height / original_height
    scale_factor = min(width_scale, height_scale, 1.0)  # Don't upscale

    if scale_factor < 1.0:
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return image

def display_image_with_info(image, key):
    """Display image with detailed information"""
    # Display optimized image
    display_image = optimize_image_for_display(image)
    st.image(display_image, use_column_width=False)

    # Image information
    with st.expander("Image Information"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Dimensions", f"{image.width} × {image.height}")
        with col2:
            st.metric("Mode", image.mode)
        with col3:
            file_size = len(image.tobytes()) / 1024
            st.metric("Size", f"{file_size:.1f} KB")
```

## Advanced Display Features

### Pixel-Level Inspection
- Magnified view with pixel grid
- Color value display on hover
- RGB/HSV value inspection
- Coordinate system overlay

### Image Metadata Display
- EXIF data visualization
- Color profile information
- Histogram display
- Processing history

### Export and Download Options
- Download processed images
- Export comparison images
- Save step-by-step results
- Generate processing reports

### Performance Indicators
- Load time display
- Rendering performance metrics
- Memory usage indicators
- Cache status indicators

## Responsive Design

### Mobile Compatibility
- Touch-friendly controls
- Responsive image sizing
- Swipe gestures for comparison
- Simplified interface for small screens

### Accessibility Features
- Keyboard navigation
- Screen reader support
- High contrast mode
- Zoom accessibility controls

## Integration Points

### With Pipeline Execution (Task 019)
- Display results as they become available
- Update comparison views in real-time
- Handle execution errors in display

### With Parameter Configuration (Task 018)
- Show parameter effects immediately
- Provide visual feedback for parameter changes
- Integration with parameter preview system

### With Export System (Task 021)
- Prepare images for export
- Generate comparison images
- Create processing documentation

## Success Criteria
- [ ] Side-by-side comparison works smoothly with different image sizes
- [ ] Interactive zoom and pan controls are responsive
- [ ] Before/after comparison tools provide clear visual feedback
- [ ] Pipeline step visualization helps understand operation effects
- [ ] Image display is optimized for performance
- [ ] Comparison tools work with various image formats
- [ ] Display adapts to different screen sizes
- [ ] Visual indicators clearly show processing status

## Test Cases to Implement
- **Image Display Test**: Various image formats and sizes display correctly
- **Comparison Tools Test**: All comparison modes work as expected
- **Zoom and Pan Test**: Interactive controls respond properly
- **Step Visualization Test**: Pipeline steps display correctly
- **Performance Test**: Display performance is acceptable for large images
- **Responsive Design Test**: Interface works on different screen sizes
- **Accessibility Test**: Interface is accessible via keyboard and screen readers

## Dependencies
- Builds on: Task 019 (Real-time Pipeline Execution)
- Required: PIL/Pillow for image manipulation
- Blocks: Task 021 (Export and Sharing System)

## Notes
- Prioritize smooth user experience over feature complexity
- Ensure comparison tools are intuitive for non-technical users
- Optimize for common image sizes and formats
- Consider bandwidth and loading times for large images
- Make visual feedback immediate and clear
