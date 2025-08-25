"""Advanced image display components with real-time updates."""

import io

import streamlit as st
from PIL import Image

from ui.utils.image_utils import optimize_image_for_display


def render_advanced_image_display():
    """Render advanced image display with comparison tools."""

    if not st.session_state.get("original_image"):
        st.info("👆 Upload an image to see the advanced display features")
        return

    # Display controls
    render_display_controls()

    # Get display mode from session state
    display_mode = st.session_state.get("display_mode", "side_by_side")

    if display_mode == "original_only":
        render_single_image_advanced(st.session_state.original_image, "Original Image")

    elif display_mode == "processed_only" and st.session_state.get("processed_image"):
        render_single_image_advanced(st.session_state.processed_image, "Processed Image")

    elif display_mode == "side_by_side" and st.session_state.get("processed_image"):
        render_side_by_side_advanced()

    elif display_mode == "comparison_tools" and st.session_state.get("processed_image"):
        render_interactive_comparison()

    elif display_mode == "step_viewer" and st.session_state.get("last_execution_result"):
        render_step_by_step_viewer()

    else:
        # Fallback to original image
        render_single_image_advanced(st.session_state.original_image, "Original Image")


def render_display_controls():
    """Render display mode and zoom controls."""

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        # Display mode selector
        mode_labels = {
            "side_by_side": "📱 Side by Side",
            "original_only": "📷 Original Only",
            "processed_only": "✨ Processed Only",
            "comparison_tools": "🔀 Interactive Compare",
            "step_viewer": "👁️ Step by Step",
        }

        # Filter available modes based on what images we have
        available_modes = ["original_only"]
        if st.session_state.get("processed_image"):
            available_modes.extend(["processed_only", "side_by_side", "comparison_tools"])
        if st.session_state.get("last_execution_result"):
            available_modes.append("step_viewer")

        st.selectbox(
            "Display Mode",
            available_modes,
            format_func=lambda x: mode_labels.get(x, str(x)) if x is not None else "Unknown",
            index=available_modes.index("side_by_side") if "side_by_side" in available_modes else 0,
            key="display_mode",
        )

    with col2:
        # Zoom level control
        st.selectbox(
            "Zoom",
            [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
            index=3,  # Default to 100%
            format_func=lambda x: f"{int(x * 100)}%",
            key="zoom_level",
        )

    with col3:
        # Image info toggle
        st.checkbox("Show Info", value=False, key="show_image_info")


def render_single_image_advanced(image: Image.Image, title: str):
    """Render a single image with advanced controls."""

    st.subheader(title)

    # Apply zoom level
    zoom = st.session_state.get("zoom_level", 1.0)
    display_width = min(800, int(800 * zoom))
    display_height = min(600, int(600 * zoom))

    display_image = optimize_image_for_display(image, display_width, display_height)
    st.image(display_image, use_container_width=(zoom == 1.0))

    # Show image info if requested
    if st.session_state.get("show_image_info"):
        render_image_info(image, title)


def render_side_by_side_advanced():
    """Render advanced side-by-side comparison."""

    st.subheader("📱 Side by Side Comparison")

    # Synchronization toggle
    st.checkbox("Sync Zoom & Pan", value=True, key="sync_images")

    col1, col2 = st.columns(2, gap="medium")

    zoom = st.session_state.get("zoom_level", 1.0)
    display_width = min(400, int(400 * zoom))
    display_height = min(400, int(400 * zoom))

    with col1:
        st.write("**🖼️ Original**")
        original_display = optimize_image_for_display(
            st.session_state.original_image, display_width, display_height
        )
        st.image(original_display, use_container_width=(zoom == 1.0))

        if st.session_state.get("show_image_info"):
            render_image_info(st.session_state.original_image, "Original")

    with col2:
        st.write("**✨ Processed**")
        processed_display = optimize_image_for_display(
            st.session_state.processed_image, display_width, display_height
        )
        st.image(processed_display, use_container_width=(zoom == 1.0))

        if st.session_state.get("show_image_info"):
            render_image_info(st.session_state.processed_image, "Processed")

    # Comparison metrics
    render_comparison_metrics()


def render_interactive_comparison():
    """Render interactive comparison tools."""

    st.subheader("🔀 Interactive Comparison")

    # Comparison tool selector
    comp_tool = st.selectbox(
        "Comparison Tool",
        ["slider", "toggle", "split_screen", "overlay"],
        format_func=lambda x: {
            "slider": "🎚️ Slider Reveal",
            "toggle": "🔄 Toggle Switch",
            "split_screen": "⚡ Split Screen",
            "overlay": "🎭 Overlay Mode",
        }[x],
        key="comparison_tool",
    )

    if comp_tool == "slider":
        render_slider_comparison()
    elif comp_tool == "toggle":
        render_toggle_comparison()
    elif comp_tool == "split_screen":
        render_split_screen_comparison()
    elif comp_tool == "overlay":
        render_overlay_comparison()


def render_slider_comparison():
    """Render slider-based before/after comparison."""

    # Slider control
    reveal_percent = st.slider(
        "Reveal Amount",
        0,
        100,
        50,
        help="Drag to reveal more of the processed image",
        key="reveal_slider",
    )

    st.write(f"**Revealing {reveal_percent}% of processed image**")

    # For now, show side by side with a visual indicator
    # In a full implementation, this would use custom HTML/JS
    col1, col2 = st.columns([reveal_percent, 100 - reveal_percent])

    with col1:
        if reveal_percent > 0:
            st.write("**✨ Processed**")
            processed_display = optimize_image_for_display(
                st.session_state.processed_image, 400, 400
            )
            st.image(processed_display, use_container_width=True)

    with col2:
        if reveal_percent < 100:
            st.write("**🖼️ Original**")
            original_display = optimize_image_for_display(st.session_state.original_image, 400, 400)
            st.image(original_display, use_container_width=True)


def render_toggle_comparison():
    """Render toggle-based comparison."""

    show_processed = st.checkbox(
        "Show Processed Image",
        value=False,
        help="Toggle between original and processed",
        key="toggle_processed",
    )

    if show_processed:
        render_single_image_advanced(st.session_state.processed_image, "✨ Processed Image")
    else:
        render_single_image_advanced(st.session_state.original_image, "🖼️ Original Image")

    # Quick toggle button
    if st.button("🔄 Quick Toggle", help="Quickly switch views"):
        st.session_state.toggle_processed = not st.session_state.get("toggle_processed", False)
        st.rerun()


def render_split_screen_comparison():
    """Render split screen comparison."""

    # Split orientation
    orientation = st.radio(
        "Split Orientation",
        ["horizontal", "vertical"],
        format_func=lambda x: "⬌ Horizontal" if x == "horizontal" else "⬍ Vertical",
        key="split_orientation",
        horizontal=True,
    )

    if orientation == "horizontal":
        col1, col2 = st.columns(2, gap="small")

        with col1:
            st.write("**🖼️ Original**")
            original_display = optimize_image_for_display(st.session_state.original_image, 400, 300)
            st.image(original_display, use_container_width=True)

        with col2:
            st.write("**✨ Processed**")
            processed_display = optimize_image_for_display(
                st.session_state.processed_image, 400, 300
            )
            st.image(processed_display, use_container_width=True)
    else:
        # Vertical split - show one above the other
        st.write("**🖼️ Original**")
        original_display = optimize_image_for_display(st.session_state.original_image, 600, 300)
        st.image(original_display, use_container_width=True)

        st.write("**✨ Processed**")
        processed_display = optimize_image_for_display(st.session_state.processed_image, 600, 300)
        st.image(processed_display, use_container_width=True)


def render_overlay_comparison():
    """Render overlay comparison with opacity control."""

    opacity = st.slider(
        "Processed Image Opacity",
        0.0,
        1.0,
        0.5,
        help="Adjust opacity of processed image overlay",
        key="overlay_opacity",
    )

    st.write(f"**Overlay with {opacity:.1%} opacity**")

    # Show base image (original)
    st.write("*Base: Original Image*")
    original_display = optimize_image_for_display(st.session_state.original_image, 600, 400)
    st.image(original_display, use_container_width=True)

    # Note about overlay (in real implementation, this would be a true overlay)
    st.info(f"💡 Processed image would be overlaid at {opacity:.1%} opacity")


def render_step_by_step_viewer():
    """Render step-by-step pipeline results."""

    st.subheader("👁️ Step by Step Results")

    result = st.session_state.get("last_execution_result")
    if not result or not result.steps:
        st.info("Execute a pipeline to see step-by-step results")
        return

    # Step selector
    step_names = [f"Step {i + 1}: {step['operation']}" for i, step in enumerate(result.steps)]
    step_names.insert(0, "Original Image")

    selected_step = st.selectbox(
        "Select Step to View",
        range(len(step_names)),
        format_func=lambda i: step_names[i],
        key="selected_step",
    )

    # Display selected step
    if selected_step == 0:
        # Show original
        render_single_image_advanced(st.session_state.original_image, "📷 Original Image")
    else:
        # Show step result
        step_result = result.steps[selected_step - 1]
        step_image = step_result["image"]

        cache_icon = "💾" if step_result["cached"] else "⚡"
        title = f"{cache_icon} Step {selected_step}: {step_result['operation']}"

        render_single_image_advanced(step_image, title)

        # Step details
        with st.expander("Step Details"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Execution Time", f"{step_result['execution_time']:.3f}s")
            with col2:
                st.metric("Status", "Cached" if step_result["cached"] else "Processed")
            with col3:
                st.metric("Step Index", step_result["step"] + 1)


def render_image_info(image: Image.Image, title: str):
    """Render detailed image information."""

    with st.expander(f"ℹ️ {title} Information"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Width", f"{image.width}px")
        with col2:
            st.metric("Height", f"{image.height}px")
        with col3:
            st.metric("Mode", image.mode)
        with col4:
            # Calculate file size
            img_buffer = io.BytesIO()
            image.save(img_buffer, format="PNG")
            size_kb = len(img_buffer.getvalue()) / 1024
            st.metric("Size", f"{size_kb:.1f} KB")

        # Additional info
        st.write(f"**Pixels:** {image.width * image.height:,}")
        if hasattr(image, "info") and image.info:
            st.write("**Metadata:**")
            for key, value in image.info.items():
                st.write(f"- {key}: {value}")


def render_comparison_metrics():
    """Render detailed comparison metrics."""

    if not (st.session_state.get("original_image") and st.session_state.get("processed_image")):
        return

    with st.expander("📊 Comparison Analysis"):
        original = st.session_state.original_image
        processed = st.session_state.processed_image

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Dimension changes
            width_change = processed.width - original.width
            height_change = processed.height - original.height
            st.metric(
                "Dimensions",
                f"{processed.width}×{processed.height}",
                delta=f"{width_change:+}×{height_change:+}"
                if width_change or height_change
                else None,
            )

        with col2:
            # Pixel count change
            original_pixels = original.width * original.height
            processed_pixels = processed.width * processed.height
            pixel_change = processed_pixels - original_pixels
            change_percent = (pixel_change / original_pixels * 100) if original_pixels > 0 else 0

            st.metric(
                "Pixel Count",
                f"{processed_pixels:,}",
                delta=f"{pixel_change:+,} ({change_percent:+.1f}%)" if pixel_change else None,
            )

        with col3:
            # Mode comparison
            mode_changed = processed.mode != original.mode
            st.metric(
                "Color Mode",
                processed.mode,
                delta=f"Changed from {original.mode}" if mode_changed else "Unchanged",
            )

        with col4:
            # File size comparison
            orig_buffer = io.BytesIO()
            proc_buffer = io.BytesIO()
            original.save(orig_buffer, format="PNG")
            processed.save(proc_buffer, format="PNG")

            orig_size = len(orig_buffer.getvalue()) / 1024
            proc_size = len(proc_buffer.getvalue()) / 1024
            size_change = proc_size - orig_size
            size_percent = (size_change / orig_size * 100) if orig_size > 0 else 0

            st.metric(
                "File Size",
                f"{proc_size:.1f} KB",
                delta=f"{size_change:+.1f} KB ({size_percent:+.1f}%)"
                if abs(size_change) > 0.1
                else None,
            )
