"""Layout components for the Streamlit interface."""

import streamlit as st

from ui.components.export_manager import render_export_manager
from ui.components.image_comparison import render_comparison_analysis
from ui.components.image_display import render_advanced_image_display
from ui.components.image_viewer import render_image_upload
from ui.components.pipeline_executor import get_pipeline_executor
from ui.components.session import get_pipeline_summary, reset_all, reset_pipeline
from ui.design_system import Colors, render_section_header, render_status_badge


def render_header():
    """Render the application header with title and controls."""

    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        st.markdown(
            f"""
            <div style="margin-bottom: 24px;">
                <h1 style="font-size: 2.5rem; font-weight: 700; color: {Colors.TEXT_PRIMARY}; margin: 0; line-height: 1.2;">
                    🎨 Pixel Perfect
                </h1>
                <p style="font-size: 1.1rem; color: {Colors.TEXT_SECONDARY}; margin: 8px 0 0 0; font-weight: 400;">
                    Sophisticated image processing with visual pipeline builder
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        if st.button(
            "🔄 Reset Pipeline",
            help="Clear all operations",
            key="header_reset_pipeline",
            type="secondary",
        ):
            reset_pipeline()
            st.rerun()

    with col3:
        if st.button("🗑️ Reset All", help="Clear everything", key="header_reset_all"):
            reset_all()
            st.rerun()


def execute_pipeline_inline():
    """Execute the pipeline inline for quick preview."""
    if not st.session_state.get("original_image"):
        st.error("No image loaded")
        return

    if not st.session_state.get("pipeline_operations"):
        st.warning("No operations in pipeline")
        return

    executor = get_pipeline_executor()

    try:
        # Execute pipeline
        result = executor.execute_pipeline_realtime(
            original_image=st.session_state.original_image,
            operations=st.session_state.pipeline_operations,
            force_refresh=False,
        )

        if result.error:
            st.error(f"❌ {result.error}")
        elif result.final_image:
            # Update processed image
            st.session_state.processed_image = result.final_image
            st.session_state.last_execution_result = result

            # Reset parameters changed flag
            st.session_state.parameters_changed = False

            # Show success toast
            cache_info = (
                f" ({result.cache_hits}/{result.total_steps} cached)"
                if result.cache_hits > 0
                else ""
            )
            st.toast(f"✅ Pipeline executed in {result.execution_time:.2f}s{cache_info}", icon="✅")

            # Force UI refresh to update preview
            st.rerun()

    except Exception as e:
        # Use user-friendly error messages
        from ui.utils.error_translator import translate_error

        error_info = translate_error(e, context="pipeline_execution")

        st.error(f"❌ {error_info['message']}")

        if error_info.get("suggestion"):
            st.info(f"💡 {error_info['suggestion']}")

        # Show technical details in expander for debugging
        with st.expander("🔧 Technical Details", expanded=False):
            st.code(str(e))


def render_sidebar_preview():
    """Render a live preview of the processed image in the sidebar."""
    from PIL import Image

    # Check what to display
    has_processed = st.session_state.get("processed_image") is not None

    if not has_processed:
        st.info("⏳ No processed result yet")
        st.caption("Execute the pipeline to see results here")
        return

    # Get processed image
    processed = st.session_state.processed_image

    # Create thumbnail for sidebar display
    max_width = 260  # Sidebar width constraint

    def create_thumbnail(img, max_size=(max_width, max_width)):
        """Create a thumbnail preserving aspect ratio."""
        img_copy = img.copy()
        img_copy.thumbnail(max_size, Image.Resampling.LANCZOS)
        return img_copy

    # Display processed image
    st.image(create_thumbnail(processed), caption="Processed Result", use_container_width=True)
    st.caption(f"📐 {processed.width}×{processed.height} | {processed.mode}")

    # Execution time if available
    if st.session_state.get("last_execution_result"):
        result = st.session_state.last_execution_result
        st.caption(f"⚡ {result.execution_time:.2f}s")
        if result.cache_hits > 0:
            st.caption(f"💾 {result.cache_hits}/{result.total_steps} cached")


def render_sidebar():
    """Render the sidebar with essential controls and live preview."""

    render_section_header("Image Upload", "📁")
    render_image_upload()

    st.markdown("<br>", unsafe_allow_html=True)

    render_section_header("Live Preview", "👁️")
    render_sidebar_preview()

    st.markdown("<br>", unsafe_allow_html=True)

    render_section_header("Pipeline Status", "📊")
    summary = get_pipeline_summary()

    # Status badges
    status_col1, status_col2 = st.columns(2)

    with status_col1:
        if summary["has_image"]:
            render_status_badge("Image Ready", "success")
        else:
            render_status_badge("No Image", "info")

    with status_col2:
        if summary["operation_count"] > 0:
            render_status_badge(f"{summary['operation_count']} Ops", "info")
        else:
            render_status_badge("No Operations", "info")

    # Results status
    if summary["operation_count"] > 0:
        if summary["has_results"]:
            render_status_badge("Results Ready", "success")
        elif summary["has_error"]:
            render_status_badge("Pipeline Error", "error")
        else:
            render_status_badge("Pending Execution", "warning")


def render_main_content():
    """Render the main content area with image display and processing."""

    # Handle operation details modal
    from ui.components.operation_browser import render_operation_details_modal
    from ui.components.parameter_forms import render_operation_parameter_editor

    render_operation_details_modal()

    # Check if we're editing operation parameters
    if st.session_state.get("selected_operation_for_editing"):
        render_operation_parameter_editor()
        return

    # Check if we have an image
    if not st.session_state.original_image:
        # Show clean welcome screen with no raw HTML
        st.markdown("# 🎨 Welcome to Pixel Perfect!")
        st.markdown("### A sophisticated image processing framework with visual pipeline builder")

        st.markdown("---")

        # Feature cards using Streamlit columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🎯 Multi-Granularity Operations")
            st.write(
                "Work at pixel, row, column, or block levels with precise control over every transformation."
            )

        with col2:
            st.markdown("### ⚡ Real-Time Preview")
            st.write(
                "See changes instantly as you adjust parameters with smart caching and optimized rendering."
            )

        with col3:
            st.markdown("### 🧩 Composable Pipeline")
            st.write(
                "Chain operations together to create complex effects with a visual pipeline builder."
            )

        st.markdown("---")

        # Get started section
        st.markdown("### Get Started in 3 Easy Steps")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 1️⃣ Upload Image")
            st.write("👆 Use the sidebar to upload your image")

        with col2:
            st.markdown("#### 2️⃣ Build Pipeline")
            st.write("Add operations to create your processing pipeline")

        with col3:
            st.markdown("#### 3️⃣ Watch Magic")
            st.write("See your transformations happen in real-time!")

        return

    # Main content with tabs for better organization
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🛠️ Pipeline Builder",
            "🖼️ Advanced Preview",
            "📊 Inspector Tools",
            "📤 Export & Share",
            "📊 Analytics",
        ]
    )

    with tab1:
        # Pipeline builder - operations and configuration
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            from ui.components.operation_browser import render_operation_browser

            render_section_header("Operations Library", "🧰")
            render_operation_browser()

        with col2:
            from ui.components.preset_browser import render_preset_browser

            render_section_header("Preset Effects", "✨")
            render_preset_browser()

        with col3:
            from ui.components.operation_browser import render_pipeline_summary

            render_section_header("Pipeline Configuration", "🔗")

            # Add real-time execution toggle
            auto_preview = st.checkbox(
                "⚡ Auto-Execute on Changes",
                value=st.session_state.get("auto_preview", False),
                help="Automatically execute pipeline when parameters change",
            )
            st.session_state.auto_preview = auto_preview

            render_pipeline_summary()

            # Pipeline actions
            if st.session_state.get("pipeline_operations"):
                # Execution controls
                col_a, col_b = st.columns(2)
                with col_a:
                    execute_button = st.button(
                        "▶️ Execute Now",
                        type="primary",
                        use_container_width=True,
                        key="pipeline_execute_now",
                    )
                with col_b:
                    if st.button("🗑️ Clear All", use_container_width=True, key="pipeline_clear_all"):
                        st.session_state.pipeline_operations = []
                        st.rerun()

                # Pipeline Management Section
                st.divider()
                from ui.components.pipeline_manager import render_pipeline_actions

                render_pipeline_actions()

                # Execute pipeline if button clicked or auto-preview is on with changes
                if execute_button or (
                    auto_preview and st.session_state.get("parameters_changed", False)
                ):
                    with st.spinner("🔄 Executing pipeline..."):
                        execute_pipeline_inline()

            # Show pipeline manager modal if requested
            if st.session_state.get("show_pipeline_manager", False):
                st.session_state.show_pipeline_manager = False  # Reset flag
                with st.container():
                    from ui.components.pipeline_manager import render_pipeline_save_load

                    render_pipeline_save_load()

    with tab2:
        # Advanced Preview System with multiple view modes
        from ui.components.enhanced_step_visualizer import render_enhanced_step_visualizer
        from ui.components.grid_viewer import render_fullscreen_viewer, render_grid_viewer

        preview_mode = st.selectbox(
            "Preview Mode",
            ["Enhanced Step Visualizer", "Grid Viewer", "Full-screen Viewer", "Classic Display"],
            help="Select advanced preview mode",
        )

        if preview_mode == "Enhanced Step Visualizer":
            render_enhanced_step_visualizer()
        elif preview_mode == "Grid Viewer":
            render_grid_viewer()
        elif preview_mode == "Full-screen Viewer":
            render_fullscreen_viewer()
        else:  # Classic Display
            render_advanced_image_display()

            # Real-time pipeline execution section
            if st.session_state.pipeline_operations:
                render_pipeline_execution_controls()
            else:
                st.info("Add operations in the Pipeline Builder tab to begin processing")

    with tab3:
        # Inspector Tools for pixel-level analysis
        from ui.components.image_inspector import render_measurement_tools, render_pixel_inspector

        inspector_mode = st.selectbox(
            "Inspector Mode",
            ["Pixel Inspector", "Measurement Tools"],
            help="Select inspection tool",
        )

        if inspector_mode == "Pixel Inspector":
            render_pixel_inspector()
        else:  # Measurement Tools
            render_measurement_tools()

    with tab4:
        # Export and sharing functionality
        render_export_manager()

    with tab5:
        # Advanced comparison analysis (if both images available)
        if st.session_state.get("original_image") and st.session_state.get("processed_image"):
            render_comparison_analysis()
        else:
            st.info("Process an image to access comparison analytics")


def render_pipeline_execution_controls():
    """Render real-time pipeline execution controls and status."""
    st.header("🚀 Real-time Execution")

    # Get pipeline executor
    executor = get_pipeline_executor()

    # First row - Main controls
    col1, col2 = st.columns([1, 2])

    with col1:
        auto_execute = st.checkbox(
            "⚡ Auto Execute",
            value=st.session_state.get("auto_execute", True),
            help="Execute pipeline automatically when parameters change",
        )
        st.session_state.auto_execute = auto_execute

    with col2:
        # Partial execution selector
        execute_up_to = st.selectbox(
            "Execute up to:",
            options=list(range(1, len(st.session_state.pipeline_operations) + 1)),
            index=len(st.session_state.pipeline_operations) - 1,
            format_func=lambda i: f"Step {i}: {st.session_state.pipeline_operations[i - 1]['name']}",
        )

    # Second row - Action buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        execute_now = st.button(
            "▶️ Execute Now", type="primary", use_container_width=True, key="realtime_execute_now"
        )

    with col2:
        execute_partial = st.button(
            "🎯 Execute to Step", use_container_width=True, key="realtime_execute_partial"
        )

    with col3:
        if st.button("🗑️ Clear Cache", use_container_width=True, key="realtime_clear_cache"):
            executor.cache.clear()
            st.toast("✅ Cache cleared!", icon="🗑️")
            st.rerun()

    # Execute based on button clicks or auto-execute
    if execute_now:
        execute_pipeline_realtime(executor, force_refresh=True)
    elif execute_partial:
        execute_pipeline_realtime(executor, execute_up_to=execute_up_to)
    elif auto_execute and should_auto_execute():
        execute_pipeline_realtime(executor)

    # Show execution statistics
    render_execution_stats(executor)


def should_auto_execute():
    """Check if pipeline should auto-execute."""
    return (
        st.session_state.get("original_image") is not None
        and len(st.session_state.get("pipeline_operations", [])) > 0
        and not st.session_state.get("execution_in_progress", False)
        and st.session_state.get("parameters_changed", False)
    )


def execute_pipeline_realtime(executor, execute_up_to=None, force_refresh=False):
    """Execute pipeline using real-time executor."""
    if not st.session_state.get("original_image"):
        st.error("No image loaded")
        return

    if not st.session_state.get("pipeline_operations"):
        st.error("No operations in pipeline")
        return

    try:
        # Execute pipeline with real-time updates
        result = executor.execute_pipeline_realtime(
            original_image=st.session_state.original_image,
            operations=st.session_state.pipeline_operations,
            execute_up_to=execute_up_to,
            force_refresh=force_refresh,
        )

        if result.error:
            st.error(f"❌ {result.error}")
        elif result.cancelled:
            st.warning("⚠️ Execution was cancelled")
        elif result.final_image:
            # Update processed image in session state
            st.session_state.processed_image = result.final_image

            # Show success message with timing info
            cache_info = (
                f" ({result.cache_hits}/{result.total_steps} cached)"
                if result.cache_hits > 0
                else ""
            )
            st.success(f"✅ Pipeline executed in {result.execution_time:.2f}s{cache_info}")

            # Store execution results for display
            st.session_state.last_execution_result = result

            # Reset parameters changed flag
            st.session_state.parameters_changed = False

            # Force UI refresh to update preview
            st.rerun()

        # Reset parameters changed flag for other cases too
        st.session_state.parameters_changed = False

    except Exception as e:
        # Use user-friendly error messages
        from ui.utils.error_translator import translate_error

        error_info = translate_error(e, context="realtime_execution")

        st.error(f"❌ {error_info['message']}")

        if error_info.get("suggestion"):
            st.info(f"💡 {error_info['suggestion']}")

        # Show technical details in expander for debugging
        with st.expander("🔧 Technical Details", expanded=False):
            st.code(str(e))


def render_execution_stats(executor):
    """Render execution statistics and performance metrics."""
    stats = executor.get_execution_stats()

    with st.expander("📊 Execution Statistics", expanded=False):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Cache Entries", stats["cache"]["entries"])

        with col2:
            st.metric("Memory Usage", f"{stats['cache']['memory_usage_mb']:.1f} MB")

        with col3:
            st.metric("Average Time", f"{stats['average_execution_time']:.2f}s")

        with col4:
            st.metric("Cache Hit Rate", f"{stats['cache_hit_rate']:.1%}")

        # Show last execution details
        if st.session_state.get("last_execution_result"):
            result = st.session_state.last_execution_result
            st.subheader("Last Execution")

            for step_result in result.steps:
                cache_icon = "💾" if step_result["cached"] else "⚡"
                time_info = (
                    f"{step_result['execution_time']:.2f}s"
                    if not step_result["cached"]
                    else "cached"
                )
                st.write(
                    f"{cache_icon} Step {step_result['step'] + 1}: {step_result['operation']} ({time_info})"
                )
