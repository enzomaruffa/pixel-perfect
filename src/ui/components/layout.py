"""Layout components for the Streamlit interface."""

import streamlit as st

from ui.components.image_viewer import render_image_display, render_image_upload
from ui.components.pipeline_executor import get_pipeline_executor
from ui.components.session import get_pipeline_summary, reset_all, reset_pipeline


def render_header():
    """Render the application header with title and controls."""

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.title("🎨 Pixel Perfect")
        st.caption("Sophisticated image processing with visual pipeline builder")

    with col2:
        if st.button("🔄 Reset Pipeline", help="Clear all operations"):
            reset_pipeline()
            st.rerun()

    with col3:
        if st.button("🗑️ Reset All", help="Clear everything"):
            reset_all()
            st.rerun()


def render_sidebar():
    """Render the sidebar with file upload and basic controls."""

    st.header("📁 Image Upload")

    # Image upload
    render_image_upload()

    # Pipeline summary
    st.header("⚙️ Pipeline")
    summary = get_pipeline_summary()

    if summary["has_image"]:
        st.success("✅ Image loaded")
    else:
        st.info("📤 Upload an image to begin")

    if summary["operation_count"] > 0:
        st.info(f"🔧 {summary['operation_count']} operation(s) in pipeline")
    else:
        st.info("➕ Add operations to build pipeline")

    if summary["has_results"]:
        st.success("✨ Pipeline results available")

    if summary["has_error"]:
        st.error("❌ Pipeline error occurred")

    # Operation browser with full functionality
    from ui.components.operation_browser import render_operation_browser, render_pipeline_summary

    render_operation_browser()

    # Pipeline summary
    st.header("🔗 Pipeline")
    render_pipeline_summary()


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
        st.info("👆 Upload an image in the sidebar to get started")

        # Show welcome message
        st.markdown("""
        ## Welcome to Pixel Perfect!

        This is a sophisticated image processing framework that applies pixel-level
        transformations through a visual pipeline builder.

        ### Features:
        - 🎯 **Multi-granularity operations** - Work at pixel, row, column, or block levels
        - ⚡ **Real-time preview** - See changes as you adjust parameters
        - 🧩 **Composable pipeline** - Chain operations for complex effects
        - 💾 **Smart caching** - Fast iteration with automatic result caching
        - 📤 **Export & Share** - Save your work and share configurations

        ### Get Started:
        1. Upload an image using the sidebar
        2. Add operations to build your processing pipeline
        3. Watch the magic happen in real-time!
        """)
        return

    # Image display
    render_image_display()

    # Real-time pipeline execution section
    if st.session_state.pipeline_operations:
        render_pipeline_execution_controls()
    else:
        st.info("Add operations to your pipeline to begin processing")


def render_pipeline_execution_controls():
    """Render real-time pipeline execution controls and status."""
    st.header("🚀 Real-time Execution")

    # Get pipeline executor
    executor = get_pipeline_executor()

    # Execution controls
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        auto_execute = st.checkbox(
            "Auto Execute",
            value=st.session_state.get("auto_execute", True),
            help="Execute pipeline automatically when parameters change",
        )
        st.session_state.auto_execute = auto_execute

    with col2:
        if st.button("▶️ Execute Now", type="primary"):
            execute_pipeline_realtime(executor, force_refresh=True)

    with col3:
        execute_up_to = st.selectbox(
            "Execute up to step",
            options=list(range(1, len(st.session_state.pipeline_operations) + 1)),
            index=len(st.session_state.pipeline_operations) - 1,
            format_func=lambda i: f"Step {i}: {st.session_state.pipeline_operations[i - 1]['name']}",
        )

    with col4:
        if st.button("🗑️ Clear Cache"):
            executor.cache.clear()
            st.success("Cache cleared!")
            st.rerun()

    # Auto-execute when enabled and parameters change
    if (
        auto_execute
        and should_auto_execute()
        or st.button("🎯 Execute to Step", key="execute_partial")
    ):
        execute_pipeline_realtime(executor, execute_up_to=execute_up_to)

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

    except Exception as e:
        st.error(f"❌ Execution failed: {str(e)}")


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
