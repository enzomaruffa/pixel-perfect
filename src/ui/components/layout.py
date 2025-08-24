"""Layout components for the Streamlit interface."""

import streamlit as st
from PIL import Image

from core.pipeline import Pipeline
from ui.components.image_viewer import render_image_display, render_image_upload
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

    render_operation_details_modal()

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

    # Pipeline execution section
    if st.session_state.pipeline_operations:
        st.header("🚀 Pipeline Execution")

        if st.button("▶️ Execute Pipeline", type="primary"):
            execute_pipeline()
    else:
        st.info("Add operations to your pipeline to begin processing")


def execute_pipeline():
    """Execute the current pipeline (basic implementation for Task 016)."""
    try:
        st.session_state.execution_in_progress = True

        # Create Pipeline instance
        # For now, we'll use a temporary file approach
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            st.session_state.original_image.save(tmp_file.name)

            try:
                # Create pipeline
                pipeline = Pipeline(tmp_file.name, verbose=False)

                # Add operations
                for op_config in st.session_state.pipeline_operations:
                    operation_class = op_config["class"]
                    params = op_config["params"]
                    operation = operation_class(**params)
                    pipeline.add(operation)

                # Execute
                with st.spinner("Processing pipeline..."):
                    pipeline.execute()

                # Load the processed image
                # For basic implementation, we'll load the final result
                final_image_path = pipeline.output_dir / "final.png"
                if final_image_path.exists():
                    st.session_state.processed_image = Image.open(final_image_path)
                    st.success("✅ Pipeline executed successfully!")
                else:
                    st.error("❌ Pipeline execution failed - no output generated")

            finally:
                # Clean up temp file
                os.unlink(tmp_file.name)

    except Exception as e:
        st.error(f"❌ Pipeline execution failed: {str(e)}")
        st.session_state.execution_error = str(e)

    finally:
        st.session_state.execution_in_progress = False
        st.rerun()
