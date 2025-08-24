"""Image upload and display components."""

import io

import streamlit as st
from PIL import Image

from ui.utils.image_utils import optimize_image_for_display, validate_uploaded_image


def render_image_upload():
    """Render the image upload interface."""

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tiff"],
        help="Upload an image to begin processing",
    )

    if uploaded_file is not None:
        try:
            # Validate the uploaded file
            if validate_uploaded_image(uploaded_file):
                # Load and store the image
                image = Image.open(uploaded_file)
                st.session_state.original_image = image

                # Show image preview in sidebar
                preview_image = optimize_image_for_display(image, max_width=300, max_height=200)
                st.image(preview_image, caption=f"Uploaded: {uploaded_file.name}")

                # Show image info
                st.write(f"**Size:** {image.width} × {image.height}")
                st.write(f"**Mode:** {image.mode}")

        except Exception as e:
            st.error(f"Error loading image: {str(e)}")


def render_image_display():
    """Render the main image display area."""

    if not st.session_state.original_image:
        return

    # Display mode selector
    display_modes = ["Original Only", "Side by Side", "Processed Only"]

    if st.session_state.processed_image:
        display_mode = st.selectbox(
            "Display Mode",
            display_modes,
            index=1,  # Default to side by side
        )
    else:
        display_mode = "Original Only"

    # Render based on display mode
    if display_mode == "Original Only" or not st.session_state.processed_image:
        render_single_image(st.session_state.original_image, "Original Image")

    elif display_mode == "Processed Only":
        render_single_image(st.session_state.processed_image, "Processed Image")

    elif display_mode == "Side by Side":
        render_side_by_side_comparison()


def render_single_image(image: Image.Image, title: str):
    """Render a single image with information."""

    st.subheader(title)

    # Display optimized image
    display_image = optimize_image_for_display(image, max_width=800, max_height=600)
    st.image(display_image, use_column_width=True)

    # Image information
    with st.expander("Image Information"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Width", f"{image.width}px")
        with col2:
            st.metric("Height", f"{image.height}px")
        with col3:
            st.metric("Mode", image.mode)
        with col4:
            # Estimate file size
            img_buffer = io.BytesIO()
            image.save(img_buffer, format="PNG")
            size_kb = len(img_buffer.getvalue()) / 1024
            st.metric("Size", f"{size_kb:.1f} KB")


def render_side_by_side_comparison():
    """Render side-by-side image comparison."""

    st.subheader("Image Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Original**")
        display_original = optimize_image_for_display(
            st.session_state.original_image, max_width=400, max_height=400
        )
        st.image(display_original, use_column_width=True)

    with col2:
        st.write("**Processed**")
        display_processed = optimize_image_for_display(
            st.session_state.processed_image, max_width=400, max_height=400
        )
        st.image(display_processed, use_column_width=True)

    # Comparison metrics
    render_image_comparison_metrics()


def render_image_comparison_metrics():
    """Render comparison metrics between original and processed images."""

    if not (st.session_state.original_image and st.session_state.processed_image):
        return

    with st.expander("Comparison Metrics"):
        original = st.session_state.original_image
        processed = st.session_state.processed_image

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Size Change",
                f"{processed.width}×{processed.height}",
                delta=f"{processed.width - original.width}×{processed.height - original.height}",
            )

        with col2:
            original_pixels = original.width * original.height
            processed_pixels = processed.width * processed.height
            pixel_change = processed_pixels - original_pixels
            st.metric("Pixels", f"{processed_pixels:,}", delta=f"{pixel_change:,}")

        with col3:
            st.metric(
                "Mode",
                processed.mode,
                delta=processed.mode if processed.mode != original.mode else None,
            )

        with col4:
            # Estimate processing impact on file size
            orig_buffer = io.BytesIO()
            proc_buffer = io.BytesIO()
            original.save(orig_buffer, format="PNG")
            processed.save(proc_buffer, format="PNG")

            orig_size = len(orig_buffer.getvalue()) / 1024
            proc_size = len(proc_buffer.getvalue()) / 1024
            size_change = proc_size - orig_size

            st.metric("Size (KB)", f"{proc_size:.1f}", delta=f"{size_change:+.1f}")
