"""Grid-based multi-image viewer for comparing pipeline variations."""

import io

import streamlit as st
from PIL import Image

from ui.design_system import Colors, Spacing, render_section_header
from ui.utils.image_utils import optimize_image_for_display


class GridViewer:
    """Multi-image grid viewer for comparing pipeline variations."""

    def __init__(self):
        self.images: list[dict] = []
        self.grid_size = (2, 2)  # columns, rows

    def add_image(self, image: Image.Image, title: str, metadata: dict | None = None):
        """Add an image to the grid."""
        self.images.append({"image": image, "title": title, "metadata": metadata or {}})

    def render(self, thumbnail_size: int = 300):
        """Render the grid viewer interface."""
        if not self.images:
            st.info("📊 No images to display in grid view")
            return

        render_section_header("Multi-Image Grid View", "🔲")

        # Grid controls
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            grid_columns = st.select_slider(
                "Grid Columns",
                options=[1, 2, 3, 4, 5],
                value=min(3, len(self.images)),
                help="Number of columns in the grid",
            )

        with col2:
            thumbnail_size = st.select_slider(
                "Thumbnail Size",
                options=[150, 200, 250, 300, 400],
                value=250,
                help="Size of thumbnails in pixels",
            )

        with col3:
            show_metadata = st.checkbox("Show Metadata", value=True)

        # Calculate grid dimensions
        total_images = len(self.images)
        grid_rows = (total_images + grid_columns - 1) // grid_columns

        st.markdown(f"**Displaying {total_images} images in {grid_columns}×{grid_rows} grid**")

        # Render grid
        for row in range(grid_rows):
            cols = st.columns(grid_columns)

            for col_idx in range(grid_columns):
                image_idx = row * grid_columns + col_idx

                if image_idx < total_images:
                    with cols[col_idx]:
                        self._render_grid_item(
                            self.images[image_idx], thumbnail_size, show_metadata, image_idx
                        )

    def _render_grid_item(self, item: dict, size: int, show_metadata: bool, index: int):
        """Render a single item in the grid."""
        image = item["image"]
        title = item["title"]
        metadata = item["metadata"]

        # Optimize image for display
        display_image = optimize_image_for_display(image, max_size=(size, size))

        # Container with border
        st.markdown(
            f"""
            <div style="
                border: 1px solid {Colors.BORDER_LIGHT};
                border-radius: 8px;
                padding: {Spacing.SM};
                margin-bottom: {Spacing.MD};
                background: {Colors.BG_PRIMARY};
            ">
            """,
            unsafe_allow_html=True,
        )

        # Image
        if st.button("🔍", key=f"inspect_{index}", help="Inspect this image"):
            st.session_state[f"selected_grid_image_{index}"] = item

        st.image(display_image, caption=title, use_container_width=True)

        # Basic info
        st.caption(f"📐 {image.width}×{image.height} | {image.mode}")

        # Metadata
        if show_metadata and metadata:
            with st.expander("ℹ️ Details", expanded=False):
                for key, value in metadata.items():
                    st.write(f"**{key}**: {value}")

        st.markdown("</div>", unsafe_allow_html=True)

    def render_comparison_mode(self):
        """Render side-by-side comparison of selected images."""
        if len(self.images) < 2:
            st.info("🔄 Need at least 2 images for comparison mode")
            return

        render_section_header("Grid Comparison Mode", "⚖️")

        # Image selection
        image_titles = [img["title"] for img in self.images]

        col1, col2 = st.columns(2)

        with col1:
            left_selection = st.selectbox(
                "Left Image:", image_titles, index=0, key="grid_left_selection"
            )

        with col2:
            right_selection = st.selectbox(
                "Right Image:",
                image_titles,
                index=min(1, len(image_titles) - 1),
                key="grid_right_selection",
            )

        # Find selected images
        left_image = next(img for img in self.images if img["title"] == left_selection)
        right_image = next(img for img in self.images if img["title"] == right_selection)

        # Display comparison
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**{left_image['title']}**")
            st.image(left_image["image"], use_container_width=True)
            self._render_image_stats(left_image["image"])

        with col2:
            st.markdown(f"**{right_image['title']}**")
            st.image(right_image["image"], use_container_width=True)
            self._render_image_stats(right_image["image"])

        # Comparison metrics
        if st.button("📊 Calculate Comparison Metrics"):
            self._calculate_comparison_metrics(left_image["image"], right_image["image"])

    def _render_image_stats(self, image: Image.Image):
        """Render basic statistics for an image."""
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Width", image.width)
            st.metric("Format", image.format or "Unknown")

        with col2:
            st.metric("Height", image.height)
            st.metric("Mode", image.mode)

    def _calculate_comparison_metrics(self, img1: Image.Image, img2: Image.Image):
        """Calculate and display comparison metrics between two images."""
        # Import here to avoid circular imports
        from ui.components.image_comparison import calculate_image_comparison

        try:
            comparison = calculate_image_comparison(img1, img2)

            st.markdown("**Comparison Metrics**")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("SSIM Score", f"{comparison.get('ssim_score', 0):.3f}")

            with col2:
                st.metric("MSE", f"{comparison.get('mse', 0):.2f}")

            with col3:
                st.metric("PSNR", f"{comparison.get('psnr', 0):.2f}")

            # Visual difference
            if "diff_image" in comparison:
                st.image(
                    comparison["diff_image"], caption="Visual Difference", use_container_width=True
                )

        except Exception as e:
            st.error(f"Failed to calculate comparison metrics: {e}")


class FullscreenViewer:
    """Professional full-screen image viewer with advanced controls."""

    def __init__(self, image: Image.Image, title: str = "Fullscreen Viewer"):
        self.image = image
        self.title = title

    def render(self):
        """Render the full-screen viewer interface."""
        render_section_header(self.title, "🖼️")

        # Viewer controls
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

        with col1:
            zoom_level = st.select_slider(
                "Zoom Level",
                options=[0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
                value=1.0,
                help="Zoom level for detailed viewing",
            )

        with col2:
            show_info = st.checkbox("Show Info", value=True)

        with col3:
            show_grid = st.checkbox("Pixel Grid", value=False)

        with col4:
            invert_colors = st.checkbox("Invert Colors", value=False)

        # Image processing
        display_image = self.image.copy()

        if invert_colors and display_image.mode in ("RGB", "RGBA"):
            # Invert RGB channels
            from PIL import ImageOps

            display_image = ImageOps.invert(display_image.convert("RGB"))
            if self.image.mode == "RGBA":
                display_image = display_image.convert("RGBA")

        # Apply zoom
        if zoom_level != 1.0:
            new_width = int(self.image.width * zoom_level)
            new_height = int(self.image.height * zoom_level)

            if zoom_level > 4.0:
                # Use nearest neighbor for pixel art effect at high zoom
                display_image = display_image.resize(
                    (new_width, new_height), Image.Resampling.NEAREST
                )
            else:
                display_image = display_image.resize(
                    (new_width, new_height), Image.Resampling.LANCZOS
                )

        # Add pixel grid for very high zoom levels
        if show_grid and zoom_level >= 8.0:
            display_image = self._add_pixel_grid(display_image, zoom_level)

        # Display the image
        st.image(display_image, use_container_width=True)

        # Image information
        if show_info:
            self._render_image_info()

        # Navigation controls
        self._render_navigation_controls(zoom_level)

    def _add_pixel_grid(self, image: Image.Image, zoom_level: float) -> Image.Image:
        """Add pixel grid overlay for high zoom levels."""
        from PIL import ImageDraw

        draw = ImageDraw.Draw(image)
        width, height = image.size

        # Calculate original pixel size in zoomed image
        pixel_size = int(zoom_level)

        # Grid color
        grid_color = (128, 128, 128, 80)  # Semi-transparent gray

        # Draw vertical lines
        for x in range(0, width, pixel_size):
            draw.line([(x, 0), (x, height)], fill=grid_color, width=1)

        # Draw horizontal lines
        for y in range(0, height, pixel_size):
            draw.line([(0, y), (width, y)], fill=grid_color, width=1)

        return image

    def _render_image_info(self):
        """Render detailed image information."""
        st.markdown("**Image Information**")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Width", f"{self.image.width} px")

        with col2:
            st.metric("Height", f"{self.image.height} px")

        with col3:
            st.metric("Mode", self.image.mode)

        with col4:
            aspect_ratio = self.image.width / self.image.height
            st.metric("Aspect Ratio", f"{aspect_ratio:.2f}")

        # Additional technical info
        with st.expander("🔧 Technical Details"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Format**: {self.image.format or 'Unknown'}")
                st.write(f"**Bands**: {len(self.image.getbands())}")

            with col2:
                file_size = len(self.image.tobytes())
                st.write(f"**Memory Size**: {file_size:,} bytes")
                st.write(f"**Bands**: {self.image.getbands()}")

    def _render_navigation_controls(self, current_zoom: float):
        """Render navigation and control buttons."""
        st.markdown("**Navigation Controls**")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("🔍➕ Zoom In"):
                zoom_options = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
                current_idx = (
                    zoom_options.index(current_zoom) if current_zoom in zoom_options else 3
                )
                if current_idx < len(zoom_options) - 1:
                    st.session_state.fullscreen_zoom = zoom_options[current_idx + 1]
                    st.rerun()

        with col2:
            if st.button("🔍➖ Zoom Out"):
                zoom_options = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
                current_idx = (
                    zoom_options.index(current_zoom) if current_zoom in zoom_options else 3
                )
                if current_idx > 0:
                    st.session_state.fullscreen_zoom = zoom_options[current_idx - 1]
                    st.rerun()

        with col3:
            if st.button("🏠 Fit to Screen"):
                st.session_state.fullscreen_zoom = 1.0
                st.rerun()

        with col4:
            if st.button("💾 Download"):
                # Create download link
                img_buffer = io.BytesIO()
                self.image.save(img_buffer, format="PNG")
                img_buffer.seek(0)

                st.download_button(
                    label="💾 Download PNG",
                    data=img_buffer.getvalue(),
                    file_name=f"{self.title.replace(' ', '_')}.png",
                    mime="image/png",
                )


def render_grid_viewer():
    """Render grid viewer for comparing multiple images."""
    if not st.session_state.get("original_image"):
        st.info("📊 Upload an image to use the grid viewer")
        return

    # Create grid viewer instance
    grid = GridViewer()

    # Add original image
    grid.add_image(
        st.session_state.original_image, "Original", {"source": "uploaded", "type": "original"}
    )

    # Add processed image if available
    if st.session_state.get("processed_image"):
        grid.add_image(
            st.session_state.processed_image,
            "Final Result",
            {"source": "pipeline", "type": "processed"},
        )

    # Add step images if available
    if (
        st.session_state.get("last_execution_result")
        and st.session_state.last_execution_result.steps
    ):
        for i, step in enumerate(st.session_state.last_execution_result.steps):
            grid.add_image(
                step["image"],
                f"Step {i + 1}: {step['operation']}",
                {
                    "operation": step["operation"],
                    "execution_time": f"{step['execution_time']:.3f}s",
                    "cached": "Yes" if step.get("cached", False) else "No",
                },
            )

    # Render grid
    tab1, tab2 = st.tabs(["🔲 Grid View", "⚖️ Compare"])

    with tab1:
        grid.render()

    with tab2:
        grid.render_comparison_mode()


def render_fullscreen_viewer():
    """Render full-screen viewer for detailed inspection."""
    if not st.session_state.get("original_image"):
        st.info("🖼️ Upload an image to use the full-screen viewer")
        return

    # Image selection
    view_options = ["Original Image"]
    if st.session_state.get("processed_image"):
        view_options.append("Processed Image")

    if (
        st.session_state.get("last_execution_result")
        and st.session_state.last_execution_result.steps
    ):
        step_names = [
            f"Step {i + 1}: {step['operation']}"
            for i, step in enumerate(st.session_state.last_execution_result.steps)
        ]
        view_options.extend(step_names)

    selected_view = st.selectbox("Select image to view:", view_options)

    # Get selected image
    if selected_view == "Original Image":
        image = st.session_state.original_image
        title = "Original Image"
    elif selected_view == "Processed Image":
        image = st.session_state.processed_image
        title = "Processed Image"
    else:
        # Step image
        step_index = int(selected_view.split(":")[0].split(" ")[1]) - 1
        step = st.session_state.last_execution_result.steps[step_index]
        image = step["image"]
        title = selected_view

    # Render full-screen viewer
    viewer = FullscreenViewer(image, title)
    viewer.render()
