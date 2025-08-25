"""Advanced image inspector tools for pixel-level analysis."""

import math

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from ui.design_system import Colors, render_section_header


class PixelInspector:
    """Pixel-level image inspector with RGB value display."""

    def __init__(self, image: Image.Image, title: str = "Image Inspector"):
        self.image = image
        self.title = title
        self.image_array = np.array(image)

    def render(self) -> None:
        """Render the pixel inspector interface."""
        render_section_header(self.title, "🔍")

        # Inspector controls
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            zoom_level = st.select_slider(
                "Zoom Level",
                options=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
                value=2.0,
                help="Zoom level for detailed inspection",
            )

        with col2:
            show_grid = st.checkbox("Show Pixel Grid", value=True)

        with col3:
            st.checkbox("Show Coordinates", value=True)

        # Region selection
        st.markdown("**Region Selection**")
        region_col1, region_col2 = st.columns(2)

        with region_col1:
            center_x = st.number_input(
                "Center X", min_value=0, max_value=self.image.width - 1, value=self.image.width // 2
            )

        with region_col2:
            center_y = st.number_input(
                "Center Y",
                min_value=0,
                max_value=self.image.height - 1,
                value=self.image.height // 2,
            )

        # Calculate region bounds
        region_size = max(50, int(100 / zoom_level))
        x1 = max(0, center_x - region_size // 2)
        y1 = max(0, center_y - region_size // 2)
        x2 = min(self.image.width, x1 + region_size)
        y2 = min(self.image.height, y1 + region_size)

        # Extract and display region
        region = self.image.crop((x1, y1, x2, y2))

        if zoom_level > 1.0:
            # Resize for pixel inspection
            display_width = int((x2 - x1) * zoom_level)
            display_height = int((y2 - y1) * zoom_level)
            region_display = region.resize(
                (display_width, display_height), Image.Resampling.NEAREST
            )

            # Add grid overlay if requested
            if show_grid and zoom_level >= 4.0:
                region_display = self._add_grid_overlay(region_display, zoom_level)
        else:
            region_display = region

        # Display the region
        st.image(region_display, caption=f"Region: ({x1},{y1}) to ({x2},{y2})")

        # Pixel value inspection
        if st.button("🎯 Inspect Center Pixel", help="Get detailed info about the center pixel"):
            self._display_pixel_info(center_x, center_y)

        # Region statistics
        self._display_region_stats(x1, y1, x2, y2)

    def _add_grid_overlay(self, image: Image.Image, zoom_level: float) -> Image.Image:
        """Add pixel grid overlay to the image."""
        draw = ImageDraw.Draw(image)
        width, height = image.size
        grid_spacing = int(zoom_level)

        # Draw vertical lines
        for x in range(0, width, grid_spacing):
            draw.line([(x, 0), (x, height)], fill=(128, 128, 128, 100), width=1)

        # Draw horizontal lines
        for y in range(0, height, grid_spacing):
            draw.line([(0, y), (width, y)], fill=(128, 128, 128, 100), width=1)

        return image

    def _display_pixel_info(self, x: int, y: int) -> None:
        """Display detailed information about a specific pixel."""
        if x >= self.image.width or y >= self.image.height:
            st.error("Coordinates out of bounds")
            return

        pixel = self.image_array[y, x]  # Note: numpy uses [y, x] indexing

        st.markdown(f"**Pixel at ({x}, {y})**")

        # Color information
        col1, col2, col3, col4 = st.columns(4)

        if len(pixel) >= 3:
            with col1:
                st.metric("Red", f"{pixel[0]}")
            with col2:
                st.metric("Green", f"{pixel[1]}")
            with col3:
                st.metric("Blue", f"{pixel[2]}")

        if len(pixel) == 4:
            with col4:
                st.metric("Alpha", f"{pixel[3]}")

        # Color preview
        if len(pixel) >= 3:
            color_hex = f"#{pixel[0]:02x}{pixel[1]:02x}{pixel[2]:02x}"
            st.markdown(
                f'<div style="background-color: {color_hex}; width: 100px; height: 50px; '
                + f'border: 1px solid {Colors.BORDER_DEFAULT}; border-radius: 4px; margin: 8px 0;"></div>',
                unsafe_allow_html=True,
            )
            st.caption(f"Hex: {color_hex}")

    def _display_region_stats(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Display statistical information about the selected region."""
        region_array = self.image_array[y1:y2, x1:x2]

        st.markdown("**Region Statistics**")

        if len(region_array.shape) == 3:  # Color image
            channels = ["Red", "Green", "Blue", "Alpha"][: region_array.shape[2]]

            stats_data = []
            for i, channel in enumerate(channels):
                channel_data = region_array[:, :, i]
                stats_data.append(
                    {
                        "Channel": channel,
                        "Mean": f"{np.mean(channel_data):.1f}",
                        "Std": f"{np.std(channel_data):.1f}",
                        "Min": f"{np.min(channel_data)}",
                        "Max": f"{np.max(channel_data)}",
                    }
                )

            st.table(stats_data)
        else:  # Grayscale
            mean_val = np.mean(region_array)
            std_val = np.std(region_array)
            min_val = np.min(region_array)
            max_val = np.max(region_array)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{mean_val:.1f}")
            with col2:
                st.metric("Std Dev", f"{std_val:.1f}")
            with col3:
                st.metric("Min", str(min_val))
            with col4:
                st.metric("Max", str(max_val))


class MeasurementTool:
    """Tools for measuring distances and areas in images."""

    def __init__(self, image: Image.Image):
        self.image = image
        self.measurements = []

    def render(self) -> None:
        """Render the measurement tools interface."""
        render_section_header("Measurement Tools", "📏")

        # Measurement controls
        col1, col2 = st.columns(2)

        with col1:
            measurement_type = st.selectbox(
                "Measurement Type",
                ["Distance", "Area", "Angle"],
                help="Select the type of measurement to perform",
            )

        with col2:
            units = st.selectbox(
                "Units", ["Pixels", "mm", "inches", "custom"], help="Units for measurements"
            )

        if units == "custom":
            scale_factor = st.number_input(
                "Pixels per unit", min_value=0.1, value=1.0, help="How many pixels equal one unit"
            )
        else:
            scale_factor = 1.0

        # Coordinate input
        if measurement_type == "Distance":
            self._render_distance_tool(scale_factor, units)
        elif measurement_type == "Area":
            self._render_area_tool(scale_factor, units)
        elif measurement_type == "Angle":
            self._render_angle_tool()

    def _render_distance_tool(self, scale_factor: float, units: str) -> None:
        """Render distance measurement tool."""
        st.markdown("**Distance Measurement**")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("*Point 1*")
            x1 = st.number_input("X1", min_value=0, max_value=self.image.width - 1, value=0)
            y1 = st.number_input("Y1", min_value=0, max_value=self.image.height - 1, value=0)

        with col2:
            st.markdown("*Point 2*")
            x2 = st.number_input("X2", min_value=0, max_value=self.image.width - 1, value=100)
            y2 = st.number_input("Y2", min_value=0, max_value=self.image.height - 1, value=100)

        # Calculate distance
        distance_pixels = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        distance_units = distance_pixels / scale_factor

        st.metric(
            f"Distance ({units})",
            f"{distance_units:.2f}",
            help=f"Distance in pixels: {distance_pixels:.2f}",
        )

        # Visual representation
        if st.button("📍 Mark on Image"):
            self._create_distance_overlay(x1, y1, x2, y2)

    def _render_area_tool(self, scale_factor: float, units: str) -> None:
        """Render area measurement tool."""
        st.markdown("**Area Measurement**")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("*Rectangle Top-Left*")
            x1 = st.number_input(
                "X", min_value=0, max_value=self.image.width - 1, value=0, key="area_x1"
            )
            y1 = st.number_input(
                "Y", min_value=0, max_value=self.image.height - 1, value=0, key="area_y1"
            )

        with col2:
            st.markdown("*Rectangle Bottom-Right*")
            x2 = st.number_input(
                "X", min_value=0, max_value=self.image.width - 1, value=100, key="area_x2"
            )
            y2 = st.number_input(
                "Y", min_value=0, max_value=self.image.height - 1, value=100, key="area_y2"
            )

        # Calculate area
        width_pixels = abs(x2 - x1)
        height_pixels = abs(y2 - y1)
        area_pixels = width_pixels * height_pixels
        area_units = area_pixels / (scale_factor * scale_factor)

        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"Width ({units})", f"{width_pixels / scale_factor:.2f}")
        with col2:
            st.metric(f"Height ({units})", f"{height_pixels / scale_factor:.2f}")

        st.metric(f"Area ({units}²)", f"{area_units:.2f}", help=f"Area in pixels²: {area_pixels}")

    def _render_angle_tool(self) -> None:
        """Render angle measurement tool."""
        st.markdown("**Angle Measurement**")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("*Point A*")
            xa = st.number_input("XA", min_value=0, max_value=self.image.width - 1, value=0)
            ya = st.number_input("YA", min_value=0, max_value=self.image.height - 1, value=0)

        with col2:
            st.markdown("*Point B (Vertex)*")
            xb = st.number_input("XB", min_value=0, max_value=self.image.width - 1, value=50)
            yb = st.number_input("YB", min_value=0, max_value=self.image.height - 1, value=50)

        with col3:
            st.markdown("*Point C*")
            xc = st.number_input("XC", min_value=0, max_value=self.image.width - 1, value=100)
            yc = st.number_input("YC", min_value=0, max_value=self.image.height - 1, value=0)

        # Calculate angle
        angle_rad = self._calculate_angle(xa, ya, xb, yb, xc, yc)
        angle_deg = math.degrees(angle_rad)

        st.metric("Angle (degrees)", f"{angle_deg:.1f}°")

    def _calculate_angle(self, xa: int, ya: int, xb: int, yb: int, xc: int, yc: int) -> float:
        """Calculate angle ABC in radians."""
        # Vectors BA and BC
        ba_x, ba_y = xa - xb, ya - yb
        bc_x, bc_y = xc - xb, yc - yb

        # Dot product and magnitudes
        dot_product = ba_x * bc_x + ba_y * bc_y
        mag_ba = math.sqrt(ba_x**2 + ba_y**2)
        mag_bc = math.sqrt(bc_x**2 + bc_y**2)

        if mag_ba == 0 or mag_bc == 0:
            return 0.0

        # Calculate angle
        cos_angle = dot_product / (mag_ba * mag_bc)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range

        return math.acos(cos_angle)

    def _create_distance_overlay(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Create image overlay with distance measurement line."""
        # Create a copy of the image for overlay
        overlay_image = self.image.copy()
        draw = ImageDraw.Draw(overlay_image)

        # Draw measurement line
        draw.line([(x1, y1), (x2, y2)], fill="red", width=2)

        # Draw endpoint markers
        marker_size = 5
        draw.ellipse(
            [x1 - marker_size, y1 - marker_size, x1 + marker_size, y1 + marker_size],
            fill="red",
            outline="white",
            width=1,
        )
        draw.ellipse(
            [x2 - marker_size, y2 - marker_size, x2 + marker_size, y2 + marker_size],
            fill="red",
            outline="white",
            width=1,
        )

        # Display the overlay
        st.image(overlay_image, caption="Distance Measurement Overlay", use_container_width=True)


def render_pixel_inspector():
    """Render the pixel inspector for the current image."""
    if not st.session_state.get("original_image"):
        st.info("📸 Upload an image to use the pixel inspector")
        return

    # Choose which image to inspect
    inspect_options = ["Original Image"]
    if st.session_state.get("processed_image"):
        inspect_options.append("Processed Image")

    if (
        st.session_state.get("last_execution_result")
        and st.session_state.last_execution_result.steps
    ):
        step_names = [
            f"Step {i + 1}: {step['operation_name']}"
            for i, step in enumerate(st.session_state.last_execution_result.steps)
        ]
        inspect_options.extend(step_names)

    selected_image = st.selectbox("Select image to inspect:", inspect_options)

    # Get the selected image
    if selected_image == "Original Image":
        image = st.session_state.original_image
        title = "Original Image Inspector"
    elif selected_image == "Processed Image":
        image = st.session_state.processed_image
        title = "Processed Image Inspector"
    else:
        # Step image
        step_index = int(selected_image.split(":")[0].split(" ")[1]) - 1
        image = st.session_state.last_execution_result.steps[step_index]["result_image"]
        title = f"{selected_image} Inspector"

    # Render inspector
    inspector = PixelInspector(image, title)
    inspector.render()


def render_measurement_tools():
    """Render measurement tools for the current image."""
    if not st.session_state.get("original_image"):
        st.info("📐 Upload an image to use measurement tools")
        return

    # Choose which image to measure
    measure_options = ["Original Image"]
    if st.session_state.get("processed_image"):
        measure_options.append("Processed Image")

    selected_image = st.selectbox("Select image to measure:", measure_options)

    # Get the selected image
    if selected_image == "Original Image":
        image = st.session_state.original_image
    else:
        image = st.session_state.processed_image

    # Render measurement tools
    tool = MeasurementTool(image)
    tool.render()
