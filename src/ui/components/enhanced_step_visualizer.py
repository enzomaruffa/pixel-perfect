"""Enhanced step-by-step pipeline visualizer with advanced features."""

import io

import streamlit as st
from PIL import Image

from ui.design_system import render_section_header, render_status_badge
from ui.utils.image_utils import optimize_image_for_display


class PipelineFlowVisualizer:
    """Visual pipeline flow with thumbnails and operation impact assessment."""

    def __init__(self, execution_result):
        self.execution_result = execution_result
        self.steps = execution_result.steps if execution_result else []

    def render(self):
        """Render the enhanced pipeline flow visualizer."""
        if not self.steps:
            st.info("🔄 Execute a pipeline to see the step visualizer")
            return

        render_section_header("Pipeline Flow Visualizer", "🔄")

        # Flow controls
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            layout_mode = st.selectbox(
                "Layout Mode",
                ["Horizontal Flow", "Vertical Stack", "Grid Layout"],
                help="How to display the pipeline steps",
            )

        with col2:
            thumbnail_size = st.select_slider(
                "Thumbnail Size",
                options=[80, 120, 160, 200],
                value=120,
                help="Size of step thumbnails",
            )

        with col3:
            show_metrics = st.checkbox("Show Metrics", value=True)

        # Render based on layout mode
        if layout_mode == "Horizontal Flow":
            self._render_horizontal_flow(thumbnail_size, show_metrics)
        elif layout_mode == "Vertical Stack":
            self._render_vertical_stack(thumbnail_size, show_metrics)
        else:  # Grid Layout
            self._render_grid_layout(thumbnail_size, show_metrics)

        # Step comparison tools
        self._render_step_comparison_tools()

    def _render_horizontal_flow(self, thumbnail_size: int, show_metrics: bool):
        """Render horizontal pipeline flow."""
        st.markdown("**Pipeline Flow** →")

        # Create columns for each step plus original
        num_cols = len(self.steps) + 1
        cols = st.columns(num_cols)

        # Original image
        with cols[0]:
            if st.session_state.get("original_image"):
                original_thumb = optimize_image_for_display(
                    st.session_state.original_image, max_size=(thumbnail_size, thumbnail_size)
                )
                st.image(original_thumb, caption="Original")
                st.markdown("🏁 **Start**")

        # Pipeline steps
        for i, step in enumerate(self.steps):
            with cols[i + 1]:
                step_thumb = optimize_image_for_display(
                    step["image"], max_size=(thumbnail_size, thumbnail_size)
                )
                st.image(step_thumb, caption=f"Step {i + 1}")

                # Step info
                st.markdown(f"**{step['operation']}**")

                if show_metrics:
                    render_status_badge(f"{step['execution_time']:.2f}s", "info")
                    if step.get("cached", False):
                        render_status_badge("Cached", "success")

                # Click to inspect
                if st.button("🔍 Inspect", key=f"flow_inspect_{i}"):
                    st.session_state.selected_step_for_inspection = i
                    st.rerun()

    def _render_vertical_stack(self, thumbnail_size: int, show_metrics: bool):
        """Render vertical pipeline stack."""
        st.markdown("**Pipeline Stack** ↓")

        # Original image
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.session_state.get("original_image"):
                original_thumb = optimize_image_for_display(
                    st.session_state.original_image, max_size=(thumbnail_size, thumbnail_size)
                )
                st.image(original_thumb)
        with col2:
            st.markdown("### 🏁 Original Image")
            if st.session_state.get("original_image"):
                img = st.session_state.original_image
                st.write(f"📐 {img.width} × {img.height} | {img.mode}")

        st.markdown("---")

        # Pipeline steps
        for i, step in enumerate(self.steps):
            col1, col2 = st.columns([1, 3])

            with col1:
                step_thumb = optimize_image_for_display(
                    step["image"], max_size=(thumbnail_size, thumbnail_size)
                )
                st.image(step_thumb)

                if st.button("🔍", key=f"stack_inspect_{i}", help="Inspect this step"):
                    st.session_state.selected_step_for_inspection = i
                    st.rerun()

            with col2:
                st.markdown(f"### {i + 1}. {step['operation']}")

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Time", f"{step['execution_time']:.3f}s")
                with col_b:
                    st.metric("Width", step["image"].width)
                with col_c:
                    st.metric("Height", step["image"].height)

                if show_metrics:
                    with st.expander("📊 Step Details"):
                        st.write(f"**Cached**: {'Yes' if step.get('cached', False) else 'No'}")
                        st.write(f"**Mode**: {step['image'].mode}")

                        # Calculate change from previous step
                        prev_image = (
                            st.session_state.original_image
                            if i == 0
                            else self.steps[i - 1]["result_image"]
                        )
                        self._display_step_impact(step["image"], prev_image)

            if i < len(self.steps) - 1:
                st.markdown("↓")

    def _render_grid_layout(self, thumbnail_size: int, show_metrics: bool):
        """Render grid layout of pipeline steps."""
        st.markdown("**Pipeline Grid**")

        # Include original image in grid
        all_images = []

        if st.session_state.get("original_image"):
            all_images.append(
                {
                    "image": st.session_state.original_image,
                    "title": "Original",
                    "index": -1,
                    "is_step": False,
                }
            )

        for i, step in enumerate(self.steps):
            all_images.append(
                {
                    "image": step["image"],
                    "title": f"Step {i + 1}: {step['operation']}",
                    "index": i,
                    "is_step": True,
                    "step": step,
                }
            )

        # Grid display
        grid_cols = min(4, len(all_images))
        grid_rows = (len(all_images) + grid_cols - 1) // grid_cols

        for row in range(grid_rows):
            cols = st.columns(grid_cols)

            for col_idx in range(grid_cols):
                img_idx = row * grid_cols + col_idx

                if img_idx < len(all_images):
                    item = all_images[img_idx]

                    with cols[col_idx]:
                        thumb = optimize_image_for_display(
                            item["image"], max_size=(thumbnail_size, thumbnail_size)
                        )
                        st.image(thumb)
                        st.markdown(f"**{item['title']}**")

                        if item["is_step"] and show_metrics:
                            step = item["step"]
                            render_status_badge(f"{step['execution_time']:.2f}s", "info")
                            if step.get("cached", False):
                                render_status_badge("Cached", "success")

                        if st.button("🔍", key=f"grid_inspect_{img_idx}"):
                            if item["is_step"]:
                                st.session_state.selected_step_for_inspection = item["index"]
                            else:
                                st.session_state.selected_step_for_inspection = -1
                            st.rerun()

    def _render_step_comparison_tools(self):
        """Render tools for comparing different steps."""
        st.markdown("---")
        render_section_header("Step Comparison Tools", "⚖️")

        if len(self.steps) < 2:
            st.info("Need at least 2 steps for comparison")
            return

        # Step selection for comparison
        step_options = ["Original"] + [
            f"Step {i + 1}: {step['operation']}" for i, step in enumerate(self.steps)
        ]

        col1, col2 = st.columns(2)

        with col1:
            step_a = st.selectbox("Compare Step A:", step_options, index=0)

        with col2:
            step_b = st.selectbox("Compare Step B:", step_options, index=1)

        if st.button("🔍 Compare Selected Steps"):
            # Get selected images
            if step_a == "Original":
                image_a = st.session_state.original_image
            else:
                idx_a = int(step_a.split(":")[0].split(" ")[1]) - 1
                image_a = self.steps[idx_a]["result_image"]

            if step_b == "Original":
                image_b = st.session_state.original_image
            else:
                idx_b = int(step_b.split(":")[0].split(" ")[1]) - 1
                image_b = self.steps[idx_b]["result_image"]

            # Side-by-side comparison
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**{step_a}**")
                st.image(image_a, use_container_width=True)

            with col2:
                st.markdown(f"**{step_b}**")
                st.image(image_b, use_container_width=True)

            # Calculate differences
            self._calculate_step_differences(image_a, image_b, step_a, step_b)

    def _display_step_impact(self, current_image: Image.Image, previous_image: Image.Image):
        """Display the impact of a step on the image."""
        # Basic change detection
        size_change = (
            current_image.width != previous_image.width
            or current_image.height != previous_image.height
        )
        mode_change = current_image.mode != previous_image.mode

        changes = []
        if size_change:
            changes.append(
                f"Size: {previous_image.width}×{previous_image.height} → {current_image.width}×{current_image.height}"
            )
        if mode_change:
            changes.append(f"Mode: {previous_image.mode} → {current_image.mode}")

        if changes:
            st.write("**Changes:**")
            for change in changes:
                st.write(f"• {change}")
        else:
            st.write("**Changes:** Pixel values modified")

    def _calculate_step_differences(
        self, image_a: Image.Image, image_b: Image.Image, name_a: str, name_b: str
    ):
        """Calculate and display differences between two steps."""
        # Import here to avoid circular imports
        try:
            from ui.components.image_comparison import calculate_image_comparison

            comparison = calculate_image_comparison(image_a, image_b)

            st.markdown(f"**Comparison: {name_a} vs {name_b}**")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("SSIM Score", f"{comparison.get('ssim_score', 0):.3f}")

            with col2:
                st.metric("MSE", f"{comparison.get('mse', 0):.2f}")

            with col3:
                st.metric("PSNR", f"{comparison.get('psnr', 0):.2f}")

            if "diff_image" in comparison:
                st.image(
                    comparison["diff_image"], caption="Visual Difference", use_container_width=True
                )

        except Exception as e:
            st.error(f"Failed to calculate comparison: {e}")


class StepInspector:
    """Detailed inspector for individual pipeline steps."""

    def __init__(self, step_index: int, execution_result):
        self.step_index = step_index
        self.execution_result = execution_result
        self.steps = execution_result.steps if execution_result else []

    def render(self):
        """Render the step inspector."""
        if self.step_index == -1:
            # Inspecting original image
            self._render_original_inspector()
        elif 0 <= self.step_index < len(self.steps):
            step = self.steps[self.step_index]
            self._render_step_inspector(step)
        else:
            st.error("Invalid step selected for inspection")

    def _render_original_inspector(self):
        """Render inspector for the original image."""
        if not st.session_state.get("original_image"):
            st.error("No original image available")
            return

        image = st.session_state.original_image

        render_section_header("Original Image Inspector", "🏁")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(image, caption="Original Image", use_container_width=True)

        with col2:
            st.markdown("**Image Properties**")
            st.metric("Width", f"{image.width} px")
            st.metric("Height", f"{image.height} px")
            st.metric("Mode", image.mode)
            st.metric("Format", image.format or "Unknown")

            # File size estimation
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")
            file_size = len(img_bytes.getvalue())
            st.metric("Size", f"{file_size:,} bytes")

    def _render_step_inspector(self, step):
        """Render inspector for a specific pipeline step."""
        step_num = self.step_index + 1

        render_section_header(f"Step {step_num}: {step['operation']}", "🔍")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(step["image"], caption=f"Step {step_num} Result", use_container_width=True)

        with col2:
            st.markdown("**Step Properties**")
            st.metric("Operation", step["operation"])
            st.metric("Execution Time", f"{step['execution_time']:.3f}s")
            st.metric("Cached", "Yes" if step.get("cached", False) else "No")

            st.markdown("**Result Properties**")
            st.metric("Width", f"{step['image'].width} px")
            st.metric("Height", f"{step['image'].height} px")
            st.metric("Mode", step["image"].mode)

        # Operation parameters (if available)
        if hasattr(step, "parameters"):
            with st.expander("⚙️ Operation Parameters"):
                for key, value in step.parameters.items():
                    st.write(f"**{key}**: {value}")

        # Comparison with previous step
        if self.step_index > 0:
            st.markdown("---")
            st.markdown("**Comparison with Previous Step**")

            prev_image = (
                st.session_state.original_image
                if self.step_index == 0
                else self.steps[self.step_index - 1]["result_image"]
            )

            self._render_step_comparison(step["image"], prev_image)

    def _render_step_comparison(self, current_image: Image.Image, previous_image: Image.Image):
        """Render comparison between current step and previous."""
        col1, col2 = st.columns(2)

        with col1:
            st.image(previous_image, caption="Previous", use_container_width=True)

        with col2:
            st.image(current_image, caption="Current", use_container_width=True)

        # Calculate basic differences
        size_changed = current_image.size != previous_image.size
        mode_changed = current_image.mode != previous_image.mode

        if size_changed:
            st.info(f"📐 Size changed: {previous_image.size} → {current_image.size}")

        if mode_changed:
            st.info(f"🎨 Color mode changed: {previous_image.mode} → {current_image.mode}")

        if not size_changed and not mode_changed:
            st.success("✓ Size and mode unchanged - pixel values modified")


def render_enhanced_step_visualizer():
    """Render the enhanced step visualizer interface."""
    if not st.session_state.get("last_execution_result"):
        st.info("🔄 Execute a pipeline to see the enhanced step visualizer")
        return

    execution_result = st.session_state.last_execution_result

    # Check if user selected a step for inspection
    if st.session_state.get("selected_step_for_inspection") is not None:
        # Render step inspector
        inspector = StepInspector(st.session_state.selected_step_for_inspection, execution_result)

        # Back button
        if st.button("← Back to Pipeline Flow"):
            del st.session_state.selected_step_for_inspection
            st.rerun()

        inspector.render()

    else:
        # Render pipeline flow visualizer
        visualizer = PipelineFlowVisualizer(execution_result)
        visualizer.render()
