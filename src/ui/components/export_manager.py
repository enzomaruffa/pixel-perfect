"""Main export functionality and user interface."""

from datetime import datetime

import streamlit as st
from PIL import Image

from ui.utils.config_serialization import (
    create_preset_from_current_pipeline,
    export_config_as_json,
    export_config_as_python,
    export_config_as_yaml,
    serialize_pipeline_config,
)
from ui.utils.image_export import (
    add_watermark,
    estimate_export_size,
    export_image,
    export_pipeline_steps,
    format_file_size,
    resize_image_for_export,
)


def render_export_manager():
    """Render the main export interface."""

    if not st.session_state.get("original_image"):
        st.info("Upload an image to access export features")
        return

    st.header("📤 Export & Share")

    # Export tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🖼️ Images", "⚙️ Configuration", "📦 Batch Export", "📋 Presets"]
    )

    with tab1:
        render_image_export()

    with tab2:
        render_config_export()

    with tab3:
        render_batch_export()

    with tab4:
        render_preset_management()


def render_image_export():
    """Render image export interface."""

    st.subheader("🖼️ Export Images")

    # Determine available images
    images_available = []
    if st.session_state.get("original_image"):
        images_available.append(("original", "Original Image"))
    if st.session_state.get("processed_image"):
        images_available.append(("processed", "Processed Image"))

    if not images_available:
        st.warning("No images available for export")
        return

    # Image selection
    col1, col2 = st.columns(2)

    with col1:
        image_to_export = st.selectbox(
            "Image to Export",
            options=[key for key, _ in images_available],
            format_func=lambda x: dict(images_available)[x],
            key="export_image_selection",
        )

    with col2:
        # Format selection
        format_options = ["PNG", "JPEG", "WebP", "TIFF"]
        export_format = st.selectbox("Format", format_options, key="export_format")

    # Get selected image
    if image_to_export == "original":
        selected_image = st.session_state.original_image
    else:
        selected_image = st.session_state.processed_image

    # Export options
    st.write("**Export Options**")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Size options
        size_options = {
            "original": "Original Size",
            "web_small": "Web Small (800×600)",
            "web_medium": "Web Medium (1200×900)",
            "web_large": "Web Large (1920×1440)",
            "thumbnail": "Thumbnail (256×256)",
            "custom": "Custom Size",
        }

        size_option = st.selectbox(
            "Size",
            options=list(size_options.keys()),
            format_func=lambda x: size_options[x],
            key="export_size_option",
        )

        # Custom size inputs
        custom_width = custom_height = None
        if size_option == "custom":
            custom_width = st.number_input(
                "Width",
                min_value=1,
                max_value=10000,
                value=selected_image.width,
                key="export_custom_width",
            )
            custom_height = st.number_input(
                "Height",
                min_value=1,
                max_value=10000,
                value=selected_image.height,
                key="export_custom_height",
            )

    with col2:
        # Quality settings
        if export_format in ["JPEG", "WebP"]:
            quality = st.slider(
                "Quality", min_value=1, max_value=100, value=95, key="export_quality"
            )
        else:
            quality = 95

        # Optimization
        optimize = st.checkbox("Optimize", value=True, key="export_optimize")

        # Format-specific options
        if export_format == "JPEG":
            st.checkbox("Progressive", value=False, key="export_progressive")
        elif export_format == "PNG":
            st.slider("Compression Level", 0, 9, 6, key="export_compress_level")
        elif export_format == "WebP":
            st.checkbox("Lossless", value=False, key="export_lossless")

    with col3:
        # Watermark options
        add_watermark_option = st.checkbox("Add Watermark", value=False, key="export_add_watermark")

        # Initialize watermark variables with defaults
        watermark_text = "Pixel Perfect"
        watermark_position = "bottom_right"
        watermark_opacity = 0.7

        if add_watermark_option:
            watermark_text = st.text_input(
                "Watermark Text", value="Pixel Perfect", key="export_watermark_text"
            )
            watermark_position = st.selectbox(
                "Position",
                ["bottom_right", "bottom_left", "top_right", "top_left", "center"],
                key="export_watermark_position",
            )
            watermark_opacity = st.slider("Opacity", 0.1, 1.0, 0.7, key="export_watermark_opacity")

    # Preview and size estimation
    st.write("**Preview & Estimation**")

    # Apply size transformation for preview
    preview_image = resize_image_for_export(
        selected_image, size_option, custom_width, custom_height
    )

    # Apply watermark for preview
    if add_watermark_option:
        preview_image = add_watermark(
            preview_image, watermark_text, watermark_position, watermark_opacity
        )

    # Show preview
    col1, col2 = st.columns([1, 2])

    with col1:
        st.write("**Preview:**")
        preview_display = preview_image.copy()
        preview_display.thumbnail((300, 300), Image.Resampling.LANCZOS)
        st.image(preview_display, caption=f"Preview ({preview_image.width}×{preview_image.height})")

    with col2:
        st.write("**Export Information:**")

        # Estimate file size
        estimated_size = estimate_export_size(preview_image, export_format, quality)
        st.info(f"📏 Estimated size: {format_file_size(estimated_size)}")

        # Show dimensions
        st.info(f"📐 Dimensions: {preview_image.width} × {preview_image.height}")

        # Format info
        format_info = {
            "PNG": "Lossless compression, supports transparency",
            "JPEG": "Lossy compression, smaller file size, no transparency",
            "WebP": "Modern format, good compression, supports transparency",
            "TIFF": "High quality, large file size, good for archival",
        }
        st.info(f"ℹ️ {format_info.get(export_format, 'Unknown format')}")

    # Export button and filename
    st.write("**Export**")

    col1, col2 = st.columns([2, 1])

    with col1:
        filename = st.text_input(
            "Filename",
            value=f"exported_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format.lower()}",
            key="export_filename",
        )

    with col2:
        if st.button("📥 Export Image", type="primary", key="export_image_button"):
            try:
                # Prepare export parameters
                export_kwargs = {"optimize": optimize}

                if export_format == "JPEG":
                    export_kwargs["progressive"] = st.session_state.get("export_progressive", False)
                elif export_format == "PNG":
                    export_kwargs["compress_level"] = st.session_state.get(
                        "export_compress_level", 6
                    )
                elif export_format == "WebP":
                    export_kwargs["lossless"] = st.session_state.get("export_lossless", False)

                # Export image
                image_data = export_image(preview_image, export_format, quality, **export_kwargs)

                # Provide download
                st.download_button(
                    label="⬇️ Download Image",
                    data=image_data,
                    file_name=filename,
                    mime=f"image/{export_format.lower()}",
                    key="download_exported_image",
                )

                actual_size = len(image_data)
                st.success(f"✅ Image exported successfully! ({format_file_size(actual_size)})")

            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")


def render_config_export():
    """Render configuration export interface."""

    st.subheader("⚙️ Export Configuration")

    if not st.session_state.get("pipeline_operations"):
        st.info("Create a pipeline to export configuration")
        return

    # Configuration preview
    with st.expander("📋 Current Pipeline", expanded=True):
        operations = st.session_state.pipeline_operations
        for i, op in enumerate(operations):
            status_icon = "✅" if op.get("enabled", True) else "❌"
            st.write(f"{i + 1}. {status_icon} **{op['name']}** - {len(op['params'])} parameters")

    # Export format selection
    col1, col2 = st.columns(2)

    with col1:
        config_format = st.selectbox(
            "Export Format", ["JSON", "YAML", "Python Script"], key="config_export_format"
        )

    with col2:
        include_metadata = st.checkbox(
            "Include Metadata", value=True, key="config_include_metadata"
        )

    # Generate configuration
    try:
        config = serialize_pipeline_config(include_metadata=include_metadata)

        if config_format == "JSON":
            config_content = export_config_as_json(config)
            file_extension = "json"
            mime_type = "application/json"
        elif config_format == "YAML":
            config_content = export_config_as_yaml(config)
            file_extension = "yaml"
            mime_type = "application/x-yaml"
        else:  # Python Script
            config_content = export_config_as_python(config)
            file_extension = "py"
            mime_type = "text/x-python"

        # Show preview
        st.write("**Preview:**")
        st.code(
            config_content[:1000] + "..." if len(config_content) > 1000 else config_content,
            language=config_format.lower() if config_format != "Python Script" else "python",
        )

        # Download
        filename = f"pipeline_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}"
        st.download_button(
            label=f"⬇️ Download {config_format} Configuration",
            data=config_content,
            file_name=filename,
            mime=mime_type,
            key="download_config",
        )

    except Exception as e:
        st.error(f"❌ Failed to generate configuration: {str(e)}")


def render_batch_export():
    """Render batch export interface."""

    st.subheader("📦 Batch Export")

    if not st.session_state.get("last_execution_result"):
        st.info("Execute a pipeline to enable batch export")
        return

    result = st.session_state.last_execution_result
    if not result.steps:
        st.warning("No pipeline steps available for export")
        return

    st.write(f"**Available for Export:** {len(result.steps)} pipeline steps + original + final")

    # Export configuration
    col1, col2 = st.columns(2)

    with col1:
        batch_format = st.selectbox("Format", ["PNG", "JPEG", "WebP"], key="batch_format")
        batch_quality = (
            st.slider("Quality", 1, 100, 95, key="batch_quality")
            if batch_format in ["JPEG", "WebP"]
            else 95
        )

    with col2:
        batch_size = st.selectbox(
            "Size",
            ["original", "web_small", "web_medium", "custom"],
            format_func=lambda x: {
                "original": "Original Size",
                "web_small": "Web Small (800×600)",
                "web_medium": "Web Medium (1200×900)",
                "custom": "Custom Size",
            }[x],
            key="batch_size",
        )

        if batch_size == "custom":
            col3, col4 = st.columns(2)
            with col3:
                st.number_input("Width", 1, 5000, 800, key="batch_width")
            with col4:
                st.number_input("Height", 1, 5000, 600, key="batch_height")

    # Additional options
    col1, col2 = st.columns(2)

    with col1:
        batch_watermark = st.checkbox("Add Watermark", key="batch_watermark")
        # Initialize with default value
        batch_watermark_text = "Pixel Perfect"
        if batch_watermark:
            batch_watermark_text = st.text_input(
                "Watermark Text", "Pixel Perfect", key="batch_watermark_text"
            )

    with col2:
        batch_include_original = st.checkbox(
            "Include Original", value=True, key="batch_include_original"
        )
        batch_include_final = st.checkbox(
            "Include Final Result", value=True, key="batch_include_final"
        )

    # Generate batch export
    if st.button("📦 Generate Batch Export", type="primary", key="batch_export_button"):
        try:
            # Prepare export configuration
            export_config = {
                "format": batch_format.lower(),
                "quality": batch_quality,
                "resize_option": batch_size,
                "add_watermark": batch_watermark,
                "include_original": batch_include_original,
                "include_final": batch_include_final,
            }

            if batch_watermark:
                export_config["watermark_text"] = batch_watermark_text

            if batch_size == "custom":
                export_config["custom_width"] = st.session_state.get("batch_width", 800)
                export_config["custom_height"] = st.session_state.get("batch_height", 600)

            # Generate ZIP file
            with st.spinner("Generating batch export..."):
                zip_data = export_pipeline_steps(result, export_config)

            # Provide download
            filename = f"pipeline_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            st.download_button(
                label="⬇️ Download ZIP Archive",
                data=zip_data,
                file_name=filename,
                mime="application/zip",
                key="download_batch_zip",
            )

            st.success(f"✅ Batch export ready! ({format_file_size(len(zip_data))})")

        except Exception as e:
            st.error(f"❌ Batch export failed: {str(e)}")


def render_preset_management():
    """Render preset creation and management."""

    st.subheader("📋 Pipeline Presets")

    if not st.session_state.get("pipeline_operations"):
        st.info("Create a pipeline to save as preset")
        return

    # Create new preset
    with st.expander("➕ Create New Preset", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            preset_name = st.text_input("Preset Name", key="preset_name")
            preset_description = st.text_area("Description (optional)", key="preset_description")

        with col2:
            # Preview current pipeline
            operations = st.session_state.pipeline_operations
            st.write("**Current Pipeline:**")
            for i, op in enumerate(operations):
                st.write(f"{i + 1}. {op['name']}")

        if preset_name and st.button("💾 Save Preset", key="save_preset_button"):
            try:
                preset_config = create_preset_from_current_pipeline(preset_name, preset_description)

                # Convert to JSON for download
                preset_json = export_config_as_json(preset_config)

                filename = f"preset_{preset_name.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d')}.json"

                st.download_button(
                    label="⬇️ Download Preset",
                    data=preset_json,
                    file_name=filename,
                    mime="application/json",
                    key="download_preset",
                )

                st.success(f"✅ Preset '{preset_name}' created successfully!")

            except Exception as e:
                st.error(f"❌ Failed to create preset: {str(e)}")

    # Load preset section
    st.write("**Load Preset**")
    uploaded_preset = st.file_uploader(
        "Upload Preset File",
        type=["json", "yaml"],
        help="Upload a previously saved preset configuration",
        key="upload_preset",
    )

    if uploaded_preset is not None:
        try:
            content = uploaded_preset.read().decode("utf-8")

            if uploaded_preset.name.endswith(".json"):
                from ui.utils.config_serialization import load_config_from_json

                config = load_config_from_json(content)
            else:
                from ui.utils.config_serialization import load_config_from_yaml

                config = load_config_from_yaml(content)

            if config:
                st.success("✅ Preset loaded successfully!")

                if st.button("🔄 Apply Preset", key="apply_preset_button"):
                    from ui.utils.config_serialization import deserialize_pipeline_config

                    if deserialize_pipeline_config(config):
                        st.success("✅ Preset applied to current pipeline!")
                        st.session_state.parameters_changed = True
                        st.rerun()
                    else:
                        st.error("❌ Failed to apply preset")

        except Exception as e:
            st.error(f"❌ Error loading preset: {str(e)}")
