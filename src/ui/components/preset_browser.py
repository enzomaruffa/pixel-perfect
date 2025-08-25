"""Preset browser for loading presets into current pipeline."""

import uuid
from typing import Any

import streamlit as st

from presets.built_in import BUILT_IN_PRESETS
from ui.components.operation_registry import get_operation_by_name


def render_preset_browser():
    """Render preset browser with load-to-pipeline functionality."""
    st.subheader("📋 Preset Browser")

    if not BUILT_IN_PRESETS:
        st.info("No presets available")
        return

    st.caption(f"Choose from {len(BUILT_IN_PRESETS)} built-in presets to add to your pipeline")

    # Preset selection
    for preset_key, preset_config in BUILT_IN_PRESETS.items():
        render_preset_card(preset_key, preset_config)


def render_preset_card(preset_key: str, preset_config: dict[str, Any]):
    """Render individual preset card with preview and load options."""

    with st.expander(f"**{preset_key.replace('-', ' ').title()}**", expanded=False):
        # Description
        st.write(preset_config.get("description", "No description available"))

        # Operation preview
        operations = preset_config.get("operations", [])
        st.write(f"**Operations ({len(operations)}):**")

        for i, op in enumerate(operations, 1):
            op_type = op.get("type", "Unknown")
            st.write(f"{i}. {op_type}")

        # Action buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "📥 Replace Pipeline",
                key=f"replace_{preset_key}",
                help=f"Replace current pipeline with {preset_key}",
                use_container_width=True,
            ):
                load_preset_to_pipeline(preset_config, replace=True, preset_name=preset_key)
                st.success(f"✅ Pipeline replaced with '{preset_key}'")
                st.rerun()

        with col2:
            if st.button(
                "➕ Add to Pipeline",
                key=f"append_{preset_key}",
                help=f"Add {preset_key} operations to current pipeline",
                use_container_width=True,
            ):
                load_preset_to_pipeline(preset_config, replace=False, preset_name=preset_key)
                st.success(f"✅ Added '{preset_key}' to pipeline")
                st.rerun()


def load_preset_to_pipeline(
    preset_config: dict[str, Any], replace: bool = True, preset_name: str = ""
):
    """Load preset operations into current pipeline.

    Args:
        preset_config: Preset configuration from BUILT_IN_PRESETS
        replace: If True, replace current pipeline. If False, append operations.
        preset_name: Name of the preset for UI feedback
    """
    try:
        # Convert preset operations to pipeline format
        pipeline_operations = convert_preset_to_pipeline(preset_config, preset_name)

        if replace:
            # Replace entire pipeline
            st.session_state.pipeline_operations = pipeline_operations
        else:
            # Append to existing pipeline
            if "pipeline_operations" not in st.session_state:
                st.session_state.pipeline_operations = []

            st.session_state.pipeline_operations.extend(pipeline_operations)

        # Mark parameters as changed for auto-execution
        st.session_state.parameters_changed = True

    except Exception as e:
        st.error(f"❌ Failed to load preset: {str(e)}")


def convert_preset_to_pipeline(
    preset_config: dict[str, Any], preset_name: str = ""
) -> list[dict[str, Any]]:
    """Convert preset format to pipeline operations format.

    Args:
        preset_config: Preset configuration with operations list
        preset_name: Name of preset for operation naming

    Returns:
        List of pipeline operation configurations

    Raises:
        ValueError: If preset format is invalid or operations not found
    """
    operations = preset_config.get("operations", [])
    pipeline_operations = []

    for i, op_config in enumerate(operations):
        # Validate operation structure
        if "type" not in op_config:
            raise ValueError(f"Operation {i + 1} missing 'type' field")

        op_type = op_config["type"]

        # Look up operation class
        operation_info = get_operation_by_name(op_type)
        if not operation_info:
            raise ValueError(f"Unknown operation type: {op_type}")

        # Generate display name
        display_name = f"{op_type} ({preset_name})" if preset_name else op_type

        # Convert parameters
        preset_params = op_config.get("params", {})
        converted_params = convert_preset_params(preset_params, op_type)

        # Create pipeline operation config
        pipeline_op = {
            "id": str(uuid.uuid4()),
            "name": display_name,
            "class": operation_info["class"],
            "params": converted_params,
            "enabled": True,
        }

        pipeline_operations.append(pipeline_op)

    return pipeline_operations


def convert_preset_params(preset_params: dict[str, Any], operation_type: str) -> dict[str, Any]:
    """Convert preset parameter format to pipeline parameter format.

    Args:
        preset_params: Parameters from preset configuration
        operation_type: Type of operation for context-specific conversion

    Returns:
        Converted parameters suitable for pipeline operations
    """
    converted = {}

    for key, value in preset_params.items():
        if isinstance(value, list) and key in ["fill_color", "color", "rgba_color"]:
            # Convert color lists to tuples
            converted[key] = tuple(value) if len(value) >= 3 else value
        elif isinstance(value, list) and key in ["block_size", "grid_size", "tile_size"]:
            # Convert size lists to tuples
            converted[key] = tuple(value) if len(value) == 2 else value
        else:
            # Keep other parameters as-is
            converted[key] = value

    return converted


def render_preset_quick_load():
    """Render quick preset loading interface for main UI integration."""

    if not BUILT_IN_PRESETS:
        return

    st.write("**Quick Load Preset:**")

    preset_names = list(BUILT_IN_PRESETS.keys())
    [name.replace("-", " ").title() for name in preset_names]

    selected_preset = st.selectbox(
        "Choose Preset",
        options=preset_names,
        format_func=lambda x: x.replace("-", " ").title(),
        key="quick_preset_select",
    )

    if selected_preset:
        preset_config = BUILT_IN_PRESETS[selected_preset]

        # Show quick info
        operations_count = len(preset_config.get("operations", []))
        st.caption(f"{operations_count} operations - {preset_config.get('description', '')}")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Replace", key="quick_replace", help="Replace current pipeline"):
                load_preset_to_pipeline(preset_config, replace=True, preset_name=selected_preset)
                st.success(f"✅ Loaded '{selected_preset}'")
                st.rerun()

        with col2:
            if st.button("➕ Add", key="quick_add", help="Add to current pipeline"):
                load_preset_to_pipeline(preset_config, replace=False, preset_name=selected_preset)
                st.success(f"✅ Added '{selected_preset}'")
                st.rerun()


def get_available_presets() -> dict[str, dict[str, Any]]:
    """Get all available presets for external use."""
    return BUILT_IN_PRESETS.copy()


def get_preset_by_name(preset_name: str) -> dict[str, Any] | None:
    """Get specific preset by name."""
    return BUILT_IN_PRESETS.get(preset_name)


def validate_preset_format(preset_config: dict[str, Any]) -> bool:
    """Validate preset configuration format.

    Args:
        preset_config: Preset configuration to validate

    Returns:
        True if format is valid, False otherwise
    """
    try:
        # Check required fields
        if "operations" not in preset_config:
            return False

        operations = preset_config["operations"]
        if not isinstance(operations, list):
            return False

        # Validate each operation
        for op in operations:
            if not isinstance(op, dict):
                return False
            if "type" not in op:
                return False

            # Check if operation type exists
            if not get_operation_by_name(op["type"]):
                return False

        return True

    except Exception:
        return False
