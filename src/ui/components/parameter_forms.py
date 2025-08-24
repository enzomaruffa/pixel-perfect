"""Parameter configuration forms for operations."""

from typing import Any

import streamlit as st
from pydantic import ValidationError

from ui.utils.form_generators import extract_pydantic_model_from_operation, generate_parameter_form


def render_operation_parameter_editor():
    """Render parameter editor for selected operation."""

    if not st.session_state.get("selected_operation_for_editing"):
        st.info("Select an operation from your pipeline to configure its parameters.")
        return

    operation_config = st.session_state.selected_operation_for_editing
    operation_class = operation_config["class"]
    current_params = operation_config["params"]

    st.header(f"🔧 Configure {operation_config['name']}")

    # Generate parameter form
    with st.form(key="parameter_form"):
        form_values = generate_parameter_form(operation_class, current_params)

        # Form submission buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            apply_changes = st.form_submit_button("✅ Apply Changes", type="primary")
        with col2:
            reset_defaults = st.form_submit_button("🔄 Reset to Defaults")
        with col3:
            cancel_editing = st.form_submit_button("❌ Cancel")

    # Handle form submissions
    if apply_changes:
        try:
            # Validate parameters using Pydantic model
            config_model = extract_pydantic_model_from_operation(operation_class)
            if config_model:
                validated_params = config_model(**form_values)
                # Update the operation in pipeline
                update_operation_parameters(operation_config["id"], validated_params.model_dump())
                st.success("✅ Parameters updated successfully!")
                st.session_state.selected_operation_for_editing = None
                st.session_state.parameters_changed = True  # Trigger auto-execution
                st.rerun()
            else:
                st.error("Could not validate parameters - no Pydantic model found")
        except ValidationError as e:
            st.error("❌ Parameter validation failed:")
            for error in e.errors():
                field = error["loc"][0] if error["loc"] else "Unknown"
                message = error["msg"]
                st.error(f"• **{field}**: {message}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

    elif reset_defaults:
        # Reset to default parameters
        from ui.components.operation_registry import get_operation_by_name

        op_info = get_operation_by_name(operation_config["name"])
        if op_info:
            update_operation_parameters(operation_config["id"], op_info["default_params"])
            st.success("🔄 Parameters reset to defaults!")
            st.session_state.selected_operation_for_editing = None
            st.session_state.parameters_changed = True  # Trigger auto-execution
            st.rerun()

    elif cancel_editing:
        st.session_state.selected_operation_for_editing = None
        st.rerun()


def update_operation_parameters(operation_id: str, new_params: dict[str, Any]):
    """Update parameters for an operation in the pipeline."""
    if "pipeline_operations" not in st.session_state:
        return

    for i, op_config in enumerate(st.session_state.pipeline_operations):
        if op_config["id"] == operation_id:
            st.session_state.pipeline_operations[i]["params"] = new_params
            break


def render_pipeline_with_edit_buttons():
    """Render pipeline summary with parameter editing buttons."""

    if not st.session_state.get("pipeline_operations"):
        st.info("No operations in pipeline yet. Add some operations to get started!")
        return

    st.subheader("🔗 Current Pipeline")

    operations = st.session_state.pipeline_operations

    for i, op_config in enumerate(operations):
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

        with col1:
            # Operation info
            enabled_icon = "✅" if op_config["enabled"] else "❌"
            st.write(f"{i + 1}. {enabled_icon} **{op_config['name']}**")

        with col2:
            # Configure button
            if st.button("⚙️", key=f"config_{op_config['id']}", help="Configure parameters"):
                st.session_state.selected_operation_for_editing = op_config
                st.rerun()

        with col3:
            # Enable/disable toggle
            new_enabled = st.checkbox(
                "On", value=op_config["enabled"], key=f"enable_{op_config['id']}"
            )
            if new_enabled != op_config["enabled"]:
                st.session_state.pipeline_operations[i]["enabled"] = new_enabled
                st.session_state.parameters_changed = True  # Trigger auto-execution
                st.rerun()

        with col4:
            # Remove button
            if st.button("🗑️", key=f"remove_{op_config['id']}", help="Remove from pipeline"):
                st.session_state.pipeline_operations.pop(i)
                st.session_state.parameters_changed = True  # Trigger auto-execution
                st.rerun()

        # Show current parameter summary
        with st.expander(f"Current Parameters - {op_config['name']}", expanded=False):
            params = op_config["params"]
            for param_name, param_value in params.items():
                st.write(f"**{param_name}**: {param_value}")

    # Clear all button
    if st.button("🗑️ Clear All", help="Remove all operations from pipeline"):
        st.session_state.pipeline_operations = []
        st.session_state.parameters_changed = True  # Trigger auto-execution
        st.rerun()


def render_parameter_presets(operation_class):
    """Render parameter presets for an operation."""

    st.subheader("📋 Parameter Presets")

    # Define some common presets for different operations
    presets = get_operation_presets(operation_class.__name__)

    if presets:
        preset_names = list(presets.keys())
        selected_preset = st.selectbox(
            "Choose a preset:",
            options=["Custom"] + preset_names,
            index=0,
            key="parameter_preset_selector",
        )

        if selected_preset != "Custom" and st.button("📥 Load Preset"):
            # Load preset parameters
            preset_params = presets[selected_preset]
            # Update session state or return values
            return preset_params
    else:
        st.info("No presets available for this operation.")

    return None


def get_operation_presets(operation_name: str) -> dict[str, dict[str, Any]]:
    """Get parameter presets for a specific operation."""

    presets = {
        "PixelFilter": {
            "Prime Red": {
                "condition": "prime",
                "fill_color": (255, 0, 0, 128),
                "preserve_alpha": True,
                "index_mode": "linear",
            },
            "Even Blue": {
                "condition": "even",
                "fill_color": (0, 0, 255, 128),
                "preserve_alpha": True,
                "index_mode": "linear",
            },
            "Fibonacci Gold": {
                "condition": "fibonacci",
                "fill_color": (255, 215, 0, 180),
                "preserve_alpha": True,
                "index_mode": "linear",
            },
        },
        "RowShift": {
            "Gentle Wave": {"pattern": "wave", "amplitude": 10, "frequency": 0.05, "wrap": True},
            "Strong Wave": {"pattern": "wave", "amplitude": 50, "frequency": 0.15, "wrap": True},
            "Linear Shift": {"pattern": "linear", "amplitude": 20, "frequency": 0.1, "wrap": True},
        },
        "PixelMath": {
            "Brighten": {"expression": "r * 1.3", "channels": ["r", "g", "b"], "clamp": True},
            "Darken": {"expression": "r * 0.7", "channels": ["r", "g", "b"], "clamp": True},
            "Increase Contrast": {
                "expression": "(r - 128) * 1.5 + 128",
                "channels": ["r", "g", "b"],
                "clamp": True,
            },
        },
        "ChannelSwap": {
            "Red to Green": {
                "mapping": {"r": "g", "g": "r", "b": "b", "a": "a"},
                "preserve_alpha": True,
            },
            "RGB Rotate": {
                "mapping": {"r": "g", "g": "b", "b": "r", "a": "a"},
                "preserve_alpha": True,
            },
            "Invert RGB": {
                "mapping": {"r": "b", "g": "g", "b": "r", "a": "a"},
                "preserve_alpha": True,
            },
        },
    }

    return presets.get(operation_name, {})


def render_parameter_validation_info():
    """Render information about parameter validation."""

    with st.expander("ℹ️ Parameter Validation Help", expanded=False):
        st.markdown("""
        ### Parameter Types and Validation

        **🔢 Numeric Parameters**
        - Use sliders for bounded values (frequencies, amplitudes)
        - Use number inputs for precise values
        - Some parameters have smart ranges based on their purpose

        **🎨 Color Parameters**
        - RGBA values are validated to be 0-255
        - Alpha channel controls transparency (0=transparent, 255=opaque)
        - Color preview shows the actual color

        **📝 Text Parameters**
        - Mathematical expressions are validated for safety
        - Channel lists must contain valid channel names (r, g, b, a)
        - Custom expressions use variables like 'i' for pixel index

        **⚙️ Enum Parameters**
        - Dropdown menus show all valid options
        - Some parameters depend on others (e.g., custom expressions)

        **🔗 Parameter Dependencies**
        - Some parameters are only used when others have specific values
        - Validation errors will show which dependencies are missing
        - The system prevents invalid combinations automatically
        """)


def validate_operation_parameters(
    operation_class, parameters: dict[str, Any]
) -> tuple[bool, list[str]]:
    """Validate operation parameters and return validation status and errors."""

    try:
        config_model = extract_pydantic_model_from_operation(operation_class)
        if not config_model:
            return False, ["Could not find parameter validation model"]

        # Validate using Pydantic
        config_model(**parameters)  # This validates the parameters
        return True, []

    except ValidationError as e:
        error_messages = []
        for error in e.errors():
            field = error["loc"][0] if error["loc"] else "unknown"
            message = error["msg"]
            error_messages.append(f"{field}: {message}")
        return False, error_messages

    except Exception as e:
        return False, [f"Unexpected validation error: {str(e)}"]
