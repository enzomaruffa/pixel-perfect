"""Dynamic form generation from Pydantic models for Streamlit."""

import inspect
from typing import Any, get_args, get_origin

import streamlit as st
from pydantic import BaseModel
from pydantic.fields import FieldInfo


def extract_pydantic_model_from_operation(operation_class) -> type[BaseModel] | None:
    """Extract the Pydantic config model from an operation class."""
    # Look for Config classes in the operation module
    module = inspect.getmodule(operation_class)
    if not module:
        return None

    # Common naming patterns for config classes
    config_name = f"{operation_class.__name__}Config"

    # Check if config class exists in the module
    if hasattr(module, config_name):
        config_class = getattr(module, config_name)
        if issubclass(config_class, BaseModel):
            return config_class

    # Alternative: look for any BaseModel subclass in the module
    for name, obj in inspect.getmembers(module):
        if (
            inspect.isclass(obj)
            and issubclass(obj, BaseModel)
            and obj != BaseModel
            and config_name.lower() in name.lower()
        ):
            return obj

    # If no separate config class, try to inspect the operation's __init__ signature
    # and create a dynamic model (fallback for operations without explicit config classes)
    try:
        signature = inspect.signature(operation_class.__init__)
        fields = {}

        for param_name, param in signature.parameters.items():
            if param_name in ("self", "kwargs"):
                continue

            # Try to infer field info from parameter
            field_type = param.annotation if param.annotation != inspect.Parameter.empty else Any
            default = param.default if param.default != inspect.Parameter.empty else None

            fields[param_name] = (field_type, default)

        if fields:
            # Create a dynamic Pydantic model
            return type(
                f"{operation_class.__name__}Config",
                (BaseModel,),
                {
                    "__annotations__": {
                        name: field_type for name, (field_type, _) in fields.items()
                    },
                    **{
                        name: default
                        for name, (_, default) in fields.items()
                        if default is not None
                    },
                },
            )
    except Exception:
        pass

    return None


def get_field_type_info(field_info: FieldInfo) -> dict[str, Any]:
    """Extract detailed type information from a Pydantic field."""
    annotation = field_info.annotation
    origin = get_origin(annotation)
    args = get_args(annotation)

    info = {
        "raw_type": annotation,
        "origin": origin,
        "args": args,
        "is_optional": False,
        "base_type": annotation,
    }

    # Handle Union types (including Optional)
    if origin is type(int | str):  # Union type
        # Check if it's Optional (Union with None)
        if len(args) == 2 and type(None) in args:
            info["is_optional"] = True
            # Get the non-None type
            info["base_type"] = args[0] if args[1] is type(None) else args[1]
        else:
            info["base_type"] = args[0]  # Use first type for multi-union

    # Handle Literal types
    elif hasattr(annotation, "__origin__") and str(annotation).startswith("typing.Literal"):
        info["is_literal"] = True
        info["literal_values"] = args
        info["base_type"] = type(args[0]) if args else str
    else:
        info["is_literal"] = False
        info["literal_values"] = None

    return info


def generate_parameter_form(
    operation_class, current_params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Generate Streamlit form inputs from an operation's Pydantic model."""
    if current_params is None:
        current_params = {}

    config_model = extract_pydantic_model_from_operation(operation_class)
    if not config_model:
        st.error(f"Could not find Pydantic model for {operation_class.__name__}")
        return {}

    st.subheader(f"⚙️ {operation_class.__name__} Parameters")

    form_values = {}

    # Generate widgets for each field
    for field_name, field_info in config_model.model_fields.items():
        current_value = current_params.get(field_name, field_info.default)

        # Create widget based on field type
        widget_value = create_widget_for_field(
            field_name=field_name, field_info=field_info, current_value=current_value
        )

        form_values[field_name] = widget_value

    return form_values


def create_widget_for_field(
    field_name: str, field_info: FieldInfo, current_value: Any = None
) -> Any:
    """Create appropriate Streamlit widget based on field type."""
    type_info = get_field_type_info(field_info)
    base_type = type_info["base_type"]

    # Use current value or default
    if current_value is None:
        current_value = (
            field_info.default
            if field_info.default is not None
            else get_default_for_type(base_type)
        )

    # Create human-readable label
    label = field_name.replace("_", " ").title()
    help_text = field_info.description

    # Handle Literal types (enums)
    if type_info.get("is_literal", False):
        return st.selectbox(
            label,
            options=list(type_info["literal_values"]),
            index=list(type_info["literal_values"]).index(current_value)
            if current_value in type_info["literal_values"]
            else 0,
            help=help_text,
            key=f"param_{field_name}",
        )

    # Handle basic types
    elif base_type is bool:
        return st.checkbox(
            label, value=bool(current_value), help=help_text, key=f"param_{field_name}"
        )

    elif base_type is int:
        return st.number_input(
            label,
            value=int(current_value) if current_value is not None else 0,
            step=1,
            help=help_text,
            key=f"param_{field_name}",
        )

    elif base_type is float:
        # Special handling for different parameter names
        if "frequency" in field_name.lower():
            return st.slider(
                label,
                min_value=0.0,
                max_value=1.0,
                value=float(current_value) if current_value is not None else 0.1,
                step=0.01,
                help=help_text,
                key=f"param_{field_name}",
            )
        elif "amplitude" in field_name.lower():
            return st.slider(
                label,
                min_value=0.0,
                max_value=100.0,
                value=float(current_value) if current_value is not None else 20.0,
                step=1.0,
                help=help_text,
                key=f"param_{field_name}",
            )
        elif "strength" in field_name.lower():
            return st.slider(
                label,
                min_value=0.1,
                max_value=5.0,
                value=float(current_value) if current_value is not None else 1.0,
                step=0.1,
                help=help_text,
                key=f"param_{field_name}",
            )
        elif "threshold" in field_name.lower():
            return st.slider(
                label,
                min_value=0.0,
                max_value=255.0,
                value=float(current_value) if current_value is not None else 128.0,
                step=1.0,
                help=help_text,
                key=f"param_{field_name}",
            )
        else:
            return st.number_input(
                label,
                value=float(current_value) if current_value is not None else 1.0,
                step=0.1,
                help=help_text,
                key=f"param_{field_name}",
            )

    elif base_type is str:
        return st.text_input(
            label,
            value=str(current_value) if current_value is not None else "",
            help=help_text,
            key=f"param_{field_name}",
        )

    # Handle tuple types (likely colors or coordinates)
    elif get_origin(base_type) is tuple:
        args = get_args(base_type)
        if len(args) == 4 and all(arg is int for arg in args):
            # RGBA color tuple
            return create_color_widget(label, current_value, help_text, field_name)
        elif len(args) == 2 and all(arg in (int, float) for arg in args):
            # Coordinate tuple or size tuple
            return create_coordinate_widget(label, current_value, help_text, field_name)
        else:
            # Generic tuple - fall back to text input
            return st.text_input(
                label,
                value=str(current_value) if current_value is not None else "()",
                help=help_text,
                key=f"param_{field_name}",
            )

    # Handle list types
    elif get_origin(base_type) is list:
        args = get_args(base_type)
        if args and args[0] is str:
            # String list - likely channels
            return st.multiselect(
                label,
                options=["r", "g", "b", "a"],
                default=current_value if isinstance(current_value, list) else ["r", "g", "b"],
                help=help_text,
                key=f"param_{field_name}",
            )
        else:
            # Generic list - text input for now
            return st.text_area(
                label,
                value=str(current_value) if current_value is not None else "[]",
                help=help_text,
                key=f"param_{field_name}",
            )

    # Handle dict types (like channel mappings)
    elif base_type is dict:
        return create_dict_widget(label, current_value, help_text, field_name)

    # Fallback to text input
    else:
        return st.text_input(
            label,
            value=str(current_value) if current_value is not None else "",
            help=help_text,
            key=f"param_{field_name}",
        )


def create_color_widget(
    label: str, current_value: Any, help_text: str | None, field_name: str
) -> tuple:
    """Create RGBA color picker widget."""
    if not isinstance(current_value, tuple | list) or len(current_value) != 4:
        current_value = (255, 0, 0, 255)

    st.write(f"**{label}**")
    if help_text:
        st.caption(help_text)

    cols = st.columns(4)
    with cols[0]:
        r = st.slider("Red", 0, 255, int(current_value[0]), key=f"{field_name}_r")
    with cols[1]:
        g = st.slider("Green", 0, 255, int(current_value[1]), key=f"{field_name}_g")
    with cols[2]:
        b = st.slider("Blue", 0, 255, int(current_value[2]), key=f"{field_name}_b")
    with cols[3]:
        a = st.slider("Alpha", 0, 255, int(current_value[3]), key=f"{field_name}_a")

    # Show color preview
    color_preview = f"rgba({r}, {g}, {b}, {a / 255:.2f})"
    st.markdown(
        f"""
    <div style="
        background-color: {color_preview};
        height: 30px;
        border: 1px solid #ccc;
        border-radius: 4px;
        margin: 5px 0;
    "></div>
    """,
        unsafe_allow_html=True,
    )

    return (r, g, b, a)


def create_coordinate_widget(
    label: str, current_value: Any, help_text: str | None, field_name: str
) -> tuple:
    """Create coordinate/size input widget."""
    if not isinstance(current_value, tuple | list) or len(current_value) != 2:
        current_value = (0, 0)

    st.write(f"**{label}**")
    if help_text:
        st.caption(help_text)

    cols = st.columns(2)
    with cols[0]:
        x = st.number_input("X/Width", value=int(current_value[0]), key=f"{field_name}_x")
    with cols[1]:
        y = st.number_input("Y/Height", value=int(current_value[1]), key=f"{field_name}_y")

    return (x, y)


def create_dict_widget(
    label: str, current_value: Any, help_text: str | None, field_name: str
) -> dict:
    """Create dictionary input widget (for channel mappings, etc)."""
    if not isinstance(current_value, dict):
        current_value = {}

    st.write(f"**{label}**")
    if help_text:
        st.caption(help_text)

    # Special handling for channel mappings
    if "mapping" in field_name.lower() or "channel" in field_name.lower():
        channels = ["r", "g", "b", "a"]
        result = {}
        cols = st.columns(len(channels))

        for i, channel in enumerate(channels):
            with cols[i]:
                result[channel] = st.selectbox(
                    f"{channel.upper()} →",
                    options=channels,
                    index=channels.index(current_value.get(channel, channel)),
                    key=f"{field_name}_{channel}",
                )
        return result
    else:
        # Generic dict as JSON text
        import json

        try:
            json_str = json.dumps(current_value, indent=2)
        except Exception:
            json_str = "{}"

        text_value = st.text_area(label, value=json_str, help=help_text, key=f"param_{field_name}")

        try:
            return json.loads(text_value)
        except Exception:
            return current_value


def get_default_for_type(type_class: type) -> Any:
    """Get sensible default value for a type."""
    if type_class is bool:
        return False
    elif type_class is int:
        return 0
    elif type_class is float:
        return 0.0
    elif type_class is str:
        return ""
    elif type_class is list:
        return []
    elif type_class is dict:
        return {}
    elif type_class is tuple:
        return ()
    else:
        return None
