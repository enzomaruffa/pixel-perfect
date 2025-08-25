"""Pipeline configuration serialization and deserialization."""

import json
from datetime import datetime
from typing import Any

import streamlit as st
import yaml


def serialize_pipeline_config(include_metadata: bool = True) -> dict[str, Any]:
    """Serialize current pipeline configuration to dictionary."""

    config = {"version": "1.0", "pipeline": {"operations": []}}

    # Add metadata if requested
    if include_metadata:
        config["metadata"] = {
            "created_at": datetime.now().isoformat(),
            "created_by": "Pixel Perfect Web Interface",
            "description": "Exported pipeline configuration",
            "original_image_info": get_original_image_info()
            if st.session_state.get("original_image")
            else None,
        }

    # Serialize operations
    operations = st.session_state.get("pipeline_operations", [])
    for op_config in operations:
        serialized_op = {
            "name": op_config["name"],
            "class_name": op_config["class"].__name__
            if hasattr(op_config["class"], "__name__")
            else str(op_config["class"]),
            "module": op_config["class"].__module__
            if hasattr(op_config["class"], "__module__")
            else None,
            "parameters": serialize_parameters(op_config["params"]),
            "enabled": op_config.get("enabled", True),
            "id": op_config.get("id"),
        }
        config["pipeline"]["operations"].append(serialized_op)

    # Add execution settings if available
    if st.session_state.get("auto_execute") is not None:
        config["execution_settings"] = {
            "auto_execute": st.session_state.auto_execute,
            "cache_enabled": True,  # Always enabled in current implementation
        }

    # Add display settings
    config["display_settings"] = {
        "display_mode": st.session_state.get("display_mode", "side_by_side"),
        "zoom_level": st.session_state.get("zoom_level", 1.0),
        "show_image_info": st.session_state.get("show_image_info", False),
    }

    return config


def deserialize_pipeline_config(config: dict[str, Any]) -> bool:
    """Load pipeline configuration into session state."""

    try:
        # Validate config structure
        if not validate_config_structure(config):
            return False

        # Clear current pipeline
        st.session_state.pipeline_operations = []

        # Import operation classes (this is simplified - in practice would need dynamic imports)
        operation_classes = get_available_operation_classes()

        # Load operations
        for op_data in config["pipeline"]["operations"]:
            class_name = op_data["class_name"]

            if class_name in operation_classes:
                operation_config = {
                    "name": op_data["name"],
                    "class": operation_classes[class_name],
                    "params": deserialize_parameters(op_data["parameters"]),
                    "enabled": op_data.get("enabled", True),
                    "id": op_data.get(
                        "id", f"{op_data['name']}_{len(st.session_state.pipeline_operations)}"
                    ),
                }
                st.session_state.pipeline_operations.append(operation_config)
            else:
                st.warning(f"Unknown operation class: {class_name}")

        # Load execution settings
        if "execution_settings" in config:
            settings = config["execution_settings"]
            st.session_state.auto_execute = settings.get("auto_execute", True)

        # Load display settings
        if "display_settings" in config:
            settings = config["display_settings"]
            st.session_state.display_mode = settings.get("display_mode", "side_by_side")
            st.session_state.zoom_level = settings.get("zoom_level", 1.0)
            st.session_state.show_image_info = settings.get("show_image_info", False)

        return True

    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        return False


def serialize_parameters(params: dict[str, Any]) -> dict[str, Any]:
    """Serialize parameters to JSON-compatible format."""

    serialized = {}

    for key, value in params.items():
        if isinstance(value, tuple | list):
            serialized[key] = list(value)
        elif hasattr(value, "__dict__"):  # Complex objects
            serialized[key] = str(value)  # Convert to string representation
        else:
            serialized[key] = value

    return serialized


def deserialize_parameters(params: dict[str, Any]) -> dict[str, Any]:
    """Deserialize parameters from JSON format."""

    deserialized = {}

    for key, value in params.items():
        # Handle common parameter types
        if (
            isinstance(value, list)
            and len(value) in (3, 4)
            and all(isinstance(x, int) for x in value)
        ):
            # Likely a color tuple
            deserialized[key] = tuple(value)
        else:
            deserialized[key] = value

    return deserialized


def validate_config_structure(config: dict[str, Any]) -> bool:
    """Validate configuration structure."""

    required_fields = ["version", "pipeline"]
    for field in required_fields:
        if field not in config:
            st.error(f"Missing required field: {field}")
            return False

    if "operations" not in config["pipeline"]:
        st.error("Missing operations in pipeline configuration")
        return False

    return True


def get_available_operation_classes() -> dict[str, Any]:
    """Get mapping of operation class names to classes."""

    # Import all operation classes
    try:
        from operations.aspect import AspectCrop, AspectPad, AspectStretch
        from operations.block import BlockFilter, BlockRotate, BlockScramble, BlockShift
        from operations.channel import AlphaGenerator, ChannelIsolate, ChannelSwap
        from operations.column import ColumnMirror, ColumnShift, ColumnStretch, ColumnWeave
        from operations.geometric import GridWarp, PerspectiveStretch, RadialStretch
        from operations.pattern import Dither, Mosaic
        from operations.pixel import PixelFilter, PixelMath, PixelSort
        from operations.row import RowRemove, RowShift, RowShuffle, RowStretch

        return {
            # Pixel operations
            "PixelFilter": PixelFilter,
            "PixelMath": PixelMath,
            "PixelSort": PixelSort,
            # Row operations
            "RowShift": RowShift,
            "RowStretch": RowStretch,
            "RowRemove": RowRemove,
            "RowShuffle": RowShuffle,
            # Column operations
            "ColumnShift": ColumnShift,
            "ColumnStretch": ColumnStretch,
            "ColumnMirror": ColumnMirror,
            "ColumnWeave": ColumnWeave,
            # Block operations
            "BlockFilter": BlockFilter,
            "BlockShift": BlockShift,
            "BlockRotate": BlockRotate,
            "BlockScramble": BlockScramble,
            # Geometric operations
            "GridWarp": GridWarp,
            "PerspectiveStretch": PerspectiveStretch,
            "RadialStretch": RadialStretch,
            # Aspect operations
            "AspectStretch": AspectStretch,
            "AspectCrop": AspectCrop,
            "AspectPad": AspectPad,
            # Channel operations
            "ChannelSwap": ChannelSwap,
            "ChannelIsolate": ChannelIsolate,
            "AlphaGenerator": AlphaGenerator,
            # Pattern operations
            "Mosaic": Mosaic,
            "Dither": Dither,
        }
    except ImportError as e:
        st.error(f"Error importing operations: {str(e)}")
        return {}


def get_original_image_info() -> dict[str, Any] | None:
    """Get information about the original image."""

    original = st.session_state.get("original_image")
    if not original:
        return None

    return {
        "width": original.width,
        "height": original.height,
        "mode": original.mode,
        "format": getattr(original, "format", "Unknown"),
        "size_bytes": len(original.tobytes()) if hasattr(original, "tobytes") else 0,
    }


def export_config_as_json(config: dict[str, Any], filename: str | None = None) -> str:
    """Export configuration as JSON string."""

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pipeline_config_{timestamp}.json"

    json_str = json.dumps(config, indent=2, ensure_ascii=False)
    return json_str


def export_config_as_yaml(config: dict[str, Any], filename: str | None = None) -> str:
    """Export configuration as YAML string."""

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pipeline_config_{timestamp}.yaml"

    yaml_str = yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return yaml_str


def export_config_as_python(config: dict[str, Any], filename: str | None = None) -> str:
    """Export configuration as Python script."""

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pipeline_script_{timestamp}.py"

    script_lines = [
        "#!/usr/bin/env python3",
        '"""',
        "Generated pipeline script from Pixel Perfect",
        f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        '"""',
        "",
        "import sys",
        "from pathlib import Path",
        "from PIL import Image",
        "",
        "# Add src to path for imports",
        'sys.path.insert(0, str(Path(__file__).parent / "src"))',
        "",
        "from core.pipeline import Pipeline",
    ]

    # Add operation imports
    operation_modules = set()
    for op_data in config["pipeline"]["operations"]:
        if op_data.get("module"):
            operation_modules.add(op_data["module"])

    for module in sorted(operation_modules):
        script_lines.append(f"import {module}")

    script_lines.extend(
        [
            "",
            "",
            "def main(input_path: str, output_path: str):",
            '    """Execute the pipeline on given input."""',
            "",
            "    # Create pipeline",
            "    pipeline = Pipeline(input_path)",
            "",
        ]
    )

    # Add operations
    for op_data in config["pipeline"]["operations"]:
        if not op_data.get("enabled", True):
            script_lines.append(f"    # Disabled: {op_data['name']}")
            continue

        class_name = op_data["class_name"]
        params = op_data["parameters"]

        # Format parameters
        param_strs = []
        for key, value in params.items():
            if isinstance(value, str):
                param_strs.append(f'{key}="{value}"')
            elif isinstance(value, list):
                param_strs.append(f"{key}={value}")
            else:
                param_strs.append(f"{key}={value}")

        param_str = ", ".join(param_strs)
        script_lines.append(f"    pipeline.add({class_name}({param_str}))")

    script_lines.extend(
        [
            "",
            "    # Execute pipeline",
            "    context = pipeline.execute(output_path)",
            "    print(f'Pipeline executed successfully: {output_path}')",
            "    return context",
            "",
            "",
            'if __name__ == "__main__":',
            "    import sys",
            "    if len(sys.argv) != 3:",
            '        print("Usage: python script.py input_image output_image")',
            "        sys.exit(1)",
            "    ",
            "    main(sys.argv[1], sys.argv[2])",
        ]
    )

    return "\n".join(script_lines)


def load_config_from_json(json_str: str) -> dict[str, Any] | None:
    """Load configuration from JSON string."""

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON format: {str(e)}")
        return None


def load_config_from_yaml(yaml_str: str) -> dict[str, Any] | None:
    """Load configuration from YAML string."""

    try:
        return yaml.safe_load(yaml_str)
    except yaml.YAMLError as e:
        st.error(f"Invalid YAML format: {str(e)}")
        return None


def create_preset_from_current_pipeline(name: str, description: str = "") -> dict[str, Any]:
    """Create a preset from the current pipeline configuration."""

    config = serialize_pipeline_config(include_metadata=True)

    # Add preset metadata
    config["preset_info"] = {
        "name": name,
        "description": description,
        "created_at": datetime.now().isoformat(),
        "operation_count": len(config["pipeline"]["operations"]),
        "categories": get_pipeline_categories(config["pipeline"]["operations"]),
    }

    return config


def get_pipeline_categories(operations: list[dict[str, Any]]) -> list[str]:
    """Extract categories from pipeline operations."""

    categories = set()

    for op in operations:
        class_name = op.get("class_name", "")

        if class_name.startswith("Pixel"):
            categories.add("Pixel")
        elif class_name.startswith("Row"):
            categories.add("Row")
        elif class_name.startswith("Column"):
            categories.add("Column")
        elif class_name.startswith("Block"):
            categories.add("Block")
        elif class_name.startswith("Aspect"):
            categories.add("Aspect")
        elif class_name.startswith("Channel"):
            categories.add("Channel")
        elif class_name in ("GridWarp", "PerspectiveStretch", "RadialStretch"):
            categories.add("Geometric")
        elif class_name in ("Mosaic", "Dither"):
            categories.add("Pattern")

    return sorted(categories)
