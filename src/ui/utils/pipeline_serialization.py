"""Pipeline serialization and deserialization utilities."""

import json
from datetime import datetime
from typing import Any, Dict, List

from ui.components.operation_registry import get_operation_by_name


def serialize_pipeline(
    operations: List[Dict[str, Any]], 
    name: str = "Untitled Pipeline", 
    description: str = ""
) -> Dict[str, Any]:
    """Serialize pipeline operations to JSON-compatible format.
    
    Args:
        operations: List of operation configurations from session state
        name: Pipeline name
        description: Optional description
        
    Returns:
        Dictionary ready for JSON serialization
    """
    serialized_operations = []
    
    for op_config in operations:
        # Convert class reference to string identifier
        class_name = op_config["class"].__name__ if hasattr(op_config["class"], "__name__") else str(op_config["class"])
        
        serialized_op = {
            "id": op_config["id"],
            "type": class_name,
            "name": op_config["name"], 
            "enabled": op_config["enabled"],
            "params": _serialize_params(op_config["params"])
        }
        serialized_operations.append(serialized_op)
    
    pipeline_data = {
        "version": "1.0",
        "name": name,
        "description": description,
        "created": datetime.now().isoformat(),
        "operations": serialized_operations
    }
    
    return pipeline_data


def deserialize_pipeline(pipeline_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Deserialize pipeline JSON back to operation configurations.
    
    Args:
        pipeline_data: JSON data from saved pipeline
        
    Returns:
        List of operation configurations ready for session state
        
    Raises:
        ValueError: If pipeline format is invalid or operations not found
    """
    # Validate pipeline format
    required_fields = ["version", "operations"]
    for field in required_fields:
        if field not in pipeline_data:
            raise ValueError(f"Invalid pipeline format: missing '{field}' field")
    
    # Check version compatibility
    version = pipeline_data.get("version", "1.0")
    if version != "1.0":
        raise ValueError(f"Unsupported pipeline version: {version}")
    
    operations = []
    
    for op_data in pipeline_data["operations"]:
        # Validate operation structure
        required_op_fields = ["type", "params"]
        for field in required_op_fields:
            if field not in op_data:
                raise ValueError(f"Invalid operation format: missing '{field}' field")
        
        # Look up operation class
        operation_info = get_operation_by_name(op_data["type"])
        if not operation_info:
            raise ValueError(f"Unknown operation type: {op_data['type']}")
        
        # Reconstruct operation config
        operation_config = {
            "id": op_data.get("id", f"imported_{len(operations)}"),
            "name": op_data.get("name", op_data["type"]),
            "class": operation_info["class"],
            "params": _deserialize_params(op_data["params"]),
            "enabled": op_data.get("enabled", True)
        }
        
        operations.append(operation_config)
    
    return operations


def _serialize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert parameters to JSON-serializable format."""
    serialized = {}
    
    for key, value in params.items():
        if isinstance(value, tuple):
            # Convert tuples to lists for JSON
            serialized[key] = list(value)
        elif hasattr(value, '__dict__'):
            # Handle complex objects by converting to dict
            serialized[key] = value.__dict__ if hasattr(value, '__dict__') else str(value)
        else:
            # Keep primitives as-is
            serialized[key] = value
    
    return serialized


def _deserialize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert parameters back from JSON format."""
    deserialized = {}
    
    for key, value in params.items():
        if isinstance(value, list) and key in ['fill_color', 'color', 'rgba_color']:
            # Convert color lists back to tuples
            deserialized[key] = tuple(value)
        elif isinstance(value, list) and len(value) == 2 and key in ['block_size', 'grid_size', 'size']:
            # Convert coordinate/size lists back to tuples  
            deserialized[key] = tuple(value)
        else:
            # Keep other values as-is
            deserialized[key] = value
    
    return deserialized


def export_pipeline_json(pipeline_data: Dict[str, Any]) -> str:
    """Export pipeline data as formatted JSON string."""
    return json.dumps(pipeline_data, indent=2, ensure_ascii=False)


def import_pipeline_json(json_str: str) -> Dict[str, Any]:
    """Import pipeline data from JSON string.
    
    Args:
        json_str: JSON string containing pipeline data
        
    Returns:
        Pipeline data dictionary
        
    Raises:
        ValueError: If JSON is invalid or malformed
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")


def validate_pipeline_format(pipeline_data: Dict[str, Any]) -> bool:
    """Validate pipeline data format without full deserialization.
    
    Args:
        pipeline_data: Pipeline data to validate
        
    Returns:
        True if format is valid, False otherwise
    """
    try:
        # Basic structure validation
        if not isinstance(pipeline_data, dict):
            return False
        
        if "operations" not in pipeline_data:
            return False
        
        if not isinstance(pipeline_data["operations"], list):
            return False
        
        # Validate each operation
        for op in pipeline_data["operations"]:
            if not isinstance(op, dict):
                return False
            if "type" not in op or "params" not in op:
                return False
        
        return True
        
    except Exception:
        return False