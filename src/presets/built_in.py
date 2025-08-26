"""Built-in presets for common artistic effects."""

from typing import Any

# Built-in presets for common effects
BUILT_IN_PRESETS = {
    "glitch-effect": {
        "description": "Create a digital glitch effect with pixel filtering and row shifting",
        "operations": [
            {
                "type": "PixelFilter",
                "params": {"condition": "prime", "fill_color": [255, 0, 0, 128]},
            },
            {"type": "RowShift", "params": {"selection": "odd", "shift_amount": 5, "wrap": False}},
            {
                "type": "ColumnShift",
                "params": {"selection": "even", "shift_amount": 3, "wrap": True},
            },
        ],
    },
    "mosaic-art": {
        "description": "Transform image into mosaic tiles with block effects",
        "operations": [
            {
                "type": "BlockFilter",
                "params": {"block_width": 8, "block_height": 8, "condition": "checkerboard"},
            },
            {"type": "Mosaic", "params": {"tile_size": (16, 16), "gap_size": 2, "mode": "average"}},
        ],
    },
    "retro-effect": {
        "description": "Create a retro computer graphics effect with channel manipulation and dithering",
        "operations": [
            {
                "type": "ChannelSwap",
                "params": {"red_source": "green", "green_source": "blue", "blue_source": "red"},
            },
            {"type": "Dither", "params": {"method": "floyd_steinberg", "levels": 4}},
        ],
    },
    "perspective-trick": {
        "description": "Create perspective distortion with geometric transformations",
        "operations": [
            {"type": "PerspectiveStretch", "params": {"top_factor": 0.8, "bottom_factor": 1.5}},
            {"type": "GridWarp", "params": {"axis": "both", "frequency": 2.0, "amplitude": 10.0}},
        ],
    },
    "color-pop": {
        "description": "Isolate specific colors and enhance with alpha effects",
        "operations": [
            {"type": "ChannelIsolate", "params": {"channels": ["red"], "mode": "enhance"}},
            {"type": "AlphaGenerator", "params": {"method": "saturation", "threshold": 0.3}},
        ],
    },
    "data-corruption": {
        "description": "Simulate data corruption with block scrambling and pixel math",
        "operations": [
            {
                "type": "BlockScramble",
                "params": {"block_width": 16, "block_height": 16, "scramble_ratio": 0.3},
            },
            {"type": "PixelMath", "params": {"expression": "r ^ 128", "channels": ["r", "g", "b"]}},
        ],
    },
    "vintage-photo": {
        "description": "Create a vintage photo effect with aspect adjustments and color manipulation",
        "operations": [
            {
                "type": "AspectPad",
                "params": {"target_ratio": "4:3", "mode": "blur", "blur_radius": 5},
            },
            {
                "type": "PixelMath",
                "params": {"expression": "r * 1.1 + 20", "channels": ["r"], "clamp": True},
            },
            {
                "type": "ChannelSwap",
                "params": {"red_source": "red", "green_source": "red", "blue_source": "green"},
            },
        ],
    },
    "abstract-art": {
        "description": "Create abstract art with radial distortion and column effects",
        "operations": [
            {
                "type": "RadialStretch",
                "params": {"factor": 1.5, "center": "auto", "falloff": "quadratic"},
            },
            {"type": "ColumnWeave", "params": {"pattern": [1, 0, 1, 0], "mode": "skip"}},
            {
                "type": "PixelSort",
                "params": {"direction": "diagonal", "sort_by": "hue", "reverse": True},
            },
        ],
    },
    "minimal-clean": {
        "description": "Clean, minimal effect with aspect ratio adjustment",
        "operations": [
            {"type": "AspectCrop", "params": {"target_ratio": "1:1", "crop_mode": "smart"}},
            {
                "type": "PixelMath",
                "params": {"expression": "r * 0.95", "channels": ["r", "g", "b"], "clamp": True},
            },
        ],
    },
    "stripe-pattern": {
        "description": "Create stripe patterns with row manipulation",
        "operations": [
            {
                "type": "RowRemove",
                "params": {"pattern": [True, False, True, False], "fill_mode": "stretch"},
            },
            {"type": "RowStretch", "params": {"rows": [0, 2, 4, 6], "factor": 2.0}},
        ],
    },
}


def get_all_presets() -> dict[str, dict[str, Any]]:
    """Get all built-in presets.

    Returns:
        Dictionary mapping preset names to their configurations
    """
    return BUILT_IN_PRESETS.copy()


def get_preset(preset_name: str) -> dict[str, Any]:
    """Get a specific preset by name.

    Args:
        preset_name: Name of the preset to retrieve

    Returns:
        Preset configuration dictionary

    Raises:
        FileNotFoundError: If preset doesn't exist
    """
    if preset_name not in BUILT_IN_PRESETS:
        available_presets = list(BUILT_IN_PRESETS.keys())
        raise FileNotFoundError(
            f"Preset '{preset_name}' not found. Available presets: {', '.join(available_presets)}"
        )

    return BUILT_IN_PRESETS[preset_name].copy()


def list_preset_names() -> list[str]:
    """Get list of all preset names.

    Returns:
        List of preset names
    """
    return list(BUILT_IN_PRESETS.keys())


def search_presets(query: str) -> dict[str, dict[str, Any]]:
    """Search presets by name or description.

    Args:
        query: Search query string

    Returns:
        Dictionary of matching presets
    """
    query_lower = query.lower()
    matching_presets = {}

    for name, preset in BUILT_IN_PRESETS.items():
        # Check if query matches name or description
        if query_lower in name.lower() or query_lower in preset.get("description", "").lower():
            matching_presets[name] = preset.copy()

    return matching_presets
