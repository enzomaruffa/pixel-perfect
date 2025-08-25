"""Operation registry for the UI with metadata and default parameters."""

from operations import (
    AlphaGenerator,
    AspectCrop,
    AspectPad,
    AspectStretch,
    BlockFilter,
    BlockRotate,
    BlockScramble,
    BlockShift,
    ChannelIsolate,
    ChannelSwap,
    ColumnMirror,
    ColumnShift,
    ColumnStretch,
    ColumnWeave,
    Dither,
    GridWarp,
    Mosaic,
    PerspectiveStretch,
    PixelFilter,
    PixelMath,
    PixelSort,
    RadialStretch,
    RowRemove,
    RowShift,
    RowShuffle,
    RowStretch,
)

OPERATION_REGISTRY = {
    "pixel": {
        "icon": "🎯",
        "name": "Pixel Operations",
        "description": "Apply transformations at individual pixel level",
        "operations": {
            "PixelFilter": {
                "class": PixelFilter,
                "description": "Filter pixels based on mathematical conditions (prime, odd, even, fibonacci)",
                "difficulty": "Beginner",
                "default_params": {
                    "condition": "prime",
                    "fill_color": (255, 0, 0, 128),
                    "preserve_alpha": True,
                    "index_mode": "linear",
                },
                "tags": ["filter", "mathematical", "selective"],
            },
            "PixelMath": {
                "class": PixelMath,
                "description": "Apply mathematical expressions to pixel values",
                "difficulty": "Intermediate",
                "default_params": {
                    "expression": "r * 1.2",
                    "channels": ["r", "g", "b"],
                    "clamp": True,
                },
                "tags": ["math", "expression", "color-adjustment"],
            },
            "PixelSort": {
                "class": PixelSort,
                "description": "Sort pixels by various criteria (brightness, hue, saturation)",
                "difficulty": "Intermediate",
                "default_params": {"key": "brightness", "direction": "ascending", "mode": "linear"},
                "tags": ["sort", "reorder", "artistic"],
            },
        },
    },
    "row": {
        "icon": "↔️",
        "name": "Row Operations",
        "description": "Manipulate horizontal rows of pixels",
        "operations": {
            "RowShift": {
                "class": RowShift,
                "description": "Shift rows horizontally with various patterns",
                "difficulty": "Beginner",
                "default_params": {
                    "pattern": "wave",
                    "amplitude": 20,
                    "frequency": 0.1,
                    "wrap": True,
                },
                "tags": ["shift", "horizontal", "wave"],
            },
            "RowStretch": {
                "class": RowStretch,
                "description": "Stretch or compress rows to create distortion effects",
                "difficulty": "Intermediate",
                "default_params": {
                    "pattern": "sine",
                    "amplitude": 0.5,
                    "frequency": 0.1,
                    "interpolation": "bilinear",
                },
                "tags": ["stretch", "distortion", "warp"],
            },
            "RowRemove": {
                "class": RowRemove,
                "description": "Remove rows based on patterns or conditions",
                "difficulty": "Beginner",
                "default_params": {"pattern": "every_nth", "n": 3, "fill_color": (0, 0, 0, 0)},
                "tags": ["remove", "pattern", "selective"],
            },
            "RowShuffle": {
                "class": RowShuffle,
                "description": "Randomly shuffle the order of rows",
                "difficulty": "Beginner",
                "default_params": {"seed": None, "blocks": 1},
                "tags": ["shuffle", "random", "reorder"],
            },
        },
    },
    "column": {
        "icon": "↕️",
        "name": "Column Operations",
        "description": "Manipulate vertical columns of pixels",
        "operations": {
            "ColumnShift": {
                "class": ColumnShift,
                "description": "Shift columns vertically with various patterns",
                "difficulty": "Beginner",
                "default_params": {
                    "pattern": "wave",
                    "amplitude": 20,
                    "frequency": 0.1,
                    "wrap": True,
                },
                "tags": ["shift", "vertical", "wave"],
            },
            "ColumnStretch": {
                "class": ColumnStretch,
                "description": "Stretch or compress columns to create distortion effects",
                "difficulty": "Intermediate",
                "default_params": {
                    "pattern": "sine",
                    "amplitude": 0.5,
                    "frequency": 0.1,
                    "interpolation": "bilinear",
                },
                "tags": ["stretch", "distortion", "warp"],
            },
            "ColumnMirror": {
                "class": ColumnMirror,
                "description": "Mirror columns to create reflection effects",
                "difficulty": "Beginner",
                "default_params": {"axis": "center", "blend_mode": "replace"},
                "tags": ["mirror", "reflection", "symmetry"],
            },
            "ColumnWeave": {
                "class": ColumnWeave,
                "description": "Interleave columns from different parts of the image",
                "difficulty": "Intermediate",
                "default_params": {"pattern": "alternating", "offset": 0},
                "tags": ["weave", "interleave", "pattern"],
            },
        },
    },
    "block": {
        "icon": "🔳",
        "name": "Block Operations",
        "description": "Work with rectangular blocks of pixels",
        "operations": {
            "BlockFilter": {
                "class": BlockFilter,
                "description": "Filter blocks based on various criteria",
                "difficulty": "Intermediate",
                "default_params": {
                    "block_size": (8, 8),
                    "condition": "brightness_threshold",
                    "threshold": 128,
                    "fill_color": (0, 0, 0, 128),
                },
                "tags": ["filter", "blocks", "selective"],
            },
            "BlockShift": {
                "class": BlockShift,
                "description": "Shift blocks in various patterns",
                "difficulty": "Beginner",
                "default_params": {
                    "block_size": (16, 16),
                    "shift_pattern": "checkerboard",
                    "shift_amount": (8, 8),
                },
                "tags": ["shift", "blocks", "pattern"],
            },
            "BlockRotate": {
                "class": BlockRotate,
                "description": "Rotate individual blocks independently",
                "difficulty": "Intermediate",
                "default_params": {
                    "block_size": (16, 16),
                    "rotation_pattern": "alternating",
                    "angle": 90,
                },
                "tags": ["rotate", "blocks", "angle"],
            },
            "BlockScramble": {
                "class": BlockScramble,
                "description": "Randomly rearrange blocks within the image",
                "difficulty": "Beginner",
                "default_params": {"block_size": (32, 32), "seed": None},
                "tags": ["scramble", "random", "blocks"],
            },
        },
    },
    "geometric": {
        "icon": "🌊",
        "name": "Geometric Operations",
        "description": "Apply geometric transformations and warping",
        "operations": {
            "GridWarp": {
                "class": GridWarp,
                "description": "Warp image using a flexible grid transformation",
                "difficulty": "Advanced",
                "default_params": {
                    "grid_size": (4, 4),
                    "warp_strength": 0.3,
                    "interpolation": "bilinear",
                },
                "tags": ["warp", "grid", "distortion"],
            },
            "PerspectiveStretch": {
                "class": PerspectiveStretch,
                "description": "Apply perspective transformation to create depth effects",
                "difficulty": "Advanced",
                "default_params": {"direction": "horizontal", "strength": 0.5, "anchor": "center"},
                "tags": ["perspective", "depth", "3d-effect"],
            },
            "RadialStretch": {
                "class": RadialStretch,
                "description": "Stretch image radially from center point",
                "difficulty": "Intermediate",
                "default_params": {
                    "center": (0.5, 0.5),
                    "strength": 1.5,
                    "interpolation": "bilinear",
                },
                "tags": ["radial", "stretch", "center"],
            },
        },
    },
    "aspect": {
        "icon": "📐",
        "name": "Aspect Operations",
        "description": "Modify image aspect ratio and dimensions",
        "operations": {
            "AspectStretch": {
                "class": AspectStretch,
                "description": "Stretch image to fit new aspect ratio",
                "difficulty": "Beginner",
                "default_params": {"target_ratio": "16:9", "interpolation": "bilinear"},
                "tags": ["aspect-ratio", "stretch", "resize"],
            },
            "AspectCrop": {
                "class": AspectCrop,
                "description": "Crop image to achieve target aspect ratio",
                "difficulty": "Beginner",
                "default_params": {"target_ratio": "1:1", "anchor": "center"},
                "tags": ["aspect-ratio", "crop", "trim"],
            },
            "AspectPad": {
                "class": AspectPad,
                "description": "Add padding to achieve target aspect ratio",
                "difficulty": "Beginner",
                "default_params": {
                    "target_ratio": "4:3",
                    "fill_color": (0, 0, 0, 255),
                    "position": "center",
                },
                "tags": ["aspect-ratio", "padding", "fill"],
            },
        },
    },
    "channel": {
        "icon": "🎨",
        "name": "Channel Operations",
        "description": "Manipulate color channels and transparency",
        "operations": {
            "ChannelSwap": {
                "class": ChannelSwap,
                "description": "Rearrange or swap color channels",
                "difficulty": "Beginner",
                "default_params": {
                    "mapping": {"r": "g", "g": "b", "b": "r"},
                    "preserve_alpha": True,
                },
                "tags": ["channels", "swap", "color"],
            },
            "ChannelIsolate": {
                "class": ChannelIsolate,
                "description": "Isolate specific color channels",
                "difficulty": "Beginner",
                "default_params": {"channels": ["r"], "mode": "isolate"},
                "tags": ["channels", "isolate", "selective"],
            },
            "AlphaGenerator": {
                "class": AlphaGenerator,
                "description": "Generate alpha channel based on various criteria",
                "difficulty": "Intermediate",
                "default_params": {"source": "brightness", "threshold": 128, "invert": False},
                "tags": ["alpha", "transparency", "generate"],
            },
        },
    },
    "pattern": {
        "icon": "🎭",
        "name": "Pattern Operations",
        "description": "Create patterns and stylized effects",
        "operations": {
            "Mosaic": {
                "class": Mosaic,
                "description": "Create mosaic effect with tiles",
                "difficulty": "Intermediate",
                "default_params": {
                    "tile_size": (16, 16),
                    "gap_size": 1,
                    "gap_color": (128, 128, 128, 255),
                },
                "tags": ["mosaic", "tiles", "artistic"],
            },
            "Dither": {
                "class": Dither,
                "description": "Apply dithering patterns for stylized effects",
                "difficulty": "Advanced",
                "default_params": {
                    "method": "floyd_steinberg",
                    "levels": 4,
                    "channels": ["r", "g", "b"],
                },
                "tags": ["dither", "quantize", "retro"],
            },
        },
    },
}


def get_operation_by_name(operation_name: str):
    """Get operation class and metadata by name."""
    for category_info in OPERATION_REGISTRY.values():
        if operation_name in category_info["operations"]:
            return category_info["operations"][operation_name]
    return None


def get_all_operations():
    """Get flat list of all operations with their metadata."""
    operations = []
    for category_key, category_info in OPERATION_REGISTRY.items():
        for op_name, op_info in category_info["operations"].items():
            operations.append(
                {
                    "name": op_name,
                    "category": category_key,
                    "category_name": category_info["name"],
                    "category_icon": category_info["icon"],
                    **op_info,
                }
            )
    return operations


def search_operations(query: str, categories: list[str] | None = None):
    """Search operations by name, description, or tags."""
    query = query.lower()
    results = []

    for category_key, category_info in OPERATION_REGISTRY.items():
        # Skip categories not in filter
        if categories and category_key not in categories:
            continue

        for op_name, op_info in category_info["operations"].items():
            # Check if query matches name, description, or tags
            matches = (
                query in op_name.lower()
                or query in op_info["description"].lower()
                or any(query in tag for tag in op_info["tags"])
            )

            if matches:
                results.append(
                    {
                        "name": op_name,
                        "category": category_key,
                        "category_name": category_info["name"],
                        "category_icon": category_info["icon"],
                        **op_info,
                    }
                )

    return results
