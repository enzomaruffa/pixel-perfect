#!/usr/bin/env python3
"""
Simple transformation pipeline for pixel-perfect image processing.

This script provides an easy interface for applying image transformations
using either built-in presets or custom pipelines.

Usage:
    python transform.py input.jpg                    # Run custom pipeline
    python transform.py input.jpg --preset glitch    # Apply a preset
    python transform.py --list-presets               # Show available presets
"""

import argparse
import sys
from pathlib import Path

from core.pipeline import Pipeline
from operations.aspect import AspectCrop, AspectPad, AspectStretch
from operations.block import BlockFilter, BlockRotate, BlockScramble, BlockShift
from operations.channel import AlphaGenerator, ChannelIsolate, ChannelSwap
from operations.column import ColumnMirror, ColumnShift, ColumnStretch, ColumnWeave
from operations.geometric import GridWarp, PerspectiveStretch, RadialStretch
from operations.pattern import Dither, Mosaic
from operations.pixel import PixelFilter, PixelMath, PixelSort
from operations.row import RowRemove, RowShift, RowShuffle, RowStretch
from presets import get_all_presets, get_preset


def custom_pipeline(input_path: str, output_dir: str | None = None, **kwargs):
    """Custom pipeline for user experimentation.

    Modify this function to create your own image transformations!

    Args:
        input_path: Path to input image
        output_dir: Directory to save all steps (auto-generated if None)
        **kwargs: Additional arguments (use_cache, verbose, etc.)

    Returns:
        ImageContext after pipeline execution
    """
    # Create pipeline
    pipeline = Pipeline(
        input_path,
        output_dir=output_dir,
        use_cache=kwargs.get("use_cache", True),
        verbose=kwargs.get("verbose", False),
    )

    # ============================================================
    # ADD YOUR OPERATIONS HERE!
    # ============================================================
    # Example (uncomment and modify):

    # pipeline.add(PixelFilter(condition="prime", fill_color=(255, 0, 0, 128)))
    # pipeline.add(RowShift(selection="every_n", n=1, shift_amount=10, wrap=True))
    # pipeline.add(ChannelSwap(red_source="green", green_source="blue", blue_source="red"))
    # pipeline.add(Mosaic(tile_size=(16, 16), gap_size=2, sample_mode="average"))
    pipeline.add(BlockScramble(block_width=256, block_height=256, seed=42, scramble_ratio=0.5))

    # ============================================================

    # Execute pipeline
    return pipeline.execute()


def apply_preset(preset_name: str, input_path: str, output_dir: str | None = None, **kwargs):
    """Apply a built-in preset to an image.

    Args:
        preset_name: Name of the preset to apply
        input_path: Path to input image
        output_dir: Directory to save all steps (auto-generated if None)
        **kwargs: Additional arguments (use_cache, verbose, etc.)

    Returns:
        ImageContext after pipeline execution
    """
    try:
        preset_config = get_preset(preset_name)
    except FileNotFoundError:
        print(f"❌ Preset '{preset_name}' not found")
        print(f"Available presets: {', '.join(get_all_presets().keys())}")
        return None

    # Create pipeline
    pipeline = Pipeline(
        input_path,
        output_dir=output_dir,
        use_cache=kwargs.get("use_cache", True),
        verbose=kwargs.get("verbose", False),
    )

    # Map operation types to classes
    operation_map = {
        "PixelFilter": PixelFilter,
        "PixelMath": PixelMath,
        "PixelSort": PixelSort,
        "RowShift": RowShift,
        "RowStretch": RowStretch,
        "RowRemove": RowRemove,
        "RowShuffle": RowShuffle,
        "ColumnShift": ColumnShift,
        "ColumnStretch": ColumnStretch,
        "ColumnMirror": ColumnMirror,
        "ColumnWeave": ColumnWeave,
        "BlockFilter": BlockFilter,
        "BlockShift": BlockShift,
        "BlockRotate": BlockRotate,
        "BlockScramble": BlockScramble,
        "GridWarp": GridWarp,
        "PerspectiveStretch": PerspectiveStretch,
        "RadialStretch": RadialStretch,
        "AspectStretch": AspectStretch,
        "AspectCrop": AspectCrop,
        "AspectPad": AspectPad,
        "ChannelSwap": ChannelSwap,
        "ChannelIsolate": ChannelIsolate,
        "AlphaGenerator": AlphaGenerator,
        "Mosaic": Mosaic,
        "Dither": Dither,
    }

    # Add operations from preset
    for op_config in preset_config["operations"]:
        op_type = op_config["type"]
        op_params = op_config.get("params", {})

        if op_type in operation_map:
            # Handle special parameter conversions
            if (
                op_type == "Mosaic"
                and "tile_size" in op_params
                and isinstance(op_params["tile_size"], int)
            ):
                # Ensure tile_size is a tuple
                op_params["tile_size"] = (op_params["tile_size"], op_params["tile_size"])

            if (
                op_type == "BlockFilter"
                and "block_size" in op_params
                and isinstance(op_params["block_size"], int)
            ):
                # Convert block_size to block_width and block_height
                op_params["block_width"] = op_params["block_size"]
                op_params["block_height"] = op_params["block_size"]
                del op_params["block_size"]

            # Create and add operation
            operation = operation_map[op_type](**op_params)
            pipeline.add(operation)
        else:
            print(f"⚠️  Unknown operation type: {op_type}")

    # Execute pipeline
    return pipeline.execute()


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Transform images with pixel-perfect effects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run custom pipeline (edit custom_pipeline function)
    python transform.py input.jpg

    # Apply a preset
    python transform.py input.jpg --preset glitch-effect

    # Disable cache for fresh computation
    python transform.py input.jpg --no-cache

    # Specify output directory
    python transform.py input.jpg --output-dir my_experiment/

    # Verbose output
    python transform.py input.jpg --verbose

    # List all available presets
    python transform.py --list-presets
        """,
    )

    parser.add_argument("input", nargs="?", help="Input image path")
    parser.add_argument("--preset", help="Apply a built-in preset")
    parser.add_argument(
        "--output-dir",
        help="Directory to save all steps (auto-generates timestamp if not specified)",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable cache (force fresh computation)"
    )
    parser.add_argument(
        "--cache-from", nargs="+", help="Additional directories to use as cache sources"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--list-presets", action="store_true", help="List all available presets")

    args = parser.parse_args()

    # Handle list presets
    if args.list_presets:
        presets = get_all_presets()
        print("📦 Available presets:")
        for name, config in presets.items():
            desc = config.get("description", "No description")
            print(f"  • {name}: {desc}")
        return 0

    # Validate input
    if not args.input:
        parser.error("Input image path is required (unless using --list-presets)")

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return 1

    # Prepare kwargs
    kwargs = {
        "use_cache": not args.no_cache,
        "verbose": args.verbose,
    }

    # Add cache directories if specified
    if args.cache_from:
        kwargs["cache_dirs"] = args.cache_from

    try:
        if args.preset:
            # Apply preset
            print(f"🎨 Applying preset: {args.preset}")
            context = apply_preset(
                args.preset, str(input_path), output_dir=args.output_dir, **kwargs
            )
        else:
            # Run custom pipeline
            print("🔧 Running custom pipeline (edit custom_pipeline function to add operations)")
            context = custom_pipeline(str(input_path), output_dir=args.output_dir, **kwargs)

        if context:
            print("✅ Pipeline complete!")
            print(f"📏 Final dimensions: {context.width}×{context.height}")
            if context.warnings:
                print(f"⚠️  Warnings: {len(context.warnings)}")
                for warning in context.warnings[:3]:
                    print(f"   • {warning}")
        else:
            return 1

    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
