#!/usr/bin/env python3
"""Main entry point for pixel-perfect image processing."""

import sys
from pathlib import Path

# Add src to path for clean imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from PIL import Image

from core.context import ImageContext
from operations.geometric import GridWarp, PerspectiveStretch, RadialStretch


def main():
    """Demonstrate geometric transformations on enzo.jpg."""

    # Load the image
    image = Image.open("enzo.jpg").convert("RGBA")
    original_size = image.size
    print(f"✅ Loaded image: {original_size[0]}x{original_size[1]} pixels")

    # Create ImageContext
    context = ImageContext(
        width=original_size[0],
        height=original_size[1],
        channels=len(image.getbands()),
        dtype="uint8",
    )

    print("\n🌊 Applying GridWarp (horizontal wave effect)...")
    warp_operation = GridWarp(axis="horizontal", frequency=2.0, amplitude=20.0)
    validated_context = warp_operation.validate_operation(context)
    warped_image, warped_context = warp_operation.apply(image, validated_context)
    warped_image.save("enzo_warped.webp", "WEBP")
    print(f"   ✓ Saved warped image: {warped_image.size}")

    print("\n🎭 Applying PerspectiveStretch (trapezoid effect)...")
    perspective_operation = PerspectiveStretch(top_factor=1.0, bottom_factor=1.8)
    validated_context = perspective_operation.validate_operation(context)
    perspective_image, perspective_context = perspective_operation.apply(image, validated_context)
    perspective_image.save("enzo_perspective.webp", "WEBP")
    print(f"   ✓ Saved perspective image: {perspective_image.size}")

    print("\n💫 Applying RadialStretch (center expansion)...")
    radial_operation = RadialStretch(factor=1.5, falloff="quadratic")
    validated_context = radial_operation.validate_operation(context)
    radial_image, radial_context = radial_operation.apply(image, validated_context)
    radial_image.save("enzo_radial.webp", "WEBP")
    print(f"   ✓ Saved radial image: {radial_image.size}")

    print("\n🎯 Chaining operations (warp + perspective + radial)...")
    # Chain all three operations
    step1_context = warp_operation.validate_operation(context)
    step1_image, step1_context = warp_operation.apply(image, step1_context)

    step2_context = perspective_operation.validate_operation(step1_context)
    step2_image, step2_context = perspective_operation.apply(step1_image, step2_context)

    step3_context = radial_operation.validate_operation(step2_context)
    final_image, final_context = radial_operation.apply(step2_image, step3_context)

    final_image.save("enzo_combined.webp", "WEBP")
    print(f"   ✓ Saved combined effects: {original_size} → {final_image.size}")

    print("\n✨ Geometric transformations complete!")
    print(f"   Original: {original_size[0]}x{original_size[1]} pixels")
    print(
        "   Files created: enzo_warped.webp, enzo_perspective.webp, enzo_radial.webp, enzo_combined.webp"
    )

    total_memory = (
        warped_context.memory_estimate
        + perspective_context.memory_estimate
        + radial_context.memory_estimate
    )
    print(f"   Total memory used: {total_memory:,} bytes ({total_memory / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
