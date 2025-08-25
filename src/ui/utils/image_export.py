"""Image export utilities with format conversion and optimization."""

import io
import zipfile
from typing import Any

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from ui.utils.image_utils import optimize_image_for_display


def export_image(
    image: Image.Image, format_name: str, quality: int = 95, optimize: bool = True, **kwargs
) -> bytes:
    """Export image to specified format with options."""

    buffer = io.BytesIO()

    # Handle format-specific options
    export_kwargs: dict[str, Any] = {"format": format_name.upper()}

    if format_name.upper() == "JPEG":
        # Ensure RGB mode for JPEG
        if image.mode in ("RGBA", "LA"):
            # Create white background for transparent images
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(
                image, mask=image.split()[-1] if image.mode in ("RGBA", "LA") else None
            )
            image = background
        elif image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        export_kwargs["quality"] = quality
        export_kwargs["optimize"] = optimize
        export_kwargs["progressive"] = kwargs.get("progressive", False)

    elif format_name.upper() == "PNG":
        export_kwargs["optimize"] = optimize
        export_kwargs["compress_level"] = kwargs.get("compress_level", 6)

    elif format_name.upper() == "WEBP":
        export_kwargs["quality"] = quality
        export_kwargs["optimize"] = optimize
        export_kwargs["lossless"] = kwargs.get("lossless", False)

    elif format_name.upper() == "TIFF":
        export_kwargs["compression"] = kwargs.get("compression", "lzw")
        export_kwargs["optimize"] = optimize

    # Export image
    image.save(buffer, **export_kwargs)
    return buffer.getvalue()


def resize_image_for_export(
    image: Image.Image,
    size_option: str,
    custom_width: int | None = None,
    custom_height: int | None = None,
) -> Image.Image:
    """Resize image based on export size option."""

    if size_option == "original":
        return image

    elif size_option == "web_small":
        return optimize_image_for_display(image, 800, 600)

    elif size_option == "web_medium":
        return optimize_image_for_display(image, 1200, 900)

    elif size_option == "web_large":
        return optimize_image_for_display(image, 1920, 1440)

    elif size_option == "thumbnail":
        return optimize_image_for_display(image, 256, 256)

    elif size_option == "custom" and custom_width and custom_height:
        return image.resize((custom_width, custom_height), Image.Resampling.LANCZOS)

    else:
        return image


def generate_filename(
    base_name: str,
    operation_name: str | None = None,
    step_number: int | None = None,
    format_name: str = "png",
) -> str:
    """Generate appropriate filename for export."""

    # Clean base name
    clean_base = "".join(c for c in base_name if c.isalnum() or c in (" ", "-", "_")).strip()
    clean_base = clean_base.replace(" ", "_")

    # Build filename parts
    parts = [clean_base]

    if step_number is not None:
        parts.append(f"step_{step_number:02d}")

    if operation_name:
        clean_op = "".join(c for c in operation_name if c.isalnum() or c in ("_",)).strip()
        parts.append(clean_op.lower())

    filename = "_".join(parts)
    return f"{filename}.{format_name.lower()}"


def add_watermark(
    image: Image.Image, text: str, position: str = "bottom_right", opacity: float = 0.7
) -> Image.Image:
    """Add text watermark to image."""

    # Create a copy to avoid modifying original
    watermarked = image.copy()

    # Create watermark layer
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Try to use a nice font, fall back to default
    try:
        max(12, min(image.width, image.height) // 40)
        # This would require font files - using default for now
        font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # Get text dimensions
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Calculate position
    margin = 10
    if position == "bottom_right":
        x = image.width - text_width - margin
        y = image.height - text_height - margin
    elif position == "bottom_left":
        x = margin
        y = image.height - text_height - margin
    elif position == "top_right":
        x = image.width - text_width - margin
        y = margin
    elif position == "top_left":
        x = margin
        y = margin
    else:  # center
        x = (image.width - text_width) // 2
        y = (image.height - text_height) // 2

    # Draw semi-transparent background
    bg_padding = 5
    bg_color = (0, 0, 0, int(128 * opacity))
    draw.rectangle(
        [x - bg_padding, y - bg_padding, x + text_width + bg_padding, y + text_height + bg_padding],
        fill=bg_color,
    )

    # Draw text
    text_color = (255, 255, 255, int(255 * opacity))
    draw.text((x, y), text, font=font, fill=text_color)

    # Composite watermark onto image
    if watermarked.mode != "RGBA":
        watermarked = watermarked.convert("RGBA")

    watermarked = Image.alpha_composite(watermarked, overlay)

    # Convert back to original mode if needed
    if image.mode != "RGBA":
        watermarked = watermarked.convert(image.mode)

    return watermarked


def export_pipeline_steps(execution_result: Any, export_config: dict[str, Any]) -> bytes:
    """Export all pipeline steps as a ZIP file."""

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Export original image
        original_image = st.session_state.get("original_image")
        if original_image:
            original_data = export_image(
                original_image, export_config.get("format", "png"), export_config.get("quality", 95)
            )
            filename = generate_filename("original", format_name=export_config.get("format", "png"))
            zip_file.writestr(filename, original_data)

        # Export each step
        if execution_result and execution_result.steps:
            for i, step in enumerate(execution_result.steps):
                step_image = step.get("image")
                if step_image:
                    # Resize if needed
                    if export_config.get("resize_option") != "original":
                        step_image = resize_image_for_export(
                            step_image,
                            export_config.get("resize_option", "original"),
                            export_config.get("custom_width"),
                            export_config.get("custom_height"),
                        )

                    # Add watermark if requested
                    if export_config.get("add_watermark", False):
                        watermark_text = export_config.get("watermark_text", "Pixel Perfect")
                        step_image = add_watermark(step_image, watermark_text)

                    # Export step image
                    step_data = export_image(
                        step_image,
                        export_config.get("format", "png"),
                        export_config.get("quality", 95),
                    )

                    filename = generate_filename(
                        "processed",
                        step.get("operation", "unknown"),
                        i + 1,
                        export_config.get("format", "png"),
                    )
                    zip_file.writestr(filename, step_data)

        # Export final image
        final_image = st.session_state.get("processed_image")
        if final_image:
            # Resize if needed
            if export_config.get("resize_option") != "original":
                final_image = resize_image_for_export(
                    final_image,
                    export_config.get("resize_option", "original"),
                    export_config.get("custom_width"),
                    export_config.get("custom_height"),
                )

            # Add watermark if requested
            if export_config.get("add_watermark", False):
                watermark_text = export_config.get("watermark_text", "Pixel Perfect")
                final_image = add_watermark(final_image, watermark_text)

            final_data = export_image(
                final_image, export_config.get("format", "png"), export_config.get("quality", 95)
            )
            filename = generate_filename("final", format_name=export_config.get("format", "png"))
            zip_file.writestr(filename, final_data)

    return zip_buffer.getvalue()


def get_image_metadata(image: Image.Image) -> dict[str, Any]:
    """Extract metadata from image for export."""

    metadata = {
        "dimensions": f"{image.width} × {image.height}",
        "mode": image.mode,
        "format": getattr(image, "format", "Unknown"),
        "size_bytes": len(image.tobytes()) if hasattr(image, "tobytes") else 0,
        "has_transparency": image.mode in ("RGBA", "LA", "P") and "transparency" in image.info,
    }

    # Add EXIF data if available
    if hasattr(image, "getexif"):
        exif_data = image.getexif()
        if exif_data:
            metadata["exif"] = dict(exif_data)

    # Add PIL info
    if hasattr(image, "info") and image.info:
        metadata["info"] = dict(image.info)

    return metadata


def estimate_export_size(image: Image.Image, format_name: str, quality: int = 95) -> int:
    """Estimate export file size without actually exporting."""

    # Quick size estimation based on format and image properties
    width, height = image.size
    pixels = width * height

    if format_name.upper() == "PNG":
        # PNG: roughly 3-4 bytes per pixel for RGBA, 1-2 for grayscale
        if image.mode == "RGBA":
            return int(pixels * 3.5)
        elif image.mode == "RGB":
            return int(pixels * 2.5)
        else:
            return int(pixels * 1.5)

    elif format_name.upper() == "JPEG":
        # JPEG: compression ratio based on quality
        base_size = pixels * 3  # RGB
        compression_ratio = (100 - quality) / 100 * 0.9 + 0.1  # 10% to 100%
        return int(base_size * compression_ratio)

    elif format_name.upper() == "WEBP":
        # WebP: generally smaller than JPEG at same quality
        base_size = pixels * 3
        compression_ratio = (100 - quality) / 100 * 0.7 + 0.05  # 5% to 75%
        return int(base_size * compression_ratio)

    elif format_name.upper() == "TIFF":
        # TIFF: usually larger, depends on compression
        return int(pixels * 4)  # Uncompressed estimate

    else:
        # Generic estimate
        return int(pixels * 3)


def format_file_size(size_bytes: int | float) -> str:
    """Format file size in human readable format."""

    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024

    return f"{size_bytes:.1f} TB"
