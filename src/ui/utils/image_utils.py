"""Image processing utilities for the UI."""

import io

from PIL import Image


def validate_uploaded_image(uploaded_file) -> bool:
    """
    Validate an uploaded image file.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        True if valid, raises exception if not

    Raises:
        ValueError: If file is invalid
    """

    # File size check (50MB limit)
    if uploaded_file.size > 50 * 1024 * 1024:
        raise ValueError("File too large (max 50MB)")

    # File type validation
    allowed_types = ["image/png", "image/jpeg", "image/webp", "image/bmp", "image/tiff"]
    if uploaded_file.type not in allowed_types:
        raise ValueError(f"Unsupported file type: {uploaded_file.type}")

    # Try to open and validate as image
    try:
        image = Image.open(uploaded_file)
        image.verify()  # Verify it's a valid image
        uploaded_file.seek(0)  # Reset file pointer after verify
        return True
    except Exception as e:
        raise ValueError(f"Invalid image file: {str(e)}") from e


def optimize_image_for_display(
    image: Image.Image, max_width: int = 800, max_height: int = 600
) -> Image.Image:
    """
    Optimize image for web display without losing aspect ratio.

    Args:
        image: PIL Image to optimize
        max_width: Maximum display width
        max_height: Maximum display height

    Returns:
        Optimized PIL Image
    """
    original_width, original_height = image.size

    # Calculate scaling factor to fit within bounds
    width_scale = max_width / original_width
    height_scale = max_height / original_height
    scale_factor = min(width_scale, height_scale, 1.0)  # Don't upscale

    if scale_factor < 1.0:
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return image


def get_image_info(image: Image.Image) -> dict:
    """
    Get detailed information about an image.

    Args:
        image: PIL Image to analyze

    Returns:
        Dictionary with image information
    """

    # Calculate file size estimate
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    size_bytes = len(img_buffer.getvalue())

    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": getattr(image, "format", "Unknown"),
        "size_bytes": size_bytes,
        "size_kb": size_bytes / 1024,
        "size_mb": size_bytes / (1024 * 1024),
        "aspect_ratio": image.width / image.height,
        "total_pixels": image.width * image.height,
    }


def create_image_thumbnail(image: Image.Image, size: tuple = (200, 200)) -> Image.Image:
    """
    Create a thumbnail of the image maintaining aspect ratio.

    Args:
        image: PIL Image to create thumbnail from
        size: Tuple of (max_width, max_height)

    Returns:
        Thumbnail PIL Image
    """
    thumbnail = image.copy()
    thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
    return thumbnail


def convert_image_mode(image: Image.Image, target_mode: str) -> Image.Image:
    """
    Convert image to target mode if needed.

    Args:
        image: PIL Image to convert
        target_mode: Target mode ('RGB', 'RGBA', etc.)

    Returns:
        Converted PIL Image
    """
    if image.mode == target_mode:
        return image

    if target_mode == "RGB" and image.mode == "RGBA":
        # Create white background for transparent images
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
        return background

    return image.convert(target_mode)
