"""Aspect ratio transformation operations."""

import math
from typing import Literal

import numpy as np
from PIL import Image, ImageFilter
from pydantic import Field, field_validator

from core.base import BaseOperation
from core.context import ImageContext
from exceptions import ProcessingError, ValidationError
from utils.validation import validate_color_tuple

# Common aspect ratios
COMMON_RATIOS = {
    "1:1": 1.0,  # Square
    "4:5": 0.8,  # Portrait (Instagram)
    "5:4": 1.25,  # Landscape (Instagram)
    "9:16": 0.5625,  # Vertical video
    "16:9": 1.7778,  # Horizontal video/widescreen
    "21:9": 2.3333,  # Ultrawide
    "3:2": 1.5,  # Traditional photo
    "4:3": 1.3333,  # Classic TV/monitor
    "2:3": 0.6667,  # Portrait photo
}


def _parse_ratio_string(ratio: str) -> float:
    """Convert ratio string like '16:9' to decimal ratio.

    Args:
        ratio: Ratio string in format "W:H"

    Returns:
        Decimal ratio (width/height)

    Raises:
        ValidationError: If ratio format is invalid
    """
    if ratio in COMMON_RATIOS:
        return COMMON_RATIOS[ratio]

    try:
        if ":" in ratio:
            width_str, height_str = ratio.split(":", 1)
            width = float(width_str.strip())
            height = float(height_str.strip())

            if width <= 0 or height <= 0:
                raise ValueError("Ratio components must be positive")

            return width / height
        else:
            # Try to parse as decimal
            decimal_ratio = float(ratio)
            if decimal_ratio <= 0:
                raise ValueError("Ratio must be positive")
            return decimal_ratio

    except (ValueError, ZeroDivisionError) as e:
        raise ValidationError(f"Invalid ratio format '{ratio}': {e}") from e


def _calculate_target_dimensions(
    current_size: tuple[int, int], target_ratio: float
) -> tuple[int, int]:
    """Calculate target dimensions for given aspect ratio.

    Args:
        current_size: Current (width, height)
        target_ratio: Target width/height ratio

    Returns:
        Target (width, height) maintaining current area when possible
    """
    current_width, current_height = current_size
    current_ratio = current_width / current_height

    if abs(current_ratio - target_ratio) < 0.001:
        # Already correct ratio
        return current_size

    # Calculate current area for reference
    # current_area = current_width * current_height  # May be needed for future enhancement

    if target_ratio > current_ratio:
        # Need to be wider - increase width
        new_height = current_height
        new_width = int(new_height * target_ratio)
    else:
        # Need to be taller - increase height
        new_width = current_width
        new_height = int(new_width / target_ratio)

    return (new_width, new_height)


def _get_crop_bounds(
    image_size: tuple[int, int], target_size: tuple[int, int], anchor: str
) -> tuple[int, int, int, int]:
    """Calculate crop rectangle bounds.

    Args:
        image_size: Current (width, height)
        target_size: Target (width, height)
        anchor: Crop anchor position

    Returns:
        Crop bounds as (left, top, right, bottom)
    """
    img_width, img_height = image_size
    crop_width, crop_height = target_size

    # Ensure crop doesn't exceed image size
    crop_width = min(crop_width, img_width)
    crop_height = min(crop_height, img_height)

    if anchor == "center":
        left = (img_width - crop_width) // 2
        top = (img_height - crop_height) // 2
    elif anchor == "top":
        left = (img_width - crop_width) // 2
        top = 0
    elif anchor == "bottom":
        left = (img_width - crop_width) // 2
        top = img_height - crop_height
    elif anchor == "left":
        left = 0
        top = (img_height - crop_height) // 2
    elif anchor == "right":
        left = img_width - crop_width
        top = (img_height - crop_height) // 2
    else:
        raise ValidationError(f"Invalid anchor position: {anchor}")

    return (left, top, left + crop_width, top + crop_height)


def _analyze_image_detail(image: Image.Image, regions: list) -> int:
    """Analyze regions for detail content and return best region index.

    Args:
        image: PIL Image to analyze
        regions: List of (left, top, right, bottom) region bounds

    Returns:
        Index of region with highest detail score
    """
    try:
        # Convert to grayscale for edge detection
        gray_image = image.convert("L")
        gray_array = np.array(gray_image)

        # Use simple gradient calculation for detail detection
        # Future enhancement: Could use Sobel operators for better edge detection
        # sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        # sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Apply simple gradient calculation
        grad_x = np.abs(np.gradient(gray_array, axis=1))
        grad_y = np.abs(np.gradient(gray_array, axis=0))
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        best_score = -1
        best_region = 0

        for i, (left, top, right, bottom) in enumerate(regions):
            # Extract region
            region_grad = gradient_magnitude[top:bottom, left:right]

            # Calculate detail score (mean gradient magnitude)
            detail_score = np.mean(region_grad)

            # Add center bias (prefer regions closer to center)
            img_center_x = image.width / 2
            img_center_y = image.height / 2
            region_center_x = (left + right) / 2
            region_center_y = (top + bottom) / 2

            distance_from_center = math.sqrt(
                (region_center_x - img_center_x) ** 2 + (region_center_y - img_center_y) ** 2
            )
            max_distance = math.sqrt(img_center_x**2 + img_center_y**2)
            center_bias = 1.0 - (distance_from_center / max_distance) * 0.2

            final_score = detail_score * center_bias

            if final_score > best_score:
                best_score = final_score
                best_region = i

        return best_region

    except Exception:
        # Fallback to center region (first region should be center)
        return 0


def _generate_padding(
    image: Image.Image,
    pad_mode: str,
    pad_color: tuple[int, int, int, int],
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> Image.Image:
    """Generate padded image with specified padding mode.

    Args:
        image: Source image
        pad_mode: Padding mode ("solid", "gradient", "mirror", "blur")
        pad_color: RGBA color for solid padding
        left, top, right, bottom: Padding amounts

    Returns:
        Padded PIL Image
    """
    if pad_mode == "solid":
        # Simple solid color padding
        new_size = (image.width + left + right, image.height + top + bottom)
        padded = Image.new(image.mode, new_size, pad_color[: len(image.getbands())])
        padded.paste(image, (left, top))
        return padded

    elif pad_mode == "mirror":
        # Mirror edge pixels outward
        new_size = (image.width + left + right, image.height + top + bottom)
        padded = Image.new(image.mode, new_size)

        # Paste original image
        padded.paste(image, (left, top))

        # Mirror edges
        if left > 0:
            # Left edge
            left_strip = image.crop((0, 0, min(left, image.width), image.height))
            left_strip = left_strip.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            padded.paste(left_strip, (left - left_strip.width, top))

        if right > 0:
            # Right edge
            right_strip = image.crop((max(0, image.width - right), 0, image.width, image.height))
            right_strip = right_strip.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            padded.paste(right_strip, (left + image.width, top))

        if top > 0:
            # Top edge
            top_strip = image.crop((0, 0, image.width, min(top, image.height)))
            top_strip = top_strip.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            padded.paste(top_strip, (left, top - top_strip.height))

        if bottom > 0:
            # Bottom edge
            bottom_strip = image.crop((0, max(0, image.height - bottom), image.width, image.height))
            bottom_strip = bottom_strip.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            padded.paste(bottom_strip, (left, top + image.height))

        return padded

    elif pad_mode == "blur":
        # Blur and extend edge regions
        new_size = (image.width + left + right, image.height + top + bottom)
        padded = Image.new(image.mode, new_size, pad_color[: len(image.getbands())])

        # Create blurred version of original image
        blurred = image.filter(ImageFilter.GaussianBlur(radius=5))

        # Paste original in center
        padded.paste(image, (left, top))

        # Extend blurred edges
        if left > 0:
            left_strip = blurred.crop((0, 0, 1, image.height))
            left_strip = left_strip.resize((left, image.height))
            padded.paste(left_strip, (0, top))

        if right > 0:
            right_strip = blurred.crop((image.width - 1, 0, image.width, image.height))
            right_strip = right_strip.resize((right, image.height))
            padded.paste(right_strip, (left + image.width, top))

        if top > 0:
            top_strip = blurred.crop((0, 0, image.width, 1))
            top_strip = top_strip.resize((image.width, top))
            padded.paste(top_strip, (left, 0))

        if bottom > 0:
            bottom_strip = blurred.crop((0, image.height - 1, image.width, image.height))
            bottom_strip = bottom_strip.resize((image.width, bottom))
            padded.paste(bottom_strip, (left, top + image.height))

        return padded

    else:  # gradient mode
        # Simple gradient fallback (fade to pad_color)
        return _generate_padding(image, "solid", pad_color, left, top, right, bottom)


class AspectStretch(BaseOperation):
    """Force image to specific aspect ratio via non-uniform scaling."""

    target_ratio: str = Field("16:9", description="Target aspect ratio (e.g., '16:9', '1:1')")
    method: Literal["simple", "segment"] = Field(
        "simple", description="Scaling factor (>1 for stretching, <1 for compressing)"
    )
    segment_count: int = Field(3, ge=1, description="Number of segments for segment method")
    preserve_center: bool = Field(True, description="Minimize center distortion")

    @field_validator("target_ratio")
    @classmethod
    def validate_ratio(cls, v):
        """Validate target ratio format."""
        _parse_ratio_string(v)  # Will raise ValidationError if invalid
        return v

    def validate_operation(self, context: ImageContext) -> ImageContext:
        """Validate operation against image context."""
        target_ratio = _parse_ratio_string(self.target_ratio)
        current_ratio = context.width / context.height

        if abs(current_ratio - target_ratio) < 0.001:
            context.add_warning("Image already has target aspect ratio")

        new_width, new_height = _calculate_target_dimensions(
            (context.width, context.height), target_ratio
        )

        return context.copy_with_updates(width=new_width, height=new_height)

    def get_cache_key(self, image_hash: str) -> str:
        """Generate cache key for this operation."""
        config_str = (
            f"{self.target_ratio}_{self.method}_{self.segment_count}_{self.preserve_center}"
        )
        return f"aspectstretch_{image_hash}_{hash(config_str)}"

    def estimate_memory(self, context: ImageContext) -> int:
        """Estimate memory usage for this operation."""
        target_ratio = _parse_ratio_string(self.target_ratio)
        new_width, new_height = _calculate_target_dimensions(
            (context.width, context.height), target_ratio
        )
        return new_width * new_height * context.channels

    def apply(self, image: Image.Image, context: ImageContext) -> tuple[Image.Image, ImageContext]:
        """Apply aspect stretch transformation."""
        try:
            target_ratio = _parse_ratio_string(self.target_ratio)
            current_ratio = image.width / image.height

            if abs(current_ratio - target_ratio) < 0.001:
                # Already correct ratio
                updated_context = context.copy_with_updates(
                    memory_estimate=self.estimate_memory(context)
                )
                return image, updated_context

            new_width, new_height = _calculate_target_dimensions(image.size, target_ratio)

            if self.method == "simple":
                # Direct resize to target dimensions
                result_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            else:  # segment method
                # Divide into vertical segments and stretch each
                segment_width = image.width // self.segment_count
                segments = []

                for i in range(self.segment_count):
                    left = i * segment_width
                    right = (i + 1) * segment_width if i < self.segment_count - 1 else image.width

                    segment = image.crop((left, 0, right, image.height))

                    # Calculate segment target width
                    segment_target_width = (new_width * (right - left)) // image.width

                    # Resize segment
                    stretched_segment = segment.resize(
                        (segment_target_width, new_height), Image.Resampling.LANCZOS
                    )
                    segments.append(stretched_segment)

                # Combine segments
                result_image = Image.new(image.mode, (new_width, new_height))
                current_x = 0
                for segment in segments:
                    result_image.paste(segment, (current_x, 0))
                    current_x += segment.width

            # Update context
            updated_context = context.copy_with_updates(
                width=new_width, height=new_height, memory_estimate=self.estimate_memory(context)
            )

            return result_image, updated_context

        except Exception as e:
            raise ProcessingError(f"AspectStretch failed: {e}") from e


class AspectCrop(BaseOperation):
    """Crop image to achieve target aspect ratio."""

    target_ratio: str = Field("16:9", description="Target aspect ratio (e.g., '16:9', '1:1')")
    anchor: Literal["center", "top", "bottom", "left", "right"] = Field(
        "center", description="Crop anchor position"
    )
    smart_crop: bool = Field(False, description="Use smart cropping to preserve detail")

    @field_validator("target_ratio")
    @classmethod
    def validate_ratio(cls, v):
        """Validate target ratio format."""
        _parse_ratio_string(v)  # Will raise ValidationError if invalid
        return v

    def validate_operation(self, context: ImageContext) -> ImageContext:
        """Validate operation against image context."""
        target_ratio = _parse_ratio_string(self.target_ratio)
        current_ratio = context.width / context.height

        if abs(current_ratio - target_ratio) < 0.001:
            context.add_warning("Image already has target aspect ratio")
            return context

        # Calculate crop dimensions
        if target_ratio > current_ratio:
            # Target is wider - crop height
            new_width = context.width
            new_height = int(context.width / target_ratio)
        else:
            # Target is taller - crop width
            new_height = context.height
            new_width = int(context.height * target_ratio)

        return context.copy_with_updates(width=new_width, height=new_height)

    def get_cache_key(self, image_hash: str) -> str:
        """Generate cache key for this operation."""
        config_str = f"{self.target_ratio}_{self.anchor}_{self.smart_crop}"
        return f"aspectcrop_{image_hash}_{hash(config_str)}"

    def estimate_memory(self, context: ImageContext) -> int:
        """Estimate memory usage for this operation."""
        target_ratio = _parse_ratio_string(self.target_ratio)
        current_ratio = context.width / context.height

        if target_ratio > current_ratio:
            new_width = context.width
            new_height = int(context.width / target_ratio)
        else:
            new_height = context.height
            new_width = int(context.height * target_ratio)

        return new_width * new_height * context.channels

    def apply(self, image: Image.Image, context: ImageContext) -> tuple[Image.Image, ImageContext]:
        """Apply aspect crop transformation."""
        try:
            target_ratio = _parse_ratio_string(self.target_ratio)
            current_ratio = image.width / image.height

            if abs(current_ratio - target_ratio) < 0.001:
                # Already correct ratio
                updated_context = context.copy_with_updates(
                    memory_estimate=self.estimate_memory(context)
                )
                return image, updated_context

            # Calculate crop dimensions
            if target_ratio > current_ratio:
                # Target is wider - crop height
                crop_width = image.width
                crop_height = int(image.width / target_ratio)
            else:
                # Target is taller - crop width
                crop_height = image.height
                crop_width = int(image.height * target_ratio)

            if self.smart_crop:
                # Generate multiple crop candidates
                candidates = []

                # Center crop
                center_bounds = _get_crop_bounds(image.size, (crop_width, crop_height), "center")
                candidates.append(center_bounds)

                # Add offset candidates if image is large enough
                if image.width > crop_width * 1.2:
                    left_bounds = _get_crop_bounds(image.size, (crop_width, crop_height), "left")
                    right_bounds = _get_crop_bounds(image.size, (crop_width, crop_height), "right")
                    candidates.extend([left_bounds, right_bounds])

                if image.height > crop_height * 1.2:
                    top_bounds = _get_crop_bounds(image.size, (crop_width, crop_height), "top")
                    bottom_bounds = _get_crop_bounds(
                        image.size, (crop_width, crop_height), "bottom"
                    )
                    candidates.extend([top_bounds, bottom_bounds])

                # Choose best candidate based on detail analysis
                best_index = _analyze_image_detail(image, candidates)
                crop_bounds = candidates[best_index]
            else:
                # Simple anchor-based crop
                crop_bounds = _get_crop_bounds(image.size, (crop_width, crop_height), self.anchor)

            # Apply crop
            result_image = image.crop(crop_bounds)

            # Update context
            updated_context = context.copy_with_updates(
                width=crop_width, height=crop_height, memory_estimate=self.estimate_memory(context)
            )

            return result_image, updated_context

        except Exception as e:
            raise ProcessingError(f"AspectCrop failed: {e}") from e


class AspectPad(BaseOperation):
    """Add padding to achieve target aspect ratio."""

    target_ratio: str = Field("16:9", description="Target aspect ratio (e.g., '16:9', '1:1')")
    pad_color: tuple[int, int, int, int] = Field(
        (0, 0, 0, 255), description="RGBA color for padding"
    )
    pad_mode: Literal["solid", "gradient", "mirror", "blur"] = Field(
        "solid", description="Padding mode"
    )

    @field_validator("target_ratio")
    @classmethod
    def validate_ratio(cls, v):
        """Validate target ratio format."""
        _parse_ratio_string(v)  # Will raise ValidationError if invalid
        return v

    @field_validator("pad_color")
    @classmethod
    def validate_pad_color(cls, v):
        """Validate padding color."""
        return validate_color_tuple(v, channels=4)

    def validate_operation(self, context: ImageContext) -> ImageContext:
        """Validate operation against image context."""
        target_ratio = _parse_ratio_string(self.target_ratio)
        current_ratio = context.width / context.height

        if abs(current_ratio - target_ratio) < 0.001:
            context.add_warning("Image already has target aspect ratio")
            return context

        # Calculate padding dimensions
        if target_ratio > current_ratio:
            # Target is wider - add horizontal padding
            new_width = int(context.height * target_ratio)
            new_height = context.height
        else:
            # Target is taller - add vertical padding
            new_width = context.width
            new_height = int(context.width / target_ratio)

        return context.copy_with_updates(width=new_width, height=new_height)

    def get_cache_key(self, image_hash: str) -> str:
        """Generate cache key for this operation."""
        config_str = f"{self.target_ratio}_{self.pad_color}_{self.pad_mode}"
        return f"aspectpad_{image_hash}_{hash(config_str)}"

    def estimate_memory(self, context: ImageContext) -> int:
        """Estimate memory usage for this operation."""
        target_ratio = _parse_ratio_string(self.target_ratio)
        current_ratio = context.width / context.height

        if target_ratio > current_ratio:
            new_width = int(context.height * target_ratio)
            new_height = context.height
        else:
            new_width = context.width
            new_height = int(context.width / target_ratio)

        return new_width * new_height * context.channels

    def apply(self, image: Image.Image, context: ImageContext) -> tuple[Image.Image, ImageContext]:
        """Apply aspect pad transformation."""
        try:
            target_ratio = _parse_ratio_string(self.target_ratio)
            current_ratio = image.width / image.height

            if abs(current_ratio - target_ratio) < 0.001:
                # Already correct ratio
                updated_context = context.copy_with_updates(
                    memory_estimate=self.estimate_memory(context)
                )
                return image, updated_context

            # Calculate padding amounts
            if target_ratio > current_ratio:
                # Target is wider - add horizontal padding
                new_width = int(image.height * target_ratio)
                new_height = image.height
                left_pad = (new_width - image.width) // 2
                right_pad = new_width - image.width - left_pad
                top_pad = 0
                bottom_pad = 0
            else:
                # Target is taller - add vertical padding
                new_width = image.width
                new_height = int(image.width / target_ratio)
                left_pad = 0
                right_pad = 0
                top_pad = (new_height - image.height) // 2
                bottom_pad = new_height - image.height - top_pad

            # Generate padded image
            result_image = _generate_padding(
                image, self.pad_mode, self.pad_color, left_pad, top_pad, right_pad, bottom_pad
            )

            # Update context
            updated_context = context.copy_with_updates(
                width=new_width, height=new_height, memory_estimate=self.estimate_memory(context)
            )

            return result_image, updated_context

        except Exception as e:
            raise ProcessingError(f"AspectPad failed: {e}") from e
