"""Color channel manipulation operations."""

import math
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import Field, field_validator, model_validator

from core.base import BaseOperation
from core.context import ImageContext
from exceptions import ProcessingError, ValidationError
from utils.validation import validate_color_tuple


def _ensure_image_mode(image: Image.Image, required_mode: str) -> Image.Image:
    """Convert image mode safely.

    Args:
        image: PIL Image to convert
        required_mode: Target mode ("L", "RGB", "RGBA")

    Returns:
        Image in required mode
    """
    if image.mode == required_mode:
        return image

    try:
        if required_mode == "RGBA" and image.mode in ["L", "RGB"]:
            # Add opaque alpha channel
            return image.convert("RGBA")
        elif required_mode == "RGB" and image.mode == "L":
            # Convert grayscale to RGB
            return image.convert("RGB")
        elif required_mode == "L" and image.mode in ["RGB", "RGBA"]:
            # Convert to grayscale
            return image.convert("L")
        else:
            # General conversion
            return image.convert(required_mode)
    except Exception as e:
        raise ProcessingError(
            f"Failed to convert image from {image.mode} to {required_mode}: {e}"
        ) from e


def _extract_channel(image: Image.Image, channel_name: str) -> np.ndarray:
    """Extract single channel as numpy array.

    Args:
        image: PIL Image to extract from
        channel_name: Channel to extract ("r", "g", "b", "a", "l")

    Returns:
        2D numpy array of channel values
    """
    if image.mode == "L":
        if channel_name in ["l", "r", "g", "b"]:
            return np.array(image)
        else:
            raise ValidationError(f"Cannot extract channel '{channel_name}' from grayscale image")

    elif image.mode == "RGB":
        if channel_name == "r":
            return np.array(image)[:, :, 0]
        elif channel_name == "g":
            return np.array(image)[:, :, 1]
        elif channel_name == "b":
            return np.array(image)[:, :, 2]
        elif channel_name == "a":
            # RGB doesn't have alpha, return opaque
            return np.full(image.size[::-1], 255, dtype=np.uint8)
        else:
            raise ValidationError(f"Cannot extract channel '{channel_name}' from RGB image")

    elif image.mode == "RGBA":
        if channel_name == "r":
            return np.array(image)[:, :, 0]
        elif channel_name == "g":
            return np.array(image)[:, :, 1]
        elif channel_name == "b":
            return np.array(image)[:, :, 2]
        elif channel_name == "a":
            return np.array(image)[:, :, 3]
        else:
            raise ValidationError(f"Cannot extract channel '{channel_name}' from RGBA image")

    else:
        raise ValidationError(f"Unsupported image mode: {image.mode}")


def _combine_channels(channels_dict: dict[str, np.ndarray], target_mode: str) -> Image.Image:
    """Combine channels into PIL Image.

    Args:
        channels_dict: Dict mapping channel names to numpy arrays
        target_mode: Target image mode ("L", "RGB", "RGBA")

    Returns:
        Combined PIL Image
    """
    if target_mode == "L":
        if "l" in channels_dict:
            return Image.fromarray(channels_dict["l"], "L")
        elif "r" in channels_dict:
            # Use red channel as luminance
            return Image.fromarray(channels_dict["r"], "L")
        else:
            raise ValidationError("No suitable channel for grayscale image")

    elif target_mode == "RGB":
        if all(ch in channels_dict for ch in ["r", "g", "b"]):
            rgb_array = np.stack(
                [channels_dict["r"], channels_dict["g"], channels_dict["b"]], axis=-1
            )
            return Image.fromarray(rgb_array, "RGB")
        else:
            raise ValidationError("Missing RGB channels for RGB image")

    elif target_mode == "RGBA":
        if all(ch in channels_dict for ch in ["r", "g", "b", "a"]):
            rgba_array = np.stack(
                [channels_dict["r"], channels_dict["g"], channels_dict["b"], channels_dict["a"]],
                axis=-1,
            )
            return Image.fromarray(rgba_array, "RGBA")
        else:
            raise ValidationError("Missing RGBA channels for RGBA image")

    else:
        raise ValidationError(f"Unsupported target mode: {target_mode}")


def _calculate_luminance(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calculate luminance using ITU-R BT.709 formula.

    Args:
        r, g, b: RGB channel arrays

    Returns:
        Luminance array
    """
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)


def _calculate_saturation(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calculate HSV saturation.

    Args:
        r, g, b: RGB channel arrays

    Returns:
        Saturation array (0-255)
    """
    max_vals = np.maximum(np.maximum(r, g), b)
    min_vals = np.minimum(np.minimum(r, g), b)

    # Avoid division by zero
    saturation = np.where(max_vals > 0, (max_vals - min_vals) / max_vals * 255, 0)

    return saturation.astype(np.uint8)


def _color_distance(image_array: np.ndarray, target_color: tuple[int, int, int]) -> np.ndarray:
    """Calculate Euclidean distance from target color.

    Args:
        image_array: RGB image array (H, W, 3)
        target_color: Target RGB color

    Returns:
        Distance array (0-255, normalized)
    """
    target = np.array(target_color)
    distances = np.sqrt(np.sum((image_array - target) ** 2, axis=-1))

    # Normalize to 0-255 range (max possible distance is sqrt(3*255^2))
    max_distance = math.sqrt(3 * 255 * 255)
    normalized = (distances / max_distance * 255).astype(np.uint8)

    return normalized


class ChannelSwap(BaseOperation):
    """Rearrange color channels according to mapping."""

    mapping: dict[str, str] = Field(
        {"r": "b", "b": "r"}, description="Channel mapping (destination: source)"
    )

    @field_validator("mapping")
    @classmethod
    def validate_mapping(cls, v):
        """Validate channel mapping."""
        valid_channels = {"r", "g", "b", "a"}

        # Check all keys and values are valid channels
        for dest, src in v.items():
            if dest not in valid_channels:
                raise ValueError(f"Invalid destination channel: {dest}")
            if src not in valid_channels:
                raise ValueError(f"Invalid source channel: {src}")

        # Check for circular references (optional - could be useful)
        # For now, allow any mapping including circular ones

        return v

    def validate_operation(self, context: ImageContext) -> ImageContext:
        """Validate operation against image context."""
        # Determine if we need to convert image mode
        max_channels_needed = 3  # RGB
        if any(ch in self.mapping for ch in ["a"]) or any(
            ch == "a" for ch in self.mapping.values()
        ):
            max_channels_needed = 4  # RGBA

        new_channels = max(context.channels, max_channels_needed)

        return context.copy_with_updates(channels=new_channels)

    def get_cache_key(self, image_hash: str) -> str:
        """Generate cache key for this operation."""
        mapping_str = ",".join(f"{k}:{v}" for k, v in sorted(self.mapping.items()))
        return f"channelswap_{image_hash}_{hash(mapping_str)}"

    def estimate_memory(self, context: ImageContext) -> int:
        """Estimate memory usage for this operation."""
        # May need to convert to RGBA if alpha channel is involved
        channels_needed = (
            4
            if any(ch == "a" for ch in list(self.mapping.keys()) + list(self.mapping.values()))
            else 3
        )
        return context.width * context.height * max(context.channels, channels_needed)

    def apply(self, image: Image.Image, context: ImageContext) -> tuple[Image.Image, ImageContext]:
        """Apply channel swap transformation."""
        try:
            # Determine required image mode
            needs_alpha = any(ch in self.mapping for ch in ["a"]) or any(
                ch == "a" for ch in self.mapping.values()
            )

            # Also preserve alpha if the original image has it and we're not explicitly removing it
            has_original_alpha = image.mode in ("RGBA", "LA")

            if needs_alpha or has_original_alpha:
                working_image = _ensure_image_mode(image, "RGBA")
                channels_list = ["r", "g", "b", "a"]
            else:
                working_image = _ensure_image_mode(image, "RGB")
                channels_list = ["r", "g", "b"]

            # Extract all channels
            channels = {}
            for ch in channels_list:
                channels[ch] = _extract_channel(working_image, ch)

            # Apply mapping
            new_channels = {}
            for ch in channels_list:
                if ch in self.mapping:
                    # This channel gets mapped from another channel
                    source_ch = self.mapping[ch]
                    new_channels[ch] = channels[source_ch]
                else:
                    # This channel stays the same
                    new_channels[ch] = channels[ch]

            # Combine channels back into image
            result_image = _combine_channels(new_channels, working_image.mode)

            # Update context
            updated_context = context.copy_with_updates(
                channels=len(working_image.getbands()),
                memory_estimate=self.estimate_memory(context),
            )

            return result_image, updated_context

        except Exception as e:
            raise ProcessingError(f"ChannelSwap failed: {e}") from e


class ChannelIsolate(BaseOperation):
    """Keep only specific channels, fill others with specified value."""

    keep_channels: list[str] = Field(["r"], description="Channels to preserve")
    fill_value: int = Field(0, ge=0, le=255, description="Value for removed channels")

    @field_validator("keep_channels")
    @classmethod
    def validate_keep_channels(cls, v):
        """Validate channel list."""
        valid_channels = {"r", "g", "b", "a"}

        if not v:
            raise ValueError("Must keep at least one channel")

        for ch in v:
            if ch not in valid_channels:
                raise ValueError(f"Invalid channel: {ch}")

        return v

    def validate_operation(self, context: ImageContext) -> ImageContext:
        """Validate operation against image context."""
        # Determine if we need to convert image mode
        needs_alpha = "a" in self.keep_channels
        new_channels = 4 if needs_alpha else max(3, context.channels)

        return context.copy_with_updates(channels=new_channels)

    def get_cache_key(self, image_hash: str) -> str:
        """Generate cache key for this operation."""
        channels_str = ",".join(sorted(self.keep_channels))
        return f"channelisolate_{image_hash}_{hash(f'{channels_str}_{self.fill_value}')}"

    def estimate_memory(self, context: ImageContext) -> int:
        """Estimate memory usage for this operation."""
        channels_needed = 4 if "a" in self.keep_channels else 3
        return context.width * context.height * max(context.channels, channels_needed)

    def apply(self, image: Image.Image, context: ImageContext) -> tuple[Image.Image, ImageContext]:
        """Apply channel isolation."""
        try:
            # Determine required image mode
            needs_alpha = "a" in self.keep_channels

            if needs_alpha:
                working_image = _ensure_image_mode(image, "RGBA")
                all_channels = ["r", "g", "b", "a"]
            else:
                working_image = _ensure_image_mode(image, "RGB")
                all_channels = ["r", "g", "b"]

            # Extract all channels
            channels = {}
            for ch in all_channels:
                channels[ch] = _extract_channel(working_image, ch)

            # Fill non-preserved channels
            fill_array = np.full_like(channels[all_channels[0]], self.fill_value)

            new_channels = {}
            for ch in all_channels:
                if ch in self.keep_channels:
                    new_channels[ch] = channels[ch]
                else:
                    new_channels[ch] = fill_array

            # Combine channels back into image
            result_image = _combine_channels(new_channels, working_image.mode)

            # Update context
            updated_context = context.copy_with_updates(
                channels=len(working_image.getbands()),
                memory_estimate=self.estimate_memory(context),
            )

            return result_image, updated_context

        except Exception as e:
            raise ProcessingError(f"ChannelIsolate failed: {e}") from e


class AlphaGenerator(BaseOperation):
    """Generate alpha channel from image content."""

    source: Literal["luminance", "saturation", "specific_color"] = Field(
        "luminance", description="Source for alpha generation"
    )
    threshold: int = Field(128, ge=0, le=255, description="Threshold for alpha generation")
    invert: bool = Field(False, description="Invert alpha values")
    color_target: tuple[int, int, int] | None = Field(
        None, description="Target color for specific_color mode"
    )

    @field_validator("color_target")
    @classmethod
    def validate_color_target(cls, v):
        """Validate target color."""
        if v is not None:
            return validate_color_tuple(v, channels=3)
        return v

    @model_validator(mode="after")
    def validate_specific_color_requirements(self):
        """Validate that color_target is provided for specific_color mode."""
        if self.source == "specific_color" and self.color_target is None:
            raise ValueError("color_target is required when source='specific_color'")
        return self

    def validate_operation(self, context: ImageContext) -> ImageContext:
        """Validate operation against image context."""
        # Always results in RGBA image
        return context.copy_with_updates(channels=4)

    def get_cache_key(self, image_hash: str) -> str:
        """Generate cache key for this operation."""
        config_str = f"{self.source}_{self.threshold}_{self.invert}_{self.color_target}"
        return f"alphagen_{image_hash}_{hash(config_str)}"

    def estimate_memory(self, context: ImageContext) -> int:
        """Estimate memory usage for this operation."""
        # Always produces RGBA
        return context.width * context.height * 4

    def apply(self, image: Image.Image, context: ImageContext) -> tuple[Image.Image, ImageContext]:
        """Apply alpha generation."""
        try:
            # Ensure RGB mode for analysis
            rgb_image = _ensure_image_mode(image, "RGB")
            rgb_array = np.array(rgb_image)

            r_channel = rgb_array[:, :, 0]
            g_channel = rgb_array[:, :, 1]
            b_channel = rgb_array[:, :, 2]

            # Generate alpha based on source mode
            if self.source == "luminance":
                alpha_values = _calculate_luminance(r_channel, g_channel, b_channel)

            elif self.source == "saturation":
                alpha_values = _calculate_saturation(r_channel, g_channel, b_channel)

            elif self.source == "specific_color":
                # Calculate distance from target color
                distances = _color_distance(rgb_array, self.color_target)
                # Convert distance to alpha (closer = more opaque)
                alpha_values = 255 - distances

            else:
                raise ValidationError(f"Unknown source mode: {self.source}")

            # Apply threshold
            if self.threshold > 0:
                alpha_values = np.where(alpha_values >= self.threshold, 255, 0)

            # Apply inversion if requested
            if self.invert:
                alpha_values = 255 - alpha_values

            # Combine RGB + generated alpha
            rgba_array = np.concatenate([rgb_array, alpha_values[:, :, np.newaxis]], axis=-1)

            result_image = Image.fromarray(rgba_array, "RGBA")

            # Update context
            updated_context = context.copy_with_updates(
                channels=4, memory_estimate=self.estimate_memory(context)
            )

            return result_image, updated_context

        except Exception as e:
            raise ProcessingError(f"AlphaGenerator failed: {e}") from e
