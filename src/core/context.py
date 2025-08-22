"""Image context that flows through the pipeline."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ImageContext(BaseModel):
    """Context carrying accumulated state through the pipeline."""

    width: int = Field(gt=0, description="Current image width in pixels")
    height: int = Field(gt=0, description="Current image height in pixels")
    channels: int = Field(ge=1, le=4, description="Number of channels (1=grayscale, 3=RGB, 4=RGBA)")
    dtype: Literal["uint8", "float32"] = Field(
        default="uint8", description="Data type of pixel values"
    )
    warnings: list[str] = Field(default_factory=list, description="Non-fatal issues encountered")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Optional metadata (EXIF, etc.)"
    )

    @field_validator("channels")
    @classmethod
    def validate_channels(cls, v: int) -> int:
        """Validate that channels is 1, 3, or 4."""
        if v not in (1, 3, 4):
            raise ValueError(f"Channels must be 1 (grayscale), 3 (RGB), or 4 (RGBA), got {v}")
        return v

    def add_warning(self, warning: str) -> None:
        """Add a warning to the context."""
        self.warnings.append(warning)

    def copy_with_updates(self, **kwargs) -> "ImageContext":
        """Create a copy of the context with updated values."""
        data = self.model_dump()
        data.update(kwargs)
        return ImageContext(**data)

    @property
    def total_pixels(self) -> int:
        """Total number of pixels in the image."""
        return self.width * self.height

    @property
    def memory_estimate(self) -> int:
        """Estimate memory usage in bytes."""
        bytes_per_pixel = 1 if self.dtype == "uint8" else 4
        return self.total_pixels * self.channels * bytes_per_pixel
