"""Custom exceptions for pixel-perfect."""


class PixelPerfectError(Exception):
    """Base exception for all pixel-perfect errors."""


class ValidationError(PixelPerfectError):
    """Raised when operation validation fails."""


class ProcessingError(PixelPerfectError):
    """Raised when image processing fails."""


class PipelineError(PixelPerfectError):
    """Raised when pipeline execution fails."""


class DimensionError(ValidationError):
    """Raised when image dimensions are incompatible with operation."""


class ChannelError(ValidationError):
    """Raised when channel requirements are not met."""


class MemoryError(PixelPerfectError):
    """Raised when estimated memory usage exceeds limits."""
