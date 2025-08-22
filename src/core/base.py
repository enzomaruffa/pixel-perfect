"""Base operation class for all image transformations."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from PIL import Image
from pydantic import BaseModel, ConfigDict

from core.context import ImageContext

if TYPE_CHECKING:
    pass


class BaseOperation(BaseModel, ABC):
    """Abstract base class for all image operations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def validate_operation(self, context: ImageContext) -> ImageContext:
        """Validate operation against current image state.

        Args:
            context: Current image context

        Returns:
            Updated context with any changes or warnings

        Raises:
            ValidationError: If operation is invalid for current context
        """
        ...

    @abstractmethod
    def get_cache_key(self, image_hash: str) -> str:
        """Generate unique cache key for this operation.

        Args:
            image_hash: Hash of the input image

        Returns:
            Unique string identifying this operation and its parameters
        """
        ...

    @abstractmethod
    def estimate_memory(self, context: ImageContext) -> int:
        """Estimate memory usage in bytes.

        Args:
            context: Current image context

        Returns:
            Estimated memory usage in bytes
        """
        ...

    @abstractmethod
    def apply(self, image: Image.Image, context: ImageContext) -> tuple[Image.Image, ImageContext]:
        """Transform image and update context.

        Args:
            image: Input PIL Image
            context: Current image context

        Returns:
            Tuple of (transformed image, updated context)
        """
        ...

    def generate_param_hash(self) -> str:
        """Generate hash from operation parameters."""
        import hashlib

        # Use model dump to get all parameters
        params = self.model_dump(exclude_none=True)
        # Sort keys for consistent hashing
        param_str = str(sorted(params.items()))
        return hashlib.md5(param_str.encode()).hexdigest()

    @property
    def operation_name(self) -> str:
        """Get the name of this operation."""
        return self.__class__.__name__
