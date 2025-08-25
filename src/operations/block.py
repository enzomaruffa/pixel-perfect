"""Block-based image processing operations."""

import random
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import Field, field_validator, model_validator

from core.base import BaseOperation
from core.context import ImageContext
from exceptions import ProcessingError, ValidationError
from utils.validation import validate_color_tuple


def _calculate_grid_dimensions(
    image_size: tuple[int, int], block_size: tuple[int, int]
) -> tuple[int, int]:
    """Calculate grid dimensions for block operations.

    Args:
        image_size: (width, height) of image
        block_size: (width, height) of blocks

    Returns:
        (grid_cols, grid_rows) number of blocks in each dimension
    """
    width, height = image_size
    block_width, block_height = block_size

    grid_cols = width // block_width
    grid_rows = height // block_height

    return grid_cols, grid_rows


def _get_block_bounds(
    block_index: int, grid_dims: tuple[int, int], block_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Get pixel coordinates for a block.

    Args:
        block_index: Block index in row-major order
        grid_dims: (grid_cols, grid_rows)
        block_size: (block_width, block_height)

    Returns:
        (left, top, right, bottom) pixel coordinates
    """
    grid_cols, grid_rows = grid_dims
    block_width, block_height = block_size

    row = block_index // grid_cols
    col = block_index % grid_cols

    left = col * block_width
    top = row * block_height
    right = left + block_width
    bottom = top + block_height

    return left, top, right, bottom


def _extract_block(image: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    """Extract block from image array.

    Args:
        image: Image array (H, W, C)
        bounds: (left, top, right, bottom)

    Returns:
        Block array
    """
    left, top, right, bottom = bounds
    return image[top:bottom, left:right].copy()


def _place_block(image: np.ndarray, block: np.ndarray, bounds: tuple[int, int, int, int]) -> None:
    """Place block into image array in-place.

    Args:
        image: Target image array (H, W, C)
        block: Block array to place
        bounds: (left, top, right, bottom)
    """
    left, top, right, bottom = bounds
    image[top:bottom, left:right] = block


def _handle_padding(
    image: Image.Image,
    block_size: tuple[int, int],
    padding_mode: str,
    fill_color: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> Image.Image:
    """Handle non-divisible dimensions with padding.

    Args:
        image: Input image
        block_size: (block_width, block_height)
        padding_mode: "crop", "extend", or "fill"
        fill_color: Color for fill mode

    Returns:
        Padded/cropped image
    """
    width, height = image.size
    block_width, block_height = block_size

    new_width = (width // block_width) * block_width
    new_height = (height // block_height) * block_height

    if padding_mode == "crop":
        # Crop to largest divisible size
        return image.crop((0, 0, new_width, new_height))

    elif padding_mode == "extend":
        # Extend to next block boundary by replicating edge pixels
        if new_width == width and new_height == height:
            return image  # Already divisible

        # Calculate target size
        target_width = ((width + block_width - 1) // block_width) * block_width
        target_height = ((height + block_height - 1) // block_height) * block_height

        # Create new image and paste original
        new_image = Image.new(image.mode, (target_width, target_height))
        new_image.paste(image, (0, 0))

        # Fill right edge
        if target_width > width:
            edge_col = image.crop((width - 1, 0, width, height))
            for x in range(width, target_width):
                new_image.paste(edge_col, (x, 0))

        # Fill bottom edge
        if target_height > height:
            edge_row = new_image.crop((0, height - 1, target_width, height))
            for y in range(height, target_height):
                new_image.paste(edge_row, (0, y))

        return new_image

    elif padding_mode == "fill":
        # Extend to next block boundary with fill color
        if new_width == width and new_height == height:
            return image  # Already divisible

        target_width = ((width + block_width - 1) // block_width) * block_width
        target_height = ((height + block_height - 1) // block_height) * block_height

        # Create new image with fill color
        new_image = Image.new("RGBA", (target_width, target_height), fill_color)
        new_image.paste(image, (0, 0))

        # Convert back to original mode if needed
        if image.mode != "RGBA":
            new_image = new_image.convert(image.mode)

        return new_image

    else:
        raise ValidationError(f"Unknown padding mode: {padding_mode}")


def _select_blocks(selection: str, grid_dims: tuple[int, int], **kwargs) -> np.ndarray:
    """Select block indices based on selection criteria.

    Args:
        selection: Selection method
        grid_dims: (grid_cols, grid_rows)
        **kwargs: Additional parameters

    Returns:
        Array of selected block indices
    """
    grid_cols, grid_rows = grid_dims
    total_blocks = grid_cols * grid_rows

    if selection == "odd":
        return np.arange(1, total_blocks, 2)
    elif selection == "even":
        return np.arange(0, total_blocks, 2)
    elif selection == "checkerboard":
        indices = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                if (row + col) % 2 == 0:
                    indices.append(row * grid_cols + col)
        return np.array(indices)
    elif selection == "diagonal":
        indices = []
        for i in range(min(grid_rows, grid_cols)):
            indices.append(i * grid_cols + i)
        return np.array(indices)
    elif selection == "corners":
        indices = [0]  # Top-left
        if grid_cols > 1:
            indices.append(grid_cols - 1)  # Top-right
        if grid_rows > 1:
            indices.append((grid_rows - 1) * grid_cols)  # Bottom-left
        if grid_cols > 1 and grid_rows > 1:
            indices.append(grid_rows * grid_cols - 1)  # Bottom-right
        return np.array(indices)
    elif selection == "custom":
        indices = kwargs.get("indices", [])
        if not indices:
            raise ValidationError("Custom selection requires 'indices' parameter")
        indices_array = np.array(indices)
        if np.any(indices_array < 0) or np.any(indices_array >= total_blocks):
            raise ValidationError(f"Block indices must be in range [0, {total_blocks})")
        return indices_array
    else:
        raise ValidationError(f"Unknown selection method: {selection}")


class BlockFilter(BaseOperation):
    """Filter blocks in virtual grid."""

    block_width: int = Field(8, gt=0, description="Width of each block in pixels")
    block_height: int = Field(8, gt=0, description="Height of each block in pixels")
    keep_blocks: list[int] | None = Field(None, description="List of block indices to preserve")
    condition: Literal["checkerboard", "diagonal", "corners", "custom"] = "checkerboard"
    fill_color: tuple[int, int, int, int] = Field(
        (0, 0, 0, 0), description="RGBA for hidden blocks"
    )
    padding_mode: Literal["crop", "extend", "fill"] = "crop"

    @field_validator("fill_color")
    @classmethod
    def validate_fill_color(cls, v):
        return validate_color_tuple(v)

    @model_validator(mode="after")
    def validate_block_parameters(self) -> "BlockFilter":
        """Validate block-related parameters."""
        if self.condition == "custom" and self.keep_blocks is None:
            raise ValueError("keep_blocks required when condition='custom'")
        return self

    def validate_operation(self, context: ImageContext) -> ImageContext:
        """Validate operation against image context."""
        if self.block_width > context.width or self.block_height > context.height:
            raise ValidationError(
                f"Block size ({self.block_width}x{self.block_height}) cannot exceed image size ({context.width}x{context.height})"
            )
        return context.copy_with_updates()

    def get_cache_key(self, image_hash: str) -> str:
        """Generate cache key for this operation."""
        config_str = f"{self.block_width}_{self.block_height}_{self.keep_blocks}"
        config_str += f"_{self.condition}_{self.fill_color}_{self.padding_mode}"
        return f"blockfilter_{image_hash}_{hash(config_str)}"

    def estimate_memory(self, context: ImageContext) -> int:
        """Estimate memory usage in bytes."""
        return context.memory_estimate * 2  # Input + output

    def apply(self, image: Image.Image, context: ImageContext) -> tuple[Image.Image, ImageContext]:
        """Apply block filter to image."""
        try:
            # Handle padding for non-divisible dimensions
            padded_image = _handle_padding(
                image, (self.block_width, self.block_height), self.padding_mode, self.fill_color
            )

            # Convert to RGBA for consistent processing
            rgba_image = padded_image.convert("RGBA")
            pixels = np.array(rgba_image)
            height, width = pixels.shape[:2]

            # Calculate grid dimensions
            grid_dims = _calculate_grid_dimensions(
                (width, height), (self.block_width, self.block_height)
            )
            grid_cols, grid_rows = grid_dims
            total_blocks = grid_cols * grid_rows

            # Determine which blocks to keep
            if self.condition == "custom":
                if self.keep_blocks is None:
                    raise ValidationError("keep_blocks is required when condition is 'custom'")
                keep_indices = set(self.keep_blocks)
            else:
                keep_indices = set(_select_blocks(self.condition, grid_dims))

            result_pixels = pixels.copy()

            # Process each block
            for block_idx in range(total_blocks):
                if block_idx not in keep_indices:
                    # Fill this block with fill_color
                    bounds = _get_block_bounds(
                        block_idx, grid_dims, (self.block_width, self.block_height)
                    )
                    left, top, right, bottom = bounds
                    result_pixels[top:bottom, left:right] = self.fill_color

            # Create result image
            result_image = Image.fromarray(result_pixels)
            if result_image.mode != "RGBA":
                result_image = result_image.convert("RGBA")

            # Convert back to original mode if fill_color is opaque
            if image.mode != "RGBA" and self.fill_color[3] == 255:
                result_image = result_image.convert(image.mode)

            # Update context with potentially new dimensions
            new_width, new_height = result_image.size
            updated_context = context.copy_with_updates(width=new_width, height=new_height)

            return result_image, updated_context

        except Exception as e:
            raise ProcessingError(f"BlockFilter failed: {e}") from e


class BlockShift(BaseOperation):
    """Rearrange blocks within virtual grid."""

    block_width: int = Field(8, gt=0, description="Width of blocks")
    block_height: int = Field(8, gt=0, description="Height of blocks")
    shift_map: dict[int, int] = Field(..., description="Mapping source index → destination index")
    swap_mode: Literal["move", "swap"] = "move"
    padding_mode: Literal["crop", "extend", "fill"] = "crop"

    def validate_operation(self, context: ImageContext) -> ImageContext:
        """Validate operation against image context."""
        if self.block_width > context.width or self.block_height > context.height:
            raise ValidationError(
                f"Block size ({self.block_width}x{self.block_height}) cannot exceed image size ({context.width}x{context.height})"
            )

        # Validate shift_map indices (we'll check against actual grid size in apply)
        if not self.shift_map:
            raise ValidationError("shift_map cannot be empty")

        return context.copy_with_updates()

    def get_cache_key(self, image_hash: str) -> str:
        """Generate cache key for this operation."""
        config_str = f"{self.block_width}_{self.block_height}_{sorted(self.shift_map.items())}"
        config_str += f"_{self.swap_mode}_{self.padding_mode}"
        return f"blockshift_{image_hash}_{hash(config_str)}"

    def estimate_memory(self, context: ImageContext) -> int:
        """Estimate memory usage in bytes."""
        return context.memory_estimate * 3  # Input + output + temp blocks

    def apply(self, image: Image.Image, context: ImageContext) -> tuple[Image.Image, ImageContext]:
        """Apply block shifting to image."""
        try:
            # Handle padding for non-divisible dimensions
            padded_image = _handle_padding(
                image, (self.block_width, self.block_height), self.padding_mode
            )

            # Convert to RGBA for consistent processing
            rgba_image = padded_image.convert("RGBA")
            pixels = np.array(rgba_image)
            height, width = pixels.shape[:2]

            # Calculate grid dimensions
            grid_dims = _calculate_grid_dimensions(
                (width, height), (self.block_width, self.block_height)
            )
            grid_cols, grid_rows = grid_dims
            total_blocks = grid_cols * grid_rows

            # Validate shift_map indices
            for src, dst in self.shift_map.items():
                if src < 0 or src >= total_blocks:
                    raise ValidationError(
                        f"Source block index {src} out of range [0, {total_blocks})"
                    )
                if dst < 0 or dst >= total_blocks:
                    raise ValidationError(
                        f"Destination block index {dst} out of range [0, {total_blocks})"
                    )

            result_pixels = pixels.copy()

            if self.swap_mode == "move":
                # Extract all blocks that will be moved
                moved_blocks = {}
                for src_idx, dst_idx in self.shift_map.items():
                    bounds = _get_block_bounds(
                        src_idx, grid_dims, (self.block_width, self.block_height)
                    )
                    moved_blocks[dst_idx] = _extract_block(pixels, bounds)

                # Place moved blocks at their destinations
                for dst_idx, block in moved_blocks.items():
                    bounds = _get_block_bounds(
                        dst_idx, grid_dims, (self.block_width, self.block_height)
                    )
                    _place_block(result_pixels, block, bounds)

            else:  # swap mode
                # Perform swaps
                swapped = set()
                for src_idx, dst_idx in self.shift_map.items():
                    if src_idx in swapped or dst_idx in swapped:
                        continue  # Already handled in a previous swap

                    # Extract both blocks
                    src_bounds = _get_block_bounds(
                        src_idx, grid_dims, (self.block_width, self.block_height)
                    )
                    dst_bounds = _get_block_bounds(
                        dst_idx, grid_dims, (self.block_width, self.block_height)
                    )

                    src_block = _extract_block(pixels, src_bounds)
                    dst_block = _extract_block(pixels, dst_bounds)

                    # Swap them
                    _place_block(result_pixels, dst_block, src_bounds)
                    _place_block(result_pixels, src_block, dst_bounds)

                    swapped.add(src_idx)
                    swapped.add(dst_idx)

            # Create result image
            result_image = Image.fromarray(result_pixels)
            if result_image.mode != "RGBA":
                result_image = result_image.convert("RGBA")

            # Convert back to original mode if appropriate
            if image.mode != "RGBA":
                result_image = result_image.convert(image.mode)

            # Update context with potentially new dimensions
            new_width, new_height = result_image.size
            updated_context = context.copy_with_updates(width=new_width, height=new_height)

            return result_image, updated_context

        except Exception as e:
            raise ProcessingError(f"BlockShift failed: {e}") from e


class BlockRotate(BaseOperation):
    """Rotate individual blocks."""

    block_width: int = Field(8, gt=0, description="Width of blocks")
    block_height: int = Field(8, gt=0, description="Height of blocks")
    rotation: Literal[90, 180, 270] = 90
    selection: Literal["all", "odd", "even", "checkerboard", "diagonal", "corners", "custom"] = (
        "all"
    )
    indices: list[int] | None = Field(None, description="Custom block indices to rotate")
    padding_mode: Literal["crop", "extend", "fill"] = "crop"

    @model_validator(mode="after")
    def validate_rotation_parameters(self) -> "BlockRotate":
        """Validate rotation-related parameters."""
        if self.selection == "custom" and self.indices is None:
            raise ValueError("indices required when selection='custom'")
        return self

    def validate_operation(self, context: ImageContext) -> ImageContext:
        """Validate operation against image context."""
        if self.block_width > context.width or self.block_height > context.height:
            raise ValidationError(
                f"Block size ({self.block_width}x{self.block_height}) cannot exceed image size ({context.width}x{context.height})"
            )
        return context.copy_with_updates()

    def get_cache_key(self, image_hash: str) -> str:
        """Generate cache key for this operation."""
        config_str = f"{self.block_width}_{self.block_height}_{self.rotation}"
        config_str += f"_{self.selection}_{self.indices}_{self.padding_mode}"
        return f"blockrotate_{image_hash}_{hash(config_str)}"

    def estimate_memory(self, context: ImageContext) -> int:
        """Estimate memory usage in bytes."""
        return context.memory_estimate * 2  # Input + output

    def apply(self, image: Image.Image, context: ImageContext) -> tuple[Image.Image, ImageContext]:
        """Apply block rotation to image."""
        try:
            # Handle padding for non-divisible dimensions
            padded_image = _handle_padding(
                image, (self.block_width, self.block_height), self.padding_mode
            )

            # Convert to RGBA for consistent processing
            rgba_image = padded_image.convert("RGBA")
            pixels = np.array(rgba_image)
            height, width = pixels.shape[:2]

            # Calculate grid dimensions
            grid_dims = _calculate_grid_dimensions(
                (width, height), (self.block_width, self.block_height)
            )
            grid_cols, grid_rows = grid_dims
            total_blocks = grid_cols * grid_rows

            # Determine which blocks to rotate
            if self.selection == "all":
                rotate_indices = set(range(total_blocks))
            elif self.selection == "custom":
                if self.indices is None:
                    raise ValidationError("indices is required when selection is 'custom'")
                rotate_indices = set(self.indices)
            else:
                rotate_indices = set(_select_blocks(self.selection, grid_dims))

            result_pixels = pixels.copy()

            # Rotate selected blocks
            for block_idx in rotate_indices:
                if block_idx >= total_blocks:
                    continue  # Skip invalid indices

                bounds = _get_block_bounds(
                    block_idx, grid_dims, (self.block_width, self.block_height)
                )
                block = _extract_block(pixels, bounds)

                # Rotate the block
                if self.rotation == 90:
                    rotated_block = np.rot90(block, k=1)  # 90° counter-clockwise
                elif self.rotation == 180:
                    rotated_block = np.rot90(block, k=2)  # 180°
                elif self.rotation == 270:
                    rotated_block = np.rot90(block, k=3)  # 270° counter-clockwise (90° clockwise)
                else:
                    rotated_block = block  # No rotation

                _place_block(result_pixels, rotated_block, bounds)

            # Create result image
            result_image = Image.fromarray(result_pixels)
            if result_image.mode != "RGBA":
                result_image = result_image.convert("RGBA")

            # Convert back to original mode if appropriate
            if image.mode != "RGBA":
                result_image = result_image.convert(image.mode)

            # Update context with potentially new dimensions
            new_width, new_height = result_image.size
            updated_context = context.copy_with_updates(width=new_width, height=new_height)

            return result_image, updated_context

        except Exception as e:
            raise ProcessingError(f"BlockRotate failed: {e}") from e


class BlockScramble(BaseOperation):
    """Randomly shuffle blocks."""

    block_width: int = Field(8, gt=0, description="Width of blocks")
    block_height: int = Field(8, gt=0, description="Height of blocks")
    seed: int | None = Field(None, description="Random seed for reproducibility")
    exclude: list[int] | None = Field(None, description="Block indices to keep in place")
    padding_mode: Literal["crop", "extend", "fill"] = "crop"

    def validate_operation(self, context: ImageContext) -> ImageContext:
        """Validate operation against image context."""
        if self.block_width > context.width or self.block_height > context.height:
            raise ValidationError(
                f"Block size ({self.block_width}x{self.block_height}) cannot exceed image size ({context.width}x{context.height})"
            )
        return context.copy_with_updates()

    def get_cache_key(self, image_hash: str) -> str:
        """Generate cache key for this operation."""
        config_str = f"{self.block_width}_{self.block_height}_{self.seed}"
        config_str += f"_{self.exclude}_{self.padding_mode}"
        return f"blockscramble_{image_hash}_{hash(config_str)}"

    def estimate_memory(self, context: ImageContext) -> int:
        """Estimate memory usage in bytes."""
        return context.memory_estimate * 3  # Input + output + temp blocks

    def apply(self, image: Image.Image, context: ImageContext) -> tuple[Image.Image, ImageContext]:
        """Apply block scrambling to image."""
        try:
            # Handle padding for non-divisible dimensions
            padded_image = _handle_padding(
                image, (self.block_width, self.block_height), self.padding_mode
            )

            # Convert to RGBA for consistent processing
            rgba_image = padded_image.convert("RGBA")
            pixels = np.array(rgba_image)
            height, width = pixels.shape[:2]

            # Calculate grid dimensions
            grid_dims = _calculate_grid_dimensions(
                (width, height), (self.block_width, self.block_height)
            )
            grid_cols, grid_rows = grid_dims
            total_blocks = grid_cols * grid_rows

            # Set random seed if provided
            if self.seed is not None:
                random.seed(self.seed)
                np.random.seed(self.seed)

            # Create list of block indices to shuffle
            block_indices = list(range(total_blocks))
            exclude_set = set(self.exclude) if self.exclude else set()

            # Remove excluded blocks from shuffling
            shuffleable_indices = [idx for idx in block_indices if idx not in exclude_set]

            # Shuffle the indices
            shuffled_indices = shuffleable_indices.copy()
            random.shuffle(shuffled_indices)

            # Extract all blocks
            blocks = {}
            for block_idx in range(total_blocks):
                bounds = _get_block_bounds(
                    block_idx, grid_dims, (self.block_width, self.block_height)
                )
                blocks[block_idx] = _extract_block(pixels, bounds)

            result_pixels = pixels.copy()

            # Place shuffled blocks
            for i, src_idx in enumerate(shuffleable_indices):
                dst_idx = shuffled_indices[i]
                dst_bounds = _get_block_bounds(
                    dst_idx, grid_dims, (self.block_width, self.block_height)
                )
                _place_block(result_pixels, blocks[src_idx], dst_bounds)

            # Create result image
            result_image = Image.fromarray(result_pixels)
            if result_image.mode != "RGBA":
                result_image = result_image.convert("RGBA")

            # Convert back to original mode if appropriate
            if image.mode != "RGBA":
                result_image = result_image.convert(image.mode)

            # Update context with potentially new dimensions
            new_width, new_height = result_image.size
            updated_context = context.copy_with_updates(width=new_width, height=new_height)

            return result_image, updated_context

        except Exception as e:
            raise ProcessingError(f"BlockScramble failed: {e}") from e
