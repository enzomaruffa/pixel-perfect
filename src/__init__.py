"""Pixel-perfect image processing framework."""

__version__ = "0.1.0"
__author__ = "Claude Code"
__description__ = (
    "A composable pipeline framework for sophisticated pixel-level image transformations"
)

# Make core components easily accessible
from .core.base import BaseOperation
from .core.context import ImageContext
from .core.pipeline import Pipeline

# Import operations explicitly to avoid star import warnings
from .operations.aspect import AspectCrop, AspectPad, AspectStretch
from .operations.block import BlockFilter, BlockRotate, BlockScramble, BlockShift
from .operations.channel import AlphaGenerator, ChannelIsolate, ChannelSwap
from .operations.column import ColumnMirror, ColumnShift, ColumnStretch, ColumnWeave
from .operations.geometric import GridWarp, PerspectiveStretch, RadialStretch
from .operations.pattern import Dither, Mosaic
from .operations.pixel import PixelFilter, PixelMath, PixelSort
from .operations.row import RowRemove, RowShift, RowShuffle, RowStretch
from .presets import get_all_presets, get_preset

__all__ = [
    "Pipeline",
    "ImageContext",
    "BaseOperation",
    "get_preset",
    "get_all_presets",
    # Re-export all operations
    "PixelFilter",
    "PixelMath",
    "PixelSort",
    "RowShift",
    "RowStretch",
    "RowRemove",
    "RowShuffle",
    "ColumnShift",
    "ColumnStretch",
    "ColumnMirror",
    "ColumnWeave",
    "BlockFilter",
    "BlockShift",
    "BlockRotate",
    "BlockScramble",
    "GridWarp",
    "PerspectiveStretch",
    "RadialStretch",
    "AspectStretch",
    "AspectCrop",
    "AspectPad",
    "ChannelSwap",
    "ChannelIsolate",
    "AlphaGenerator",
    "Mosaic",
    "Dither",
]
