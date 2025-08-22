# Task 001: Set Up Basic Project Structure

## Objective
Create the foundational project structure with core modules, base classes, and essential dependencies.

## Requirements
1. Set up proper Python package structure
2. Install core dependencies (Pillow, Pydantic, NumPy)
3. Create base module files
4. Implement core base classes (ImageContext, BaseOperation, Pipeline)
5. Set up basic error handling and validation infrastructure

Image Processing Pipeline - Starter Code
File Structure
image_pipeline/
├── main.py                     # Entry point
├── pipeline/
│   ├── __init__.py
│   ├── core.py                 # Pipeline and ImageContext
│   ├── base_operation.py       # BaseOperation class
│   └── operations/
│       ├── __init__.py
│       ├── pixel_filter/
│       │   ├── __init__.py
│       │   └── pixel_filter.py     # PixelFilter operation
│       │   └── test_pixel_filter.py     # PixelFilter tests
│       └── row_shift/
│           ├── __init__.py
│           └── row_shift.py        # RowShift operation
│           └── test_row_shift.py        # RowShift tests

txtPillow>=10.0.0
pydantic>=2.0.0
numpy>=1.24.0
pytest>=7.0.0
rich>=13.0.0

main.py
python#!/usr/bin/env python3
"""
Simple example showing pipeline usage
"""
from pipeline.core import Pipeline
from pipeline.operations.pixel_filter import PixelFilter
from pipeline.operations.row_shift import RowShift

def main():
    # Example 1: Simple pixel filter
    print("Running pixel filter example...")
    pipeline = Pipeline("input.jpg", debug=True)
    pipeline.add(PixelFilter(
        condition="odd",
        fill_color=(0, 0, 0, 0)  # Transparent fill
    ))
    result = pipeline.execute("output_odd_pixels.png")
    print(f"Saved to {result.output_path}")

    # Example 2: Combination of operations
    print("\nRunning combined operations...")
    pipeline = Pipeline("input.jpg", debug=True)
    pipeline.add(RowShift(
        selection="even",
        shift_amount=5,
        wrap=True
    ))
    pipeline.add(PixelFilter(
        condition="prime",
        fill_color=(255, 0, 0, 128)  # Semi-transparent red
    ))
    result = pipeline.execute("output_combined.png")

    # Example 3: Validation demo
    print("\nDemonstrating validation...")
    pipeline = Pipeline("input.jpg")
    pipeline.add(RowShift(
        selection="custom",
        indices=[5, 10, 15],  # Will validate against actual image height
        shift_amount=3
    ))

    try:
        # Dry run first
        pipeline.execute("test.png", dry_run=True)
        print("Pipeline validated successfully!")
        # Actually execute
        pipeline.execute("output_validated.png")
    except Exception as e:
        print(f"Validation failed: {e}")

if __name__ == "__main__":
    main()
pipeline/init.py
pythonfrom .core import Pipeline, ImageContext
from .base_operation import BaseOperation

__all__ = ['Pipeline', 'ImageContext', 'BaseOperation']
pipeline/core.py
python"""
Core pipeline implementation
"""
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from PIL import Image
import hashlib

from .base_operation import BaseOperation

@dataclass
class ImageContext:
    """Context that flows through pipeline"""
    width: int
    height: int
    channels: int  # 1=L, 3=RGB, 4=RGBA
    mode: str  # PIL mode string
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def from_image(cls, img: Image.Image) -> 'ImageContext':
        mode_to_channels = {'L': 1, 'RGB': 3, 'RGBA': 4, 'P': 1}
        return cls(
            width=img.width,
            height=img.height,
            channels=mode_to_channels.get(img.mode, 4),
            mode=img.mode
        )

@dataclass
class PipelineResult:
    """Result of pipeline execution"""
    output_path: str
    context: ImageContext
    success: bool
    warnings: List[str]

class Pipeline:
    """Main pipeline for chaining image operations"""

    def __init__(self, input_path: str, debug: bool = False, cache_dir: Optional[str] = None):
        self.input_path = Path(input_path)
        self.debug = debug
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.operations: List[BaseOperation] = []
        self._image: Optional[Image.Image] = None
        self._context: Optional[ImageContext] = None

        # Debug setup
        if self.debug:
            self.debug_dir = Path("./debug")
            self.debug_dir.mkdir(exist_ok=True)

    def add(self, operation: BaseOperation) -> 'Pipeline':
        """Add an operation to the pipeline"""
        self.operations.append(operation)
        return self

    def _load_image(self):
        """Load image and create initial context"""
        if self._image is None:
            self._image = Image.open(self.input_path)
            # Convert to RGBA for consistent processing
            if self._image.mode != 'RGBA':
                self._image = self._image.convert('RGBA')
            self._context = ImageContext.from_image(self._image)

            # Validate minimum size
            if self._image.width == 0 or self._image.height == 0:
                raise ValueError("Cannot process 0x0 images")

    def _validate_pipeline(self) -> ImageContext:
        """Validate all operations against image context"""
        context = self._context.copy()

        for i, op in enumerate(self.operations):
            try:
                context = op.validate(context)
            except Exception as e:
                raise ValueError(f"Operation {i} ({op.__class__.__name__}): {e}")

        return context

    def _get_image_hash(self, img: Image.Image) -> str:
        """Generate hash for image state"""
        return hashlib.md5(img.tobytes()).hexdigest()[:8]

    def execute(self, output_path: str, dry_run: bool = False) -> PipelineResult:
        """Execute the pipeline"""
        # Load image if not loaded
        self._load_image()

        # Validate pipeline
        final_context = self._validate_pipeline()

        if dry_run:
            return PipelineResult(
                output_path=output_path,
                context=final_context,
                success=True,
                warnings=final_context.warnings
            )

        # Process operations
        current_image = self._image.copy()
        current_context = self._context.copy()

        for i, op in enumerate(self.operations):
            # Apply operation
            current_image, current_context = op.apply(current_image, current_context)

            # Debug save
            if self.debug:
                debug_path = self.debug_dir / f"step_{i:02d}_{op.__class__.__name__}.png"
                current_image.save(debug_path)
                print(f"  Debug: saved {debug_path}")

        # Save final result
        current_image.save(output_path)

        return PipelineResult(
            output_path=output_path,
            context=current_context,
            success=True,
            warnings=current_context.warnings
        )
pipeline/base_operation.py
python"""
Base operation class that all operations inherit from
"""
from abc import ABC, abstractmethod
from typing import Tuple
from PIL import Image
from pydantic import BaseModel

from .core import ImageContext

class BaseOperation(ABC):
    """Base class for all image operations"""

    def __init__(self, **kwargs):
        """Initialize with parameters"""
        # Subclasses should define their Pydantic model
        if hasattr(self, 'Model'):
            self.params = self.Model(**kwargs)
        else:
            self.params = None

    @abstractmethod
    def validate(self, context: ImageContext) -> ImageContext:
        """
        Validate operation against current image context.
        Returns updated context if operation changes dimensions.
        Raises exception if validation fails.
        """
        pass

    @abstractmethod
    def apply(self, image: Image.Image, context: ImageContext) -> Tuple[Image.Image, ImageContext]:
        """
        Apply operation to image.
        Returns modified image and updated context.
        """
        pass

    def estimate_memory(self, context: ImageContext) -> int:
        """Estimate memory usage in bytes"""
        # Basic estimation: width * height * channels * bytes_per_channel
        return context.width * context.height * context.channels
pipeline/operations/init.py
pythonfrom .pixel_filter import PixelFilter
from .row_shift import RowShift

__all__ = ['PixelFilter', 'RowShift']
pipeline/operations/pixel_filter.py
python"""
PixelFilter operation - filter pixels based on index conditions
"""
from typing import Tuple, Literal, Optional
from PIL import Image
import numpy as np
from pydantic import BaseModel, validator

from ..base_operation import BaseOperation
from ..core import ImageContext

class PixelFilterParams(BaseModel):
    """Parameters for PixelFilter operation"""
    condition: Literal["odd", "even", "prime", "custom"]
    fill_color: Tuple[int, int, int, int] = (0, 0, 0, 0)
    custom_expression: Optional[str] = None

    @validator('fill_color')
    def validate_color(cls, v):
        if len(v) != 4:
            raise ValueError("fill_color must be RGBA tuple")
        if not all(0 <= c <= 255 for c in v):
            raise ValueError("color values must be 0-255")
        return v

    @validator('custom_expression')
    def validate_expression(cls, v, values):
        if values.get('condition') == 'custom' and not v:
            raise ValueError("custom_expression required when condition='custom'")
        return v

class PixelFilter(BaseOperation):
    """Filter pixels based on their index"""

    Model = PixelFilterParams

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _is_prime(self, n: int) -> bool:
        """Check if number is prime"""
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    def _should_keep_pixel(self, index: int) -> bool:
        """Determine if pixel should be kept based on condition"""
        if self.params.condition == "odd":
            return index % 2 == 1
        elif self.params.condition == "even":
            return index % 2 == 0
        elif self.params.condition == "prime":
            return self._is_prime(index)
        elif self.params.condition == "custom":
            # Safe evaluation of simple expressions
            try:
                return eval(self.params.custom_expression, {"__builtins__": {}}, {"i": index})
            except:
                return False
        return True

    def validate(self, context: ImageContext) -> ImageContext:
        """Validate operation - always valid for pixel filter"""
        return context

    def apply(self, image: Image.Image, context: ImageContext) -> Tuple[Image.Image, ImageContext]:
        """Apply pixel filtering"""
        # Convert to numpy for easier manipulation
        arr = np.array(image)
        height, width = arr.shape[:2]

        # Create mask for pixels to filter
        for y in range(height):
            for x in range(width):
                index = y * width + x
                if not self._should_keep_pixel(index):
                    arr[y, x] = self.params.fill_color

        # Convert back to PIL
        result = Image.fromarray(arr, mode=image.mode)
        return result, context


class TestPixelFilter:
    """Test cases for PixelFilter"""

    @staticmethod
    def create_test_image(width: int = 4, height: int = 4) -> Image.Image:
        """Create a numbered test image where each pixel value equals its index"""
        arr = np.zeros((height, width, 4), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                index = y * width + x
                # Use index as red channel value for testing
                arr[y, x] = [index * 16, index * 16, index * 16, 255]
        return Image.fromarray(arr, mode='RGBA')

    def test_odd_filter(self):
        """Test filtering odd-indexed pixels"""
        img = self.create_test_image(4, 4)
        context = ImageContext.from_image(img)

        op = PixelFilter(condition="odd", fill_color=(0, 0, 0, 0))
        result, _ = op.apply(img, context)

        # Check that odd indices are kept and even are transparent
        arr = np.array(result)
        for y in range(4):
            for x in range(4):
                index = y * 4 + x
                if index % 2 == 1:
                    # Odd indices should be preserved
                    assert arr[y, x, 3] == 255, f"Pixel {index} should be opaque"
                else:
                    # Even indices should be transparent
                    assert arr[y, x, 3] == 0, f"Pixel {index} should be transparent"

        print("✓ Odd filter test passed")

    def test_prime_filter(self):
        """Test filtering prime-indexed pixels"""
        img = self.create_test_image(4, 4)  # Indices 0-15
        context = ImageContext.from_image(img)

        op = PixelFilter(condition="prime", fill_color=(255, 0, 0, 255))
        result, _ = op.apply(img, context)

        # Prime indices in 0-15: 2, 3, 5, 7, 11, 13
        primes = {2, 3, 5, 7, 11, 13}

        arr = np.array(result)
        for y in range(4):
            for x in range(4):
                index = y * 4 + x
                if index in primes:
                    # Should be preserved (not red)
                    assert arr[y, x, 0] != 255 or arr[y, x, 1] != 0, f"Prime pixel {index} should be preserved"
                else:
                    # Should be red fill
                    assert tuple(arr[y, x]) == (255, 0, 0, 255), f"Non-prime pixel {index} should be red"

        print("✓ Prime filter test passed")

    def run_all_tests(self):
        """Run all tests"""
        self.test_odd_filter()
        self.test_prime_filter()
        print("All PixelFilter tests passed!")

# Quick test runner
if __name__ == "__main__":
    tester = TestPixelFilter()
    tester.run_all_tests()
pipeline/operations/row_shift.py
python"""
RowShift operation - shift entire rows horizontally
"""
from typing import Tuple, List, Literal, Optional
from PIL import Image
import numpy as np
from pydantic import BaseModel, validator

from ..base_operation import BaseOperation
from ..core import ImageContext

class RowShiftParams(BaseModel):
    """Parameters for RowShift operation"""
    selection: Literal["odd", "even", "all", "custom"]
    shift_amount: int
    wrap: bool = True
    fill_color: Tuple[int, int, int, int] = (0, 0, 0, 0)
    indices: Optional[List[int]] = None

    @validator('indices')
    def validate_indices(cls, v, values):
        if values.get('selection') == 'custom' and not v:
            raise ValueError("indices required when selection='custom'")
        return v

    @validator('fill_color')
    def validate_color(cls, v):
        if len(v) != 4:
            raise ValueError("fill_color must be RGBA tuple")
        if not all(0 <= c <= 255 for c in v):
            raise ValueError("color values must be 0-255")
        return v

class RowShift(BaseOperation):
    """Shift rows horizontally"""

    Model = RowShiftParams

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_selected_rows(self, height: int) -> List[int]:
        """Get list of row indices to shift"""
        if self.params.selection == "odd":
            return [i for i in range(height) if i % 2 == 1]
        elif self.params.selection == "even":
            return [i for i in range(height) if i % 2 == 0]
        elif self.params.selection == "all":
            return list(range(height))
        elif self.params.selection == "custom":
            return self.params.indices
        return []

    def validate(self, context: ImageContext) -> ImageContext:
        """Validate that row indices are within bounds"""
        if self.params.selection == "custom":
            for idx in self.params.indices:
                if idx < 0 or idx >= context.height:
                    raise ValueError(f"Row index {idx} out of bounds (height={context.height})")
        return context

    def apply(self, image: Image.Image, context: ImageContext) -> Tuple[Image.Image, ImageContext]:
        """Apply row shifting"""
        arr = np.array(image)
        height, width = arr.shape[:2]
        result = arr.copy()

        rows_to_shift = self._get_selected_rows(height)
        shift = self.params.shift_amount

        for row_idx in rows_to_shift:
            if self.params.wrap:
                # Circular shift
                result[row_idx] = np.roll(arr[row_idx], shift, axis=0)
            else:
                # Shift with fill
                if shift > 0:
                    # Shift right
                    result[row_idx, shift:] = arr[row_idx, :-shift]
                    result[row_idx, :shift] = self.params.fill_color
                else:
                    # Shift left
                    result[row_idx, :shift] = arr[row_idx, -shift:]
                    result[row_idx, shift:] = self.params.fill_color

        return Image.fromarray(result, mode=image.mode), context


class TestRowShift:
    """Test cases for RowShift"""

    @staticmethod
    def create_test_image(width: int = 4, height: int = 4) -> Image.Image:
        """Create test image where each row has a distinct pattern"""
        arr = np.zeros((height, width, 4), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                # Each row has a different base color
                arr[y, x] = [y * 60, x * 60, 128, 255]
        return Image.fromarray(arr, mode='RGBA')

    def test_even_rows_shift(self):
        """Test shifting even rows with wrap"""
        img = self.create_test_image(4, 4)
        context = ImageContext.from_image(img)

        # Get original array for comparison
        original = np.array(img)

        op = RowShift(selection="even", shift_amount=1, wrap=True)
        result, _ = op.apply(img, context)

        result_arr = np.array(result)

        # Check even rows are shifted, odd rows unchanged
        for y in range(4):
            if y % 2 == 0:
                # Even row should be shifted right by 1
                for x in range(4):
                    expected_x = (x - 1) % 4
                    assert np.array_equal(result_arr[y, x], original[y, expected_x]), \
                        f"Even row {y} not shifted correctly at x={x}"
            else:
                # Odd row should be unchanged
                assert np.array_equal(result_arr[y], original[y]), \
                    f"Odd row {y} should be unchanged"

        print("✓ Even rows shift test passed")

    def test_custom_indices_validation(self):
        """Test validation of custom row indices"""
        img = self.create_test_image(4, 4)
        context = ImageContext.from_image(img)

        # Valid indices
        op = RowShift(selection="custom", indices=[0, 2], shift_amount=1)
        context_valid = op.validate(context)
        assert context_valid == context

        # Invalid indices
        op_invalid = RowShift(selection="custom", indices=[0, 10], shift_amount=1)
        try:
            op_invalid.validate(context)
            assert False, "Should have raised validation error"
        except ValueError as e:
            assert "out of bounds" in str(e)

        print("✓ Custom indices validation test passed")

    def test_no_wrap_fill(self):
        """Test shifting without wrap uses fill color"""
        img = self.create_test_image(4, 2)
        context = ImageContext.from_image(img)

        op = RowShift(
            selection="all",
            shift_amount=2,
            wrap=False,
            fill_color=(255, 0, 0, 255)
        )
        result, _ = op.apply(img, context)

        result_arr = np.array(result)

        # First 2 pixels of each row should be red fill
        for y in range(2):
            for x in range(2):
                assert tuple(result_arr[y, x]) == (255, 0, 0, 255), \
                    f"Pixel at ({x}, {y}) should be fill color"

        print("✓ No-wrap fill test passed")

    def run_all_tests(self):
        """Run all tests"""
        self.test_even_rows_shift()
        self.test_custom_indices_validation()
        self.test_no_wrap_fill()
        print("All RowShift tests passed!")

# Quick test runner
if __name__ == "__main__":
    tester = TestRowShift()
    tester.run_all_tests()
How to Use This Starter Code

Install dependencies:
bashpip install -r requirements.txt

Run the example:
bashpython main.py

Run tests for an operation:
bashpython -m pipeline.operations.pixel_filter
python -m pipeline.operations.row_shift

Run all tests with pytest:
bashpytest pipeline/operations/ -v

Add a new operation:

Create a new file in pipeline/operations/
Copy the structure from pixel_filter.py or row_shift.py
Define your parameters with Pydantic
Implement validate() and apply() methods
Add test cases in the same file
Import in pipeline/operations/__init__.py



Key Design Points

Each operation is self-contained with its parameters, logic, and tests
Pydantic models validate parameters at construction time
ImageContext flows through the pipeline carrying state
Validation happens early - entire pipeline is validated before processing
Tests use synthetic images for predictable verification
Debug mode saves intermediate results for inspection
Operations work on RGBA for consistency (convert on load)

## Notes
- Use Pydantic v2 for modern validation features
- Keep initial implementation minimal but extensible
- Focus on clean interfaces that operations will implement
- Ensure Python 3.10+ compatibility
