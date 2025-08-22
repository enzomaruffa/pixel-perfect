"""Main pipeline orchestrator for image processing."""

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from core.base import BaseOperation
from core.context import ImageContext
from exceptions import PipelineError, ValidationError

console = Console()


class Pipeline:
    """Main orchestrator for image processing operations."""

    def __init__(
        self,
        input_path: str | Path,
        *,
        debug: bool = False,
        cache_dir: str | Path | None = None,
    ):
        """Initialize pipeline with input image.

        Args:
            input_path: Path to input image
            debug: Enable debug output
            cache_dir: Directory for caching operation results
        """
        self.input_path = Path(input_path)
        self.debug = debug
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.operations: list[BaseOperation] = []
        self._image: Image.Image | None = None
        self._context: ImageContext | None = None

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input image not found: {self.input_path}")

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def add(self, operation: BaseOperation) -> "Pipeline":
        """Add an operation to the pipeline.

        Args:
            operation: Operation to add

        Returns:
            Self for method chaining
        """
        self.operations.append(operation)
        return self

    def _load_image(self) -> tuple[Image.Image, ImageContext]:
        """Load image and create initial context."""
        if self._image is None or self._context is None:
            self._image = Image.open(self.input_path)
            # Convert to RGBA if needed
            if self._image.mode not in ("RGBA", "RGB", "L"):
                self._image = self._image.convert("RGBA")

            # Create initial context
            channels = {"L": 1, "RGB": 3, "RGBA": 4}[self._image.mode]
            self._context = ImageContext(
                width=self._image.width,
                height=self._image.height,
                channels=channels,
                dtype="uint8",
            )

            if self.debug:
                console.print(f"[green]Loaded image:[/green] {self.input_path}")
                console.print(f"  Size: {self._image.width}×{self._image.height}")
                console.print(f"  Mode: {self._image.mode}")
                console.print(f"  Channels: {channels}")

        return self._image, self._context

    def _get_image_hash(self, image: Image.Image) -> str:
        """Generate hash for an image."""
        # Convert to numpy array and hash
        arr = np.array(image)
        return hashlib.md5(arr.tobytes()).hexdigest()

    def _validate_pipeline(self) -> ImageContext:
        """Run validation pass through all operations.

        Returns:
            Final context after all validations

        Raises:
            ValidationError: If any operation fails validation
        """
        _, context = self._load_image()

        if self.debug:
            console.print("\n[yellow]Running validation pass...[/yellow]")

        for i, operation in enumerate(self.operations, 1):
            try:
                context = operation.validate_operation(context)
                if self.debug:
                    console.print(f"  ✓ {operation.operation_name} validated")
            except Exception as e:
                raise ValidationError(
                    f"Validation failed at operation {i} ({operation.operation_name}): {e}"
                ) from e

        if context.warnings and self.debug:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in context.warnings:
                console.print(f"  ⚠ {warning}")

        return context

    def execute(
        self,
        output_path: str | Path,
        *,
        dry_run: bool = False,
    ) -> ImageContext:
        """Execute the pipeline.

        Args:
            output_path: Path to save output image
            dry_run: Only run validation, don't process

        Returns:
            Final image context

        Raises:
            PipelineError: If execution fails
        """
        output_path = Path(output_path)

        # Validate pipeline
        try:
            final_context = self._validate_pipeline()
        except ValidationError as e:
            raise PipelineError(f"Pipeline validation failed: {e}") from e

        if dry_run:
            if self.debug:
                console.print("\n[green]Dry run complete - pipeline is valid[/green]")
            return final_context

        # Execute operations
        image, context = self._load_image()

        if self.debug:
            console.print("\n[cyan]Executing pipeline...[/cyan]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=not self.debug,
        ) as progress:
            for i, operation in enumerate(self.operations, 1):
                task = progress.add_task(
                    f"[{i}/{len(self.operations)}] {operation.operation_name}...",
                    total=None,
                )

                try:
                    # Check cache if available
                    if self.cache_dir:
                        cache_key = operation.get_cache_key(self._get_image_hash(image))
                        cache_path = self.cache_dir / f"{cache_key}.png"

                        if cache_path.exists():
                            if self.debug:
                                progress.update(
                                    task,
                                    description=f"[{i}/{len(self.operations)}] {operation.operation_name} (cached)",
                                )
                            image = Image.open(cache_path)
                            # Still need to update context
                            _, context = operation.apply(image, context)
                            progress.remove_task(task)
                            continue

                    # Apply operation
                    image, context = operation.apply(image, context)

                    # Cache result if enabled
                    if self.cache_dir:
                        cache_key = operation.get_cache_key(self._get_image_hash(image))
                        cache_path = self.cache_dir / f"{cache_key}.png"
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        image.save(cache_path)

                except Exception as e:
                    raise PipelineError(
                        f"Execution failed at operation {i} ({operation.operation_name}): {e}"
                    ) from e

                progress.remove_task(task)

        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)

        if self.debug:
            console.print("\n[green]✓ Pipeline complete![/green]")
            console.print(f"Output saved to: {output_path}")
            console.print(f"Final size: {context.width}×{context.height}")

        return context

    def estimate_memory(self) -> int:
        """Estimate total memory usage for the pipeline.

        Returns:
            Estimated memory usage in bytes
        """
        _, context = self._load_image()
        total_memory = context.memory_estimate

        for operation in self.operations:
            op_memory = operation.estimate_memory(context)
            total_memory = max(total_memory, op_memory)
            # Update context for next operation
            context = operation.validate_operation(context)

        return total_memory
