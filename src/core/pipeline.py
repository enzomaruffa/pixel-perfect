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
from utils.cache_manager import CacheManager, CachePolicy

console = Console()


class Pipeline:
    """Main orchestrator for image processing operations."""

    def __init__(
        self,
        input_path: str | Path,
        *,
        debug: bool = False,
        cache_dir: str | Path | None = None,
        cache_policy: CachePolicy | None = None,
        enable_memory_cache: bool = True,
    ):
        """Initialize pipeline with input image.

        Args:
            input_path: Path to input image
            debug: Enable debug output
            cache_dir: Directory for caching operation results
            cache_policy: Cache policy configuration
            enable_memory_cache: Whether to use in-memory caching
        """
        self.input_path = Path(input_path)
        self.debug = debug
        self.operations: list[BaseOperation] = []
        self._image: Image.Image | None = None
        self._context: ImageContext | None = None

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input image not found: {self.input_path}")

        # Initialize cache manager if cache directory is provided
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_manager = CacheManager(
                cache_dir=cache_dir,
                policy=cache_policy or CachePolicy(),
                enable_memory_cache=enable_memory_cache,
            )
        else:
            self.cache_dir = None
            self.cache_manager = None

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

            # Convert to RGB/RGBA for consistent processing
            # Only grayscale (L) gets converted to RGB, others to RGBA if needed
            if self._image.mode == "L":
                self._image = self._image.convert("RGB")
            elif self._image.mode not in ("RGB", "RGBA"):
                self._image = self._image.convert("RGBA")

            # Create initial context
            channels = {"RGB": 3, "RGBA": 4}[self._image.mode]
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
        # For cache consistency, we should hash the original file content
        # rather than the processed image which might have conversion differences
        try:
            with open(self.input_path, "rb") as f:
                file_content = f.read()
            return hashlib.md5(file_content).hexdigest()
        except Exception:
            # Fallback to image data if file reading fails
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
            if self.cache_manager:
                cache_stats = self.cache_manager.get_statistics()
                console.print(
                    f"Cache status: {cache_stats.entry_count} entries, {cache_stats.human_readable_size}"
                )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=not self.debug,
        ) as progress:
            cache_hits = 0
            cache_misses = 0

            for i, operation in enumerate(self.operations, 1):
                task = progress.add_task(
                    f"[{i}/{len(self.operations)}] {operation.operation_name}...",
                    total=None,
                )

                try:
                    # Check cache if available
                    cached_result = None
                    if self.cache_manager:
                        image_hash = self._get_image_hash(image)
                        cached_result = self.cache_manager.get_cached_result(operation, image_hash)

                        if cached_result is not None:
                            cache_hits += 1
                            image, context = cached_result
                            if self.debug:
                                progress.update(
                                    task,
                                    description=f"[{i}/{len(self.operations)}] {operation.operation_name} [green](cached)[/green]",
                                )
                            progress.remove_task(task)
                            continue

                    # Apply operation
                    cache_misses += 1
                    validated_context = operation.validate_operation(context)
                    image, context = operation.apply(image, validated_context)

                    # Cache result if enabled
                    if self.cache_manager:
                        image_hash = self._get_image_hash(image)
                        self.cache_manager.save_result(operation, image_hash, image, context)

                except Exception as e:
                    raise PipelineError(
                        f"Execution failed at operation {i} ({operation.operation_name}): {e}"
                    ) from e

                progress.remove_task(task)

            # Show cache statistics if debug enabled
            if self.debug and self.cache_manager:
                total_ops = cache_hits + cache_misses
                hit_rate = cache_hits / total_ops if total_ops > 0 else 0
                console.print(
                    f"\nCache performance: {cache_hits}/{total_ops} hits ({hit_rate:.1%})"
                )

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

    def get_cache_statistics(self) -> dict | None:
        """Get cache statistics.

        Returns:
            Cache statistics dictionary or None if caching disabled
        """
        if not self.cache_manager:
            return None

        return self.cache_manager.get_statistics().to_dict()

    def print_cache_report(self) -> None:
        """Print detailed cache report to console."""
        if not self.cache_manager:
            console.print("[yellow]Caching is not enabled[/yellow]")
            return

        report = self.cache_manager.export_cache_report()
        console.print(report)

    def cleanup_cache(self, max_age_days: int | None = None) -> int:
        """Clean up old cache entries.

        Args:
            max_age_days: Maximum age in days (uses policy default if None)

        Returns:
            Number of entries removed
        """
        if not self.cache_manager:
            return 0

        return self.cache_manager.cleanup_old_entries(max_age_days)

    def clear_cache(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries removed
        """
        if not self.cache_manager:
            return 0

        return self.cache_manager.clear_cache()

    def warm_cache(self, test_images: list[Image.Image] | None = None) -> dict:
        """Pre-populate cache with operation results.

        Args:
            test_images: Optional list of test images

        Returns:
            Dict with warming statistics
        """
        if not self.cache_manager:
            return {"warmed": 0, "errors": 0}

        return self.cache_manager.warm_cache(self.operations, test_images)

    def set_cache_size_limit(self, max_size_bytes: int) -> None:
        """Set cache size limit.

        Args:
            max_size_bytes: Maximum cache size in bytes
        """
        if self.cache_manager:
            self.cache_manager.set_size_limit(max_size_bytes)

    def invalidate_cache_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern.

        Args:
            pattern: Glob pattern to match cache keys

        Returns:
            Number of entries invalidated
        """
        if not self.cache_manager:
            return 0

        return self.cache_manager.invalidate_pattern(pattern)
