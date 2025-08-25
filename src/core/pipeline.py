"""Main pipeline orchestrator for image processing."""

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from core.base import BaseOperation
from core.context import ImageContext
from exceptions import PipelineError, ValidationError

console = Console()


class Pipeline:
    """Main orchestrator for image processing operations.

    Every pipeline run creates a directory with all intermediate steps saved.
    These steps serve as both debug output and cache for future runs.
    """

    def __init__(
        self,
        input_path: str | Path,
        *,
        output_dir: str | Path | None = None,
        use_cache: bool = True,
        verbose: bool = False,
        cache_dirs: list[str | Path] | None = None,
    ):
        """Initialize pipeline with input image.

        Args:
            input_path: Path to input image
            output_dir: Directory to save all steps (auto-generated if None)
            use_cache: Whether to reuse results from previous runs
            verbose: Enable verbose console output
            cache_dirs: Additional directories to search for cached results
        """
        self.input_path = Path(input_path)
        self.verbose = verbose
        self.use_cache = use_cache
        self.operations: list[BaseOperation] = []
        self._image: Image.Image | None = None
        self._context: ImageContext | None = None
        self._manifest: dict[str, Any] = {
            "input": str(input_path),
            "timestamp": None,
            "operations": [],
            "cache_hits": 0,
            "cache_misses": 0,
        }

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input image not found: {self.input_path}")

        # Set up output directory
        if output_dir is None:
            # Auto-generate timestamped directory
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            self.output_dir = Path("runs") / timestamp
            self._manifest["timestamp"] = timestamp
        else:
            self.output_dir = Path(output_dir)
            self._manifest["timestamp"] = datetime.now().isoformat()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up cache directories to search
        self.cache_dirs = []
        if use_cache:
            # Add previous runs as cache sources
            runs_dir = Path("runs")
            if runs_dir.exists():
                self.cache_dirs.extend(sorted(runs_dir.iterdir(), reverse=True))

            # Add any explicitly specified cache directories
            if cache_dirs:
                self.cache_dirs.extend([Path(d) for d in cache_dirs])

        # Create .cache subdirectory for hash mappings
        self.cache_index_dir = self.output_dir / ".cache"
        self.cache_index_dir.mkdir(exist_ok=True)

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

            # Save original image
            original_path = self.output_dir / "00_original.png"
            self._image.save(original_path)

            if self.verbose:
                console.print(f"[green]Loaded image:[/green] {self.input_path}")
                console.print(f"  Size: {self._image.width}×{self._image.height}")
                console.print(f"  Mode: {self._image.mode}")
                console.print(f"  Channels: {channels}")
                console.print(f"  Output dir: {self.output_dir}")

        return self._image, self._context

    def _get_image_hash(self, image: Image.Image) -> str:
        """Generate hash for an image."""
        return hashlib.md5(image.tobytes()).hexdigest()

    def _get_operation_cache_key(self, operation: BaseOperation, image_hash: str) -> str:
        """Generate cache key for operation and image combination."""
        param_hash = operation.generate_param_hash()
        cache_data = {
            "operation": operation.operation_name,
            "params": param_hash,
            "image": image_hash,
            "version": "1.0",
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()

    def _find_cached_result(self, cache_key: str) -> tuple[Image.Image, ImageContext] | None:
        """Look for cached result in cache directories."""
        if not self.use_cache:
            return None

        for cache_dir in self.cache_dirs:
            cache_index = cache_dir / ".cache"
            if not cache_index.exists():
                continue

            # Check if this cache directory has the result
            cache_ref = cache_index / cache_key
            if cache_ref.exists():
                # Read the reference to find the actual file
                with open(cache_ref) as f:
                    ref_data = json.load(f)
                    image_file = cache_dir / ref_data["image_file"]
                    context_file = cache_dir / ref_data["context_file"]

                    if image_file.exists() and context_file.exists():
                        # Load cached image and context
                        image = Image.open(image_file)
                        with open(context_file) as cf:
                            context_data = json.load(cf)
                            context = ImageContext(
                                **{
                                    k: v
                                    for k, v in context_data.items()
                                    if k
                                    in [
                                        "width",
                                        "height",
                                        "channels",
                                        "dtype",
                                        "warnings",
                                        "metadata",
                                    ]
                                }
                            )
                        return image, context

        return None

    def _save_step(
        self,
        step_num: int,
        operation: BaseOperation,
        image: Image.Image,
        context: ImageContext,
        cache_key: str,
    ):
        """Save a pipeline step with both human-readable and cache-indexed names."""
        # Save with human-readable name
        step_name = f"{step_num:02d}_{operation.operation_name}"
        image_file = f"{step_name}.png"
        context_file = f"{step_name}.json"

        image_path = self.output_dir / image_file
        context_path = self.output_dir / context_file

        # Save image
        image.save(image_path)

        # Save context with operation details
        context_data = {
            "step": step_num,
            "operation": operation.operation_name,
            "parameters": operation.model_dump(),
            "width": context.width,
            "height": context.height,
            "channels": context.channels,
            "dtype": context.dtype,
            "warnings": context.warnings,
            "metadata": context.metadata,
            "timestamp": time.time(),
        }

        with open(context_path, "w") as f:
            json.dump(context_data, f, indent=2)

        # Create cache index entry
        cache_ref_path = self.cache_index_dir / cache_key
        with open(cache_ref_path, "w") as f:
            json.dump(
                {
                    "image_file": image_file,
                    "context_file": context_file,
                    "step": step_num,
                    "operation": operation.operation_name,
                },
                f,
            )

        # Add to manifest
        self._manifest["operations"].append(
            {
                "step": step_num,
                "operation": operation.operation_name,
                "parameters": operation.model_dump(),
                "cache_key": cache_key,
                "files": {
                    "image": image_file,
                    "context": context_file,
                },
            }
        )

    def validate(self) -> ImageContext:
        """Validate the pipeline without executing.

        Returns:
            Final expected context after all operations

        Raises:
            ValidationError: If validation fails
        """
        image, context = self._load_image()

        if self.verbose:
            console.print("\n[cyan]Running validation pass...[/cyan]")

        for operation in self.operations:
            try:
                context = operation.validate_operation(context)
                if self.verbose:
                    console.print(f"  ✓ {operation.operation_name} validated")
            except ValidationError as e:
                raise ValidationError(
                    f"Validation failed for {operation.operation_name}: {e}"
                ) from e

        return context

    def execute(
        self,
        output_path: str | Path | None = None,
        *,
        dry_run: bool = False,
    ) -> ImageContext:
        """Execute the pipeline.

        Args:
            output_path: Path for final output (optional, defaults to output_dir/final.png)
            dry_run: Only run validation, don't process

        Returns:
            Final image context

        Raises:
            PipelineError: If execution fails
        """
        # Set default output path if not provided
        output_path = self.output_dir / "final.png" if output_path is None else Path(output_path)

        # Validate pipeline
        try:
            final_context = self.validate()
        except ValidationError as e:
            raise PipelineError(f"Pipeline validation failed: {e}") from e

        if dry_run:
            if self.verbose:
                console.print("\n[green]Dry run complete - pipeline is valid[/green]")
            return final_context

        # Execute operations
        image, context = self._load_image()

        if self.verbose:
            console.print("\n[cyan]Executing pipeline...[/cyan]")
            if self.use_cache:
                console.print(f"  Cache dirs: {len(self.cache_dirs)} directories available")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=not self.verbose,
        ) as progress:
            for i, operation in enumerate(self.operations, 1):
                task = progress.add_task(
                    f"[{i}/{len(self.operations)}] {operation.operation_name}...",
                    total=None,
                )

                try:
                    # Generate cache key
                    image_hash = self._get_image_hash(image)
                    cache_key = self._get_operation_cache_key(operation, image_hash)

                    # Check for cached result
                    cached_result = self._find_cached_result(cache_key)

                    if cached_result is not None:
                        # Use cached result
                        image, context = cached_result
                        self._manifest["cache_hits"] += 1

                        if self.verbose:
                            progress.update(
                                task,
                                description=f"[{i}/{len(self.operations)}] {operation.operation_name} [green](cached)[/green]",
                            )
                    else:
                        # Compute result
                        self._manifest["cache_misses"] += 1
                        validated_context = operation.validate_operation(context)
                        image, context = operation.apply(image, validated_context)

                    # Save step (even if cached, to have complete record)
                    self._save_step(i, operation, image, context, cache_key)

                except Exception as e:
                    raise PipelineError(
                        f"Execution failed at operation {i} ({operation.operation_name}): {e}"
                    ) from e

                progress.remove_task(task)

        # Save final output (create directory if it doesn't exist)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        if output_path != self.output_dir / "final.png":
            # Also save in output_dir for consistency
            image.save(self.output_dir / "final.png")

        # Save manifest
        manifest_path = self.output_dir / "manifest.json"
        self._manifest["output"] = str(output_path)
        self._manifest["final_size"] = {"width": context.width, "height": context.height}

        with open(manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2)

        if self.verbose:
            cache_total = self._manifest["cache_hits"] + self._manifest["cache_misses"]
            hit_rate = self._manifest["cache_hits"] / cache_total if cache_total > 0 else 0

            console.print("\n[green]✓ Pipeline complete![/green]")
            console.print(f"Output saved to: {output_path}")
            console.print(f"All steps saved in: {self.output_dir}")
            console.print(
                f"Cache performance: {self._manifest['cache_hits']}/{cache_total} hits ({hit_rate:.1%})"
            )

        return context

    def get_pipeline_config(self) -> dict[str, Any]:
        """Get the complete pipeline configuration.

        Returns:
            Dictionary with pipeline configuration
        """
        return {
            "input": str(self.input_path),
            "operations": [
                {"type": op.__class__.__name__, "params": op.model_dump()} for op in self.operations
            ],
        }
