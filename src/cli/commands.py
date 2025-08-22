"""Command implementations for the pixel-perfect CLI."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.tree import Tree

from core.pipeline import Pipeline
from operations import (
    AspectCrop, AspectPad, AspectStretch,
    AlphaGenerator, ChannelIsolate, ChannelSwap,
    BlockFilter, BlockRotate, BlockScramble, BlockShift,
    ColumnMirror, ColumnShift, ColumnStretch, ColumnWeave,
    GridWarp, PerspectiveStretch, RadialStretch,
    Dither, Mosaic,
    PixelFilter, PixelMath, PixelSort,
    RowRemove, RowShift, RowShuffle, RowStretch,
)
from utils.cache_manager import CachePolicy
from .config import load_config, save_config, ConfigManager

console = Console()

# Map of all available operations
OPERATION_CLASSES = {
    # Pixel operations
    "PixelFilter": PixelFilter,
    "PixelMath": PixelMath,
    "PixelSort": PixelSort,
    
    # Row operations
    "RowShift": RowShift,
    "RowStretch": RowStretch,
    "RowRemove": RowRemove,
    "RowShuffle": RowShuffle,
    
    # Column operations
    "ColumnShift": ColumnShift,
    "ColumnStretch": ColumnStretch,
    "ColumnMirror": ColumnMirror,
    "ColumnWeave": ColumnWeave,
    
    # Block operations
    "BlockFilter": BlockFilter,
    "BlockShift": BlockShift,
    "BlockRotate": BlockRotate,
    "BlockScramble": BlockScramble,
    
    # Geometric operations
    "GridWarp": GridWarp,
    "PerspectiveStretch": PerspectiveStretch,
    "RadialStretch": RadialStretch,
    
    # Aspect ratio operations
    "AspectStretch": AspectStretch,
    "AspectCrop": AspectCrop,
    "AspectPad": AspectPad,
    
    # Channel operations
    "ChannelSwap": ChannelSwap,
    "ChannelIsolate": ChannelIsolate,
    "AlphaGenerator": AlphaGenerator,
    
    # Pattern operations
    "Mosaic": Mosaic,
    "Dither": Dither,
}

# Operation categories for organized display
OPERATION_CATEGORIES = {
    "Pixel Operations": ["PixelFilter", "PixelMath", "PixelSort"],
    "Row Operations": ["RowShift", "RowStretch", "RowRemove", "RowShuffle"],
    "Column Operations": ["ColumnShift", "ColumnStretch", "ColumnMirror", "ColumnWeave"],
    "Block Operations": ["BlockFilter", "BlockShift", "BlockRotate", "BlockScramble"],
    "Geometric Operations": ["GridWarp", "PerspectiveStretch", "RadialStretch"],
    "Aspect Ratio Operations": ["AspectStretch", "AspectCrop", "AspectPad"],
    "Channel Operations": ["ChannelSwap", "ChannelIsolate", "AlphaGenerator"],
    "Pattern Operations": ["Mosaic", "Dither"],
}


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("--config", "-c", type=click.Path(exists=True, path_type=Path), 
              help="Pipeline configuration file (YAML)")
@click.option("--cache-dir", type=click.Path(path_type=Path), 
              help="Directory for caching operation results")
@click.option("--cache-size", type=str, 
              help="Maximum cache size (e.g., '100MB', '1GB')")
@click.option("--dry-run", is_flag=True, 
              help="Validate pipeline without processing")
@click.option("--debug", is_flag=True, 
              help="Enable debug output")
@click.pass_context
def process(ctx, input_path, output_path, config, cache_dir, cache_size, dry_run, debug):
    """Process an image with a specified pipeline.
    
    INPUT_PATH: Path to the input image file
    OUTPUT_PATH: Path where the processed image will be saved
    """
    verbose = ctx.obj.get("verbose", False)
    
    try:
        # Load configuration if provided
        if config:
            with open(config) as f:
                config_data = yaml.safe_load(f)
            
            if verbose:
                console.print(f"📄 Loaded configuration from {config}")
        else:
            config_data = {}
        
        # Set up cache policy
        cache_policy = None
        if cache_dir or cache_size:
            cache_policy = CachePolicy()
            if cache_size:
                # Parse cache size string
                cache_policy.max_size_bytes = _parse_size_string(cache_size)
        
        # Create pipeline
        pipeline = Pipeline(
            input_path,
            debug=debug or verbose,
            cache_dir=cache_dir,
            cache_policy=cache_policy,
        )
        
        # Add operations from config
        operations = config_data.get("operations", [])
        for op_config in operations:
            op_type = op_config["type"]
            op_params = op_config.get("params", {})
            
            if op_type not in OPERATION_CLASSES:
                raise click.ClickException(f"Unknown operation type: {op_type}")
            
            operation = OPERATION_CLASSES[op_type](**op_params)
            pipeline.add(operation)
        
        if not operations:
            console.print("[yellow]Warning: No operations specified. Output will be identical to input.[/yellow]")
        
        # Execute pipeline
        if verbose:
            console.print(f"🖼️  Processing: {input_path} → {output_path}")
            if dry_run:
                console.print("🔍 [yellow]Dry run mode enabled[/yellow]")
        
        result_context = pipeline.execute(output_path, dry_run=dry_run)
        
        if dry_run:
            console.print("✅ Pipeline validation successful!")
            console.print(f"Final dimensions: {result_context.width}×{result_context.height}")
            if result_context.warnings:
                console.print(f"Warnings: {len(result_context.warnings)}")
        else:
            console.print("✅ [green]Processing complete![/green]")
            
            # Show cache statistics if available
            if pipeline.cache_manager and verbose:
                stats = pipeline.get_cache_statistics()
                if stats and stats.get("hits", 0) > 0:
                    console.print(f"📊 Cache hit rate: {stats['hit_rate']:.1%}")
    
    except Exception as e:
        raise click.ClickException(f"Processing failed: {e}")


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("--cache-dir", type=click.Path(path_type=Path), 
              help="Directory for caching operation results")
@click.option("--simple", is_flag=True,
              help="Use simple text-based interface instead of advanced TUI")
@click.pass_context
def build(ctx, input_path, output_path, cache_dir, simple):
    """Interactive pipeline builder with live preview.
    
    🎨 Advanced TUI (default): Live preview, step-by-step execution, real-time editing
    📝 Simple mode (--simple): Text-based prompts for basic usage
    
    Keyboard shortcuts in TUI mode:
    - Ctrl+S: Save pipeline    - Ctrl+R: Run pipeline
    - Ctrl+O: Load pipeline    - Space: Toggle step
    - Enter: Add operation     - F1: Help
    """
    verbose = ctx.obj.get("verbose", False)
    
    if simple:
        # Use the original simple interface
        _build_simple_interface(input_path, output_path, cache_dir, verbose)
    else:
        # Launch the amazing TUI interface
        from .interactive import launch_interactive_builder
        
        console.print("🚀 [bold cyan]Launching Interactive Pipeline Builder...[/bold cyan]")
        console.print("💡 [yellow]Tip: Press F1 for help once the interface loads[/yellow]\n")
        
        try:
            launch_interactive_builder(input_path, output_path, cache_dir)
        except KeyboardInterrupt:
            console.print("\n👋 Interactive builder closed")
        except Exception as e:
            raise click.ClickException(f"TUI failed to launch: {e}")


@click.command("list-operations")
@click.option("--category", "-c", help="Filter by category")
@click.option("--search", "-s", help="Search operations by name or description")
def list_operations(category, search):
    """List all available operations."""
    console.print("🎨 [bold cyan]Available Operations[/bold cyan]\n")
    
    for cat_name, operations in OPERATION_CATEGORIES.items():
        if category and category.lower() not in cat_name.lower():
            continue
        
        # Filter by search term
        if search:
            operations = [
                op for op in operations 
                if search.lower() in op.lower() or 
                   (OPERATION_CLASSES[op].__doc__ and search.lower() in OPERATION_CLASSES[op].__doc__.lower())
            ]
        
        if not operations:
            continue
        
        console.print(f"[bold yellow]{cat_name}[/bold yellow]")
        for op_name in operations:
            op_class = OPERATION_CLASSES[op_name]
            doc = op_class.__doc__ or "No description available"
            # Get first line of docstring
            short_desc = doc.split('\n')[0].strip()
            console.print(f"  • [cyan]{op_name}[/cyan]: {short_desc}")
        console.print()


@click.command()
@click.argument("operation_name")
def describe(operation_name):
    """Show detailed information about a specific operation."""
    if operation_name not in OPERATION_CLASSES:
        # Try case-insensitive search
        matches = [name for name in OPERATION_CLASSES.keys() 
                  if name.lower() == operation_name.lower()]
        if not matches:
            console.print(f"❌ Operation '{operation_name}' not found")
            console.print("\n💡 Use 'pixel-perfect list-operations' to see available operations")
            return
        operation_name = matches[0]
    
    op_class = OPERATION_CLASSES[operation_name]
    
    console.print(f"📖 [bold cyan]{operation_name}[/bold cyan]\n")
    
    # Show documentation
    if op_class.__doc__:
        console.print(f"[yellow]Description:[/yellow]")
        console.print(f"  {op_class.__doc__.strip()}\n")
    
    # Show parameters using Pydantic model
    try:
        # Get model fields
        model_fields = op_class.model_fields
        
        if model_fields:
            console.print("[yellow]Parameters:[/yellow]")
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Parameter", style="cyan")
            table.add_column("Type", style="green")
            table.add_column("Default", style="yellow")
            table.add_column("Description", style="white")
            
            for field_name, field_info in model_fields.items():
                field_type = str(field_info.annotation).replace("typing.", "")
                default_val = str(field_info.default) if field_info.default is not None else "Required"
                description = field_info.description or "No description"
                
                table.add_row(field_name, field_type, default_val, description)
            
            console.print(table)
        else:
            console.print("  No parameters required")
    
    except Exception as e:
        console.print(f"  [red]Error displaying parameters: {e}[/red]")


@click.command()
@click.argument("operation_name")
def examples(operation_name):
    """Show usage examples for a specific operation."""
    if operation_name not in OPERATION_CLASSES:
        console.print(f"❌ Operation '{operation_name}' not found")
        return
    
    console.print(f"💡 [bold cyan]Examples for {operation_name}[/bold cyan]\n")
    
    # Generate example configurations
    examples_data = _generate_operation_examples(operation_name)
    
    for i, example in enumerate(examples_data, 1):
        console.print(f"[yellow]Example {i}: {example['title']}[/yellow]")
        console.print("```yaml")
        console.print(yaml.dump(example["config"], default_flow_style=False))
        console.print("```")
        if example.get("description"):
            console.print(f"ℹ️  {example['description']}")
        console.print()


@click.group()
def presets():
    """Manage operation presets for common effects."""
    pass


@presets.command("list")
def presets_list():
    """List all available presets."""
    from presets.built_in import get_all_presets
    
    console.print("🎭 [bold cyan]Available Presets[/bold cyan]\n")
    
    all_presets = get_all_presets()
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Operations", style="yellow")
    
    for preset_name, preset_data in all_presets.items():
        operations_count = len(preset_data.get("operations", []))
        table.add_row(
            preset_name,
            preset_data.get("description", "No description"),
            str(operations_count)
        )
    
    console.print(table)


@presets.command("apply")
@click.argument("preset_name")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("--cache-dir", type=click.Path(path_type=Path),
              help="Directory for caching operation results")
@click.pass_context
def presets_apply(ctx, preset_name, input_path, output_path, cache_dir):
    """Apply a preset to an image."""
    from presets.built_in import get_preset
    
    verbose = ctx.obj.get("verbose", False)
    
    try:
        preset_config = get_preset(preset_name)
        
        if verbose:
            console.print(f"🎭 Applying preset: [cyan]{preset_name}[/cyan]")
            console.print(f"📄 {preset_config.get('description', 'No description')}")
        
        # Create pipeline
        pipeline = Pipeline(input_path, debug=verbose, cache_dir=cache_dir)
        
        # Add operations from preset
        for op_config in preset_config["operations"]:
            op_type = op_config["type"]
            op_params = op_config.get("params", {})
            
            if op_type not in OPERATION_CLASSES:
                raise click.ClickException(f"Unknown operation type in preset: {op_type}")
            
            operation = OPERATION_CLASSES[op_type](**op_params)
            pipeline.add(operation)
        
        # Execute pipeline
        pipeline.execute(output_path)
        console.print("✅ [green]Preset applied successfully![/green]")
        
    except FileNotFoundError:
        raise click.ClickException(f"Preset '{preset_name}' not found")
    except Exception as e:
        raise click.ClickException(f"Failed to apply preset: {e}")


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option("--config", "-c", type=click.Path(exists=True, path_type=Path),
              help="Pipeline configuration file (YAML)")
@click.option("--preset", "-p", help="Use a built-in preset")
@click.option("--pattern", default="*.{jpg,jpeg,png,bmp,tiff}",
              help="File pattern to match (default: *.{jpg,jpeg,png,bmp,tiff})")
@click.option("--parallel", "-j", type=int, default=1,
              help="Number of parallel processes (default: 1)")
@click.option("--cache-dir", type=click.Path(path_type=Path),
              help="Directory for caching operation results")
@click.pass_context
def batch(ctx, input_dir, output_dir, config, preset, pattern, parallel, cache_dir):
    """Process multiple images with the same pipeline.
    
    INPUT_DIR: Directory containing input images
    OUTPUT_DIR: Directory where processed images will be saved
    """
    verbose = ctx.obj.get("verbose", False)
    
    # Find input files
    input_files = []
    for ext in ["jpg", "jpeg", "png", "bmp", "tiff"]:
        input_files.extend(input_dir.glob(f"*.{ext}"))
        input_files.extend(input_dir.glob(f"*.{ext.upper()}"))
    
    if not input_files:
        console.print(f"❌ No image files found in {input_dir}")
        return
    
    console.print(f"📁 Found {len(input_files)} images to process")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    if config and preset:
        raise click.ClickException("Cannot specify both --config and --preset")
    
    if config:
        with open(config) as f:
            config_data = yaml.safe_load(f)
        console.print(f"📄 Using configuration: {config}")
    elif preset:
        from presets.built_in import get_preset
        config_data = get_preset(preset)
        console.print(f"🎭 Using preset: {preset}")
    else:
        raise click.ClickException("Must specify either --config or --preset")
    
    # Process files
    operations = config_data.get("operations", [])
    if not operations:
        console.print("⚠️  No operations specified")
        return
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("Processing images...", total=len(input_files))
        
        for i, input_file in enumerate(input_files):
            try:
                # Generate output path
                output_file = output_dir / input_file.name
                
                progress.update(task, description=f"Processing {input_file.name}...")
                
                # Create and execute pipeline
                pipeline = Pipeline(input_file, debug=False, cache_dir=cache_dir)
                
                for op_config in operations:
                    op_type = op_config["type"]
                    op_params = op_config.get("params", {})
                    
                    operation = OPERATION_CLASSES[op_type](**op_params)
                    pipeline.add(operation)
                
                pipeline.execute(output_file)
                
                progress.update(task, advance=1)
                
            except Exception as e:
                console.print(f"❌ Failed to process {input_file.name}: {e}")
                progress.update(task, advance=1)
    
    console.print(f"✅ [green]Batch processing complete![/green]")
    console.print(f"📁 Results saved to: {output_dir}")


def _select_operation_from_category(pipeline, category):
    """Helper function to select and configure an operation from a category."""
    operations = OPERATION_CATEGORIES[category]
    
    console.print(f"\n🔧 [bold]{category}[/bold]")
    for i, op_name in enumerate(operations, 1):
        op_class = OPERATION_CLASSES[op_name]
        doc = op_class.__doc__ or "No description"
        short_desc = doc.split('\n')[0].strip()
        console.print(f"  {i}. [cyan]{op_name}[/cyan]: {short_desc}")
    
    try:
        choice = click.prompt("Select operation", type=int)
        if 1 <= choice <= len(operations):
            op_name = operations[choice - 1]
            _configure_operation(pipeline, op_name)
        else:
            console.print("❌ Invalid choice")
    except (ValueError, click.Abort):
        console.print("❌ Invalid input")


def _configure_operation(pipeline, op_name):
    """Helper function to configure operation parameters."""
    op_class = OPERATION_CLASSES[op_name]
    
    console.print(f"\n⚙️  Configuring [cyan]{op_name}[/cyan]")
    
    # For now, create with default parameters
    # In a full implementation, this would prompt for each parameter
    try:
        operation = op_class()
        pipeline.add(operation)
        console.print(f"✅ Added {op_name} with default parameters")
    except Exception as e:
        console.print(f"❌ Failed to add operation: {e}")


def _generate_operation_examples(operation_name):
    """Generate example configurations for an operation."""
    # This would be expanded with real examples for each operation
    examples = [
        {
            "title": "Basic usage",
            "config": {
                "operations": [
                    {
                        "type": operation_name,
                        "params": {}
                    }
                ]
            },
            "description": f"Basic {operation_name} with default parameters"
        }
    ]
    
    # Add operation-specific examples
    if operation_name == "PixelFilter":
        examples.extend([
            {
                "title": "Filter prime numbered pixels",
                "config": {
                    "operations": [
                        {
                            "type": "PixelFilter",
                            "params": {
                                "condition": "prime",
                                "fill_color": [255, 0, 0, 0]
                            }
                        }
                    ]
                },
                "description": "Makes non-prime pixels transparent red"
            },
            {
                "title": "Custom filter expression",
                "config": {
                    "operations": [
                        {
                            "type": "PixelFilter",
                            "params": {
                                "condition": "custom",
                                "custom_expression": "i % 3 == 0"
                            }
                        }
                    ]
                },
                "description": "Filters pixels where index is divisible by 3"
            }
        ])
    
    return examples


def _build_simple_interface(input_path: Path, output_path: Path, cache_dir: Optional[Path], verbose: bool) -> None:
    """Simple text-based pipeline builder interface."""
    console.print("🔧 [bold blue]Simple Pipeline Builder[/bold blue]")
    console.print(f"Input: {input_path}")
    console.print(f"Output: {output_path}\n")
    
    # Create pipeline
    pipeline = Pipeline(input_path, debug=verbose, cache_dir=cache_dir)
    
    # Interactive operation selection
    while True:
        console.print("\n📋 [bold]Available Operation Categories:[/bold]")
        categories = list(OPERATION_CATEGORIES.keys())
        
        for i, category in enumerate(categories, 1):
            console.print(f"  {i}. {category}")
        
        console.print(f"  {len(categories) + 1}. Finish and execute pipeline")
        console.print(f"  {len(categories) + 2}. Cancel")
        
        try:
            choice = click.prompt("\nSelect category", type=int)
            
            if choice == len(categories) + 1:
                # Finish and execute
                break
            elif choice == len(categories) + 2:
                # Cancel
                console.print("❌ Pipeline building cancelled")
                return
            elif 1 <= choice <= len(categories):
                # Show operations in selected category
                category = categories[choice - 1]
                _select_operation_from_category(pipeline, category)
            else:
                console.print("❌ Invalid choice")
        
        except (ValueError, click.Abort):
            console.print("❌ Invalid input or cancelled")
            return
    
    # Execute pipeline
    if len(pipeline.operations) == 0:
        console.print("⚠️  No operations added. Nothing to process.")
        return
    
    console.print(f"\n🚀 Executing pipeline with {len(pipeline.operations)} operations...")
    try:
        pipeline.execute(output_path)
        console.print("✅ [green]Pipeline execution complete![/green]")
    except Exception as e:
        raise click.ClickException(f"Pipeline execution failed: {e}")


def _parse_size_string(size_str: str) -> int:
    """Parse size strings like '100MB', '1GB' into bytes."""
    size_str = size_str.upper().strip()
    
    multipliers = {
        'B': 1,
        'KB': 1024,
        'MB': 1024**2,
        'GB': 1024**3,
        'TB': 1024**4,
    }
    
    for suffix, multiplier in multipliers.items():
        if size_str.endswith(suffix):
            number_part = size_str[:-len(suffix)].strip()
            try:
                return int(float(number_part) * multiplier)
            except ValueError:
                raise click.ClickException(f"Invalid size format: {size_str}")
    
    # If no suffix, assume bytes
    try:
        return int(size_str)
    except ValueError:
        raise click.ClickException(f"Invalid size format: {size_str}")