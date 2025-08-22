"""Main CLI entry point for pixel-perfect."""

import click
from rich.console import Console

from .commands import build, describe, examples, list_operations, process, presets, batch

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="pixel-perfect")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx, verbose):
    """Pixel Perfect - Image processing pipeline framework.
    
    A sophisticated image processing toolkit that applies pixel-level
    transformations through a composable pipeline architecture.
    """
    # Ensure context object exists and add config
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    
    if verbose:
        console.print("🎨 [bold cyan]Pixel Perfect[/bold cyan] - Image Processing Framework")
        console.print("Version 0.1.0\n")


# Add command groups and commands
cli.add_command(process)
cli.add_command(build)
cli.add_command(list_operations)
cli.add_command(describe)
cli.add_command(examples)
cli.add_command(presets)
cli.add_command(batch)


if __name__ == "__main__":
    cli()