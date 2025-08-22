"""Cache debugging and inspection tools."""

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from utils.cache import (
    format_bytes,
    get_cache_size,
    verify_cache_integrity,
)
from utils.cache_manager import CacheManager

console = Console()


class CacheInspector:
    """Cache inspection and debugging utility."""

    def __init__(self, cache_dir: Path):
        """Initialize cache inspector.

        Args:
            cache_dir: Cache directory to inspect
        """
        self.cache_dir = Path(cache_dir)

    def print_cache_tree(self) -> None:
        """Print cache directory structure as a tree."""
        if not self.cache_dir.exists():
            console.print(f"[red]Cache directory does not exist: {self.cache_dir}[/red]")
            return

        tree = Tree(f"📁 {self.cache_dir.name}")

        # Get all files
        image_files = list(self.cache_dir.glob("*.png"))
        context_files = list(self.cache_dir.glob("*.json"))
        other_files = [
            f
            for f in self.cache_dir.rglob("*")
            if f.is_file() and f.suffix not in (".png", ".json")
        ]

        # Add image files
        if image_files:
            images_branch = tree.add("🖼️  Images")
            for img_file in sorted(image_files):
                file_size = format_bytes(img_file.stat().st_size)
                images_branch.add(f"{img_file.name} ({file_size})")

        # Add context files
        if context_files:
            contexts_branch = tree.add("📄 Context Files")
            for ctx_file in sorted(context_files):
                file_size = format_bytes(ctx_file.stat().st_size)
                contexts_branch.add(f"{ctx_file.name} ({file_size})")

        # Add other files
        if other_files:
            others_branch = tree.add("📋 Other Files")
            for other_file in sorted(other_files):
                file_size = format_bytes(other_file.stat().st_size)
                others_branch.add(f"{other_file.name} ({file_size})")

        console.print(tree)

    def print_cache_summary(self) -> None:
        """Print cache summary statistics."""
        if not self.cache_dir.exists():
            console.print(f"[red]Cache directory does not exist: {self.cache_dir}[/red]")
            return

        # Get basic statistics
        size_info = get_cache_size(self.cache_dir)
        integrity_info = verify_cache_integrity(self.cache_dir)

        # Create summary table
        table = Table(title="Cache Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Entries", str(size_info["entry_count"]))
        table.add_row("Total Files", str(size_info["files"]))
        table.add_row("Total Size", size_info.get("human_readable", "Unknown"))
        table.add_row("Valid Entries", str(integrity_info.get("valid", 0)))
        table.add_row("Invalid Entries", str(integrity_info.get("invalid", 0)))
        table.add_row("Orphaned Files", str(integrity_info.get("orphaned", 0)))

        console.print(table)

        # Show orphaned files if any
        orphaned_files = integrity_info.get("orphaned_files", [])
        if orphaned_files:
            console.print("\n[yellow]Orphaned Files:[/yellow]")
            for file_path in orphaned_files:
                console.print(f"  • {file_path}")

    def print_cache_entries(self, limit: int = 20) -> None:
        """Print detailed cache entry information.

        Args:
            limit: Maximum number of entries to show
        """
        if not self.cache_dir.exists():
            console.print(f"[red]Cache directory does not exist: {self.cache_dir}[/red]")
            return

        # Get context files with metadata
        context_files = list(self.cache_dir.glob("*.json"))

        if not context_files:
            console.print("[yellow]No cache entries found[/yellow]")
            return

        # Sort by modification time (newest first)
        context_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        # Create entries table
        table = Table(
            title=f"Cache Entries (showing {min(limit, len(context_files))} of {len(context_files)})"
        )
        table.add_column("Key", style="cyan", width=12)
        table.add_column("Size", style="green", width=10)
        table.add_column("Dimensions", style="yellow", width=12)
        table.add_column("Channels", style="blue", width=8)
        table.add_column("Age", style="magenta", width=10)
        table.add_column("Valid", style="red", width=8)

        import json
        import time

        for context_file in context_files[:limit]:
            try:
                # Get file info
                key = context_file.stem
                image_file = self.cache_dir / f"{key}.png"

                # Check if valid
                is_valid = image_file.exists()

                # Get sizes
                ctx_size = context_file.stat().st_size
                img_size = image_file.stat().st_size if is_valid else 0
                total_size = format_bytes(ctx_size + img_size)

                # Get age
                age_seconds = time.time() - context_file.stat().st_mtime
                if age_seconds < 3600:
                    age_str = f"{age_seconds / 60:.0f}m"
                elif age_seconds < 86400:
                    age_str = f"{age_seconds / 3600:.0f}h"
                else:
                    age_str = f"{age_seconds / 86400:.0f}d"

                # Get dimensions from context
                dimensions = "Unknown"
                channels = "Unknown"
                try:
                    with open(context_file) as f:
                        ctx_data = json.load(f)
                    dimensions = f"{ctx_data.get('width', '?')}×{ctx_data.get('height', '?')}"
                    channels = str(ctx_data.get("channels", "?"))
                except Exception:
                    pass

                table.add_row(
                    key[:12], total_size, dimensions, channels, age_str, "✓" if is_valid else "✗"
                )

            except Exception:
                # Skip problematic entries
                continue

        console.print(table)

    def find_large_entries(self, min_size_mb: float = 1.0) -> list[dict[str, Any]]:
        """Find cache entries larger than specified size.

        Args:
            min_size_mb: Minimum size in megabytes

        Returns:
            List of large entry information
        """
        if not self.cache_dir.exists():
            return []

        min_size_bytes = int(min_size_mb * 1024 * 1024)
        large_entries = []

        for context_file in self.cache_dir.glob("*.json"):
            try:
                key = context_file.stem
                image_file = self.cache_dir / f"{key}.png"

                if not image_file.exists():
                    continue

                total_size = context_file.stat().st_size + image_file.stat().st_size

                if total_size >= min_size_bytes:
                    large_entries.append(
                        {
                            "key": key,
                            "size_bytes": total_size,
                            "size_human": format_bytes(total_size),
                            "image_path": str(image_file),
                            "context_path": str(context_file),
                        }
                    )

            except Exception:
                continue

        # Sort by size (largest first)
        large_entries.sort(key=lambda x: x["size_bytes"], reverse=True)

        return large_entries

    def find_old_entries(self, max_age_days: int = 7) -> list[dict[str, Any]]:
        """Find cache entries older than specified age.

        Args:
            max_age_days: Maximum age in days

        Returns:
            List of old entry information
        """
        if not self.cache_dir.exists():
            return []

        import time

        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        old_entries = []

        for context_file in self.cache_dir.glob("*.json"):
            try:
                if context_file.stat().st_mtime < cutoff_time:
                    key = context_file.stem
                    image_file = self.cache_dir / f"{key}.png"

                    age_days = (time.time() - context_file.stat().st_mtime) / 86400

                    total_size = context_file.stat().st_size
                    if image_file.exists():
                        total_size += image_file.stat().st_size

                    old_entries.append(
                        {
                            "key": key,
                            "age_days": age_days,
                            "size_bytes": total_size,
                            "size_human": format_bytes(total_size),
                            "context_path": str(context_file),
                            "image_path": str(image_file) if image_file.exists() else None,
                        }
                    )

            except Exception:
                continue

        # Sort by age (oldest first)
        old_entries.sort(key=lambda x: x["age_days"], reverse=True)

        return old_entries


def print_cache_performance_analysis(cache_manager: CacheManager) -> None:
    """Print detailed cache performance analysis.

    Args:
        cache_manager: Cache manager to analyze
    """
    overall_stats = cache_manager.get_statistics()

    # Overall performance table
    table = Table(title="Cache Performance Analysis", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Details", style="yellow")

    table.add_row(
        "Hit Rate", f"{overall_stats.hit_rate:.1%}", "Percentage of cache hits vs total requests"
    )
    table.add_row(
        "Total Requests",
        str(overall_stats.hits + overall_stats.misses),
        f"{overall_stats.hits} hits, {overall_stats.misses} misses",
    )
    table.add_row(
        "Cache Size", overall_stats.human_readable_size, f"{overall_stats.entry_count} entries"
    )
    table.add_row("Save Operations", str(overall_stats.saves), "Number of results cached")
    table.add_row("Errors", str(overall_stats.errors), "Cache operation failures")

    console.print(table)

    # Performance recommendations
    console.print("\n[bold]Performance Recommendations:[/bold]")

    if overall_stats.hit_rate < 0.3:
        console.print("🔴 Low hit rate - consider cache warming or reviewing operation parameters")
    elif overall_stats.hit_rate < 0.7:
        console.print("🟡 Moderate hit rate - cache is working but could be optimized")
    else:
        console.print("🟢 Good hit rate - cache is performing well")

    if overall_stats.errors > 0:
        console.print(
            f"⚠️  {overall_stats.errors} cache errors detected - check disk space and permissions"
        )

    if overall_stats.entry_count > 1000:
        console.print("💾 Large cache - consider periodic cleanup or size limits")


def compare_cache_keys(key1: str, key2: str) -> None:
    """Compare two cache keys and show differences.

    Args:
        key1: First cache key
        key2: Second cache key
    """
    console.print("[bold]Comparing cache keys:[/bold]")
    console.print(f"Key 1: [cyan]{key1}[/cyan]")
    console.print(f"Key 2: [cyan]{key2}[/cyan]")

    if key1 == key2:
        console.print("[green]✓ Keys are identical[/green]")
    else:
        console.print("[red]✗ Keys are different[/red]")

        # Show character-by-character differences
        console.print("\n[bold]Character differences:[/bold]")
        min_len = min(len(key1), len(key2))

        for i in range(min_len):
            if key1[i] != key2[i]:
                console.print(f"Position {i}: '{key1[i]}' vs '{key2[i]}'")

        if len(key1) != len(key2):
            console.print(f"Length difference: {len(key1)} vs {len(key2)}")
