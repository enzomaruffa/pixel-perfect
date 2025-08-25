"""Advanced cache management for the pixel-perfect framework."""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from core.base import BaseOperation
from core.context import ImageContext
from exceptions import ProcessingError, ValidationError

# Import at module level to avoid circular imports
# We'll import these functions when needed in methods


@dataclass
class CacheStats:
    """Cache statistics and metrics."""

    hits: int = 0
    misses: int = 0
    saves: int = 0
    errors: int = 0
    total_bytes: int = 0
    entry_count: int = 0
    hit_rate: float = field(init=False)

    def __post_init__(self):
        """Calculate derived statistics."""
        total_requests = self.hits + self.misses
        self.hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

    @property
    def human_readable_size(self) -> str:
        """Get human-readable cache size."""
        from utils.cache import format_bytes

        return format_bytes(self.total_bytes)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "saves": self.saves,
            "errors": self.errors,
            "total_bytes": self.total_bytes,
            "entry_count": self.entry_count,
            "hit_rate": self.hit_rate,
            "human_readable_size": self.human_readable_size,
        }


@dataclass
class CachePolicy:
    """Cache policy configuration."""

    enabled: bool = True
    max_size_bytes: int | None = None  # None = unlimited
    max_age_days: int = 7
    auto_cleanup: bool = True
    compression_enabled: bool = False
    memory_cache_size: int = 100  # Number of results to keep in memory

    def validate(self) -> None:
        """Validate cache policy settings."""
        if self.max_size_bytes is not None and self.max_size_bytes <= 0:
            raise ValidationError("Cache max_size_bytes must be positive")

        if self.max_age_days <= 0:
            raise ValidationError("Cache max_age_days must be positive")

        if self.memory_cache_size < 0:
            raise ValidationError("Memory cache size must be non-negative")


class CacheManager:
    """Advanced cache management system."""

    def __init__(
        self,
        cache_dir: str | Path,
        policy: CachePolicy | None = None,
        enable_memory_cache: bool = True,
    ):
        """Initialize cache manager.

        Args:
            cache_dir: Directory for cache storage
            policy: Cache policy configuration
            enable_memory_cache: Whether to use in-memory caching
        """
        self.cache_dir = Path(cache_dir)
        self.policy = policy or CachePolicy()
        self.policy.validate()

        # Statistics tracking
        self._stats = CacheStats()
        self._operation_stats: dict[str, CacheStats] = defaultdict(CacheStats)

        # Memory cache for small results
        self._memory_cache: dict[str, tuple[Image.Image, ImageContext]] = {}
        self._memory_cache_access: dict[str, float] = {}
        self._enable_memory_cache = enable_memory_cache

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load existing cache statistics
        self._load_stats()

    def get_cached_result(
        self,
        operation: BaseOperation,
        image_hash: str,
    ) -> tuple[Image.Image, ImageContext] | None:
        """Get cached result for operation.

        Args:
            operation: Operation instance
            image_hash: Hash of input image

        Returns:
            Cached result tuple or None if not found
        """
        if not self.policy.enabled:
            return None

        cache_key = operation.get_cache_key(image_hash)

        # Check memory cache first
        if self._enable_memory_cache and cache_key in self._memory_cache:
            self._record_hit(operation.operation_name)
            self._memory_cache_access[cache_key] = time.time()
            return self._memory_cache[cache_key]

        # Check disk cache
        try:
            from utils.cache import load_cached_result

            result = load_cached_result(self.cache_dir, cache_key)
            if result is not None:
                self._record_hit(operation.operation_name)

                # Add to memory cache if enabled
                if self._enable_memory_cache:
                    self._add_to_memory_cache(cache_key, result)

                return result
            else:
                self._record_miss(operation.operation_name)
                return None

        except Exception:
            self._record_error(operation.operation_name)
            # Log error but don't raise - treat as cache miss
            return None

    def save_result(
        self,
        operation: BaseOperation,
        image_hash: str,
        image: Image.Image,
        context: ImageContext,
    ) -> None:
        """Save operation result to cache.

        Args:
            operation: Operation instance
            image_hash: Hash of input image
            image: Result image
            context: Result context
        """
        if not self.policy.enabled:
            return

        cache_key = operation.get_cache_key(image_hash)

        try:
            # Check size limits before saving
            if self.policy.max_size_bytes is not None:
                from utils.cache import get_cache_size

                current_size = get_cache_size(self.cache_dir)["total_bytes"]
                if current_size >= self.policy.max_size_bytes:
                    self._enforce_size_limit()

            # Save to disk
            from utils.cache import save_cached_result

            save_cached_result(self.cache_dir, cache_key, image, context)
            self._record_save(operation.operation_name)

            # Add to memory cache if enabled
            if self._enable_memory_cache:
                self._add_to_memory_cache(cache_key, (image.copy(), context))

            # Update statistics
            self._update_cache_stats()

        except Exception:
            self._record_error(operation.operation_name)
            # Don't raise - caching is not critical

    def get_statistics(self) -> CacheStats:
        """Get overall cache statistics."""
        self._update_cache_stats()
        return self._stats

    def get_operation_statistics(self, operation_name: str) -> CacheStats:
        """Get statistics for specific operation type."""
        return self._operation_stats.get(operation_name, CacheStats())

    def cleanup_old_entries(self, max_age_days: int | None = None) -> int:
        """Clean up old cache entries.

        Args:
            max_age_days: Maximum age in days (uses policy default if None)

        Returns:
            Number of entries removed
        """
        age_limit = max_age_days or self.policy.max_age_days

        try:
            from utils.cache import cleanup_old_cache

            removed_count = cleanup_old_cache(self.cache_dir, age_limit)

            # Update statistics
            self._update_cache_stats()

            return removed_count

        except Exception as e:
            raise ProcessingError(f"Cache cleanup failed: {e}") from e

    def set_size_limit(self, max_size_bytes: int) -> None:
        """Set cache size limit and enforce it.

        Args:
            max_size_bytes: Maximum cache size in bytes
        """
        if max_size_bytes <= 0:
            raise ValidationError("Cache size limit must be positive")

        self.policy.max_size_bytes = max_size_bytes
        self._enforce_size_limit()

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern.

        Args:
            pattern: Glob pattern to match cache keys

        Returns:
            Number of entries invalidated
        """
        try:
            from utils.cache import invalidate_cache_pattern

            removed_count = invalidate_cache_pattern(self.cache_dir, pattern)

            # Remove from memory cache too
            if self._enable_memory_cache:
                keys_to_remove = [
                    key for key in self._memory_cache if key.startswith(pattern.replace("*", ""))
                ]
                for key in keys_to_remove:
                    self._memory_cache.pop(key, None)
                    self._memory_cache_access.pop(key, None)

            # Update statistics
            self._update_cache_stats()

            return removed_count

        except Exception as e:
            raise ProcessingError(f"Cache invalidation failed: {e}") from e

    def warm_cache(
        self,
        common_operations: list[BaseOperation],
        test_images: list[Image.Image] | None = None,
    ) -> dict[str, int]:
        """Pre-populate cache with common operation results.

        Args:
            common_operations: List of operations to warm cache for
            test_images: Optional list of test images (creates synthetic if None)

        Returns:
            Dict with warming statistics
        """
        if not self.policy.enabled:
            return {"warmed": 0, "errors": 0}

        # Create test images if none provided
        if test_images is None:
            test_images = self._create_test_images()

        warmed_count = 0
        error_count = 0

        for operation in common_operations:
            for image in test_images:
                try:
                    # Generate image hash
                    image_hash = self._generate_image_hash(image)

                    # Check if already cached
                    cache_key = operation.get_cache_key(image_hash)
                    from utils.cache import load_cached_result

                    if (
                        cache_key in self._memory_cache
                        or load_cached_result(self.cache_dir, cache_key) is not None
                    ):
                        continue

                    # Apply operation and cache result
                    channels = {"L": 1, "RGB": 3, "RGBA": 4}[image.mode]
                    context = ImageContext(
                        width=image.width,
                        height=image.height,
                        channels=channels,
                        dtype="uint8",
                    )
                    validated_context = operation.validate_operation(context)
                    result_image, result_context = operation.apply(image, validated_context)

                    self.save_result(operation, image_hash, result_image, result_context)
                    warmed_count += 1

                except Exception:
                    error_count += 1
                    continue

        return {"warmed": warmed_count, "errors": error_count}

    def export_cache_report(self) -> str:
        """Export detailed cache report.

        Returns:
            Formatted cache report string
        """
        self._update_cache_stats()

        # Overall statistics
        overall_stats = self._stats
        from utils.cache import verify_cache_integrity

        integrity_info = verify_cache_integrity(self.cache_dir)

        # Generate report
        report_lines = [
            "# Cache Report",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Overall Statistics",
            f"Total entries: {overall_stats.entry_count}",
            f"Cache size: {overall_stats.human_readable_size}",
            f"Hit rate: {overall_stats.hit_rate:.1%}",
            f"Total requests: {overall_stats.hits + overall_stats.misses}",
            f"Cache hits: {overall_stats.hits}",
            f"Cache misses: {overall_stats.misses}",
            f"Save operations: {overall_stats.saves}",
            f"Errors: {overall_stats.errors}",
            "",
            "## Cache Integrity",
            f"Valid entries: {integrity_info.get('valid', 0)}",
            f"Invalid entries: {integrity_info.get('invalid', 0)}",
            f"Orphaned files: {integrity_info.get('orphaned', 0)}",
            "",
            "## Memory Cache",
            f"Entries in memory: {len(self._memory_cache)}",
            f"Memory cache enabled: {self._enable_memory_cache}",
            "",
            "## Per-Operation Statistics",
        ]

        # Add per-operation stats
        for op_name, op_stats in self._operation_stats.items():
            if op_stats.hits + op_stats.misses > 0:
                report_lines.extend(
                    [
                        f"### {op_name}",
                        f"Hit rate: {op_stats.hit_rate:.1%}",
                        f"Hits: {op_stats.hits}, Misses: {op_stats.misses}",
                        f"Saves: {op_stats.saves}, Errors: {op_stats.errors}",
                        "",
                    ]
                )

        # Configuration
        report_lines.extend(
            [
                "## Configuration",
                f"Cache directory: {self.cache_dir}",
                f"Cache enabled: {self.policy.enabled}",
                f"Max size: {self._format_bytes_local(self.policy.max_size_bytes) if self.policy.max_size_bytes else 'Unlimited'}",
                f"Max age: {self.policy.max_age_days} days",
                f"Auto cleanup: {self.policy.auto_cleanup}",
                f"Memory cache size limit: {self.policy.memory_cache_size}",
            ]
        )

        return "\n".join(report_lines)

    def clear_cache(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries removed
        """
        removed_count = 0

        # Clear disk cache
        for file_path in self.cache_dir.rglob("*"):
            if file_path.is_file():
                try:
                    file_path.unlink()
                    removed_count += 1
                except Exception:
                    continue

        # Clear memory cache
        if self._enable_memory_cache:
            self._memory_cache.clear()
            self._memory_cache_access.clear()

        # Reset statistics
        self._stats = CacheStats()
        self._operation_stats.clear()

        return removed_count

    def _record_hit(self, operation_name: str) -> None:
        """Record cache hit."""
        self._stats.hits += 1
        self._operation_stats[operation_name].hits += 1

    def _record_miss(self, operation_name: str) -> None:
        """Record cache miss."""
        self._stats.misses += 1
        self._operation_stats[operation_name].misses += 1

    def _record_save(self, operation_name: str) -> None:
        """Record cache save."""
        self._stats.saves += 1
        self._operation_stats[operation_name].saves += 1

    def _record_error(self, operation_name: str) -> None:
        """Record cache error."""
        self._stats.errors += 1
        self._operation_stats[operation_name].errors += 1

    def _update_cache_stats(self) -> None:
        """Update cache size statistics."""
        from utils.cache import get_cache_size

        size_info = get_cache_size(self.cache_dir)
        self._stats.total_bytes = size_info["total_bytes"]
        self._stats.entry_count = size_info["entry_count"]

        # Recalculate hit rate
        total_requests = self._stats.hits + self._stats.misses
        self._stats.hit_rate = self._stats.hits / total_requests if total_requests > 0 else 0.0

    def _enforce_size_limit(self) -> None:
        """Enforce cache size limit by removing oldest entries."""
        if self.policy.max_size_bytes is None:
            return

        from utils.cache import get_cache_size

        current_size = get_cache_size(self.cache_dir)["total_bytes"]

        while current_size > self.policy.max_size_bytes:
            # Find oldest entries and remove them
            oldest_files = []

            for context_file in self.cache_dir.glob("*.json"):
                try:
                    mtime = context_file.stat().st_mtime
                    oldest_files.append((mtime, context_file))
                except Exception:
                    continue

            if not oldest_files:
                break

            # Sort by modification time and remove oldest
            oldest_files.sort(key=lambda x: x[0])

            # Remove oldest 10% or at least 1 entry
            remove_count = max(1, len(oldest_files) // 10)

            for _, context_file in oldest_files[:remove_count]:
                try:
                    key = context_file.stem
                    image_file = self.cache_dir / f"{key}.png"

                    if image_file.exists():
                        image_file.unlink()
                    context_file.unlink()

                    # Remove from memory cache too
                    self._memory_cache.pop(key, None)
                    self._memory_cache_access.pop(key, None)

                except Exception:
                    continue

            # Update current size
            current_size = get_cache_size(self.cache_dir)["total_bytes"]

    def _add_to_memory_cache(
        self,
        key: str,
        result: tuple[Image.Image, ImageContext],
    ) -> None:
        """Add result to memory cache with LRU eviction."""
        if not self._enable_memory_cache:
            return

        # Evict old entries if at limit
        while len(self._memory_cache) >= self.policy.memory_cache_size:
            if not self._memory_cache_access:
                break

            # Find least recently used entry
            oldest_key = min(
                self._memory_cache_access.keys(), key=lambda k: self._memory_cache_access[k]
            )

            self._memory_cache.pop(oldest_key, None)
            self._memory_cache_access.pop(oldest_key, None)

        # Add new entry
        self._memory_cache[key] = result
        self._memory_cache_access[key] = time.time()

    def _create_test_images(self) -> list[Image.Image]:
        """Create synthetic test images for cache warming."""
        test_images = []

        # Different sizes and modes
        configs = [
            (4, 4, "RGB"),
            (8, 8, "RGB"),
            (16, 16, "RGBA"),
            (4, 8, "L"),  # Different aspect ratio
        ]

        for width, height, mode in configs:
            if mode == "L":
                image = Image.new(mode, (width, height), 128)
            elif mode == "RGB":
                image = Image.new(mode, (width, height), (128, 64, 192))
            else:  # RGBA
                image = Image.new(mode, (width, height), (128, 64, 192, 255))

            test_images.append(image)

        return test_images

    def _generate_image_hash(self, image: Image.Image) -> str:
        """Generate hash for image content."""
        import hashlib
        import io

        # Convert image to bytes for consistent hashing
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_data = img_bytes.getvalue()

        # Generate hash
        return hashlib.md5(img_data).hexdigest()

    def _format_bytes_local(self, bytes_value: int) -> str:
        """Format bytes as human-readable string (local version to avoid circular imports)."""
        if bytes_value == 0:
            return "0 B"

        units = ["B", "KB", "MB", "GB", "TB"]
        unit_index = 0
        value = float(bytes_value)

        while value >= 1024 and unit_index < len(units) - 1:
            value /= 1024
            unit_index += 1

        if unit_index == 0:
            return f"{int(value)} {units[unit_index]}"
        else:
            return f"{value:.1f} {units[unit_index]}"

    def _load_stats(self) -> None:
        """Load existing statistics from disk."""
        stats_file = self.cache_dir / "cache_stats.json"

        if not stats_file.exists():
            return

        try:
            with open(stats_file) as f:
                stats_data = json.load(f)

            # Load overall stats
            if "overall" in stats_data:
                overall = stats_data["overall"]
                self._stats.hits = overall.get("hits", 0)
                self._stats.misses = overall.get("misses", 0)
                self._stats.saves = overall.get("saves", 0)
                self._stats.errors = overall.get("errors", 0)

            # Load per-operation stats
            if "operations" in stats_data:
                for op_name, op_data in stats_data["operations"].items():
                    stats = self._operation_stats[op_name]
                    stats.hits = op_data.get("hits", 0)
                    stats.misses = op_data.get("misses", 0)
                    stats.saves = op_data.get("saves", 0)
                    stats.errors = op_data.get("errors", 0)

        except Exception:
            # If loading fails, start with fresh stats
            pass

    def _save_stats(self) -> None:
        """Save statistics to disk."""
        stats_file = self.cache_dir / "cache_stats.json"

        try:
            stats_data = {
                "overall": self._stats.to_dict(),
                "operations": {
                    op_name: op_stats.to_dict()
                    for op_name, op_stats in self._operation_stats.items()
                },
                "last_updated": time.time(),
            }

            with open(stats_file, "w") as f:
                json.dump(stats_data, f, indent=2)

        except Exception:
            # If saving fails, don't raise - statistics are not critical
            pass

    def __del__(self):
        """Save statistics when manager is destroyed."""
        import contextlib

        with contextlib.suppress(Exception):
            self._save_stats()
