"""Caching utilities for operation results."""

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

from core.base import BaseOperation
from core.context import ImageContext
from exceptions import ProcessingError


def generate_operation_hash(operation: BaseOperation, image_hash: str) -> str:
    """Generate unique cache key for operation and image combination.

    Args:
        operation: Operation instance
        image_hash: Hash of input image

    Returns:
        Unique cache key string
    """
    # Get operation parameters hash
    param_hash = operation.generate_param_hash()

    # Combine operation class name, parameters, and image hash
    cache_data = {
        "operation": operation.operation_name,
        "params": param_hash,
        "image": image_hash,
        "version": "1.0",  # For cache invalidation when implementations change
    }

    cache_string = json.dumps(cache_data, sort_keys=True)
    return hashlib.md5(cache_string.encode()).hexdigest()


def save_cached_result(
    cache_dir: Path, key: str, image: Image.Image, context: ImageContext
) -> None:
    """Save operation result to cache.

    Args:
        cache_dir: Cache directory path
        key: Cache key
        image: Result image
        context: Result context

    Raises:
        ProcessingError: If caching fails
    """
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Save image
        image_path = cache_dir / f"{key}.png"
        image.save(image_path, "PNG")

        # Save context metadata
        context_path = cache_dir / f"{key}.json"
        context_data = {
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

    except Exception as e:
        raise ProcessingError(f"Failed to save cache entry {key}: {e}") from e


def load_cached_result(cache_dir: Path, key: str) -> tuple[Image.Image, ImageContext] | None:
    """Load operation result from cache.

    Args:
        cache_dir: Cache directory path
        key: Cache key

    Returns:
        Tuple of (image, context) if found, None otherwise

    Raises:
        ProcessingError: If cache loading fails
    """
    try:
        image_path = cache_dir / f"{key}.png"
        context_path = cache_dir / f"{key}.json"

        # Check if both files exist
        if not image_path.exists() or not context_path.exists():
            return None

        # Load image
        image = Image.open(image_path)

        # Load context
        with open(context_path) as f:
            context_data = json.load(f)

        # Reconstruct context
        context = ImageContext(
            width=context_data["width"],
            height=context_data["height"],
            channels=context_data["channels"],
            dtype=context_data.get("dtype", "uint8"),
            warnings=context_data.get("warnings", []),
            metadata=context_data.get("metadata", {}),
        )

        return image, context

    except Exception:
        # If cache loading fails, treat as cache miss
        return None


def is_cache_entry_valid(cache_dir: Path, key: str, max_age_hours: int = 24) -> bool:
    """Check if cache entry is valid and not expired.

    Args:
        cache_dir: Cache directory path
        key: Cache key
        max_age_hours: Maximum age in hours before expiration

    Returns:
        True if cache entry is valid
    """
    try:
        context_path = cache_dir / f"{key}.json"

        if not context_path.exists():
            return False

        # Check file modification time
        stat = context_path.stat()
        age_hours = (time.time() - stat.st_mtime) / 3600

        if age_hours > max_age_hours:
            return False

        # Check if context file is valid JSON
        with open(context_path) as f:
            context_data = json.load(f)

        # Verify required fields exist
        required_fields = ["width", "height", "channels"]
        return all(field in context_data for field in required_fields)

    except Exception:
        return False


def cleanup_old_cache(cache_dir: Path, max_age_days: int = 7) -> int:
    """Remove cache entries older than specified age.

    Args:
        cache_dir: Cache directory path
        max_age_days: Maximum age in days

    Returns:
        Number of entries removed

    Raises:
        ProcessingError: If cleanup fails
    """
    if not cache_dir.exists():
        return 0

    try:
        removed_count = 0
        cutoff_time = time.time() - (max_age_days * 24 * 3600)

        # Find all .json files (context files)
        for context_file in cache_dir.glob("*.json"):
            try:
                if context_file.stat().st_mtime < cutoff_time:
                    # Remove both image and context files
                    key = context_file.stem
                    image_file = cache_dir / f"{key}.png"

                    if image_file.exists():
                        image_file.unlink()
                    context_file.unlink()
                    removed_count += 1

            except Exception:
                # Skip files that can't be processed
                continue

        return removed_count

    except Exception as e:
        raise ProcessingError(f"Failed to cleanup cache: {e}") from e


def get_cache_size(cache_dir: Path) -> dict[str, Any]:
    """Get cache directory size and entry count.

    Args:
        cache_dir: Cache directory path

    Returns:
        Dict with size statistics
    """
    if not cache_dir.exists():
        return {"total_bytes": 0, "entry_count": 0, "files": 0}

    try:
        total_bytes = 0
        entry_count = 0
        file_count = 0

        for file_path in cache_dir.rglob("*"):
            if file_path.is_file():
                file_count += 1
                total_bytes += file_path.stat().st_size

                if file_path.suffix == ".json":
                    entry_count += 1

        return {
            "total_bytes": total_bytes,
            "entry_count": entry_count,
            "files": file_count,
            "human_readable": format_bytes(total_bytes),
        }

    except Exception:
        return {"total_bytes": 0, "entry_count": 0, "files": 0, "error": True}


def format_bytes(bytes_value: int) -> str:
    """Format bytes as human-readable string.

    Args:
        bytes_value: Number of bytes

    Returns:
        Formatted string (e.g., "1.5 MB")
    """
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


def invalidate_cache_pattern(cache_dir: Path, pattern: str) -> int:
    """Invalidate cache entries matching pattern.

    Args:
        cache_dir: Cache directory path
        pattern: Glob pattern to match cache keys

    Returns:
        Number of entries invalidated

    Raises:
        ProcessingError: If invalidation fails
    """
    if not cache_dir.exists():
        return 0

    try:
        removed_count = 0

        # Find matching files
        for context_file in cache_dir.glob(f"{pattern}.json"):
            try:
                key = context_file.stem
                image_file = cache_dir / f"{key}.png"

                if image_file.exists():
                    image_file.unlink()
                context_file.unlink()
                removed_count += 1

            except Exception:
                continue

        return removed_count

    except Exception as e:
        raise ProcessingError(f"Failed to invalidate cache pattern {pattern}: {e}") from e


def verify_cache_integrity(cache_dir: Path) -> dict[str, Any]:
    """Verify integrity of cache entries.

    Args:
        cache_dir: Cache directory path

    Returns:
        Dict with integrity check results
    """
    if not cache_dir.exists():
        return {"valid": 0, "invalid": 0, "orphaned": 0}

    try:
        valid_count = 0
        invalid_count = 0
        orphaned_files = []

        # Check .json files
        for context_file in cache_dir.glob("*.json"):
            key = context_file.stem
            image_file = cache_dir / f"{key}.png"

            if not image_file.exists():
                orphaned_files.append(str(context_file))
                invalid_count += 1
                continue

            try:
                # Try to load both files
                with open(context_file) as f:
                    json.load(f)

                Image.open(image_file).verify()
                valid_count += 1

            except Exception:
                invalid_count += 1

        # Check for orphaned .png files
        for image_file in cache_dir.glob("*.png"):
            key = image_file.stem
            context_file = cache_dir / f"{key}.json"

            if not context_file.exists():
                orphaned_files.append(str(image_file))

        return {
            "valid": valid_count,
            "invalid": invalid_count,
            "orphaned": len(orphaned_files),
            "orphaned_files": orphaned_files,
        }

    except Exception as e:
        return {"error": str(e)}
