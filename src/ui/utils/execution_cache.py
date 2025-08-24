"""Smart caching system for pipeline execution."""

import hashlib
import io
import time
from typing import Any

import streamlit as st
from PIL import Image


class ExecutionCache:
    """Smart caching system for pipeline operations."""

    def __init__(self, max_size: int = 50, max_memory_mb: int = 200):
        """Initialize cache with size and memory limits."""
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: dict[str, dict[str, Any]] = {}
        self.access_times: dict[str, float] = {}
        self.memory_usage = 0

    def get(self, cache_key: str) -> tuple[Image.Image, Any] | None:
        """Get cached result if available."""
        if cache_key in self.cache:
            # Update access time for LRU
            self.access_times[cache_key] = time.time()
            cache_entry = self.cache[cache_key]

            # Deserialize image from bytes
            image_bytes = cache_entry["image_bytes"]
            image = Image.open(io.BytesIO(image_bytes))
            context = cache_entry["context"]

            return image, context
        return None

    def put(self, cache_key: str, image: Image.Image, context: Any) -> None:
        """Store result in cache with memory management."""
        # Serialize image to bytes for storage
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        image_bytes = img_buffer.getvalue()

        # Calculate memory usage
        entry_size = len(image_bytes) + self._estimate_context_size(context)

        # Make room if necessary
        while (
            len(self.cache) >= self.max_size
            or self.memory_usage + entry_size > self.max_memory_bytes
        ):
            if not self._evict_lru():
                break  # Can't evict any more

        # Store in cache
        self.cache[cache_key] = {
            "image_bytes": image_bytes,
            "context": context,
            "size": entry_size,
            "timestamp": time.time(),
        }
        self.access_times[cache_key] = time.time()
        self.memory_usage += entry_size

    def invalidate_after(self, operation_index: int, pipeline_hash: str) -> None:
        """Invalidate cache entries that depend on operations after given index."""
        keys_to_remove = []

        for key in self.cache:
            # Parse key to extract operation index and pipeline hash
            if pipeline_hash in key:
                try:
                    # Assuming key format includes operation index
                    parts = key.split("_")
                    cached_index = int(parts[-1])
                    if cached_index > operation_index:
                        keys_to_remove.append(key)
                except (ValueError, IndexError):
                    # If we can't parse the key, better to remove it
                    keys_to_remove.append(key)

        for key in keys_to_remove:
            self._remove_entry(key)

    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
        self.access_times.clear()
        self.memory_usage = 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "entries": len(self.cache),
            "max_entries": self.max_size,
            "memory_usage_mb": self.memory_usage / (1024 * 1024),
            "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
            "hit_ratio": getattr(self, "_hits", 0) / max(getattr(self, "_requests", 1), 1),
        }

    def _evict_lru(self) -> bool:
        """Evict least recently used entry."""
        if not self.cache:
            return False

        # Find LRU entry
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove_entry(lru_key)
        return True

    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self.cache:
            entry_size = self.cache[key]["size"]
            del self.cache[key]
            del self.access_times[key]
            self.memory_usage -= entry_size

    def _estimate_context_size(self, context: Any) -> int:
        """Estimate memory size of context object."""
        # Simple estimation - in practice could be more sophisticated
        try:
            return len(str(context)) * 2  # Rough estimate
        except Exception:
            return 1024  # Default estimate


def generate_cache_key(
    image: Image.Image, operation_config: dict[str, Any], step_index: int
) -> str:
    """Generate cache key for operation result."""
    # Create hash of image
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    image_hash = hashlib.md5(img_buffer.getvalue()).hexdigest()[:8]

    # Create hash of operation config including enabled state
    operation_str = f"{operation_config['name']}_{operation_config['params']}_{operation_config.get('enabled', True)}"
    operation_hash = hashlib.md5(operation_str.encode()).hexdigest()[:8]

    return f"cache_{image_hash}_{operation_hash}_{step_index}"


def get_pipeline_hash(operations: list) -> str:
    """Generate hash for entire pipeline configuration."""
    pipeline_str = ""
    for i, op in enumerate(operations):
        pipeline_str += f"{i}_{op['name']}_{op['params']}"

    return hashlib.md5(pipeline_str.encode()).hexdigest()[:12]


# Global cache instance for Streamlit session
def get_execution_cache() -> ExecutionCache:
    """Get or create global execution cache."""
    if "execution_cache" not in st.session_state:
        st.session_state.execution_cache = ExecutionCache()
    return st.session_state.execution_cache
