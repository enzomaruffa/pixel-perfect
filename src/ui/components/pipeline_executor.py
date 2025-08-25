"""Real-time pipeline execution with caching and progress tracking."""

import os
import tempfile
import time
from typing import Any

import streamlit as st
from PIL import Image

from core.pipeline import Pipeline
from ui.utils.execution_cache import generate_cache_key, get_execution_cache, get_pipeline_hash


class ExecutionResult:
    """Result of pipeline execution."""

    def __init__(self):
        self.steps: list[dict[str, Any]] = []
        self.final_image: Image.Image | None = None
        self.execution_time: float = 0.0
        self.cache_hits: int = 0
        self.total_steps: int = 0
        self.error: str | None = None
        self.cancelled: bool = False


class PipelineExecutor:
    """Real-time pipeline executor with smart caching."""

    def __init__(self):
        self.cache = get_execution_cache()
        self.current_execution_id: str | None = None
        self.execution_results: dict[str, ExecutionResult] = {}

    def execute_pipeline_realtime(
        self,
        original_image: Image.Image,
        operations: list[dict[str, Any]],
        execute_up_to: int | None = None,
        force_refresh: bool = False,
    ) -> ExecutionResult:
        """Execute pipeline with real-time updates and caching."""

        # Generate execution ID
        execution_id = f"exec_{time.time()}"
        self.current_execution_id = execution_id

        # Cancel previous execution
        self._cancel_previous_executions(execution_id)

        # Create result object
        result = ExecutionResult()
        result.total_steps = len(operations) if execute_up_to is None else execute_up_to
        self.execution_results[execution_id] = result

        if not operations:
            result.final_image = original_image
            return result

        start_time = time.time()

        try:
            # Execute pipeline steps
            current_image = original_image
            get_pipeline_hash(operations)

            # Create progress placeholder
            progress_container = st.empty()
            status_container = st.empty()

            for i, operation_config in enumerate(operations):
                # Check if execution was cancelled
                if self.current_execution_id != execution_id:
                    result.cancelled = True
                    return result

                # Check if we should stop execution
                if execute_up_to is not None and i >= execute_up_to:
                    break

                # Skip disabled operations
                if not operation_config.get("enabled", True):
                    continue

                # Update progress
                progress = (i + 1) / result.total_steps
                progress_container.progress(
                    progress, text=f"Step {i + 1}/{result.total_steps}: {operation_config['name']}"
                )

                # Try cache first (unless force refresh)
                cache_key = generate_cache_key(current_image, operation_config, i)
                cached_result = None if force_refresh else self.cache.get(cache_key)

                if cached_result:
                    # Use cached result
                    current_image, context = cached_result
                    result.cache_hits += 1

                    step_result = {
                        "step": i,
                        "operation": operation_config["name"],
                        "image": current_image.copy(),
                        "context": context,
                        "cached": True,
                        "execution_time": 0.0,
                    }

                    status_container.info(f"✅ {operation_config['name']} (cached)")

                else:
                    # Execute operation
                    step_start = time.time()
                    status_container.info(f"🔄 Executing {operation_config['name']}...")

                    try:
                        current_image, context = self._execute_operation(
                            current_image, operation_config
                        )

                        # Cache the result
                        self.cache.put(cache_key, current_image, context)

                        step_time = time.time() - step_start
                        step_result = {
                            "step": i,
                            "operation": operation_config["name"],
                            "image": current_image.copy(),
                            "context": context,
                            "cached": False,
                            "execution_time": step_time,
                        }

                        status_container.success(
                            f"✅ {operation_config['name']} ({step_time:.2f}s)"
                        )

                    except Exception as e:
                        result.error = f"Error in {operation_config['name']}: {str(e)}"
                        status_container.error(result.error)
                        break

                result.steps.append(step_result)

                # Small delay to show progress
                time.sleep(0.1)

            # Clear progress indicators
            progress_container.empty()
            status_container.empty()

            result.final_image = current_image
            result.execution_time = time.time() - start_time

        except Exception as e:
            result.error = f"Pipeline execution failed: {str(e)}"

        return result

    def execute_single_operation(
        self, image: Image.Image, operation_config: dict[str, Any]
    ) -> tuple[Image.Image, Any]:
        """Execute a single operation for testing/preview."""
        return self._execute_operation(image, operation_config)

    def _execute_operation(
        self, image: Image.Image, operation_config: dict[str, Any]
    ) -> tuple[Image.Image, Any]:
        """Execute a single operation using the core pipeline."""

        # Create temporary files for operation
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_input:
            image.save(tmp_input.name)

            tmp_output = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_output:
                    # Create pipeline with single operation
                    pipeline = Pipeline(tmp_input.name, verbose=False)

                    # Create operation instance
                    operation_class = operation_config["class"]
                    params = operation_config["params"]
                    operation = operation_class(**params)

                    # Add operation and execute
                    pipeline.add(operation)
                    context = pipeline.execute(tmp_output.name)

                    # Load result image
                    result_image = Image.open(tmp_output.name)

                    return result_image, context

            finally:
                # Clean up temp files
                try:
                    os.unlink(tmp_input.name)
                    if tmp_output is not None:
                        os.unlink(tmp_output.name)
                except Exception:
                    pass  # Ignore cleanup errors

    def _cancel_previous_executions(self, current_id: str) -> None:
        """Cancel any previous executions."""
        for exec_id, result in self.execution_results.items():
            if exec_id != current_id and not result.cancelled and not result.error:
                result.cancelled = True

    def get_execution_stats(self) -> dict[str, Any]:
        """Get execution statistics."""
        cache_stats = self.cache.get_stats()

        recent_executions = list(self.execution_results.values())[-10:]  # Last 10
        avg_time = sum(r.execution_time for r in recent_executions) / max(len(recent_executions), 1)
        total_cache_hits = sum(r.cache_hits for r in recent_executions)
        total_steps = sum(len(r.steps) for r in recent_executions)

        return {
            "cache": cache_stats,
            "recent_executions": len(recent_executions),
            "average_execution_time": avg_time,
            "cache_hit_rate": total_cache_hits / max(total_steps, 1),
            "active_execution": self.current_execution_id is not None,
        }


# Global executor instance
def get_pipeline_executor() -> PipelineExecutor:
    """Get or create global pipeline executor."""
    if "pipeline_executor" not in st.session_state:
        st.session_state.pipeline_executor = PipelineExecutor()
    return st.session_state.pipeline_executor
