"""Performance benchmarking and regression testing."""

import os
import time
from pathlib import Path

import psutil
import pytest
from PIL import Image
from tests.conftest import create_test_image

from core.pipeline import Pipeline
from operations.block import BlockFilter
from operations.channel import ChannelSwap
from operations.column import ColumnShift
from operations.pattern import Dither, Mosaic
from operations.pixel import PixelFilter
from operations.row import RowShift


@pytest.mark.performance
class TestOperationPerformance:
    """Benchmark individual operations for performance regression."""

    @pytest.fixture
    def small_image(self, temp_dir):
        """Create small test image (100x100)."""
        path = temp_dir / "small_test.png"
        img = create_test_image(100, 100, "RGBA")
        img.save(path)
        return path

    @pytest.fixture
    def medium_image(self, temp_dir):
        """Create medium test image (500x500)."""
        path = temp_dir / "medium_test.png"
        img = create_test_image(500, 500, "RGBA")
        img.save(path)
        return path

    @pytest.fixture
    def large_image(self, temp_dir):
        """Create large test image (1000x1000)."""
        path = temp_dir / "large_test.png"
        img = create_test_image(1000, 1000, "RGBA")
        img.save(path)
        return path

    def measure_operation_performance(
        self, pipeline: Pipeline, operation_name: str
    ) -> dict[str, float]:
        """Measure performance metrics for a pipeline operation."""
        process = psutil.Process(os.getpid())

        # Measure before execution
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.perf_counter()

        # Execute pipeline
        output_path = f"/tmp/perf_test_{operation_name}.png"
        result = pipeline.execute(output_path)

        # Measure after execution
        end_time = time.perf_counter()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Clean up
        if Path(output_path).exists():
            Path(output_path).unlink()

        return {
            "execution_time": end_time - start_time,
            "memory_delta": end_memory - start_memory,
            "peak_memory": end_memory,
        }

    def test_pixel_filter_performance(self, small_image, medium_image, benchmark_data):
        """Benchmark PixelFilter operation across image sizes."""
        test_cases = [("small", small_image), ("medium", medium_image)]

        for size_name, image_path in test_cases:
            pipeline = Pipeline(str(image_path))
            pipeline.add(PixelFilter(condition="prime", fill_color=(255, 0, 0, 255)))

            metrics = self.measure_operation_performance(pipeline, f"pixel_filter_{size_name}")

            # Record benchmarks
            test_name = f"pixel_filter_{size_name}"
            benchmark_data.record_benchmark(test_name, "execution_time", metrics["execution_time"])
            benchmark_data.record_benchmark(test_name, "memory_delta", metrics["memory_delta"])

            # Check for regression
            is_regression, baseline = benchmark_data.check_regression(
                test_name, "execution_time", metrics["execution_time"]
            )

            if is_regression and baseline:
                pytest.fail(
                    f"Performance regression detected: {metrics['execution_time']:.3f}s vs baseline {baseline:.3f}s"
                )

    def test_block_filter_performance(self, small_image, medium_image, benchmark_data):
        """Benchmark BlockFilter operation."""
        test_cases = [("small", small_image, 4), ("medium", medium_image, 8)]

        for size_name, image_path, block_size in test_cases:
            pipeline = Pipeline(str(image_path))
            pipeline.add(
                BlockFilter(
                    block_width=block_size, block_height=block_size, condition="checkerboard"
                )
            )

            metrics = self.measure_operation_performance(pipeline, f"block_filter_{size_name}")

            test_name = f"block_filter_{size_name}"
            benchmark_data.record_benchmark(test_name, "execution_time", metrics["execution_time"])
            benchmark_data.record_benchmark(test_name, "memory_delta", metrics["memory_delta"])

    def test_row_shift_performance(self, small_image, medium_image, benchmark_data):
        """Benchmark RowShift operation."""
        test_cases = [("small", small_image), ("medium", medium_image)]

        for size_name, image_path in test_cases:
            pipeline = Pipeline(str(image_path))
            pipeline.add(RowShift(selection="odd", shift_amount=5, wrap=True))

            metrics = self.measure_operation_performance(pipeline, f"row_shift_{size_name}")

            test_name = f"row_shift_{size_name}"
            benchmark_data.record_benchmark(test_name, "execution_time", metrics["execution_time"])
            benchmark_data.record_benchmark(test_name, "memory_delta", metrics["memory_delta"])

    def test_mosaic_performance(self, small_image, medium_image, benchmark_data):
        """Benchmark Mosaic operation."""
        test_cases = [("small", small_image, 4), ("medium", medium_image, 8)]

        for size_name, image_path, tile_size in test_cases:
            pipeline = Pipeline(str(image_path))
            pipeline.add(Mosaic(tile_size=tile_size, gap_size=1, mode="average"))

            metrics = self.measure_operation_performance(pipeline, f"mosaic_{size_name}")

            test_name = f"mosaic_{size_name}"
            benchmark_data.record_benchmark(test_name, "execution_time", metrics["execution_time"])
            benchmark_data.record_benchmark(test_name, "memory_delta", metrics["memory_delta"])

    def test_channel_swap_performance(self, small_image, medium_image, benchmark_data):
        """Benchmark ChannelSwap operation."""
        test_cases = [("small", small_image), ("medium", medium_image)]

        for size_name, image_path in test_cases:
            pipeline = Pipeline(str(image_path))
            pipeline.add(ChannelSwap(red_source="green", green_source="blue", blue_source="red"))

            metrics = self.measure_operation_performance(pipeline, f"channel_swap_{size_name}")

            test_name = f"channel_swap_{size_name}"
            benchmark_data.record_benchmark(test_name, "execution_time", metrics["execution_time"])
            benchmark_data.record_benchmark(test_name, "memory_delta", metrics["memory_delta"])


@pytest.mark.performance
@pytest.mark.slow
class TestPipelinePerformance:
    """Benchmark complete pipeline performance."""

    def test_complex_pipeline_performance(self, temp_dir, benchmark_data):
        """Benchmark a complex multi-operation pipeline."""
        input_path = temp_dir / "complex_perf_input.png"
        output_path = temp_dir / "complex_perf_output.png"

        # Medium-sized image for complex pipeline test
        test_img = create_test_image(300, 300, "RGBA")
        test_img.save(input_path)

        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        pipeline = Pipeline(str(input_path))
        context = (
            pipeline.add(PixelFilter(condition="prime", fill_color=(255, 0, 0, 128)))
            .add(BlockFilter(block_width=8, block_height=8, condition="checkerboard"))
            .add(RowShift(selection="odd", shift_amount=3, wrap=True))
            .add(ColumnShift(selection="even", shift_amount=2, wrap=False))
            .add(ChannelSwap(red_source="green", green_source="blue", blue_source="red"))
            .add(Mosaic(tile_size=(6, 6), gap_size=1, mode="average"))
            .add(Dither(method="floyd_steinberg", levels=4))
            .execute(str(output_path))
        )

        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024

        execution_time = end_time - start_time
        memory_delta = end_memory - start_memory

        # Record benchmark
        benchmark_data.record_benchmark("complex_pipeline", "execution_time", execution_time)
        benchmark_data.record_benchmark("complex_pipeline", "memory_delta", memory_delta)

        # Verify output
        assert Path(output_path).exists()
        output_img = Image.open(output_path)
        assert output_img.mode == "RGBA"

        # Performance assertions (adjust thresholds based on expected performance)
        assert execution_time < 30.0, f"Complex pipeline took too long: {execution_time:.2f}s"
        assert memory_delta < 500, f"Memory usage too high: {memory_delta:.1f}MB"

    def test_cache_performance_impact(self, temp_dir, benchmark_data):
        """Test performance impact of caching."""
        input_path = temp_dir / "cache_perf_input.png"
        cache_dir = temp_dir / "cache"

        test_img = create_test_image(200, 200, "RGBA")
        test_img.save(input_path)

        # First run without cache
        start_time = time.perf_counter()
        pipeline1 = Pipeline(str(input_path))
        (
            pipeline1.add(PixelFilter(condition="prime", fill_color=(255, 0, 0, 255)))
            .add(RowShift(selection="odd", shift_amount=3))
            .execute(str(temp_dir / "cache_test1.png"))
        )
        no_cache_time = time.perf_counter() - start_time

        # Second run with cache
        start_time = time.perf_counter()
        pipeline2 = Pipeline(str(input_path), cache_dirs=[str(cache_dir)])
        (
            pipeline2.add(PixelFilter(condition="prime", fill_color=(255, 0, 0, 255)))
            .add(RowShift(selection="odd", shift_amount=3))
            .execute(str(temp_dir / "cache_test2.png"))
        )
        first_cache_time = time.perf_counter() - start_time

        # Third run should hit cache
        start_time = time.perf_counter()
        pipeline3 = Pipeline(str(input_path), cache_dirs=[str(cache_dir)])
        (
            pipeline3.add(PixelFilter(condition="prime", fill_color=(255, 0, 0, 255)))
            .add(RowShift(selection="odd", shift_amount=3))
            .execute(str(temp_dir / "cache_test3.png"))
        )
        cached_time = time.perf_counter() - start_time

        # Record benchmarks
        benchmark_data.record_benchmark("cache_performance", "no_cache", no_cache_time)
        benchmark_data.record_benchmark("cache_performance", "first_cache", first_cache_time)
        benchmark_data.record_benchmark("cache_performance", "cached", cached_time)

        # Cache should provide performance benefit
        cache_speedup = first_cache_time / cached_time if cached_time > 0 else 1.0
        assert cache_speedup > 1.2, f"Cache speedup insufficient: {cache_speedup:.2f}x"

    def test_memory_scaling(self, temp_dir, benchmark_data):
        """Test memory usage scaling with image size."""
        sizes = [(100, 100), (200, 200), (400, 400)]
        memory_usage = []

        for width, height in sizes:
            input_path = temp_dir / f"memory_test_{width}x{height}.png"
            output_path = temp_dir / f"memory_output_{width}x{height}.png"

            test_img = create_test_image(width, height, "RGBA")
            test_img.save(input_path)

            start_memory = psutil.Process().memory_info().rss / 1024 / 1024

            pipeline = Pipeline(str(input_path))
            (
                pipeline.add(PixelFilter(condition="even", fill_color=(255, 0, 0, 255)))
                .add(RowShift(selection="odd", shift_amount=2))
                .execute(str(output_path))
            )

            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_delta = end_memory - start_memory
            memory_usage.append(memory_delta)

            benchmark_data.record_benchmark(
                f"memory_scaling_{width}x{height}", "memory_delta", memory_delta
            )

        # Memory usage should scale reasonably (not exponentially)
        if len(memory_usage) >= 2:
            scaling_factor = memory_usage[-1] / memory_usage[0] if memory_usage[0] > 0 else 1.0
            # For 4x pixel increase, memory should not increase more than 8x
            assert scaling_factor < 8.0, f"Memory scaling too steep: {scaling_factor:.2f}x"


@pytest.mark.performance
class TestCachePerformance:
    """Test cache system performance characteristics."""

    def test_cache_hit_rate(self, temp_dir):
        """Test cache hit rates for repeated operations."""
        input_path = temp_dir / "cache_hit_input.png"
        cache_dir = temp_dir / "cache"

        test_img = create_test_image(150, 150, "RGBA")
        test_img.save(input_path)

        # Run same pipeline multiple times
        operation_times = []

        for i in range(5):
            start_time = time.perf_counter()

            pipeline = Pipeline(str(input_path), cache_dirs=[str(cache_dir)])
            (
                pipeline.add(PixelFilter(condition="prime", fill_color=(255, 0, 0, 255)))
                .add(RowShift(selection="odd", shift_amount=2))
                .execute(str(temp_dir / f"cache_hit_{i}.png"))
            )

            operation_times.append(time.perf_counter() - start_time)

        # First run should be slowest, subsequent runs should be faster
        assert operation_times[0] > operation_times[-1], "Cache not providing performance benefit"

        # Average of runs 2-5 should be significantly faster than first run
        avg_cached_time = sum(operation_times[1:]) / len(operation_times[1:])
        speedup = operation_times[0] / avg_cached_time
        assert speedup > 1.5, f"Insufficient cache speedup: {speedup:.2f}x"

    def test_cache_memory_efficiency(self, temp_dir):
        """Test that cache doesn't consume excessive memory."""
        input_path = temp_dir / "cache_memory_input.png"
        cache_dir = temp_dir / "cache"

        test_img = create_test_image(100, 100, "RGBA")
        test_img.save(input_path)

        start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Run many different operations to populate cache
        for i in range(10):
            pipeline = Pipeline(str(input_path), cache_dirs=[str(cache_dir)])
            (
                pipeline.add(PixelFilter(condition="even", fill_color=(255, i * 25, 0, 255)))
                .add(RowShift(selection="odd", shift_amount=i % 5 + 1))
                .execute(str(temp_dir / f"cache_memory_{i}.png"))
            )

        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_growth = end_memory - start_memory

        # Cache should not cause excessive memory growth
        assert memory_growth < 200, f"Cache memory growth too high: {memory_growth:.1f}MB"


@pytest.mark.performance
@pytest.mark.slow
class TestScalabilityLimits:
    """Test framework limits and scalability."""

    def test_large_image_handling(self, temp_dir, benchmark_data):
        """Test handling of large images."""
        # Create a moderately large image (not too large for CI)
        input_path = temp_dir / "large_image_test.png"
        test_img = create_test_image(1500, 1000, "RGBA")  # 1.5MP
        test_img.save(input_path)

        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        pipeline = Pipeline(str(input_path))
        context = (
            pipeline.add(PixelFilter(condition="even", fill_color=(255, 0, 0, 128)))
            .add(RowShift(selection="odd", shift_amount=5))
            .execute(str(temp_dir / "large_output.png"))
        )

        execution_time = time.perf_counter() - start_time
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_delta = end_memory - start_memory

        benchmark_data.record_benchmark("large_image", "execution_time", execution_time)
        benchmark_data.record_benchmark("large_image", "memory_delta", memory_delta)

        # Verify output
        assert Path(output_path).exists()
        output_img = Image.open(output_path)
        assert output_img.size == (1500, 1000)

        # Performance bounds for large images
        assert execution_time < 60.0, f"Large image processing too slow: {execution_time:.2f}s"
        assert memory_delta < 1000, f"Large image memory usage too high: {memory_delta:.1f}MB"

    def test_operation_count_scaling(self, temp_dir):
        """Test performance with many operations."""
        input_path = temp_dir / "many_ops_input.png"
        test_img = create_test_image(100, 100, "RGBA")
        test_img.save(input_path)

        operation_counts = [5, 10, 20]
        execution_times = []

        for op_count in operation_counts:
            start_time = time.perf_counter()

            pipeline = Pipeline(str(input_path))

            for i in range(op_count):
                if i % 3 == 0:
                    pipeline.add(PixelFilter(condition="even", fill_color=(255, 0, 0, 64)))
                elif i % 3 == 1:
                    pipeline.add(RowShift(selection="odd", shift_amount=1, wrap=True))
                else:
                    pipeline.add(ColumnShift(selection="even", shift_amount=1, wrap=True))

            result = pipeline.execute(str(temp_dir / f"many_ops_{op_count}.png"))
            execution_time = time.perf_counter() - start_time
            execution_times.append(execution_time)

            assert Path(output_path).exists()

        # Execution time should scale reasonably with operation count
        if len(execution_times) >= 2:
            scaling_factor = execution_times[-1] / execution_times[0]
            op_scaling = operation_counts[-1] / operation_counts[0]

            # Time scaling should be roughly linear with operation count
            assert scaling_factor < op_scaling * 2, (
                f"Operation scaling too steep: {scaling_factor:.2f}x for {op_scaling}x operations"
            )
