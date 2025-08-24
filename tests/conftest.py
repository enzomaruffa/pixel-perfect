"""Enhanced pytest configuration for comprehensive integration testing."""

import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from skimage.metrics import structural_similarity as ssim


@pytest.fixture(scope="session")
def test_data_dir():
    """Get the test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def test_images_dir(test_data_dir):
    """Get the test images directory."""
    return test_data_dir / "test_images"


@pytest.fixture(scope="session")
def expected_outputs_dir(test_data_dir):
    """Get the expected outputs directory."""
    return test_data_dir / "expected_outputs"


@pytest.fixture(scope="session")
def benchmark_data_dir(test_data_dir):
    """Get the benchmark data directory."""
    return test_data_dir / "benchmark_data"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class ImageComparison:
    """Utilities for comparing images in tests."""

    @staticmethod
    def calculate_ssim(img1: Image.Image, img2: Image.Image) -> float:
        """Calculate structural similarity index between two images."""
        # Convert to numpy arrays
        arr1 = np.array(img1.convert("RGB"))
        arr2 = np.array(img2.convert("RGB"))

        # Ensure same dimensions
        if arr1.shape != arr2.shape:
            return 0.0

        # Calculate SSIM for each channel and average
        ssim_values = []
        for i in range(3):  # RGB channels
            ssim_val = ssim(arr1[:, :, i], arr2[:, :, i], data_range=255)
            ssim_values.append(ssim_val)

        return np.mean(ssim_values)

    @staticmethod
    def calculate_mse(img1: Image.Image, img2: Image.Image) -> float:
        """Calculate mean squared error between two images."""
        arr1 = np.array(img1.convert("RGB"))
        arr2 = np.array(img2.convert("RGB"))

        if arr1.shape != arr2.shape:
            return float("inf")

        return np.mean((arr1 - arr2) ** 2)

    @staticmethod
    def perceptual_hash(img: Image.Image) -> str:
        """Calculate perceptual hash of an image."""
        # Simple perceptual hash implementation
        img_gray = img.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
        pixels = list(img_gray.getdata())
        avg = sum(pixels) / len(pixels)

        bits = []
        for pixel in pixels:
            bits.append("1" if pixel > avg else "0")

        return "".join(bits)

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """Calculate hamming distance between two hashes."""
        if len(hash1) != len(hash2):
            return max(len(hash1), len(hash2))

        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2, strict=False))


@pytest.fixture
def image_comparison():
    """Provide image comparison utilities."""
    return ImageComparison()


class PerformanceMonitor:
    """Monitor performance metrics during tests."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_usage = []

    def start(self):
        """Start monitoring."""
        self.start_time = time.perf_counter()

    def stop(self):
        """Stop monitoring."""
        self.end_time = time.perf_counter()

    @property
    def duration(self) -> float:
        """Get execution duration in seconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time


@pytest.fixture
def performance_monitor():
    """Provide performance monitoring."""
    return PerformanceMonitor()


class BenchmarkData:
    """Manage benchmark data for regression testing."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_file = data_dir / "benchmarks.json"
        self._data = self._load_data()

    def _load_data(self) -> dict[str, Any]:
        """Load existing benchmark data."""
        if self.data_file.exists():
            try:
                with open(self.data_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        return {}

    def save_data(self):
        """Save benchmark data to file."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.data_file, "w") as f:
            json.dump(self._data, f, indent=2)

    def record_benchmark(self, test_name: str, metric: str, value: float):
        """Record a benchmark metric."""
        if test_name not in self._data:
            self._data[test_name] = {}

        if metric not in self._data[test_name]:
            self._data[test_name][metric] = []

        self._data[test_name][metric].append({"value": value, "timestamp": time.time()})

    def get_baseline(self, test_name: str, metric: str) -> float | None:
        """Get baseline value for a metric."""
        if test_name in self._data and metric in self._data[test_name]:
            values = self._data[test_name][metric]
            if values:
                # Return median of last 5 runs as baseline
                recent_values = [v["value"] for v in values[-5:]]
                return np.median(recent_values)
        return None

    def check_regression(
        self, test_name: str, metric: str, current_value: float, threshold: float = 1.5
    ) -> tuple[bool, float | None]:
        """Check if current value indicates performance regression."""
        baseline = self.get_baseline(test_name, metric)
        if baseline is None:
            return False, None

        ratio = current_value / baseline
        is_regression = ratio > threshold
        return is_regression, baseline


@pytest.fixture(scope="session")
def benchmark_data(benchmark_data_dir):
    """Provide benchmark data management."""
    return BenchmarkData(benchmark_data_dir)


def assert_image_dimensions(image: Image.Image, expected_width: int, expected_height: int):
    """Assert image has expected dimensions."""
    assert image.size == (expected_width, expected_height), (
        f"Expected dimensions {expected_width}x{expected_height}, got {image.size}"
    )


def assert_image_mode(image: Image.Image, expected_mode: str):
    """Assert image has expected color mode."""
    assert image.mode == expected_mode, f"Expected mode {expected_mode}, got {image.mode}"


def assert_images_similar(
    img1: Image.Image, img2: Image.Image, ssim_threshold: float = 0.95, mse_threshold: float = 100.0
):
    """Assert two images are visually similar."""
    comparison = ImageComparison()

    ssim_score = comparison.calculate_ssim(img1, img2)
    mse_score = comparison.calculate_mse(img1, img2)

    assert ssim_score >= ssim_threshold, (
        f"SSIM score {ssim_score:.3f} below threshold {ssim_threshold}"
    )

    assert mse_score <= mse_threshold, f"MSE score {mse_score:.1f} above threshold {mse_threshold}"


def create_test_image(width: int, height: int, mode: str = "RGBA") -> Image.Image:
    """Create a test image with specified dimensions and mode."""
    if mode == "RGBA":
        # Create a gradient pattern
        img = Image.new(mode, (width, height))
        pixels = []
        for y in range(height):
            for x in range(width):
                r = int((x / width) * 255)
                g = int((y / height) * 255)
                b = int(((x + y) / (width + height)) * 255)
                a = 255
                pixels.append((r, g, b, a))
        img.putdata(pixels)
        return img
    else:
        # Simple solid color for other modes
        return Image.new(mode, (width, height), (128, 128, 128))


def hash_image(image: Image.Image) -> str:
    """Create a hash of an image for comparison."""
    # Convert to bytes and hash
    img_bytes = image.tobytes()
    return hashlib.md5(img_bytes).hexdigest()


# Pytest markers for test categorization
pytestmark = [
    pytest.mark.integration,
]


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance benchmarks")
    config.addinivalue_line("markers", "visual: marks tests as visual regression tests")
    config.addinivalue_line("markers", "slow: marks tests as slow-running")


def execute_pipeline_and_verify(pipeline, output_path):
    """Execute pipeline and verify output file exists.

    Args:
        pipeline: Pipeline instance to execute
        output_path: Path where output should be saved

    Returns:
        Path to output file
    """
    from pathlib import Path

    # Execute pipeline (returns ImageContext)
    context = pipeline.execute(str(output_path))

    # Verify output file was created
    output_path = Path(output_path)
    assert output_path.exists(), f"Output file not created: {output_path}"

    return output_path


def pytest_runtest_teardown(item, nextitem):
    """Clean up after each test."""
    # Could add cleanup logic here if needed
    pass
