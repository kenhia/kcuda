"""Performance metrics collection utilities."""

import time
from collections.abc import Generator
from contextlib import contextmanager, suppress

try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class MetricsCollector:
    """Collect performance metrics during operations."""

    def __init__(self, device_id: int = 0):
        """
        Initialize metrics collector.

        Args:
            device_id: GPU device ID to monitor
        """
        self.device_id = device_id
        self.nvml_initialized = False
        self.handle = None

        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                self.nvml_initialized = True
            except Exception:
                self.nvml_initialized = False

    def get_gpu_memory_usage(self) -> int | None:
        """
        Get current GPU memory usage in MB.

        Returns:
            Memory usage in MB, or None if unavailable
        """
        if not self.nvml_initialized or not self.handle:
            return None

        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return mem_info.used // (1024 * 1024)  # Convert to MB
        except Exception:
            return None

    def get_gpu_utilization(self) -> float | None:
        """
        Get current GPU utilization percentage.

        Returns:
            Utilization percentage (0-100), or None if unavailable
        """
        if not self.nvml_initialized or not self.handle:
            return None

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            return float(util.gpu)
        except Exception:
            return None

    @contextmanager
    def measure_time(self) -> Generator[dict[str, float], None, None]:
        """
        Context manager to measure execution time.

        Yields:
            Dictionary that will be populated with timing data
        """
        timing = {"start_time": 0.0, "end_time": 0.0, "elapsed_sec": 0.0}
        timing["start_time"] = time.time()

        try:
            yield timing
        finally:
            timing["end_time"] = time.time()
            timing["elapsed_sec"] = timing["end_time"] - timing["start_time"]

    def shutdown(self) -> None:
        """Cleanup NVML resources."""
        if self.nvml_initialized:
            with suppress(Exception):
                pynvml.nvmlShutdown()


@contextmanager
def measure_inference_time() -> Generator[dict[str, float], None, None]:
    """
    Measure inference timing with first token detection.

    Yields:
        Dictionary with timing measurements
    """
    timing = {
        "start_time": time.time(),
        "first_token_time": None,
        "end_time": None,
        "time_to_first_token_sec": 0.0,
        "total_time_sec": 0.0,
    }

    try:
        yield timing
    finally:
        timing["end_time"] = time.time()

        if timing["first_token_time"] is not None:
            timing["time_to_first_token_sec"] = timing["first_token_time"] - timing["start_time"]

        timing["total_time_sec"] = timing["end_time"] - timing["start_time"]


def calculate_tokens_per_second(tokens: int, elapsed_sec: float) -> float:
    """
    Calculate tokens per second throughput.

    Args:
        tokens: Number of tokens generated
        elapsed_sec: Time elapsed in seconds

    Returns:
        Tokens per second
    """
    if elapsed_sec <= 0:
        return 0.0
    return tokens / elapsed_sec
