"""GPU detection service using torch.cuda and pynvml.

This service detects NVIDIA GPU hardware and returns GPUDevice instances
with comprehensive hardware information.
"""

import logging

import pynvml
import torch
import torch.cuda

from kcuda_validate.models.gpu_device import GPUDevice

logger = logging.getLogger(__name__)


class GPUDetectionError(Exception):
    """Exception raised when GPU detection fails."""

    pass


class GPUDetector:
    """Service for detecting NVIDIA GPU hardware via CUDA.

    Uses torch.cuda for basic CUDA availability and PyNVML for detailed
    hardware metrics (memory, driver versions, etc.).
    """

    def __init__(self):
        """Initialize GPU detector."""
        self._nvml_initialized = False

    def detect(self, device_id: int = 0) -> GPUDevice:
        """Detect GPU hardware and return GPUDevice instance.

        Args:
            device_id: GPU device index to query (default 0 for primary GPU)

        Returns:
            GPUDevice instance with hardware information

        Raises:
            GPUDetectionError: If CUDA unavailable, no GPUs found, or detection fails
        """
        logger.info(f"Starting GPU detection for device {device_id}")

        # Log detailed CUDA environment info
        try:
            logger.info(f"PyTorch version: {torch.version.__version__}")
            logger.info(f"CUDA compiled version: {torch.version.cuda}")
            if hasattr(torch.version, "hip"):
                logger.info(f"HIP version: {torch.version.hip}")
        except Exception as e:
            logger.debug(f"Could not retrieve PyTorch version info: {e}")

        # Check CUDA availability
        if not torch.cuda.is_available():
            logger.error("CUDA is not available")
            logger.info("Checking CUDA availability details:")
            logger.info(f"  - torch.cuda.is_available(): {torch.cuda.is_available()}")
            logger.info(f"  - PyTorch CUDA compiled: {torch.version.cuda is not None}")
            raise GPUDetectionError(
                "CUDA is not available. Ensure NVIDIA GPU drivers are installed "
                "on Windows host and WSL2 GPU passthrough is enabled."
            )

        # Log CUDA runtime information
        try:
            cuda_version = torch.version.cuda
            logger.info(f"CUDA runtime version: {cuda_version}")
        except Exception as e:
            logger.debug(f"Could not get CUDA version: {e}")

        # Check device count
        device_count = torch.cuda.device_count()
        logger.info(f"Detected {device_count} CUDA device(s)")

        if device_count == 0:
            raise GPUDetectionError("No NVIDIA GPU devices detected.")

        if device_id >= device_count:
            raise GPUDetectionError(
                f"Device ID {device_id} out of range. Only {device_count} GPU(s) available."
            )

        logger.debug(f"Found {device_count} GPU device(s)")

        # Get basic info from torch.cuda
        gpu_name = torch.cuda.get_device_name(device_id)
        compute_cap = torch.cuda.get_device_capability(device_id)
        compute_capability = f"{compute_cap[0]}.{compute_cap[1]}"

        logger.info(f"Device {device_id}: {gpu_name}")
        logger.info(f"Compute capability: {compute_capability}")

        logger.debug(f"GPU name: {gpu_name}, compute capability: {compute_capability}")

        # Initialize NVML for detailed metrics
        try:
            if not self._nvml_initialized:
                pynvml.nvmlInit()
                self._nvml_initialized = True
                logger.debug("NVML initialized successfully")
        except Exception as e:
            raise GPUDetectionError(
                f"Failed to initialize NVML: {e}. Ensure NVIDIA drivers are properly installed."
            ) from e

        # Get detailed hardware info via NVML
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

            # Memory information
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_total_mb = memory_info.total // (1024 * 1024)
            vram_free_mb = memory_info.free // (1024 * 1024)

            # Driver version
            driver_version = pynvml.nvmlSystemGetDriverVersion()

            # CUDA version
            cuda_driver_version = pynvml.nvmlSystemGetCudaDriverVersion()
            cuda_major = cuda_driver_version // 1000
            cuda_minor = (cuda_driver_version % 1000) // 10
            cuda_version = f"{cuda_major}.{cuda_minor}"

            logger.info(
                f"GPU detected: {gpu_name}, "
                f"VRAM: {vram_total_mb}MB total / {vram_free_mb}MB free, "
                f"CUDA: {cuda_version}, Driver: {driver_version}"
            )

            # Create and validate GPUDevice
            gpu_device = GPUDevice(
                name=gpu_name,
                vram_total_mb=vram_total_mb,
                vram_free_mb=vram_free_mb,
                cuda_version=cuda_version,
                driver_version=driver_version,
                compute_capability=compute_capability,
                device_id=device_id,
            )

            return gpu_device

        except ValueError as e:
            # GPUDevice validation failed
            raise GPUDetectionError(f"GPU detected but failed validation: {e}") from e
        except Exception as e:
            # Catch all other errors during detection
            raise GPUDetectionError(f"Unexpected error during GPU detection: {e}") from e
