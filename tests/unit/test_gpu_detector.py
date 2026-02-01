"""Unit tests for GPU detection service."""

from unittest.mock import MagicMock, patch

import pytest

from kcuda_validate.services.gpu_detector import GPUDetectionError, GPUDetector


class TestGPUDetector:
    """Test GPU detection service with mocked torch.cuda and pynvml."""

    @patch("kcuda_validate.services.gpu_detector.torch.cuda")
    @patch("kcuda_validate.services.gpu_detector.pynvml")
    def test_detect_gpu_success(self, mock_pynvml, mock_cuda):
        """Test successful GPU detection with CUDA available."""
        # Mock torch.cuda
        mock_cuda.is_available.return_value = True
        mock_cuda.device_count.return_value = 1
        mock_cuda.get_device_name.return_value = "NVIDIA GeForce RTX 3060"
        mock_cuda.get_device_capability.return_value = (8, 6)

        # Mock pynvml
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle

        mock_memory_info = MagicMock()
        mock_memory_info.total = 12884901888  # 12288 MB in bytes
        mock_memory_info.free = 12079595520  # 11520 MB in bytes
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory_info

        mock_pynvml.nvmlSystemGetDriverVersion.return_value = "525.60.11"
        mock_pynvml.nvmlSystemGetCudaDriverVersion.return_value = 12010  # 12.1

        # Execute
        detector = GPUDetector()
        gpu_device = detector.detect()

        # Assert
        assert gpu_device.name == "NVIDIA GeForce RTX 3060"
        assert gpu_device.vram_total_mb == 12288
        assert gpu_device.vram_free_mb == 11520
        assert gpu_device.cuda_version.startswith("12.")
        assert gpu_device.driver_version == "525.60.11"
        assert gpu_device.compute_capability == "8.6"
        assert gpu_device.device_id == 0

    @patch("kcuda_validate.services.gpu_detector.torch.cuda")
    def test_detect_no_cuda_available(self, mock_cuda):
        """Test detection fails when CUDA is not available."""
        mock_cuda.is_available.return_value = False

        detector = GPUDetector()

        with pytest.raises(GPUDetectionError, match="CUDA.*not available"):
            detector.detect()

    @patch("kcuda_validate.services.gpu_detector.torch.cuda")
    def test_detect_no_gpu_devices(self, mock_cuda):
        """Test detection fails when no GPU devices found."""
        mock_cuda.is_available.return_value = True
        mock_cuda.device_count.return_value = 0

        detector = GPUDetector()

        with pytest.raises(GPUDetectionError, match="No.*GPU.*detected"):
            detector.detect()

    @patch("kcuda_validate.services.gpu_detector.torch.cuda")
    @patch("kcuda_validate.services.gpu_detector.pynvml")
    def test_detect_multi_gpu_selects_first(self, mock_pynvml, mock_cuda):
        """Test that detection selects device 0 when multiple GPUs present."""
        mock_cuda.is_available.return_value = True
        mock_cuda.device_count.return_value = 2  # Multiple GPUs
        mock_cuda.get_device_name.return_value = "NVIDIA GeForce RTX 3060"
        mock_cuda.get_device_capability.return_value = (8, 6)

        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle

        mock_memory_info = MagicMock()
        mock_memory_info.total = 12884901888
        mock_memory_info.free = 12079595520
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory_info
        mock_pynvml.nvmlSystemGetDriverVersion.return_value = "525.60.11"
        mock_pynvml.nvmlSystemGetCudaDriverVersion.return_value = 12010

        detector = GPUDetector()
        gpu_device = detector.detect()

        # Should use device 0
        assert gpu_device.device_id == 0
        mock_pynvml.nvmlDeviceGetHandleByIndex.assert_called_with(0)

    @patch("kcuda_validate.services.gpu_detector.torch.cuda")
    @patch("kcuda_validate.services.gpu_detector.pynvml")
    def test_detect_specific_device_id(self, mock_pynvml, mock_cuda):
        """Test detection of specific GPU device by ID."""
        mock_cuda.is_available.return_value = True
        mock_cuda.device_count.return_value = 2
        mock_cuda.get_device_name.return_value = "NVIDIA GeForce RTX 3090"
        mock_cuda.get_device_capability.return_value = (8, 6)

        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle

        mock_memory_info = MagicMock()
        mock_memory_info.total = 25769803776  # 24GB
        mock_memory_info.free = 25000000000
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory_info
        mock_pynvml.nvmlSystemGetDriverVersion.return_value = "525.60.11"
        mock_pynvml.nvmlSystemGetCudaDriverVersion.return_value = 12010

        detector = GPUDetector()
        gpu_device = detector.detect(device_id=1)

        # Should use device 1
        assert gpu_device.device_id == 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.assert_called_with(1)

    @patch("kcuda_validate.services.gpu_detector.torch.cuda")
    @patch("kcuda_validate.services.gpu_detector.pynvml")
    def test_detect_pynvml_error_handling(self, mock_pynvml, mock_cuda):
        """Test graceful handling of pynvml errors."""
        mock_cuda.is_available.return_value = True
        mock_cuda.device_count.return_value = 1
        mock_cuda.get_device_name.return_value = "NVIDIA GPU"
        mock_cuda.get_device_capability.return_value = (8, 6)

        # Simulate pynvml initialization failure
        mock_pynvml.nvmlInit.side_effect = Exception("NVML not available")

        detector = GPUDetector()

        with pytest.raises(GPUDetectionError, match="NVML"):
            detector.detect()

    @patch("kcuda_validate.services.gpu_detector.torch.cuda")
    @patch("kcuda_validate.services.gpu_detector.pynvml")
    def test_detect_old_compute_capability_warning(self, mock_pynvml, mock_cuda):
        """Test that old compute capability (< 6.0) raises validation error."""
        mock_cuda.is_available.return_value = True
        mock_cuda.device_count.return_value = 1
        mock_cuda.get_device_name.return_value = "Old NVIDIA GPU"
        mock_cuda.get_device_capability.return_value = (5, 2)  # Maxwell (too old)

        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle

        mock_memory_info = MagicMock()
        mock_memory_info.total = 8589934592
        mock_memory_info.free = 8000000000
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory_info
        mock_pynvml.nvmlSystemGetDriverVersion.return_value = "525.60.11"
        mock_pynvml.nvmlSystemGetCudaDriverVersion.return_value = 12010

        detector = GPUDetector()

        # Should raise error due to GPUDevice validation
        with pytest.raises((GPUDetectionError, ValueError)):
            detector.detect()
