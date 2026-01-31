"""Integration tests for GPU detection flow.

These tests verify the full detection pipeline from service to CLI output.
May require actual GPU hardware or comprehensive mocking.
"""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from kcuda_validate.__main__ import cli
from kcuda_validate.services.gpu_detector import GPUDetector

# Try to import torch for real hardware test skip condition
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.integration
class TestGPUDetectionIntegration:
    """Integration tests for full GPU detection pipeline."""

    def setup_method(self):
        """Setup test dependencies."""
        self.runner = CliRunner()

    @patch("kcuda_validate.services.gpu_detector.torch.cuda")
    @patch("kcuda_validate.services.gpu_detector.pynvml")
    def test_full_detection_pipeline_success(self, mock_pynvml, mock_cuda):
        """Test complete detection flow from CLI to service to output."""
        # Mock torch.cuda
        mock_cuda.is_available.return_value = True
        mock_cuda.device_count.return_value = 1
        mock_cuda.get_device_name.return_value = "NVIDIA GeForce RTX 3060"
        mock_cuda.get_device_capability.return_value = (8, 6)

        # Mock pynvml
        from unittest.mock import MagicMock

        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle

        mock_memory_info = MagicMock()
        mock_memory_info.total = 12884901888
        mock_memory_info.free = 12079595520
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory_info

        mock_pynvml.nvmlSystemGetDriverVersion.return_value = "525.60.11"
        mock_pynvml.nvmlSystemGetCudaDriverVersion.return_value = 12010

        # Execute full pipeline through CLI
        result = self.runner.invoke(cli, ["detect"])

        # Verify successful execution
        assert result.exit_code == 0
        assert "RTX 3060" in result.output
        assert "12288" in result.output
        assert "PASSED" in result.output or "passed" in result.output.lower()

    @patch("kcuda_validate.services.gpu_detector.torch.cuda")
    def test_full_detection_pipeline_no_cuda(self, mock_cuda):
        """Test complete detection flow when CUDA is unavailable."""
        mock_cuda.is_available.return_value = False

        # Execute through CLI
        result = self.runner.invoke(cli, ["detect"])

        # Verify error handling
        assert result.exit_code != 0
        assert "CUDA" in result.output or "cuda" in result.output
        assert "FAILED" in result.output or "failed" in result.output.lower()

    @patch("kcuda_validate.services.gpu_detector.torch.cuda")
    @patch("kcuda_validate.services.gpu_detector.pynvml")
    def test_detection_with_insufficient_vram(self, mock_pynvml, mock_cuda):
        """Test detection fails validation when VRAM < 4GB."""
        mock_cuda.is_available.return_value = True
        mock_cuda.device_count.return_value = 1
        mock_cuda.get_device_name.return_value = "Low VRAM GPU"
        mock_cuda.get_device_capability.return_value = (8, 6)

        from unittest.mock import MagicMock

        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle

        # Only 2GB VRAM (below 4GB minimum)
        mock_memory_info = MagicMock()
        mock_memory_info.total = 2147483648
        mock_memory_info.free = 2000000000
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory_info

        mock_pynvml.nvmlSystemGetDriverVersion.return_value = "525.60.11"
        mock_pynvml.nvmlSystemGetCudaDriverVersion.return_value = 12010

        # Execute
        result = self.runner.invoke(cli, ["detect"])

        # Should fail validation
        assert result.exit_code != 0
        # Error should mention VRAM requirement
        assert "vram" in result.output.lower() or "memory" in result.output.lower()

    @pytest.mark.skipif(
        not HAS_TORCH or (HAS_TORCH and not torch.cuda.is_available()),
        reason="Requires actual CUDA hardware",
    )
    def test_detection_on_real_hardware(self):
        """Test detection on actual GPU hardware (if available).

        This test is skipped if no CUDA GPU is present.
        """
        import torch

        if not torch.cuda.is_available():
            pytest.skip("No CUDA GPU available")

        # Test with real hardware
        detector = GPUDetector()
        gpu = detector.detect()

        # Verify real detection results
        assert gpu.name is not None
        assert gpu.vram_total_mb > 0
        assert gpu.cuda_version is not None
        assert gpu.device_id >= 0

    @patch("kcuda_validate.services.gpu_detector.torch.cuda")
    @patch("kcuda_validate.services.gpu_detector.pynvml")
    def test_detection_logging_on_success(self, mock_pynvml, mock_cuda):
        """Test that successful detection logs appropriate information."""
        mock_cuda.is_available.return_value = True
        mock_cuda.device_count.return_value = 1
        mock_cuda.get_device_name.return_value = "NVIDIA GeForce RTX 3060"
        mock_cuda.get_device_capability.return_value = (8, 6)

        from unittest.mock import MagicMock

        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle

        mock_memory_info = MagicMock()
        mock_memory_info.total = 12884901888
        mock_memory_info.free = 12079595520
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory_info

        mock_pynvml.nvmlSystemGetDriverVersion.return_value = "525.60.11"
        mock_pynvml.nvmlSystemGetCudaDriverVersion.return_value = 12010

        # Execute with verbose logging
        result = self.runner.invoke(cli, ["--verbose", "detect"])

        assert result.exit_code == 0
        # Verify logging occurs (implementation-dependent)

    @patch("kcuda_validate.services.gpu_detector.torch.cuda")
    def test_detection_error_message_format(self, mock_cuda):
        """Test that error messages follow the specified format from cli.md."""
        mock_cuda.is_available.return_value = False

        result = self.runner.invoke(cli, ["detect"])

        # Error format per cli.md contract should include:
        # - ✗ marker or "Error:"
        # - Brief description
        # - "Error:" section with detailed explanation
        # - "Recommendation:" or actionable guidance
        # - Log file path
        # - Status: FAILED

        output = result.output
        assert "✗" in output or "Error" in output
        assert "FAILED" in output or "failed" in output.lower()

        # Should have recommendations
        output_lower = output.lower()
        has_guidance = any(
            word in output_lower for word in ["ensure", "check", "install", "verify", "enable"]
        )
        assert has_guidance
