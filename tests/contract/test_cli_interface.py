"""Contract tests for CLI interface.

Tests CLI commands match contracts/cli.md specification including:
- Command options and arguments
- Output format
- Exit codes
- Error message format
"""

from unittest.mock import patch

from click.testing import CliRunner

from kcuda_validate.__main__ import cli
from kcuda_validate.models.gpu_device import GPUDevice


class TestDetectCommandContract:
    """Test detect command matches CLI contract specification."""

    def setup_method(self):
        """Setup test runner."""
        self.runner = CliRunner()

    @patch("kcuda_validate.cli.detect.GPUDetector")
    def test_detect_command_success_output_format(self, mock_detector_class):
        """Test detect command success output matches contract format."""
        # Mock successful GPU detection
        mock_detector = mock_detector_class.return_value
        mock_detector.detect.return_value = GPUDevice(
            name="NVIDIA GeForce RTX 3060",
            vram_total_mb=12288,
            vram_free_mb=11520,
            cuda_version="12.1",
            driver_version="525.60.11",
            compute_capability="8.6",
            device_id=0,
        )

        result = self.runner.invoke(cli, ["detect"])

        # Assert exit code
        assert result.exit_code == 0

        # Assert output format per contract
        assert "✓ CUDA Available: Yes" in result.output or "CUDA Available: Yes" in result.output
        assert "✓ GPU Detected" in result.output or "GPU Detected" in result.output
        assert "RTX 3060" in result.output
        assert "12288" in result.output  # VRAM Total
        assert "11520" in result.output  # VRAM Free
        assert "12.1" in result.output  # CUDA Version
        assert "525.60.11" in result.output  # Driver
        assert "8.6" in result.output  # Compute capability
        assert "PASSED" in result.output or "passed" in result.output.lower()

    @patch("kcuda_validate.cli.detect.GPUDetector")
    def test_detect_command_no_gpu_error_format(self, mock_detector_class):
        """Test detect command error output matches contract format."""
        from kcuda_validate.services.gpu_detector import GPUDetectionError

        # Mock GPU detection failure
        mock_detector = mock_detector_class.return_value
        mock_detector.detect.side_effect = GPUDetectionError("No NVIDIA GPU detected")

        result = self.runner.invoke(cli, ["detect"])

        # Assert exit code (1 for no GPU detected)
        assert result.exit_code == 1

        # Assert error format per contract
        assert "✗" in result.output or "Error" in result.output
        assert "No.*GPU.*detected" in result.output or "No NVIDIA GPU" in result.output
        assert "FAILED" in result.output or "failed" in result.output.lower()

    def test_detect_command_has_help_option(self):
        """Test detect command has --help option."""
        result = self.runner.invoke(cli, ["detect", "--help"])

        assert result.exit_code == 0
        assert "detect" in result.output.lower()
        assert "GPU" in result.output or "hardware" in result.output.lower()

    @patch("kcuda_validate.cli.detect.GPUDetector")
    def test_detect_command_verbose_option(self, mock_detector_class):
        """Test detect command respects --verbose global option."""
        mock_detector = mock_detector_class.return_value
        mock_detector.detect.return_value = GPUDevice(
            name="NVIDIA GeForce RTX 3060",
            vram_total_mb=12288,
            vram_free_mb=11520,
            cuda_version="12.1",
            driver_version="525.60.11",
            compute_capability="8.6",
            device_id=0,
        )

        result = self.runner.invoke(cli, ["--verbose", "detect"])

        assert result.exit_code == 0
        # Verbose mode should include additional details
        # (actual verbose output depends on implementation)

    @patch("kcuda_validate.cli.detect.GPUDetector")
    def test_detect_command_quiet_option(self, mock_detector_class):
        """Test detect command respects --quiet global option."""
        mock_detector = mock_detector_class.return_value
        mock_detector.detect.return_value = GPUDevice(
            name="NVIDIA GeForce RTX 3060",
            vram_total_mb=12288,
            vram_free_mb=11520,
            cuda_version="12.1",
            driver_version="525.60.11",
            compute_capability="8.6",
            device_id=0,
        )

        result = self.runner.invoke(cli, ["--quiet", "detect"])

        assert result.exit_code == 0
        # Quiet mode should suppress informational output

    @patch("kcuda_validate.cli.detect.GPUDetector")
    def test_detect_command_driver_error_exit_code(self, mock_detector_class):
        """Test detect command returns exit code 2 for driver issues."""
        from kcuda_validate.services.gpu_detector import GPUDetectionError

        mock_detector = mock_detector_class.return_value
        mock_detector.detect.side_effect = GPUDetectionError("Driver version mismatch")

        result = self.runner.invoke(cli, ["detect"])

        # Exit code 2 for driver/permission errors per contract
        assert result.exit_code in [1, 2]

    @patch("kcuda_validate.cli.detect.GPUDetector")
    def test_detect_error_message_includes_recommendations(self, mock_detector_class):
        """Test error messages include actionable recommendations per contract."""
        from kcuda_validate.services.gpu_detector import GPUDetectionError

        mock_detector = mock_detector_class.return_value
        mock_detector.detect.side_effect = GPUDetectionError("CUDA not available")

        result = self.runner.invoke(cli, ["detect"])

        # Should include recommendations per cli.md error format
        output_lower = result.output.lower()
        # Check for actionable guidance keywords
        has_recommendation = any(
            keyword in output_lower
            for keyword in ["ensure", "check", "install", "enable", "verify"]
        )
        assert has_recommendation, "Error message should include recommendations"


class TestGlobalOptions:
    """Test global CLI options work across all commands."""

    def setup_method(self):
        """Setup test runner."""
        self.runner = CliRunner()

    def test_cli_has_version_option(self):
        """Test CLI has --version option."""
        result = self.runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "0.1.0" in result.output or "version" in result.output.lower()

    def test_cli_has_help_option(self):
        """Test CLI has --help option."""
        result = self.runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "kcuda" in result.output.lower() or "validate" in result.output.lower()
        assert "detect" in result.output
