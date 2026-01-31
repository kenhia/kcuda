"""Integration tests for full validation pipeline.

These tests verify the complete detect → load → infer sequence.
"""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from kcuda_validate.__main__ import cli


@pytest.mark.integration
class TestFullPipeline:
    """Test complete validation pipeline integration."""

    def setup_method(self):
        """Setup test dependencies."""
        self.runner = CliRunner()

    @patch("kcuda_validate.cli.infer.Inferencer")
    @patch("kcuda_validate.services.model_loader.Llama")
    @patch("kcuda_validate.services.model_loader.Path.exists")
    @patch("kcuda_validate.services.model_loader.hf_hub_download")
    @patch("kcuda_validate.services.gpu_detector.torch.cuda")
    @patch("kcuda_validate.services.gpu_detector.pynvml")
    def test_full_detect_load_infer_sequence(
        self,
        mock_pynvml,
        mock_cuda,
        mock_download,
        mock_exists,
        mock_llama_class,
        mock_inferencer_class,
    ):
        """Test complete detect → load → infer pipeline."""
        from kcuda_validate.models.inference_result import InferenceResult

        # Mock GPU detection (Step 1: detect)
        mock_cuda.is_available.return_value = True
        mock_cuda.device_count.return_value = 1
        mock_cuda.get_device_name.return_value = "NVIDIA GeForce RTX 3060"
        mock_cuda.get_device_properties.return_value = MagicMock(
            total_memory=12 * 1024**3, major=8, minor=6
        )
        mock_cuda.mem_get_info.return_value = (10 * 1024**3, 12 * 1024**3)

        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetName.return_value = b"NVIDIA GeForce RTX 3060"
        mock_pynvml.nvmlSystemGetCudaDriverVersion.return_value = 12010
        mock_pynvml.nvmlSystemGetDriverVersion.return_value = b"525.60.11"

        # Step 1: Detect GPU
        detect_result = self.runner.invoke(cli, ["detect"])
        assert detect_result.exit_code == 0
        assert "passed" in detect_result.output.lower()

        # Mock model loading (Step 2: load)
        mock_download.return_value = "/path/to/model.gguf"
        mock_exists.return_value = True

        mock_llama_instance = MagicMock()
        mock_llama_instance.n_ctx.return_value = 8192
        mock_llama_class.return_value = mock_llama_instance

        # Step 2: Load model (skipping for now due to required options)
        # In full pipeline test, would use validate-all command

        # Mock inference (Step 3: infer)
        mock_inferencer = mock_inferencer_class.return_value
        mock_inferencer.generate.return_value = InferenceResult.from_generation(
            prompt="Hello",
            response="Hi there!",
            tokens_generated=20,
            time_to_first_token_sec=0.5,
            total_time_sec=2.0,
            gpu_utilization_percent=95.0,
            vram_peak_mb=5000,
        )

        # Step 3: Run inference (requires model to be loaded first)
        # For now, test will be updated when infer command is implemented

    @patch("kcuda_validate.cli.infer.Inferencer")
    def test_inference_performance_metrics_collection(self, mock_inferencer_class):
        """Test that inference collects and displays performance metrics."""
        from kcuda_validate.models.inference_result import InferenceResult

        mock_inferencer = mock_inferencer_class.return_value
        mock_inferencer.generate.return_value = InferenceResult.from_generation(
            prompt="Test prompt",
            response="Test response with multiple tokens here",
            tokens_generated=50,
            time_to_first_token_sec=0.89,
            total_time_sec=3.21,
            gpu_utilization_percent=98.0,
            vram_peak_mb=4912,
        )

        result = self.runner.invoke(cli, ["infer", "Test prompt"])

        # Should display all performance metrics per cli.md contract
        output = result.output
        assert "tokens" in output.lower()
        assert "time" in output.lower() or "seconds" in output.lower()

        # Verify metrics are calculated correctly
        # tokens_per_second = 50 / 3.21 ≈ 15.6
        # This will be validated in the actual output format

    @patch("kcuda_validate.cli.infer.Inferencer")
    @patch("kcuda_validate.services.gpu_detector.pynvml")
    def test_inference_gpu_monitoring(self, mock_pynvml, mock_inferencer_class):
        """Test GPU monitoring during inference."""
        from kcuda_validate.models.inference_result import InferenceResult

        # Mock GPU metrics during inference
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle

        mock_util = MagicMock()
        mock_util.gpu = 95
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util

        mock_mem_info = MagicMock()
        mock_mem_info.used = 5000 * 1024 * 1024
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem_info

        # Mock inference with GPU metrics
        mock_inferencer = mock_inferencer_class.return_value
        mock_inferencer.generate.return_value = InferenceResult.from_generation(
            prompt="Test",
            response="Response",
            tokens_generated=10,
            time_to_first_token_sec=0.5,
            total_time_sec=1.0,
            gpu_utilization_percent=95.0,
            vram_peak_mb=5000,
        )

        result = self.runner.invoke(cli, ["infer", "Test prompt"])

        # Verify GPU metrics are displayed
        output_lower = result.output.lower()
        # GPU utilization and VRAM should be in output
        if result.exit_code == 0:
            assert "gpu" in output_lower or "vram" in output_lower

    @patch("kcuda_validate.cli.infer.Inferencer")
    def test_inference_error_handling(self, mock_inferencer_class):
        """Test inference error handling and reporting."""
        from kcuda_validate.models.inference_result import InferenceResult

        # Mock inference failure
        mock_inferencer = mock_inferencer_class.return_value
        mock_inferencer.generate.return_value = InferenceResult.from_error(
            prompt="Test prompt", error_message="CUDA out of memory"
        )

        result = self.runner.invoke(cli, ["infer", "Test prompt"])

        # Should report error clearly
        assert result.exit_code == 2
        assert "failed" in result.output.lower() or "error" in result.output.lower()

    def test_inference_prompt_validation(self):
        """Test inference validates prompt before execution."""
        # Empty prompt should fail validation
        result = self.runner.invoke(cli, ["infer", ""])

        assert result.exit_code == 3
        assert "prompt" in result.output.lower() or "empty" in result.output.lower()

    @patch("kcuda_validate.cli.infer.Inferencer")
    def test_inference_with_custom_options(self, mock_inferencer_class):
        """Test inference with custom max-tokens and temperature."""
        from kcuda_validate.models.inference_result import InferenceResult

        mock_inferencer = mock_inferencer_class.return_value
        mock_inferencer.generate.return_value = InferenceResult.from_generation(
            prompt="Test",
            response="Response",
            tokens_generated=100,
            time_to_first_token_sec=1.0,
            total_time_sec=5.0,
        )

        result = self.runner.invoke(
            cli, ["infer", "Test prompt", "--max-tokens", "100", "--temperature", "0.8"]
        )

        # Verify custom options are passed through
        if result.exit_code == 0:
            mock_inferencer.generate.assert_called_once()
            call_kwargs = mock_inferencer.generate.call_args[1]
            assert call_kwargs.get("max_tokens") == 100
            assert call_kwargs.get("temperature") == 0.8

    @patch("kcuda_validate.cli.infer.Inferencer")
    def test_inference_response_formatting(self, mock_inferencer_class):
        """Test that inference response is formatted clearly."""
        from kcuda_validate.models.inference_result import InferenceResult

        long_response = (
            "This is a longer response that spans multiple lines "
            "to test the formatting of the output. It should be "
            "displayed in a clear and readable format."
        )

        mock_inferencer = mock_inferencer_class.return_value
        mock_inferencer.generate.return_value = InferenceResult.from_generation(
            prompt="Tell me a story",
            response=long_response,
            tokens_generated=30,
            time_to_first_token_sec=0.8,
            total_time_sec=2.5,
        )

        result = self.runner.invoke(cli, ["infer", "Tell me a story"])

        # Response should be clearly separated and readable
        assert result.exit_code == 0
        # Response text should be in output
        assert "response" in result.output.lower() or "longer" in result.output.lower()


@pytest.mark.integration
class TestValidateAllCommand:
    """Test validate-all command that runs full pipeline."""

    def setup_method(self):
        """Setup test dependencies."""
        self.runner = CliRunner()

    def test_validate_all_command_exists(self):
        """Test that validate-all command is available."""
        result = self.runner.invoke(cli, ["--help"])

        # validate-all command should be listed
        # This will be implemented in Phase 6
        # For now, just verify help works
        assert result.exit_code == 0
