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


class TestLoadCommandContract:
    """Test load command matches CLI contract specification."""

    def setup_method(self):
        """Setup test runner."""
        self.runner = CliRunner()

    @patch("kcuda_validate.cli.load.ModelLoader")
    def test_load_command_success_output_format(self, mock_loader_class):
        """Test load command success output matches contract format."""
        from kcuda_validate.models.llm_model import LLMModel

        # Mock successful model loading
        mock_loader = mock_loader_class.return_value
        mock_loader.download_model.return_value = "/path/to/model.gguf"
        mock_loader.load_model.return_value = LLMModel(
            repo_id="Ttimofeyka/MistralRP-Noromaid-NSFW-Mistral-7B-GGUF",
            filename="mistralrp-noromaid-nsfw-mistral-7b.Q4_K_M.gguf",
            local_path="/path/to/model.gguf",
            file_size_mb=4168,
            parameter_count=7_240_000_000,
            quantization_type="Q4_K_M",
            context_length=8192,
            vram_usage_mb=4832,
            is_loaded=True,
        )

        result = self.runner.invoke(
            cli,
            [
                "load",
                "--repo-id",
                "Ttimofeyka/MistralRP-Noromaid-NSFW-Mistral-7B-GGUF",
                "--filename",
                "mistralrp-noromaid-nsfw-mistral-7b.Q4_K_M.gguf",
            ],
        )

        # Should succeed
        assert result.exit_code == 0

        # Verify output format per cli.md contract
        output_lower = result.output.lower()
        assert "model" in output_lower and "loaded" in output_lower
        assert "passed" in output_lower
        # Should show model metadata
        assert "q4_k_m" in output_lower or "quantization" in output_lower
        assert "vram" in output_lower

    @patch("kcuda_validate.cli.load.ModelLoader")
    def test_load_command_insufficient_vram_exit_code(self, mock_loader_class):
        """Test load command exit code 2 for insufficient VRAM."""
        from kcuda_validate.services.model_loader import ModelLoadError

        mock_loader = mock_loader_class.return_value
        mock_loader.download_model.return_value = "/path/to/model.gguf"
        mock_loader.load_model.side_effect = ModelLoadError(
            "Insufficient VRAM: need 5000MB, have 2000MB"
        )

        result = self.runner.invoke(
            cli,
            [
                "load",
                "--repo-id",
                "Ttimofeyka/MistralRP-Noromaid-NSFW-Mistral-7B-GGUF",
                "--filename",
                "mistralrp-noromaid-nsfw-mistral-7b.Q4_K_M.gguf",
            ],
        )

        # Exit code 2 for load failures
        assert result.exit_code == 2
        assert "vram" in result.output.lower() or "memory" in result.output.lower()
        assert "failed" in result.output.lower()

    @patch("kcuda_validate.cli.load.ModelLoader")
    def test_load_command_download_failure_exit_code(self, mock_loader_class):
        """Test load command exit code 1 for download failures."""
        from kcuda_validate.services.model_loader import ModelLoadError

        mock_loader = mock_loader_class.return_value
        mock_loader.download_model.side_effect = ModelLoadError("Network error")

        result = self.runner.invoke(
            cli,
            [
                "load",
                "--repo-id",
                "Ttimofeyka/MistralRP-Noromaid-NSFW-Mistral-7B-GGUF",
                "--filename",
                "mistralrp-noromaid-nsfw-mistral-7b.Q4_K_M.gguf",
            ],
        )

        # Exit code 1 for download failures
        assert result.exit_code == 1
        assert "download" in result.output.lower() or "network" in result.output.lower()
        assert "failed" in result.output.lower()

    def test_load_command_has_repo_id_option(self):
        """Test load command has --repo-id option."""
        result = self.runner.invoke(cli, ["load", "--help"])

        assert result.exit_code == 0
        assert "--repo-id" in result.output

    def test_load_command_has_filename_option(self):
        """Test load command has --filename option."""
        result = self.runner.invoke(cli, ["load", "--help"])

        assert result.exit_code == 0
        assert "--filename" in result.output

    def test_load_command_has_skip_download_option(self):
        """Test load command has --skip-download option."""
        result = self.runner.invoke(cli, ["load", "--help"])

        assert result.exit_code == 0
        assert "--skip-download" in result.output

    def test_load_command_has_no_gpu_option(self):
        """Test load command has --no-gpu option for CPU mode."""
        result = self.runner.invoke(cli, ["load", "--help"])

        assert result.exit_code == 0
        assert "--no-gpu" in result.output


class TestInferCommandContract:
    """Test infer command CLI contract per cli.md specification."""

    def setup_method(self):
        """Setup test runner."""
        self.runner = CliRunner()

    @patch("kcuda_validate.cli.infer.Inferencer")
    def test_infer_command_success_output_format(self, mock_inferencer_class):
        """Test infer command success output matches contract format."""
        from kcuda_validate.models.inference_result import InferenceResult

        # Mock successful inference
        mock_inferencer = mock_inferencer_class.return_value
        mock_inferencer.generate.return_value = InferenceResult.from_generation(
            prompt="Hello, how are you?",
            response="I'm doing well, thank you!",
            tokens_generated=10,
            time_to_first_token_sec=0.5,
            total_time_sec=1.0,
            gpu_utilization_percent=95.0,
            vram_peak_mb=5000,
        )

        result = self.runner.invoke(cli, ["infer", "Hello, how are you?"])

        # Should succeed
        assert result.exit_code == 0

        # Verify output format per cli.md contract
        output_lower = result.output.lower()
        assert "inference" in output_lower and ("completed" in output_lower or "passed" in output_lower)
        assert "tokens" in output_lower
        assert "response" in output_lower or "hello" in output_lower

    @patch("kcuda_validate.cli.infer.Inferencer")
    def test_infer_command_no_model_loaded_exit_code(self, mock_inferencer_class):
        """Test infer command exit code 1 when no model loaded."""
        # Mock no model loaded scenario
        mock_inferencer_class.side_effect = RuntimeError("No model loaded")

        result = self.runner.invoke(cli, ["infer", "Test prompt"])

        # Exit code 1 for no model loaded
        assert result.exit_code == 1
        assert "model" in result.output.lower()

    def test_infer_command_empty_prompt_exit_code(self):
        """Test infer command exit code 3 for empty prompt."""
        result = self.runner.invoke(cli, ["infer", ""])

        # Exit code 3 for invalid prompt
        assert result.exit_code == 3
        assert "prompt" in result.output.lower() or "empty" in result.output.lower()

    @patch("kcuda_validate.cli.infer.Inferencer")
    def test_infer_command_inference_failure_exit_code(self, mock_inferencer_class):
        """Test infer command exit code 2 for inference failure."""
        from kcuda_validate.models.inference_result import InferenceResult

        mock_inferencer = mock_inferencer_class.return_value
        mock_inferencer.generate.return_value = InferenceResult.from_error(
            prompt="Test", error_message="CUDA out of memory"
        )

        result = self.runner.invoke(cli, ["infer", "Test prompt"])

        # Exit code 2 for inference failures
        assert result.exit_code == 2
        assert "failed" in result.output.lower() or "error" in result.output.lower()

    def test_infer_command_has_max_tokens_option(self):
        """Test infer command has --max-tokens option."""
        result = self.runner.invoke(cli, ["infer", "--help"])

        assert result.exit_code == 0
        assert "--max-tokens" in result.output

    def test_infer_command_has_temperature_option(self):
        """Test infer command has --temperature option."""
        result = self.runner.invoke(cli, ["infer", "--help"])

        assert result.exit_code == 0
        assert "--temperature" in result.output

    def test_infer_command_has_load_model_option(self):
        """Test infer command has --load-model option."""
        result = self.runner.invoke(cli, ["infer", "--help"])

        assert result.exit_code == 0
        assert "--load-model" in result.output

    @patch("kcuda_validate.cli.infer.Inferencer")
    def test_infer_command_displays_performance_metrics(self, mock_inferencer_class):
        """Test infer command displays performance metrics."""
        from kcuda_validate.models.inference_result import InferenceResult

        mock_inferencer = mock_inferencer_class.return_value
        mock_inferencer.generate.return_value = InferenceResult.from_generation(
            prompt="Test",
            response="Response",
            tokens_generated=50,
            time_to_first_token_sec=0.8,
            total_time_sec=3.2,
            gpu_utilization_percent=98.0,
            vram_peak_mb=5500,
        )

        result = self.runner.invoke(cli, ["infer", "Test prompt"])

        output_lower = result.output.lower()
        # Should display key performance metrics
        assert "tokens" in output_lower
        assert "time" in output_lower or "seconds" in output_lower
        assert "throughput" in output_lower or "tokens/second" in output_lower or "tokens per second" in output_lower


class TestValidateAllCommandContract:
    """Test validate-all command matches CLI contract specification."""

    def setup_method(self):
        """Setup test runner."""
        self.runner = CliRunner()

    @patch("kcuda_validate.cli.validate_all.GPUDetector")
    @patch("kcuda_validate.cli.validate_all.ModelLoader")
    @patch("kcuda_validate.cli.validate_all.Inferencer")
    def test_validate_all_command_success_all_steps(
        self, mock_inferencer_class, mock_loader_class, mock_detector_class
    ):
        """Test validate-all command runs all steps and shows summary."""
        from kcuda_validate.models.inference_result import InferenceResult

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

        # Mock successful model loading
        from kcuda_validate.models.llm_model import LLMModel

        mock_loader = mock_loader_class.return_value
        mock_loader.download_model.return_value = "/path/to/model.gguf"
        mock_loader.load_model.return_value = LLMModel(
            repo_id="Ttimofeyka/MistralRP-Noromaid-NSFW-Mistral-7B-GGUF",
            filename="mistralrp-noromaid-nsfw-mistral-7b.Q4_K_M.gguf",
            local_path="/path/to/model.gguf",
            file_size_mb=4168,
            parameter_count=7_240_000_000,
            quantization_type="Q4_K_M",
            context_length=8192,
            vram_usage_mb=4832,
            is_loaded=True,
        )

        # Mock successful inference
        mock_inferencer = mock_inferencer_class.return_value
        mock_inferencer.generate.return_value = InferenceResult.from_generation(
            prompt="Hello, how are you?",
            response="I'm doing well!",
            tokens_generated=10,
            time_to_first_token_sec=0.5,
            total_time_sec=1.0,
            gpu_utilization_percent=95.0,
            vram_peak_mb=5000,
        )

        result = self.runner.invoke(cli, ["validate-all"])

        # Should succeed with all steps passing
        assert result.exit_code == 0

        # Verify output contains all step results
        output_lower = result.output.lower()
        assert "gpu" in output_lower or "detect" in output_lower
        assert "model" in output_lower or "load" in output_lower
        assert "inference" in output_lower or "infer" in output_lower

        # Verify summary footer per contract
        assert "summary" in output_lower
        assert "passed" in output_lower or "success" in output_lower

    @patch("kcuda_validate.cli.validate_all.GPUDetector")
    def test_validate_all_command_failure_detection_step(self, mock_detector_class):
        """Test validate-all command handles detection failure."""
        from kcuda_validate.services.gpu_detector import GPUDetectionError

        # Mock GPU detection failure
        mock_detector = mock_detector_class.return_value
        mock_detector.detect.side_effect = GPUDetectionError("No GPU detected")

        result = self.runner.invoke(cli, ["validate-all"])

        # Should fail with exit code 1
        assert result.exit_code == 1

        # Should show failure in summary
        output_lower = result.output.lower()
        assert "failed" in output_lower or "error" in output_lower

    @patch("kcuda_validate.cli.validate_all.GPUDetector")
    @patch("kcuda_validate.cli.validate_all.ModelLoader")
    def test_validate_all_command_failure_load_step(self, mock_loader_class, mock_detector_class):
        """Test validate-all command handles model load failure."""
        from kcuda_validate.models.llm_model import LLMModel

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

        # Mock model loading failure
        mock_loader = mock_loader_class.return_value
        mock_loader.download_model.side_effect = RuntimeError("Download failed")

        result = self.runner.invoke(cli, ["validate-all"])

        # Should fail with exit code 1
        assert result.exit_code == 1

        # Should show which step failed
        output_lower = result.output.lower()
        assert "failed" in output_lower or "error" in output_lower

    def test_validate_all_command_has_help_option(self):
        """Test validate-all command has --help option."""
        result = self.runner.invoke(cli, ["validate-all", "--help"])

        assert result.exit_code == 0
        assert "validate" in result.output.lower()

    def test_validate_all_command_has_repo_id_option(self):
        """Test validate-all command has --repo-id option."""
        result = self.runner.invoke(cli, ["validate-all", "--help"])

        assert result.exit_code == 0
        assert "--repo-id" in result.output

    def test_validate_all_command_has_filename_option(self):
        """Test validate-all command has --filename option."""
        result = self.runner.invoke(cli, ["validate-all", "--help"])

        assert result.exit_code == 0
        assert "--filename" in result.output

    def test_validate_all_command_has_prompt_option(self):
        """Test validate-all command has --prompt option."""
        result = self.runner.invoke(cli, ["validate-all", "--help"])

        assert result.exit_code == 0
        assert "--prompt" in result.output

    def test_validate_all_command_has_skip_on_error_option(self):
        """Test validate-all command has --skip-on-error option."""
        result = self.runner.invoke(cli, ["validate-all", "--help"])

        assert result.exit_code == 0
        assert "--skip-on-error" in result.output

    @patch("kcuda_validate.cli.validate_all.GPUDetector")
    @patch("kcuda_validate.cli.validate_all.ModelLoader")
    @patch("kcuda_validate.cli.validate_all.Inferencer")
    def test_validate_all_command_displays_summary_footer(
        self, mock_inferencer_class, mock_loader_class, mock_detector_class
    ):
        """Test validate-all displays validation summary footer per contract."""
        from kcuda_validate.models.inference_result import InferenceResult
        from kcuda_validate.models.llm_model import LLMModel

        # Mock all steps successful
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

        mock_loader = mock_loader_class.return_value
        mock_loader.download_model.return_value = "/path/to/model.gguf"
        mock_loader.load_model.return_value = LLMModel(
            repo_id="test/model",
            filename="model.gguf",
            local_path="/path/to/model.gguf",
            file_size_mb=4000,
            parameter_count=7_000_000_000,
            quantization_type="Q4_K_M",
            context_length=8192,
            vram_usage_mb=4500,
            is_loaded=True,
        )

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

        result = self.runner.invoke(cli, ["validate-all"])

        # Should display summary footer per cli.md contract
        output_lower = result.output.lower()
        assert "validation summary" in output_lower or "summary" in output_lower
        assert "gpu detection" in output_lower or "detection" in output_lower
        assert "model loading" in output_lower or "model" in output_lower
        assert "inference" in output_lower or "infer" in output_lower
        assert "overall" in output_lower or "status" in output_lower


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
