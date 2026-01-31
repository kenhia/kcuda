"""Integration tests for model lifecycle (download and load).

These tests verify the full model download → load pipeline.
"""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from kcuda_validate.__main__ import cli


@pytest.mark.integration
class TestModelLifecycleIntegration:
    """Integration tests for full model download and load pipeline."""

    def setup_method(self):
        """Setup test dependencies."""
        self.runner = CliRunner()
        self.default_repo = "Ttimofeyka/MistralRP-Noromaid-NSFW-Mistral-7B-GGUF"
        self.default_filename = "mistralrp-noromaid-nsfw-mistral-7b.Q4_K_M.gguf"
        self.default_load_args = [
            "load",
            "--repo-id",
            self.default_repo,
            "--filename",
            self.default_filename,
        ]

    @patch("kcuda_validate.services.model_loader.hf_hub_download")
    @patch("kcuda_validate.services.model_loader.Path.stat")
    @patch("kcuda_validate.services.model_loader.Path.exists")
    @patch("kcuda_validate.services.model_loader.Llama")
    @patch("kcuda_validate.services.model_loader.torch.cuda")
    def test_full_model_load_pipeline_success(
        self, mock_cuda, mock_llama_class, mock_exists, mock_stat, mock_download
    ):
        """Test complete download → load flow through CLI."""
        # Mock CUDA availability
        mock_cuda.is_available.return_value = True
        mock_cuda.mem_get_info.return_value = (
            8000 * 1024 * 1024,
            12000 * 1024 * 1024,
        )

        # Mock download
        mock_path = "/home/user/.cache/huggingface/hub/model.gguf"
        mock_download.return_value = mock_path

        # Mock file operations
        mock_exists.return_value = True
        mock_stat_result = MagicMock()
        mock_stat_result.st_size = 4168 * 1024 * 1024
        mock_stat.return_value = mock_stat_result

        # Mock Llama loading
        mock_llama_instance = MagicMock()
        mock_llama_instance.n_ctx.return_value = 8192
        mock_llama_class.return_value = mock_llama_instance

        # Execute through CLI
        result = self.runner.invoke(cli, self.default_load_args)

        # Debug output
        if result.exit_code != 0:
            print(f"\nExit code: {result.exit_code}")
            print(f"Output: {result.output}")
            if result.exception:
                print(f"Exception: {result.exception}")
                import traceback

                traceback.print_exception(
                    type(result.exception), result.exception, result.exception.__traceback__
                )

        # Should succeed
        assert result.exit_code == 0
        assert "passed" in result.output.lower()

    @patch("kcuda_validate.services.model_loader.hf_hub_download")
    def test_download_failure_handling(self, mock_download):
        """Test graceful handling of download failures."""
        mock_download.side_effect = Exception("Network timeout")

        result = self.runner.invoke(cli, self.default_load_args)

        # Should fail with exit code 1
        assert result.exit_code == 1
        assert "download" in result.output.lower() or "failed" in result.output.lower()

    @patch("kcuda_validate.services.model_loader.hf_hub_download")
    @patch("kcuda_validate.services.model_loader.torch.cuda")
    def test_insufficient_vram_detection(self, mock_cuda, mock_download):
        """Test VRAM insufficiency detection before load."""
        mock_download.return_value = "/path/to/model.gguf"

        # Mock insufficient VRAM (only 2GB available)
        mock_cuda.is_available.return_value = True
        mock_cuda.mem_get_info.return_value = (
            2000 * 1024 * 1024,
            12000 * 1024 * 1024,
        )

        result = self.runner.invoke(cli, self.default_load_args)

        # Should fail with exit code 2
        assert result.exit_code == 2
        assert "vram" in result.output.lower() or "memory" in result.output.lower()

    @patch("kcuda_validate.services.model_loader.hf_hub_download")
    @patch("kcuda_validate.services.model_loader.Path.stat")
    @patch("kcuda_validate.services.model_loader.Path.exists")
    @patch("kcuda_validate.services.model_loader.Llama")
    @patch("kcuda_validate.services.model_loader.torch.cuda")
    def test_custom_repo_and_filename(
        self, mock_cuda, mock_llama_class, mock_exists, mock_stat, mock_download
    ):
        """Test loading with custom repo-id and filename options."""
        custom_repo = "custom/model-repo"
        custom_filename = "custom-model.Q5_K_M.gguf"

        # Mock CUDA availability
        mock_cuda.is_available.return_value = True
        mock_cuda.mem_get_info.return_value = (
            8000 * 1024 * 1024,
            12000 * 1024 * 1024,
        )

        mock_path = f"/cache/{custom_filename}"
        mock_download.return_value = mock_path

        mock_exists.return_value = True
        mock_stat_result = MagicMock()
        mock_stat_result.st_size = 5000 * 1024 * 1024
        mock_stat.return_value = mock_stat_result

        mock_llama_instance = MagicMock()
        mock_llama_instance.n_ctx.return_value = 8192
        mock_llama_class.return_value = mock_llama_instance

        # Execute with custom options
        self.runner.invoke(cli, ["load", "--repo-id", custom_repo, "--filename", custom_filename])

        # Verify download was called with custom parameters
        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args[1]
        assert call_kwargs["repo_id"] == custom_repo
        assert call_kwargs["filename"] == custom_filename

    @patch("kcuda_validate.services.model_loader.hf_hub_download")
    @patch("kcuda_validate.services.model_loader.Path.stat")
    @patch("kcuda_validate.services.model_loader.Path.exists")
    @patch("kcuda_validate.services.model_loader.Llama")
    def test_skip_download_option(self, mock_llama_class, mock_exists, mock_stat, mock_download):
        """Test --skip-download option skips download step."""
        # Mock file already exists locally
        mock_exists.return_value = True
        mock_stat_result = MagicMock()
        mock_stat_result.st_size = 4168 * 1024 * 1024
        mock_stat.return_value = mock_stat_result

        mock_llama_instance = MagicMock()
        mock_llama_instance.n_ctx.return_value = 8192
        mock_llama_class.return_value = mock_llama_instance

        self.runner.invoke(
            cli,
            [
                "load",
                "--repo-id",
                self.default_repo,
                "--filename",
                self.default_filename,
                "--skip-download",
            ],
        )

        # Download should not be called
        mock_download.assert_not_called()

    @patch("kcuda_validate.services.model_loader.hf_hub_download")
    @patch("kcuda_validate.services.model_loader.Path.stat")
    @patch("kcuda_validate.services.model_loader.Path.exists")
    @patch("kcuda_validate.services.model_loader.Llama")
    def test_no_gpu_option_cpu_mode(self, mock_llama_class, mock_exists, mock_stat, mock_download):
        """Test --no-gpu option loads model in CPU mode."""
        mock_path = "/path/to/model.gguf"
        mock_download.return_value = mock_path

        mock_exists.return_value = True
        mock_stat_result = MagicMock()
        mock_stat_result.st_size = 4168 * 1024 * 1024
        mock_stat.return_value = mock_stat_result

        mock_llama_instance = MagicMock()
        mock_llama_instance.n_ctx.return_value = 8192
        mock_llama_class.return_value = mock_llama_instance

        self.runner.invoke(
            cli,
            [
                "load",
                "--repo-id",
                self.default_repo,
                "--filename",
                self.default_filename,
                "--no-gpu",
            ],
        )

        # Verify Llama was called with n_gpu_layers=0 for CPU mode
        mock_llama_class.assert_called_once()
        call_kwargs = mock_llama_class.call_args[1]
        assert call_kwargs["n_gpu_layers"] == 0

    @patch("kcuda_validate.services.model_loader.hf_hub_download")
    @patch("kcuda_validate.services.model_loader.Path.exists")
    @patch("kcuda_validate.services.model_loader.Llama")
    def test_invalid_gguf_file_handling(self, mock_llama_class, mock_exists, mock_download):
        """Test handling of invalid/corrupt GGUF files."""
        mock_path = "/path/to/corrupt.gguf"
        mock_download.return_value = mock_path
        mock_exists.return_value = True

        # Llama raises error for invalid file
        mock_llama_class.side_effect = ValueError("Invalid GGUF format")

        result = self.runner.invoke(cli, self.default_load_args)

        # Should fail with exit code 2 for load errors (corrupt file)
        assert result.exit_code == 2
        assert "invalid" in result.output.lower() or "corrupt" in result.output.lower()

    @patch("kcuda_validate.services.model_loader.hf_hub_download")
    @patch("kcuda_validate.services.model_loader.Path.stat")
    @patch("kcuda_validate.services.model_loader.Path.exists")
    @patch("kcuda_validate.services.model_loader.Llama")
    @patch("kcuda_validate.services.model_loader.torch.cuda")
    def test_model_metadata_displayed(
        self, mock_cuda, mock_llama_class, mock_exists, mock_stat, mock_download
    ):
        """Test that model metadata is displayed in output."""
        mock_cuda.is_available.return_value = True
        mock_cuda.mem_get_info.return_value = (
            8000 * 1024 * 1024,
            12000 * 1024 * 1024,
        )

        mock_download.return_value = "/path/to/model.Q4_K_M.gguf"

        mock_exists.return_value = True
        mock_stat_result = MagicMock()
        mock_stat_result.st_size = 4168 * 1024 * 1024
        mock_stat.return_value = mock_stat_result

        mock_llama_instance = MagicMock()
        mock_llama_instance.n_ctx.return_value = 8192
        mock_llama_class.return_value = mock_llama_instance

        result = self.runner.invoke(cli, self.default_load_args)

        output_lower = result.output.lower()
        # Should display key metadata
        assert "q4_k_m" in output_lower  # Quantization type
        assert "8192" in result.output  # Context length
        assert "vram" in output_lower  # VRAM usage
