"""Unit tests for model_loader service.

Tests model download and loading with mocked dependencies
(huggingface_hub and llama-cpp-python).
"""

from unittest.mock import MagicMock, patch

import pytest

from kcuda_validate.services.model_loader import (
    ModelLoader,
    ModelLoadError,
)


class TestModelLoader:
    """Test ModelLoader service with mocked dependencies."""

    def setup_method(self):
        """Setup test dependencies."""
        self.default_repo = "Ttimofeyka/MistralRP-Noromaid-NSFW-Mistral-7B-GGUF"
        self.default_filename = "MistralRP-Noromaid-NSFW-7B-Q4_0.gguf"

    @patch("kcuda_validate.services.model_loader.hf_hub_download")
    def test_download_model_success(self, mock_download):
        """Test successful model download from Hugging Face."""
        mock_path = "/home/user/.cache/huggingface/hub/models/model.gguf"
        mock_download.return_value = mock_path

        loader = ModelLoader()
        local_path = loader.download_model(self.default_repo, self.default_filename)

        assert local_path == mock_path
        mock_download.assert_called_once_with(
            repo_id=self.default_repo,
            filename=self.default_filename,
            repo_type="model",
        )

    @patch("kcuda_validate.services.model_loader.hf_hub_download")
    def test_download_model_network_error(self, mock_download):
        """Test model download with network error."""
        mock_download.side_effect = Exception("Network error")

        loader = ModelLoader()
        with pytest.raises(ModelLoadError, match="[Dd]ownload.*failed"):
            loader.download_model(self.default_repo, self.default_filename)

    @patch("kcuda_validate.services.model_loader.hf_hub_download")
    def test_download_model_file_not_found(self, mock_download):
        """Test model download when file doesn't exist in repo."""
        mock_download.side_effect = Exception("404")

        loader = ModelLoader()
        with pytest.raises(ModelLoadError, match="[Ff]ile not found|404"):
            loader.download_model(self.default_repo, "nonexistent.gguf")

    @patch("kcuda_validate.services.model_loader.torch.cuda")
    @patch("kcuda_validate.services.model_loader.Path.stat")
    @patch("kcuda_validate.services.model_loader.Llama")
    def test_load_model_success(self, mock_llama_class, mock_stat, mock_cuda):
        """Test successful model loading into GPU memory."""
        # Mock CUDA availability
        mock_cuda.is_available.return_value = True
        mock_cuda.mem_get_info.return_value = (10 * 1024**3, 12 * 1024**3)  # 10GB free, 12GB total
        
        # Mock file stats
        mock_stat_result = MagicMock()
        mock_stat_result.st_size = 4168 * 1024 * 1024  # 4168 MB in bytes
        mock_stat.return_value = mock_stat_result

        # Mock Llama model instance
        mock_llama_instance = MagicMock()
        mock_llama_instance.n_ctx.return_value = 8192
        mock_llama_class.return_value = mock_llama_instance

        loader = ModelLoader()
        model_path = "/path/to/model.Q4_K_M.gguf"
        llm_model = loader.load_model(
            local_path=model_path,
            repo_id=self.default_repo,
            filename=self.default_filename,
        )

        # Verify LLMModel attributes
        assert llm_model.repo_id == self.default_repo
        assert llm_model.filename == self.default_filename
        assert llm_model.local_path == model_path
        assert llm_model.file_size_mb == 4168
        assert llm_model.quantization_type == "Q4_0"  # Extracted from default_filename
        assert llm_model.context_length == 8192
        assert llm_model.is_loaded is True
        assert llm_model.vram_usage_mb > 0

        # Verify Llama was called with correct parameters
        mock_llama_class.assert_called_once()
        call_kwargs = mock_llama_class.call_args[1]
        assert call_kwargs["model_path"] == model_path
        assert call_kwargs["n_gpu_layers"] == -1  # All layers on GPU
        assert call_kwargs["verbose"] is False

    @patch("kcuda_validate.services.model_loader.torch.cuda")
    @patch("kcuda_validate.services.model_loader.Path.stat")
    @patch("kcuda_validate.services.model_loader.Path.exists")
    @patch("kcuda_validate.services.model_loader.Llama")
    def test_load_model_insufficient_vram(self, mock_llama_class, mock_exists, mock_stat, mock_cuda):
        """Test model loading failure due to insufficient VRAM."""
        # Mock CUDA with insufficient VRAM
        mock_cuda.is_available.return_value = True
        mock_cuda.mem_get_info.return_value = (1 * 1024**3, 4 * 1024**3)  # Only 1GB free
        
        mock_exists.return_value = True
        mock_stat.return_value.st_size = 4_000_000_000  # 4GB
        mock_llama_class.side_effect = RuntimeError("CUDA out of memory")

        loader = ModelLoader()
        with pytest.raises(ModelLoadError, match="[Ii]nsufficient.*[Vv][Rr][Aa][Mm]|memory"):
            loader.load_model(
                local_path="/path/to/model.gguf",
                repo_id=self.default_repo,
                filename=self.default_filename,
            )

    @patch("kcuda_validate.services.model_loader.Path.exists")
    def test_load_model_file_not_found(self, mock_exists):
        """Test model loading when file doesn't exist locally."""
        mock_exists.return_value = False

        loader = ModelLoader()
        with pytest.raises(ModelLoadError, match="[Ff]ile not found"):
            loader.load_model(
                local_path="/nonexistent/model.gguf",
                repo_id=self.default_repo,
                filename=self.default_filename,
            )

    @patch("kcuda_validate.services.model_loader.Path.exists")
    @patch("kcuda_validate.services.model_loader.Path.stat")
    def test_load_model_corrupt_file(self, mock_stat, mock_exists):
        """Test model loading with corrupt/invalid GGUF file."""
        mock_exists.return_value = True
        mock_stat.side_effect = Exception("Invalid file format")

        loader = ModelLoader()
        with pytest.raises(ModelLoadError, match="[Cc]orrupt|[Ii]nvalid"):
            loader.load_model(
                local_path="/path/to/corrupt.gguf",
                repo_id=self.default_repo,
                filename="corrupt.gguf",
            )

    @patch("kcuda_validate.services.model_loader.Path.stat")
    @patch("kcuda_validate.services.model_loader.Llama")
    def test_load_model_cpu_mode(self, mock_llama_class, mock_stat):
        """Test model loading in CPU mode (no GPU)."""
        mock_stat_result = MagicMock()
        mock_stat_result.st_size = 4168 * 1024 * 1024
        mock_stat.return_value = mock_stat_result

        mock_llama_instance = MagicMock()
        mock_llama_instance.n_ctx.return_value = 8192
        mock_llama_class.return_value = mock_llama_instance

        loader = ModelLoader()
        loader.load_model(
            local_path="/path/to/model.gguf",
            repo_id=self.default_repo,
            filename=self.default_filename,
            use_gpu=False,  # CPU mode
        )

        # Verify Llama was called with n_gpu_layers=0 for CPU mode
        call_kwargs = mock_llama_class.call_args[1]
        assert call_kwargs["n_gpu_layers"] == 0

    def test_extract_quantization_from_filename(self):
        """Test extracting quantization type from filename."""
        loader = ModelLoader()

        test_cases = [
            ("model.Q4_K_M.gguf", "Q4_K_M"),
            ("model.Q4_K_S.gguf", "Q4_K_S"),
            ("model.Q5_K_M.gguf", "Q5_K_M"),
            ("model.Q8_0.gguf", "Q8_0"),
            ("model.F16.gguf", "F16"),
            ("MistralRP-Noromaid-NSFW-7B-Q4_0.gguf", "Q4_0"),
            ("model-Q4_K_M.gguf", "Q4_K_M"),
        ]

        for filename, expected_quant in test_cases:
            quant = loader._extract_quantization(filename)
            assert quant == expected_quant

    def test_extract_quantization_invalid_filename(self):
        """Test quantization extraction with invalid filename."""
        loader = ModelLoader()

        with pytest.raises(ValueError, match="[Qq]uantization.*filename"):
            loader._extract_quantization("model.gguf")  # No quantization marker

    @patch("kcuda_validate.services.model_loader.torch.cuda")
    def test_check_vram_availability_sufficient(self, mock_cuda):
        """Test VRAM availability check with sufficient memory."""
        mock_cuda.is_available.return_value = True
        mock_cuda.mem_get_info.return_value = (
            8000 * 1024 * 1024,
            12000 * 1024 * 1024,
        )  # 8GB free, 12GB total

        loader = ModelLoader()
        # Should not raise - 5GB required, 8GB available
        loader.check_vram_availability(required_mb=5000)

    @patch("kcuda_validate.services.model_loader.torch.cuda")
    def test_check_vram_availability_insufficient(self, mock_cuda):
        """Test VRAM availability check with insufficient memory."""
        mock_cuda.is_available.return_value = True
        mock_cuda.mem_get_info.return_value = (
            2000 * 1024 * 1024,
            12000 * 1024 * 1024,
        )  # 2GB free, 12GB total

        loader = ModelLoader()
        with pytest.raises(ModelLoadError, match="[Ii]nsufficient.*[Vv][Rr][Aa][Mm]"):
            loader.check_vram_availability(required_mb=5000)  # Need 5GB, only 2GB free

    @patch("kcuda_validate.services.model_loader.torch.cuda")
    def test_check_vram_availability_no_cuda(self, mock_cuda):
        """Test VRAM check when CUDA is not available."""
        mock_cuda.is_available.return_value = False

        loader = ModelLoader()
        with pytest.raises(ModelLoadError, match="CUDA.*not available"):
            loader.check_vram_availability(required_mb=5000)
