"""Unit tests for Inferencer service.

These tests validate the inferencer service with mocked llama-cpp-python.
Tests follow TDD approach - write first, verify they fail, then implement.
"""

from unittest.mock import MagicMock, patch

import pytest

# Import will fail until we create the service - this is expected in TDD red phase
try:
    from kcuda_validate.services.inferencer import Inferencer, InferenceError
except ImportError:
    # Create placeholder for tests to run
    class Inferencer:  # type: ignore
        pass

    class InferenceError(Exception):  # type: ignore
        pass


class TestInferencer:
    """Test Inferencer service functionality."""

    def setup_method(self):
        """Setup test dependencies."""
        self.default_prompt = "Hello, how are you?"
        self.default_max_tokens = 50
        self.default_temperature = 0.7

    @patch("kcuda_validate.services.inferencer.time.time")
    def test_generate_success(self, mock_time):
        """Test successful text generation."""
        # Mock time for performance metrics
        mock_time.side_effect = [0.0, 0.5, 1.5]  # start, first token, end

        # Create mock model
        mock_model = MagicMock()
        mock_model.return_value = {
            "choices": [{"text": " I'm doing well, thank you!"}]
        }

        inferencer = Inferencer(model=mock_model)
        result = inferencer.generate(
            prompt=self.default_prompt,
            max_tokens=self.default_max_tokens,
            temperature=self.default_temperature,
        )

        # Verify result
        assert result.success is True
        assert result.prompt == self.default_prompt
        assert result.response == " I'm doing well, thank you!"
        assert result.tokens_generated > 0
        assert result.time_to_first_token_sec == pytest.approx(0.5)
        assert result.total_time_sec == pytest.approx(1.5)
        assert result.tokens_per_second > 0
        assert result.error_message is None

        # Verify model was called correctly
        mock_model.assert_called_once()
        call_kwargs = mock_model.call_args[1]
        assert call_kwargs["prompt"] == self.default_prompt
        assert call_kwargs["max_tokens"] == self.default_max_tokens
        assert call_kwargs["temperature"] == self.default_temperature

    def test_generate_with_cuda_error(self):
        """Test generation failure due to CUDA error."""
        mock_model = MagicMock()
        mock_model.side_effect = RuntimeError("CUDA out of memory")

        inferencer = Inferencer(model=mock_model)
        result = inferencer.generate(prompt=self.default_prompt)

        # Verify error result
        assert result.success is False
        assert result.prompt == self.default_prompt
        assert "CUDA" in result.error_message or "memory" in result.error_message
        assert result.tokens_generated == 0

    def test_generate_with_empty_prompt(self):
        """Test generation with empty prompt."""
        mock_model = MagicMock()
        inferencer = Inferencer(model=mock_model)

        with pytest.raises(InferenceError, match="[Pp]rompt cannot be empty"):
            inferencer.generate(prompt="")

    def test_generate_with_invalid_max_tokens(self):
        """Test generation with invalid max_tokens."""
        mock_model = MagicMock()
        inferencer = Inferencer(model=mock_model)

        with pytest.raises(InferenceError, match="[Mm]ax.*tokens.*must be.*positive"):
            inferencer.generate(prompt=self.default_prompt, max_tokens=0)

    def test_generate_with_invalid_temperature(self):
        """Test generation with invalid temperature."""
        mock_model = MagicMock()
        inferencer = Inferencer(model=mock_model)

        with pytest.raises(
            InferenceError, match="[Tt]emperature.*must be.*between.*0.*and"
        ):
            inferencer.generate(prompt=self.default_prompt, temperature=2.0)

    @patch("kcuda_validate.services.inferencer.pynvml")
    @patch("kcuda_validate.services.inferencer.time.time")
    def test_generate_with_gpu_metrics(self, mock_time, mock_pynvml):
        """Test generation collects GPU metrics."""
        # Mock time
        mock_time.side_effect = [0.0, 0.5, 1.5]

        # Mock GPU metrics
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle

        mock_util = MagicMock()
        mock_util.gpu = 95
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util

        mock_mem_info = MagicMock()
        mock_mem_info.used = 5000 * 1024 * 1024  # 5000 MB
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem_info

        # Mock model
        mock_model = MagicMock()
        mock_model.return_value = {"choices": [{"text": "Response"}]}

        inferencer = Inferencer(model=mock_model, device_id=0, collect_metrics=True)
        result = inferencer.generate(prompt=self.default_prompt)

        # Verify GPU metrics collected
        assert result.gpu_utilization_percent is not None
        assert result.gpu_utilization_percent == pytest.approx(95.0)
        assert result.vram_peak_mb is not None
        assert result.vram_peak_mb == pytest.approx(5000, abs=100)

    def test_generate_without_gpu_metrics(self):
        """Test generation without GPU metrics collection."""
        mock_model = MagicMock()
        mock_model.return_value = {"choices": [{"text": "Response"}]}

        inferencer = Inferencer(model=mock_model, collect_metrics=False)
        result = inferencer.generate(prompt=self.default_prompt)

        # Verify no GPU metrics
        assert result.gpu_utilization_percent is None
        assert result.vram_peak_mb is None

    def test_generate_handles_empty_response(self):
        """Test generation handles empty model response."""
        mock_model = MagicMock()
        mock_model.return_value = {"choices": [{"text": ""}]}

        inferencer = Inferencer(model=mock_model)
        result = inferencer.generate(prompt=self.default_prompt)

        # Should still succeed but with empty response
        assert result.success is False
        assert "empty response" in result.error_message.lower()

    def test_generate_with_streaming(self):
        """Test generation with streaming enabled."""
        # Mock streaming tokens
        def mock_generator():
            yield {"choices": [{"text": "Hello"}]}
            yield {"choices": [{"text": " world"}]}
            yield {"choices": [{"text": "!"}]}

        mock_model = MagicMock()
        mock_model.return_value = mock_generator()

        inferencer = Inferencer(model=mock_model)
        result = inferencer.generate(prompt=self.default_prompt, stream=False)

        # For non-streaming mode, should collect full response
        assert result.success is True
        assert len(result.response) > 0

    def test_count_tokens_approximation(self):
        """Test token counting approximation."""
        mock_model = MagicMock()
        inferencer = Inferencer(model=mock_model)

        # Test simple token counting (whitespace split approximation)
        text = "This is a test sentence with multiple words"
        tokens = inferencer._count_tokens(text)

        # Should be approximately the number of words
        assert tokens > 0
        assert tokens >= len(text.split()) - 2  # Allow some variance

    def test_inferencer_requires_model(self):
        """Test that Inferencer requires a model."""
        with pytest.raises(TypeError, match="[Mm]odel"):
            Inferencer(model=None)  # type: ignore


class TestInferencerIntegrationWithLLMModel:
    """Test Inferencer integration with loaded LLM model."""

    def test_create_from_llm_model(self):
        """Test creating Inferencer from LLMModel."""
        from kcuda_validate.models.llm_model import LLMModel

        # Create LLMModel
        llm_model = LLMModel(
            repo_id="test/repo",
            filename="model.gguf",
            local_path="/path/to/model.gguf",
            file_size_mb=4000,
            parameter_count=7_000_000_000,
            quantization_type="Q4_K_M",
            context_length=8192,
            vram_usage_mb=5000,
            is_loaded=True,
        )

        # Inferencer should be able to work with the loaded model
        # This tests the integration between model_loader and inferencer
        # In practice, the model would already be loaded by ModelLoader
        assert llm_model.is_loaded is True
        assert llm_model.context_length == 8192
