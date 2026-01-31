"""Unit tests for InferenceResult model validation.

These tests validate the InferenceResult data model per data-model.md specification.
"""

import pytest

from kcuda_validate.models.inference_result import InferenceResult


class TestInferenceResultValidation:
    """Test InferenceResult validation rules."""

    def test_valid_successful_inference_result(self):
        """Test creating a valid successful inference result."""
        result = InferenceResult(
            prompt="Hello, how are you?",
            response="I'm doing well, thank you!",
            tokens_generated=10,
            time_to_first_token_sec=0.5,
            total_time_sec=1.0,
            tokens_per_second=10.0,
            gpu_utilization_percent=95.0,
            vram_peak_mb=5000,
            success=True,
        )

        assert result.prompt == "Hello, how are you?"
        assert result.response == "I'm doing well, thank you!"
        assert result.tokens_generated == 10
        assert result.time_to_first_token_sec == 0.5
        assert result.total_time_sec == 1.0
        assert result.tokens_per_second == 10.0
        assert result.gpu_utilization_percent == 95.0
        assert result.vram_peak_mb == 5000
        assert result.success is True
        assert result.error_message is None

    def test_valid_failed_inference_result(self):
        """Test creating a valid failed inference result."""
        result = InferenceResult(
            prompt="Test prompt",
            success=False,
            error_message="CUDA out of memory",
        )

        assert result.prompt == "Test prompt"
        assert result.response == ""
        assert result.tokens_generated == 0
        assert result.success is False
        assert result.error_message == "CUDA out of memory"

    def test_prompt_cannot_be_empty(self):
        """Test that prompt cannot be empty."""
        with pytest.raises(ValueError, match="[Pp]rompt cannot be empty"):
            InferenceResult(
                prompt="",
                success=False,
                error_message="Error",
            )

    def test_tokens_generated_must_be_positive_for_success(self):
        """Test that tokens_generated must be > 0 when success=True."""
        with pytest.raises(
            ValueError, match="[Ii]nvalid tokens_generated.*must be > 0"
        ):
            InferenceResult(
                prompt="Test",
                response="Response",
                tokens_generated=0,
                time_to_first_token_sec=0.5,
                total_time_sec=1.0,
                tokens_per_second=10.0,
                success=True,
            )

    def test_tokens_generated_can_be_zero_for_failure(self):
        """Test that tokens_generated can be 0 when success=False."""
        result = InferenceResult(
            prompt="Test",
            tokens_generated=0,
            success=False,
            error_message="Generation failed",
        )

        assert result.tokens_generated == 0
        assert result.success is False

    def test_time_to_first_token_cannot_exceed_total_time(self):
        """Test that time_to_first_token must be <= total_time."""
        with pytest.raises(
            ValueError,
            match="time_to_first_token.*>.*total_time",
        ):
            InferenceResult(
                prompt="Test",
                response="Response",
                tokens_generated=10,
                time_to_first_token_sec=2.0,  # Greater than total_time
                total_time_sec=1.0,
                tokens_per_second=10.0,
                success=True,
            )

    def test_tokens_per_second_must_be_positive_for_success(self):
        """Test that tokens_per_second must be > 0 when success=True."""
        with pytest.raises(
            ValueError, match="[Ii]nvalid tokens_per_second.*must be > 0"
        ):
            InferenceResult(
                prompt="Test",
                response="Response",
                tokens_generated=10,
                time_to_first_token_sec=0.5,
                total_time_sec=1.0,
                tokens_per_second=0.0,  # Invalid for success
                success=True,
            )

    def test_optional_gpu_metrics_can_be_none(self):
        """Test that GPU metrics can be None."""
        result = InferenceResult(
            prompt="Test",
            response="Response",
            tokens_generated=10,
            time_to_first_token_sec=0.5,
            total_time_sec=1.0,
            tokens_per_second=10.0,
            gpu_utilization_percent=None,
            vram_peak_mb=None,
            success=True,
        )

        assert result.gpu_utilization_percent is None
        assert result.vram_peak_mb is None


class TestInferenceResultFactoryMethods:
    """Test InferenceResult factory methods."""

    def test_from_generation_creates_successful_result(self):
        """Test from_generation() creates a successful result with calculated metrics."""
        result = InferenceResult.from_generation(
            prompt="Hello",
            response="Hi there!",
            tokens_generated=20,
            time_to_first_token_sec=0.8,
            total_time_sec=2.0,
            gpu_utilization_percent=98.0,
            vram_peak_mb=5500,
        )

        assert result.prompt == "Hello"
        assert result.response == "Hi there!"
        assert result.tokens_generated == 20
        assert result.time_to_first_token_sec == 0.8
        assert result.total_time_sec == 2.0
        assert result.tokens_per_second == 10.0  # 20 tokens / 2.0 seconds
        assert result.gpu_utilization_percent == 98.0
        assert result.vram_peak_mb == 5500
        assert result.success is True
        assert result.error_message is None

    def test_from_generation_without_gpu_metrics(self):
        """Test from_generation() works without GPU metrics."""
        result = InferenceResult.from_generation(
            prompt="Test",
            response="Response",
            tokens_generated=10,
            time_to_first_token_sec=0.5,
            total_time_sec=1.0,
        )

        assert result.tokens_per_second == 10.0
        assert result.gpu_utilization_percent is None
        assert result.vram_peak_mb is None
        assert result.success is True

    def test_from_generation_handles_zero_time(self):
        """Test from_generation() with zero total_time creates invalid result (should raise)."""
        # Zero time with generated tokens would result in tokens_per_second=0
        # which violates the validation rule for success=True
        # This test verifies the validation catches this edge case
        with pytest.raises(ValueError, match="[Ii]nvalid tokens_per_second"):
            InferenceResult.from_generation(
                prompt="Test",
                response="Response",
                tokens_generated=10,
                time_to_first_token_sec=0.0,
                total_time_sec=0.0,
            )

    def test_from_error_creates_failed_result(self):
        """Test from_error() creates a failed result."""
        result = InferenceResult.from_error(
            prompt="Test prompt", error_message="CUDA error: out of memory"
        )

        assert result.prompt == "Test prompt"
        assert result.response == ""
        assert result.tokens_generated == 0
        assert result.success is False
        assert result.error_message == "CUDA error: out of memory"

    def test_from_error_requires_prompt(self):
        """Test from_error() validates prompt."""
        with pytest.raises(ValueError, match="[Pp]rompt cannot be empty"):
            InferenceResult.from_error(prompt="", error_message="Error")


class TestInferenceResultImmutability:
    """Test that InferenceResult follows immutability requirements."""

    def test_inference_result_is_not_frozen(self):
        """Test that InferenceResult is mutable (not frozen) per model definition."""
        result = InferenceResult(
            prompt="Test",
            success=False,
            error_message="Error",
        )

        # InferenceResult is not frozen, so we can modify it
        # This is different from GPUDevice and LLMModel which are frozen
        result.response = "Updated response"
        assert result.response == "Updated response"
