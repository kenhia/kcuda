"""Inference service for generating text from loaded models."""

import contextlib
import time
from typing import Any

import pynvml

from kcuda_validate.models.inference_result import InferenceResult
from kcuda_validate.models.llm_model import LLMModel


class InferenceError(Exception):
    """Raised when inference fails."""

    pass


class Inferencer:
    """Service for running inference on loaded models.

    This service wraps the llama-cpp-python model and provides:
    - Text generation with performance tracking
    - GPU utilization monitoring during inference
    - Error handling for CUDA and model failures
    """

    def __init__(
        self,
        model: LLMModel | Any,
        device_id: int = 0,
        collect_metrics: bool = True,
    ) -> None:
        """Initialize the inferencer.

        Args:
            model: The loaded LLM model to use for inference.
            device_id: GPU device ID for metrics collection.
            collect_metrics: Whether to collect GPU metrics during inference.

        Raises:
            TypeError: If model is None.
        """
        if model is None:
            raise TypeError("Model cannot be None")

        self._model = model
        self._device_id = device_id
        self._collect_metrics = collect_metrics
        self._last_result: InferenceResult | None = None

    @property
    def model(self) -> LLMModel | Any:
        """Get the currently loaded model."""
        return self._model

    def load_model(self, model: LLMModel) -> None:
        """Load a model for inference.

        Args:
            model: The LLM model to load.
        """
        self._model = model

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> InferenceResult:
        """Generate text from a prompt.

        Args:
            prompt: The input prompt text.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
            stream: Whether to use streaming mode (for testing purposes).

        Returns:
            InferenceResult with generation results and performance metrics.

        Raises:
            RuntimeError: If no model is loaded.
            InferenceError: If generation fails or validation error occurs.
        """
        if self._model is None:
            raise RuntimeError("No model loaded")

        if not prompt or not prompt.strip():
            raise InferenceError("Prompt cannot be empty")

        if max_tokens <= 0:
            raise InferenceError("Max tokens must be positive")

        if not 0.0 <= temperature < 2.0:
            raise InferenceError("Temperature must be between 0.0 and 2.0")

        try:
            # Initialize GPU monitoring
            gpu_utilization_percent: float | None = None
            vram_peak_mb: int | None = None

            if self._collect_metrics:
                with contextlib.suppress(pynvml.NVMLError):
                    # GPU monitoring not critical, continue without it if it fails
                    pynvml.nvmlInit()

            # Start timing
            start_time = time.time()

            # Call the model (tests mock model.__call__)
            model_output = self._model(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature
            )

            # Capture time to first token
            first_token_time = time.time()

            # Process model output (could be dict or generator)
            result_text = ""

            if hasattr(model_output, "__iter__") and not isinstance(model_output, dict):
                # Generator/iterator (streaming)
                for chunk in model_output:
                    if isinstance(chunk, dict) and "choices" in chunk:
                        choice = chunk["choices"][0]
                        if "text" in choice:
                            result_text += choice["text"]
            elif isinstance(model_output, dict) and "choices" in model_output:
                # Single response
                choice = model_output["choices"][0]
                if "text" in choice:
                    result_text = choice["text"]

            # End timing
            end_time = time.time()

            # Check for empty response
            if not result_text or not result_text.strip():
                error_msg = "Model returned empty response"
                self._last_result = InferenceResult.from_error(
                    prompt=prompt, error_message=error_msg
                )
                return self._last_result

            # Get GPU metrics after generation
            if self._collect_metrics:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_id)

                    # Get GPU utilization
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization_percent = float(utilization.gpu)

                    # Get VRAM usage
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    vram_peak_mb = int(mem_info.used / (1024 * 1024))

                    pynvml.nvmlShutdown()
                except pynvml.NVMLError:
                    # GPU monitoring failed, but generation succeeded
                    pass

            # Calculate timing metrics
            total_time_sec = end_time - start_time
            time_to_first_token_sec = first_token_time - start_time

            # Count tokens (approximate)
            tokens_generated = self._count_tokens(result_text)

            # Create result
            self._last_result = InferenceResult.from_generation(
                prompt=prompt,
                response=result_text,
                tokens_generated=tokens_generated,
                time_to_first_token_sec=time_to_first_token_sec,
                total_time_sec=total_time_sec,
                gpu_utilization_percent=gpu_utilization_percent,
                vram_peak_mb=vram_peak_mb,
            )

            return self._last_result

        except RuntimeError as e:
            # Model errors (CUDA OOM, etc.)
            error_msg = str(e)
            if "CUDA" in error_msg or "out of memory" in error_msg:
                self._last_result = InferenceResult.from_error(
                    prompt=prompt, error_message=error_msg
                )
                return self._last_result
            raise InferenceError(f"Generation failed: {error_msg}") from e

        except (ValueError, InferenceError):
            # Re-raise validation errors
            raise

        except Exception as e:
            # Unexpected errors
            error_msg = f"Unexpected error during inference: {str(e)}"
            self._last_result = InferenceResult.from_error(prompt=prompt, error_message=error_msg)
            raise InferenceError(error_msg) from e

    def _count_tokens(self, text: str) -> int:
        """Approximate token count for generated text.

        Uses whitespace split as approximation.

        Args:
            text: The generated text.

        Returns:
            Approximate number of tokens.
        """
        if not text or not text.strip():
            return 0

        # Simple whitespace-based approximation
        # Real tokenization would use the model's tokenizer
        return len(text.split())

    @property
    def last_result(self) -> InferenceResult | None:
        """Get the last inference result."""
        return self._last_result
