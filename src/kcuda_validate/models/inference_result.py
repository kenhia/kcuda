"""InferenceResult model representing text generation outcomes."""

from dataclasses import dataclass


@dataclass
class InferenceResult:
    """
    Represents the outcome of a single text generation request.

    Attributes:
        prompt: Input text provided by user
        response: Generated text output from model
        tokens_generated: Number of tokens in generated response
        time_to_first_token_sec: Seconds elapsed before first token generated
        total_time_sec: Total generation time in seconds
        tokens_per_second: Calculated throughput
        gpu_utilization_percent: GPU utilization during generation
        vram_peak_mb: Peak GPU memory usage during inference
        success: Boolean indicating successful completion
        error_message: Error details if success=False
    """

    prompt: str
    response: str = ""
    tokens_generated: int = 0
    time_to_first_token_sec: float = 0.0
    total_time_sec: float = 0.0
    tokens_per_second: float = 0.0
    gpu_utilization_percent: float | None = None
    vram_peak_mb: int | None = None
    success: bool = False
    error_message: str | None = None

    def __post_init__(self) -> None:
        """Validate inference result attributes."""
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")

        if self.success:
            if self.tokens_generated <= 0:
                raise ValueError(
                    f"Invalid tokens_generated: {self.tokens_generated} must be > 0 for success=True"
                )

            if self.time_to_first_token_sec > self.total_time_sec:
                raise ValueError(
                    f"time_to_first_token ({self.time_to_first_token_sec}s) "
                    f"> total_time ({self.total_time_sec}s)"
                )

            if self.tokens_per_second <= 0:
                raise ValueError(
                    f"Invalid tokens_per_second: {self.tokens_per_second} must be > 0 for success=True"
                )

    @classmethod
    def from_generation(
        cls,
        prompt: str,
        response: str,
        tokens_generated: int,
        time_to_first_token_sec: float,
        total_time_sec: float,
        gpu_utilization_percent: float | None = None,
        vram_peak_mb: int | None = None,
    ) -> "InferenceResult":
        """
        Create successful InferenceResult from generation data.

        Args:
            prompt: Input prompt
            response: Generated text
            tokens_generated: Number of tokens generated
            time_to_first_token_sec: Time to first token
            total_time_sec: Total generation time
            gpu_utilization_percent: GPU utilization
            vram_peak_mb: Peak VRAM usage

        Returns:
            InferenceResult instance marked as successful
        """
        tokens_per_second = tokens_generated / total_time_sec if total_time_sec > 0 else 0.0

        return cls(
            prompt=prompt,
            response=response,
            tokens_generated=tokens_generated,
            time_to_first_token_sec=time_to_first_token_sec,
            total_time_sec=total_time_sec,
            tokens_per_second=tokens_per_second,
            gpu_utilization_percent=gpu_utilization_percent,
            vram_peak_mb=vram_peak_mb,
            success=True,
        )

    @classmethod
    def from_error(cls, prompt: str, error_message: str) -> "InferenceResult":
        """
        Create failed InferenceResult from error.

        Args:
            prompt: Input prompt
            error_message: Error description

        Returns:
            InferenceResult instance marked as failed
        """
        return cls(prompt=prompt, success=False, error_message=error_message)
