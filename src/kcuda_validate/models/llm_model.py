"""LLMModel model representing a loaded GGUF model file."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMModel:
    """
    Represents a loaded GGUF model file and its metadata.

    Attributes:
        repo_id: Hugging Face repository identifier
        filename: GGUF file name within repository
        local_path: Absolute filesystem path to cached model file
        file_size_mb: Model file size in megabytes
        parameter_count: Number of model parameters
        quantization_type: Quantization method (e.g., "Q4_K_M", "Q5_K_S")
        context_length: Maximum context window size in tokens
        vram_usage_mb: Measured GPU memory consumption after loading (0 if not loaded)
        is_loaded: Boolean indicating if model is currently loaded in GPU memory
        instance: The actual Llama model instance (for inference)
    """

    repo_id: str
    filename: str
    local_path: str
    file_size_mb: int
    parameter_count: int
    quantization_type: str
    context_length: int
    vram_usage_mb: int = 0
    is_loaded: bool = False
    instance: Any = None

    def __post_init__(self) -> None:
        """Validate LLM model attributes."""
        # Validate repo_id
        if not self.repo_id or not self.repo_id.strip():
            raise ValueError("Repo ID cannot be empty")

        # Validate filename
        if not self.filename or not self.filename.strip():
            raise ValueError("Filename cannot be empty")

        # Validate file size
        if self.file_size_mb <= 0:
            raise ValueError(f"File size must be positive, got {self.file_size_mb}MB")

        # Validate context length
        if self.context_length <= 0:
            raise ValueError(f"Context length must be positive, got {self.context_length}")

        # Validate quantization type
        if not self.quantization_type or not self.quantization_type.strip():
            raise ValueError("Quantization type cannot be empty")

        # Validate VRAM usage when loaded
        if self.is_loaded and self.vram_usage_mb <= 0:
            raise ValueError("VRAM usage must be positive when model is loaded")
