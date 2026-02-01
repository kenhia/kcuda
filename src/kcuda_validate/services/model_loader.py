"""Model loader service for downloading and loading GGUF models.

Handles model download from Hugging Face and loading into GPU memory
using llama-cpp-python.
"""

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from huggingface_hub import hf_hub_download

from kcuda_validate.models.llm_model import LLMModel

# Lazy import to avoid requiring llama-cpp-python at import time
# (needed for CI environments without CUDA build tools)
if TYPE_CHECKING:
    from llama_cpp import Llama

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Exception raised when model loading fails."""

    pass


class ModelLoader:
    """Service for downloading and loading GGUF models."""

    def __init__(self):
        """Initialize model loader."""
        self._loaded_model: Llama | None = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.cleanup()
        return False  # Don't suppress exceptions

    def cleanup(self) -> None:
        """Clean up GPU memory by unloading model.

        This method ensures graceful cleanup of GPU resources.
        Safe to call multiple times.
        """
        if self._loaded_model is not None:
            try:
                # Delete the model instance to free GPU memory
                del self._loaded_model
                self._loaded_model = None

                # Force CUDA cache cleanup if torch is available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                logger.info("GPU memory cleaned up successfully")
            except Exception as e:
                logger.warning(f"Error during GPU cleanup: {e}")

    def download_model(self, repo_id: str, filename: str, repo_type: str = "model") -> str:
        """Download model from Hugging Face Hub.

        Args:
            repo_id: Hugging Face repository ID
            filename: Model filename in repository
            repo_type: Repository type (default: "model")

        Returns:
            Local path to downloaded model file

        Raises:
            ModelLoadError: If download fails
        """
        logger.info(f"Downloading model: {repo_id}/{filename}")

        try:
            local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
            logger.info(f"Model downloaded to: {local_path}")
            return local_path

        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "not found" in error_msg.lower():
                raise ModelLoadError(f"File not found: {filename} in repository {repo_id}") from e
            else:
                raise ModelLoadError(f"Download failed: {e}") from e

    def load_model(
        self,
        local_path: str,
        repo_id: str,
        filename: str,
        use_gpu: bool = True,
        n_ctx: int = 8192,
    ) -> LLMModel:
        """Load GGUF model into memory.

        Args:
            local_path: Path to local model file
            repo_id: Hugging Face repository ID
            filename: Model filename
            use_gpu: Whether to use GPU (default: True)
            n_ctx: Context window size (default: 8192)

        Returns:
            LLMModel instance with metadata

        Raises:
            ModelLoadError: If loading fails
        """
        logger.info(f"Loading model from: {local_path}")

        # Check file exists
        path_obj = Path(local_path)
        if not path_obj.exists():
            raise ModelLoadError(f"File not found: {local_path}")

        # Log file size
        try:
            file_size_gb = path_obj.stat().st_size / (1024**3)
            logger.info(f"Model file size: {file_size_gb:.2f} GB")
        except Exception as e:
            logger.debug(f"Could not get file size: {e}")

        # Log CUDA memory status before loading
        if use_gpu and torch.cuda.is_available():
            try:
                mem_info = torch.cuda.mem_get_info()
                free_mb = mem_info[0] // (1024 * 1024)
                total_mb = mem_info[1] // (1024 * 1024)
                used_mb = total_mb - free_mb
                logger.info(
                    f"GPU memory before load: {used_mb}/{total_mb} MB used ({free_mb} MB free)"
                )
            except Exception as e:
                logger.debug(f"Could not get GPU memory info: {e}")

        # Get file size
        try:
            file_size_bytes = path_obj.stat().st_size
            file_size_mb = file_size_bytes // (1024 * 1024)
        except Exception as e:
            raise ModelLoadError(f"Corrupt or invalid file: {e}") from e

        # Extract quantization type from filename
        try:
            quantization_type = self._extract_quantization(filename)
            logger.info(f"Detected quantization: {quantization_type}")
        except ValueError as e:
            raise ModelLoadError(str(e)) from e

        # Check VRAM availability if using GPU
        if use_gpu:
            # Estimate VRAM requirement (roughly 1.2x file size for Q4 quantization)
            estimated_vram_mb = int(file_size_mb * 1.2)
            logger.info(f"Estimated VRAM required: {estimated_vram_mb} MB")
            try:
                self.check_vram_availability(required_mb=estimated_vram_mb)
            except ModelLoadError:
                raise

        # Load model with llama-cpp-python
        try:
            # Import at runtime to avoid requiring llama-cpp-python at module import time
            from llama_cpp import Llama

            n_gpu_layers = -1 if use_gpu else 0  # -1 = all layers on GPU
            self._loaded_model = Llama(
                model_path=local_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
            logger.info("Model loaded successfully")

            # Log GPU memory after loading
            if use_gpu and torch.cuda.is_available():
                try:
                    mem_info = torch.cuda.mem_get_info()
                    free_mb = mem_info[0] // (1024 * 1024)
                    total_mb = mem_info[1] // (1024 * 1024)
                    used_mb = total_mb - free_mb
                    logger.info(
                        f"GPU memory after load: {used_mb}/{total_mb} MB used ({free_mb} MB free)"
                    )
                except Exception as e:
                    logger.debug(f"Could not get GPU memory info: {e}")

        except RuntimeError as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
                raise ModelLoadError(f"Insufficient VRAM to load model: {e}") from e
            else:
                raise ModelLoadError(f"Failed to load model: {e}") from e
        except ValueError as e:
            raise ModelLoadError(f"Invalid GGUF format: {e}") from e
        except Exception as e:
            raise ModelLoadError(f"Unexpected error loading model: {e}") from e

        # Get context length from loaded model
        context_length = self._loaded_model.n_ctx()

        # Estimate VRAM usage (actual measurement would require pynvml queries)
        # For now, use file size as approximation
        vram_usage_mb = int(file_size_mb * 1.15)  # Slight overhead

        # Estimate parameter count from file size and quantization
        # Q4_K_M is roughly 0.5 bytes per parameter
        parameter_count = self._estimate_parameters(file_size_mb, quantization_type)

        # Create LLMModel instance
        llm_model = LLMModel(
            repo_id=repo_id,
            filename=filename,
            local_path=local_path,
            file_size_mb=file_size_mb,
            parameter_count=parameter_count,
            quantization_type=quantization_type,
            context_length=context_length,
            vram_usage_mb=vram_usage_mb,
            is_loaded=True,
            instance=self._loaded_model,  # Store the actual Llama instance
        )

        return llm_model

    def check_vram_availability(self, required_mb: int) -> None:
        """Check if sufficient VRAM is available.

        Args:
            required_mb: Required VRAM in megabytes

        Raises:
            ModelLoadError: If insufficient VRAM or CUDA unavailable
        """
        if not torch.cuda.is_available():
            raise ModelLoadError("CUDA is not available. Cannot check VRAM availability.")

        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_mb = free_bytes // (1024 * 1024)

            logger.debug(f"VRAM check: {free_mb}MB free, {required_mb}MB required")

            if free_mb < required_mb:
                raise ModelLoadError(
                    f"Insufficient VRAM: need {required_mb}MB, only {free_mb}MB available"
                )

        except Exception as e:
            if isinstance(e, ModelLoadError):
                raise
            raise ModelLoadError(f"Failed to check VRAM: {e}") from e

    def _extract_quantization(self, filename: str) -> str:
        """Extract quantization type from filename.

        Args:
            filename: Model filename (e.g., "model.Q4_K_M.gguf")

        Returns:
            Quantization type string

        Raises:
            ValueError: If quantization type cannot be determined
        """
        # Common GGUF quantization patterns (match with word boundaries, not just dots)
        patterns = [
            r"[.-]Q([2-8])_([KM])_([SML])[.-]",  # Q4_K_M, Q5_K_S, etc.
            r"[.-]Q([2-8])_([01])[.-]",  # Q4_0, Q4_1, Q8_0
            r"[.-]F(16|32)[.-]",  # F16, F32
        ]

        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                # Extract and normalize, removing leading/trailing separators
                quant = match.group(0).strip(".-")
                return quant.upper()

        raise ValueError(f"Cannot determine quantization type from filename: {filename}")

    def _estimate_parameters(self, file_size_mb: int, quantization_type: str) -> int:
        """Estimate parameter count from file size and quantization.

        Args:
            file_size_mb: Model file size in MB
            quantization_type: Quantization type

        Returns:
            Estimated parameter count
        """
        # Rough estimates: bytes per parameter for different quantizations
        bytes_per_param = {
            "Q4_K_M": 0.55,
            "Q4_K_S": 0.50,
            "Q5_K_M": 0.65,
            "Q5_K_S": 0.60,
            "Q8_0": 1.0,
            "F16": 2.0,
            "F32": 4.0,
        }

        # Default to Q4_K_M estimate if not found
        bytes_pp = bytes_per_param.get(quantization_type, 0.55)

        file_size_bytes = file_size_mb * 1024 * 1024
        estimated_params = int(file_size_bytes / bytes_pp)

        return estimated_params

    def unload_model(self) -> None:
        """Unload model from memory."""
        if self._loaded_model:
            del self._loaded_model
            self._loaded_model = None
            logger.info("Model unloaded from memory")
