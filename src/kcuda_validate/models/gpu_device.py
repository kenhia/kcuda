"""GPUDevice model representing NVIDIA GPU hardware."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GPUDevice:
    """
    Represents detected NVIDIA GPU hardware and its capabilities.

    Attributes:
        name: GPU model name (e.g., "NVIDIA GeForce RTX 3060")
        vram_total_mb: Total video memory in megabytes
        vram_free_mb: Currently available video memory in megabytes
        cuda_version: CUDA runtime version string (e.g., "12.1")
        driver_version: NVIDIA driver version
        compute_capability: GPU compute capability (e.g., "8.6")
        device_id: Integer device index (0 for single GPU systems)
    """

    name: str
    vram_total_mb: int
    vram_free_mb: int
    cuda_version: str
    driver_version: str
    compute_capability: str
    device_id: int = 0

    def __post_init__(self) -> None:
        """Validate GPU device attributes."""
        # Validate VRAM minimum
        if self.vram_total_mb < 4096:
            raise ValueError(f"Insufficient VRAM: {self.vram_total_mb}MB < 4096MB minimum required")

        # Validate free VRAM doesn't exceed total
        if self.vram_free_mb > self.vram_total_mb:
            raise ValueError(
                f"Free VRAM ({self.vram_free_mb}MB) cannot exceed total VRAM ({self.vram_total_mb}MB)"
            )

        # Validate device ID is non-negative
        if self.device_id < 0:
            raise ValueError(f"Device ID must be non-negative, got {self.device_id}")

        # Parse and validate compute capability
        try:
            major, minor = map(int, self.compute_capability.split("."))
            if major < 6:
                raise ValueError(
                    f"Compute capability {self.compute_capability} < 6.0 (Pascal or newer required)"
                ) from None
        except ValueError:
            # Re-raise with proper message
            if major < 6:
                raise ValueError(
                    f"Compute capability {self.compute_capability} < 6.0 (Pascal or newer required)"
                ) from None
            else:
                raise ValueError(
                    f"Invalid compute capability format: {self.compute_capability}"
                ) from None
        except AttributeError as e:
            raise ValueError(f"Invalid compute capability format: {self.compute_capability}") from e

        # Validate CUDA version is present
        if not self.cuda_version or not self.cuda_version.strip():
            raise ValueError("CUDA version is required")

    @property
    def vram_used_mb(self) -> int:
        """Calculate currently used VRAM."""
        return self.vram_total_mb - self.vram_free_mb

    @property
    def vram_usage_percent(self) -> float:
        """Calculate VRAM usage percentage."""
        return (self.vram_used_mb / self.vram_total_mb) * 100

    def has_sufficient_vram(self, required_mb: int) -> bool:
        """Check if GPU has sufficient free VRAM for requested amount."""
        return self.vram_free_mb >= required_mb
