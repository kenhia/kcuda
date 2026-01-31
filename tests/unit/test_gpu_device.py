"""Unit tests for GPUDevice model validation."""

import pytest

from kcuda_validate.models.gpu_device import GPUDevice


class TestGPUDeviceValidation:
    """Test GPUDevice model validation rules."""

    def test_valid_gpu_device_creation(self):
        """Test creating a valid GPUDevice instance."""
        gpu = GPUDevice(
            name="NVIDIA GeForce RTX 3060",
            vram_total_mb=12288,
            vram_free_mb=11520,
            cuda_version="12.1",
            driver_version="525.60.11",
            compute_capability="8.6",
            device_id=0,
        )

        assert gpu.name == "NVIDIA GeForce RTX 3060"
        assert gpu.vram_total_mb == 12288
        assert gpu.vram_free_mb == 11520
        assert gpu.cuda_version == "12.1"
        assert gpu.driver_version == "525.60.11"
        assert gpu.compute_capability == "8.6"
        assert gpu.device_id == 0

    def test_vram_minimum_requirement(self):
        """Test that VRAM must be >= 4096 MB (4GB minimum)."""
        with pytest.raises(ValueError, match="VRAM.*4096"):
            GPUDevice(
                name="Low VRAM GPU",
                vram_total_mb=2048,  # Below minimum
                vram_free_mb=2048,
                cuda_version="12.1",
                driver_version="525.60.11",
                compute_capability="8.6",
                device_id=0,
            )

    def test_compute_capability_minimum(self):
        """Test that compute capability must be >= 6.0 (Pascal or newer)."""
        with pytest.raises(ValueError, match=r"[Cc]ompute capability.*6\.0"):
            GPUDevice(
                name="Old GPU",
                vram_total_mb=8192,
                vram_free_mb=8192,
                cuda_version="12.1",
                driver_version="525.60.11",
                compute_capability="5.2",  # Below minimum (Maxwell)
                device_id=0,
            )

    def test_cuda_version_required(self):
        """Test that CUDA version must be present."""
        with pytest.raises(ValueError, match="CUDA version.*required"):
            GPUDevice(
                name="NVIDIA GPU",
                vram_total_mb=8192,
                vram_free_mb=8192,
                cuda_version="",  # Empty string
                driver_version="525.60.11",
                compute_capability="8.6",
                device_id=0,
            )

    def test_vram_free_less_than_total(self):
        """Test that free VRAM cannot exceed total VRAM."""
        with pytest.raises(ValueError, match=r"[Ff]ree VRAM.*total"):
            GPUDevice(
                name="NVIDIA GPU",
                vram_total_mb=8192,
                vram_free_mb=10000,  # More than total
                cuda_version="12.1",
                driver_version="525.60.11",
                compute_capability="8.6",
                device_id=0,
            )

    def test_device_id_non_negative(self):
        """Test that device ID must be non-negative."""
        with pytest.raises(ValueError, match=r"[Dd]evice [Ii][Dd].*non-negative"):
            GPUDevice(
                name="NVIDIA GPU",
                vram_total_mb=8192,
                vram_free_mb=8000,
                cuda_version="12.1",
                driver_version="525.60.11",
                compute_capability="8.6",
                device_id=-1,  # Negative
            )

    def test_gpu_device_immutability(self):
        """Test that GPUDevice is immutable after creation (per data-model.md)."""
        gpu = GPUDevice(
            name="NVIDIA GeForce RTX 3060",
            vram_total_mb=12288,
            vram_free_mb=11520,
            cuda_version="12.1",
            driver_version="525.60.11",
            compute_capability="8.6",
            device_id=0,
        )

        # Should not be able to modify attributes
        with pytest.raises(AttributeError):
            gpu.vram_free_mb = 5000

    def test_gpu_device_repr(self):
        """Test string representation includes key attributes."""
        gpu = GPUDevice(
            name="NVIDIA GeForce RTX 3060",
            vram_total_mb=12288,
            vram_free_mb=11520,
            cuda_version="12.1",
            driver_version="525.60.11",
            compute_capability="8.6",
            device_id=0,
        )

        repr_str = repr(gpu)
        assert "RTX 3060" in repr_str
        assert "12288" in repr_str
