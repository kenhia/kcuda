"""CLI command for detecting NVIDIA GPU hardware."""

import logging
import sys

import click

from kcuda_validate.lib.formatters import format_gpu_error, format_gpu_info
from kcuda_validate.services.gpu_detector import GPUDetectionError, GPUDetector

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--device-id",
    type=int,
    default=0,
    help="GPU device index to query (default: 0)",
)
def detect(device_id: int) -> None:
    """Detect NVIDIA GPU hardware and validate CUDA compatibility.

    Checks for CUDA availability, detects GPU hardware, and validates
    minimum requirements for LLM inference (4GB VRAM, compute capability 6.0+).

    Exit codes:
      0 - GPU detected and meets minimum requirements
      1 - No GPU detected or CUDA unavailable
      2 - GPU detected but does not meet minimum requirements
    """
    logger.info(f"Running GPU detection for device {device_id}")

    try:
        detector = GPUDetector()
        gpu_device = detector.detect(device_id=device_id)

        # Check minimum requirements
        meets_requirements = True
        warnings = []

        if gpu_device.vram_total_mb < 4096:
            meets_requirements = False
            warnings.append(
                f"Insufficient VRAM: {gpu_device.vram_total_mb}MB "
                "(minimum 4096MB required for LLM inference)"
            )

        compute_major = int(gpu_device.compute_capability.split(".")[0])
        if compute_major < 6:
            meets_requirements = False
            warnings.append(
                f"Compute capability {gpu_device.compute_capability} is below "
                "minimum requirement 6.0 for efficient LLM operations"
            )

        # Display results
        output = format_gpu_info(gpu_device, warnings=warnings if warnings else None)
        click.echo(output)

        if not meets_requirements:
            logger.warning(f"GPU does not meet minimum requirements: {warnings}")
            sys.exit(2)

        logger.info("GPU detection successful, all requirements met")
        sys.exit(0)

    except GPUDetectionError as e:
        error_output = format_gpu_error(str(e))
        click.echo(error_output, err=True)
        logger.error(f"GPU detection failed: {e}")
        sys.exit(1)

    except Exception as e:
        error_output = format_gpu_error(f"Unexpected error: {e}")
        click.echo(error_output, err=True)
        logger.exception("Unexpected error during GPU detection")
        sys.exit(1)
