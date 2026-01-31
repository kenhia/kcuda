"""CLI command for running full validation pipeline."""

import logging
import sys

import click

from kcuda_validate.lib.formatters import format_validation_summary
from kcuda_validate.services.gpu_detector import GPUDetectionError, GPUDetector
from kcuda_validate.services.inferencer import Inferencer
from kcuda_validate.services.model_loader import ModelLoader

logger = logging.getLogger(__name__)

# Default model configuration
DEFAULT_REPO_ID = "Ttimofeyka/MistralRP-Noromaid-NSFW-Mistral-7B-GGUF"
DEFAULT_FILENAME = "MistralRP-Noromaid-NSFW-7B-Q4_0.gguf"
DEFAULT_PROMPT = "Hello, how are you?"


@click.command()
@click.option(
    "--repo-id",
    type=str,
    default=DEFAULT_REPO_ID,
    help=f"Hugging Face repository ID [default: {DEFAULT_REPO_ID}]",
)
@click.option(
    "--filename",
    type=str,
    default=DEFAULT_FILENAME,
    help=f"Specific GGUF file to download [default: {DEFAULT_FILENAME}]",
)
@click.option(
    "--prompt",
    type=str,
    default=DEFAULT_PROMPT,
    help=f"Inference test prompt [default: {DEFAULT_PROMPT}]",
)
@click.option(
    "--skip-on-error",
    is_flag=True,
    help="Continue to next step even if previous failed",
)
def validate_all(repo_id: str, filename: str, prompt: str, skip_on_error: bool) -> None:
    """Run complete validation pipeline: detect → load → infer.

    Validates the full CUDA LLM stack by executing all validation steps
    in sequence and displaying a comprehensive summary.

    Exit codes:
      0 - All validation steps passed
      1 - One or more validation steps failed
      2 - Critical error prevented validation from completing
    """
    logger.info("Starting full validation pipeline")

    # Track step results
    steps = {
        "gpu_detection": {"passed": False, "error": None},
        "model_loading": {"passed": False, "error": None},
        "inference_test": {"passed": False, "error": None},
    }

    # Step 1: GPU Detection
    click.echo("=" * 60)
    click.echo("Step 1/3: GPU Detection")
    click.echo("=" * 60)

    try:
        detector = GPUDetector()
        gpu_device = detector.detect(device_id=0)

        # Display GPU info
        from kcuda_validate.lib.formatters import format_gpu_info

        gpu_output = format_gpu_info(gpu_device)
        click.echo(gpu_output)

        steps["gpu_detection"]["passed"] = True
        logger.info("GPU detection passed")

    except GPUDetectionError as e:
        from kcuda_validate.lib.formatters import format_gpu_error

        error_output = format_gpu_error(str(e))
        click.echo(error_output, err=True)

        steps["gpu_detection"]["error"] = str(e)
        logger.error(f"GPU detection failed: {e}")

        if not skip_on_error:
            summary = format_validation_summary(steps)
            click.echo("\n" + summary)
            sys.exit(1)

    except Exception as e:
        from kcuda_validate.lib.formatters import format_gpu_error

        error_output = format_gpu_error(f"Unexpected error: {e}")
        click.echo(error_output, err=True)

        steps["gpu_detection"]["error"] = str(e)
        logger.exception("Unexpected error during GPU detection")

        if not skip_on_error:
            summary = format_validation_summary(steps)
            click.echo("\n" + summary)
            sys.exit(2)

    # Step 2: Model Loading
    click.echo("\n" + "=" * 60)
    click.echo("Step 2/3: Model Loading")
    click.echo("=" * 60)

    try:
        loader = ModelLoader()

        # Download model if needed
        click.echo("\n→ Checking model cache...")
        click.echo(f"  Model: {repo_id}")
        click.echo(f"  File: {filename}")

        model_path = loader.download_model(repo_id=repo_id, filename=filename)
        click.echo(f"\n✓ Model available: {model_path}")

        # Load model into GPU memory
        click.echo("\n→ Loading model into GPU memory...")
        model = loader.load_model(
            local_path=model_path,
            repo_id=repo_id,
            filename=filename,
            use_gpu=True,
        )

        # Display model info
        from kcuda_validate.lib.formatters import format_model_info

        model_output = format_model_info(model)
        click.echo(model_output)

        steps["model_loading"]["passed"] = True
        logger.info("Model loading passed")

    except Exception as e:
        from kcuda_validate.lib.formatters import format_model_error

        error_output = format_model_error(str(e))
        click.echo(error_output, err=True)

        steps["model_loading"]["error"] = str(e)
        logger.error(f"Model loading failed: {e}")

        if not skip_on_error:
            summary = format_validation_summary(steps)
            click.echo("\n" + summary)
            sys.exit(1)

    # Step 3: Inference Test
    click.echo("\n" + "=" * 60)
    click.echo("Step 3/3: Inference Test")
    click.echo("=" * 60)

    try:
        # Run inference
        click.echo("\n→ Running inference...")
        click.echo(f'  Prompt: "{prompt}"')

        inferencer = Inferencer(model=model.instance, collect_metrics=True)
        result = inferencer.generate(
            prompt=prompt,
            max_tokens=50,
            temperature=0.7,
        )

        # Check if inference succeeded
        if not result.success:
            raise RuntimeError(result.error_message)

        # Display inference results
        from kcuda_validate.lib.formatters import format_inference_result

        inference_output = format_inference_result(result)
        click.echo(inference_output)

        steps["inference_test"]["passed"] = True
        logger.info("Inference test passed")

    except Exception as e:
        from kcuda_validate.lib.formatters import format_inference_error

        error_output = format_inference_error(str(e))
        click.echo(error_output, err=True)

        steps["inference_test"]["error"] = str(e)
        logger.error(f"Inference test failed: {e}")

        if not skip_on_error:
            summary = format_validation_summary(steps)
            click.echo("\n" + summary)
            sys.exit(1)

    # Display validation summary
    summary = format_validation_summary(steps)
    click.echo("\n" + summary)

    # Determine exit code
    all_passed = all(step["passed"] for step in steps.values())

    if all_passed:
        logger.info("Full validation pipeline passed")
        sys.exit(0)
    else:
        logger.warning("One or more validation steps failed")
        sys.exit(1)
