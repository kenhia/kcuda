"""CLI command for running inference with loaded model."""

import click

from kcuda_validate.lib.logger import setup_logger
from kcuda_validate.services.inferencer import Inferencer, InferenceError

# Will be accessed from module state (loaded by load command)
_loaded_model = None

logger = setup_logger(__name__)


@click.command()
@click.argument("prompt", type=str, required=False, default="Hello, how are you?")
@click.option(
    "--max-tokens",
    type=int,
    default=50,
    help="Maximum tokens to generate [default: 50]",
)
@click.option(
    "--temperature",
    type=float,
    default=0.7,
    help="Sampling temperature [default: 0.7]",
)
@click.option(
    "--load-model",
    is_flag=True,
    help="Automatically load default model if not loaded",
)
@click.option(
    "--repo-id",
    type=str,
    default=None,
    help="Model repo (if --load-model used)",
)
@click.option(
    "--filename",
    type=str,
    default=None,
    help="Model file (if --load-model used)",
)
def infer(
    prompt: str,
    max_tokens: int,
    temperature: float,
    load_model: bool,
    repo_id: str | None,
    filename: str | None,
) -> None:
    """Execute text generation with loaded model to validate GPU acceleration.

    \b
    PROMPT: Text prompt for generation [default: "Hello, how are you?"]

    \b
    Examples:
      kcuda-validate infer "Tell me a story"
      kcuda-validate infer --max-tokens 100 --temperature 0.8 "Explain quantum physics"
      kcuda-validate infer --load-model "What is AI?"
    """
    try:
        # Validate prompt
        if not prompt or not prompt.strip():
            click.echo("✗ Inference failed: Empty prompt", err=True)
            click.echo("")
            click.echo("Error: Prompt cannot be empty. Provide text for generation.")
            click.echo('Example: kcuda-validate infer "Tell me a story"')
            click.echo("")
            click.echo("Inference test: FAILED")
            raise SystemExit(3)

        # Check if model is loaded or auto-load
        click.echo("→ Checking model status...")

        model_to_use = _loaded_model

        if model_to_use is None and load_model:
            click.echo("→ Loading model...")
            # TODO: Implement auto-load functionality
            click.echo("✗ Auto-load not yet implemented", err=True)
            raise SystemExit(1)

        # Display model status
        if model_to_use is not None:
            click.echo(f"✓ Model loaded: {model_to_use.filename}")
            click.echo("")

        # Run inference
        click.echo("→ Running inference...")
        click.echo(f'  Prompt: "{prompt}"')
        click.echo("")

        # Create inferencer (will raise RuntimeError if no model)
        try:
            # Get model instance - if None, Inferencer will handle error
            model_instance = model_to_use.instance if model_to_use else None
            inferencer = Inferencer(model=model_instance)
        except TypeError as e:
            # Model is None - Inferencer requires non-None model
            if "Model cannot be None" in str(e) or "None" in str(e):
                click.echo("✗ No model loaded", err=True)
                click.echo("")
                click.echo("Error: Must load model before running inference.")
                click.echo("Run: kcuda-validate load")
                click.echo("")
                click.echo('Or use: kcuda-validate infer --load-model "Your prompt"')
                click.echo("")
                click.echo("Inference test: FAILED")
                raise SystemExit(1)
            raise
        except RuntimeError as e:
            # Model not loaded or other runtime error
            click.echo(f"✗ Failed to create inferencer: {e}", err=True)
            click.echo("")
            click.echo("Inference test: FAILED")
            raise SystemExit(1)

        # Generate response
        try:
            result = inferencer.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except InferenceError as e:
            # Validation errors
            click.echo(f"✗ Inference failed: {e}", err=True)
            click.echo("")
            click.echo("Inference test: FAILED")
            raise SystemExit(3)

        # Check if generation succeeded
        if not result.success:
            click.echo("✗ Inference failed", err=True)
            click.echo("")
            click.echo(f"Error: {result.error_message}")
            click.echo("")
            click.echo("Inference test: FAILED")
            raise SystemExit(2)

        # Display response
        click.echo("─" * 80)
        click.echo("Response:")
        click.echo(result.response)
        click.echo("─" * 80)
        click.echo("")
        click.echo("✓ Inference completed successfully")
        click.echo("")

        # Display performance metrics
        click.echo("Performance Metrics:")
        click.echo(f"  - Tokens Generated: {result.tokens_generated}")
        click.echo(f"  - Time to First Token: {result.time_to_first_token_sec:.2f} seconds")
        click.echo(f"  - Total Time: {result.total_time_sec:.2f} seconds")
        click.echo(f"  - Throughput: {result.tokens_per_second:.1f} tokens/second")

        if result.gpu_utilization_percent is not None:
            click.echo(f"  - GPU Utilization: {result.gpu_utilization_percent:.0f}% (peak)")

        if result.vram_peak_mb is not None:
            click.echo(f"  - VRAM Peak: {result.vram_peak_mb} MB")

        click.echo("")
        click.echo("Inference test: PASSED")

    except SystemExit:
        raise
    except Exception as e:
        logger.exception("Unexpected error during inference")
        click.echo(f"✗ Unexpected error: {e}", err=True)
        click.echo("")
        click.echo("Inference test: FAILED")
        raise SystemExit(2)
