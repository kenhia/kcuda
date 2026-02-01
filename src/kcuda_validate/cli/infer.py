"""CLI command for running inference with loaded model."""

import click

from kcuda_validate.lib.logger import setup_logger
from kcuda_validate.services.inferencer import InferenceError, Inferencer
from kcuda_validate.services.model_loader import ModelLoader, ModelLoadError

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
    "--repo-id",
    type=str,
    default=None,
    help="Model repository to auto-load [default: Ttimofeyka/MistralRP-Noromaid-NSFW-Mistral-7B-GGUF]",
)
@click.option(
    "--filename",
    type=str,
    default=None,
    help="Model filename to auto-load [default: MistralRP-Noromaid-NSFW-7B-Q4_0.gguf]",
)
def infer(
    prompt: str,
    max_tokens: int,
    temperature: float,
    repo_id: str | None,
    filename: str | None,
) -> None:
    """Execute text generation with loaded model to validate GPU acceleration.

    If no model is currently loaded, one will be automatically downloaded and loaded.
    Use --repo-id and --filename to specify a different model than the default.

    \b
    PROMPT: Text prompt for generation [default: "Hello, how are you?"]

    \b
    Examples:
      kcuda-validate infer "Tell me a story"
      kcuda-validate infer --max-tokens 100 --temperature 0.8 "Explain quantum physics"
      kcuda-validate infer --repo-id TheBloke/Mistral-7B-GGUF --filename model.gguf "What is AI?"
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

        if model_to_use is None:
            click.echo("→ No model loaded, loading model automatically...")

            # Set defaults if not provided
            if repo_id is None:
                repo_id = "Ttimofeyka/MistralRP-Noromaid-NSFW-Mistral-7B-GGUF"
            if filename is None:
                filename = "MistralRP-Noromaid-NSFW-7B-Q4_0.gguf"

            # Load model using ModelLoader
            try:
                loader = ModelLoader()

                # Download model if needed
                model_path = loader.download_model(repo_id=repo_id, filename=filename)
                click.echo(f"✓ Model downloaded: {model_path}")

                # Load model into GPU memory
                model_to_use = loader.load_model(
                    local_path=model_path,
                    repo_id=repo_id,
                    filename=filename,
                    use_gpu=True,
                )
                click.echo(f"✓ Model loaded: {model_to_use.filename}")

            except ModelLoadError as e:
                click.echo(f"✗ Failed to load model: {e}", err=True)
                click.echo("")
                click.echo("Inference test: FAILED")
                raise SystemExit(2) from e
            except Exception as e:
                click.echo(f"✗ Unexpected error loading model: {e}", err=True)
                click.echo("")
                click.echo("Inference test: FAILED")
                raise SystemExit(2) from e

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
            # This should not happen as we auto-load, but handle it anyway
            if "Model cannot be None" in str(e) or "None" in str(e):
                click.echo("✗ No model loaded", err=True)
                click.echo("")
                click.echo("Error: Failed to load model for inference.")
                click.echo("Try explicitly loading with: kcuda-validate load")
                click.echo("")
                click.echo("Inference test: FAILED")
                raise SystemExit(1) from None
            raise
        except RuntimeError as e:
            # Model not loaded or other runtime error
            click.echo(f"✗ Failed to create inferencer: {e}", err=True)
            click.echo("")
            click.echo("Inference test: FAILED")
            raise SystemExit(1) from e

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
            raise SystemExit(3) from e

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
        raise SystemExit(2) from e
