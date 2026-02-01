"""CLI command for loading GGUF models into GPU memory."""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from kcuda_validate.lib.formatters import format_error, format_model_info
from kcuda_validate.lib.logger import get_logger
from kcuda_validate.services.model_loader import ModelLoader, ModelLoadError

logger = get_logger(__name__)
console = Console()


@click.command(name="load")
@click.option(
    "--repo-id",
    default="Ttimofeyka/MistralRP-Noromaid-NSFW-Mistral-7B-GGUF",
    help="Hugging Face repository ID [default: Ttimofeyka/MistralRP-Noromaid-NSFW-Mistral-7B-GGUF]",
)
@click.option(
    "--filename",
    default="MistralRP-Noromaid-NSFW-7B-Q4_0.gguf",
    help="Model filename within repository [default: MistralRP-Noromaid-NSFW-7B-Q4_0.gguf]",
)
@click.option(
    "--skip-download",
    is_flag=True,
    help="Skip download step if model already exists locally",
)
@click.option(
    "--no-gpu",
    is_flag=True,
    help="Load model in CPU-only mode (slower but works without CUDA)",
)
@click.option(
    "--model-dir",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    default=Path("./models"),
    help="Directory to store downloaded models (default: ./models)",
)
@click.option(
    "--n-ctx",
    type=int,
    default=8192,
    help="Context window size (default: 8192)",
)
def load(
    repo_id: str,
    filename: str,
    skip_download: bool,
    no_gpu: bool,
    model_dir: Path,
    n_ctx: int,
) -> None:
    """Load a GGUF model from Hugging Face into GPU memory.

    This command downloads (if needed) and loads a quantized GGUF model
    into GPU memory for inference. It validates VRAM availability and
    provides detailed model metadata.

    Example:
        kcuda-validate load \\
            --repo-id TheBloke/Mistral-7B-Instruct-v0.2-GGUF \\
            --filename mistral-7b-instruct-v0.2.Q4_K_M.gguf

    Exit Codes:
        0: Success - model loaded successfully
        1: Download failure (network error, 404, etc.)
        2: Load failure (insufficient VRAM, corrupt file, etc.)
        3: General error (invalid arguments, etc.)
    """
    logger.info(f"Loading model: {repo_id}/{filename}")

    # Initialize loader
    loader = ModelLoader()

    # Determine local path
    model_dir.mkdir(parents=True, exist_ok=True)
    local_path = model_dir / filename

    # Download phase
    if not skip_download or not local_path.exists():
        try:
            console.print("\n[cyan]Downloading model from Hugging Face...[/cyan]")
            console.print(f"Repository: [bold]{repo_id}[/bold]")
            console.print(f"File: [bold]{filename}[/bold]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Downloading...", total=None)
                downloaded_path = loader.download_model(
                    repo_id=repo_id,
                    filename=filename,
                )
                progress.update(task, completed=True)

            console.print(f"[green]✓[/green] Downloaded to: {downloaded_path}\n")
            local_path = Path(downloaded_path)

        except ModelLoadError as e:
            # Print to stderr to ensure ANSI codes are visible
            console.print(format_error("Download Failed", str(e)), markup=False)
            logger.error(f"Download failed: {e}")
            sys.exit(1)
    else:
        console.print(f"[cyan]Using existing model:[/cyan] {local_path}\n")

    # Load phase
    try:
        console.print("[cyan]Loading model into memory...[/cyan]")

        use_gpu = not no_gpu
        if no_gpu:
            console.print("[yellow]⚠[/yellow] Running in CPU-only mode (slower)\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading...", total=None)
            model = loader.load_model(
                local_path=str(local_path),
                repo_id=repo_id,
                filename=filename,
                use_gpu=use_gpu,
                n_ctx=n_ctx,
            )
            progress.update(task, completed=True)

        # Display model information (already formatted with ANSI codes)
        print(format_model_info(model))
        console.print("\n[green]✓ Model loaded successfully - PASSED![/green]")

        logger.info(f"Model loaded: {model.repo_id}/{model.filename}")
        sys.exit(0)

    except ModelLoadError as e:
        # Print to stderr to ensure ANSI codes are visible
        console.print(format_error("Load Failed", str(e)), markup=False)
        logger.error(f"Load failed: {e}")

        # Determine exit code based on error type
        error_msg = str(e).lower()
        if (
            "vram" in error_msg
            or "memory" in error_msg
            or "corrupt" in error_msg
            or "invalid" in error_msg
        ):
            sys.exit(2)
        else:
            sys.exit(2)

    except Exception as e:
        # Print to stderr to ensure ANSI codes are visible
        console.print(format_error("Error", str(e)), markup=False)
        logger.error(f"Unexpected error: {e}")
        sys.exit(3)
