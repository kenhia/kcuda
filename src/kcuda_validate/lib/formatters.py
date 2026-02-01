"""CLI output formatting utilities using rich library."""

from io import StringIO
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from kcuda_validate.models.gpu_device import GPUDevice
from kcuda_validate.models.inference_result import InferenceResult
from kcuda_validate.models.llm_model import LLMModel

console = Console()


def _get_log_file_message() -> str:
    """Get log file path message if logging is enabled.

    Returns:
        Message about log file location, or empty string if no log file
    """
    try:
        from kcuda_validate.lib.logger import get_log_file_path

        log_path = get_log_file_path()
        if log_path:
            return f"\nFor detailed diagnostics, see: {log_path}"
    except Exception:
        # Log file path retrieval is optional; if it fails, continue without it
        pass
    return ""


def format_gpu_info(gpu: GPUDevice, warnings: list[str] | None = None) -> str:
    """Format GPU detection success output as string.

    Args:
        gpu: Detected GPU device
        warnings: Optional list of warning messages for requirement violations

    Returns:
        Formatted output string
    """
    buffer = StringIO()
    temp_console = Console(file=buffer, force_terminal=True)

    temp_console.print("✓ CUDA Available: Yes", style="bold green")
    temp_console.print(f"✓ GPU Detected: {gpu.name}", style="bold green")
    temp_console.print(f"  - VRAM Total: {gpu.vram_total_mb} MB")
    temp_console.print(f"  - VRAM Free: {gpu.vram_free_mb} MB")
    temp_console.print(f"  - CUDA Version: {gpu.cuda_version}")
    temp_console.print(f"  - Driver Version: {gpu.driver_version}")
    temp_console.print(f"  - Compute Capability: {gpu.compute_capability}")
    temp_console.print(f"  - Device ID: {gpu.device_id}")
    temp_console.print()

    if warnings:
        temp_console.print("⚠ Warnings:", style="bold yellow")
        for warning in warnings:
            temp_console.print(f"  - {warning}", style="yellow")
        temp_console.print()
        temp_console.print("Hardware validation: FAILED (requirements not met)", style="bold red")
    else:
        temp_console.print("Hardware validation: PASSED", style="bold green")

    return buffer.getvalue()


def format_gpu_error(error_message: str) -> str:
    """Format GPU detection error output as string.

    Args:
        error_message: Error message to display

    Returns:
        Formatted error output string
    """
    buffer = StringIO()
    temp_console = Console(file=buffer, force_terminal=True)

    temp_console.print("✗ CUDA Available: No", style="bold red")
    temp_console.print("✗ GPU Detected: None", style="bold red")
    temp_console.print()
    temp_console.print(f"Error: {error_message}", style="red")
    temp_console.print()
    temp_console.print("Recommendation: Ensure:", style="yellow")
    temp_console.print("  1. NVIDIA GPU drivers are installed on Windows host (WSL2)")
    temp_console.print("  2. WSL2 GPU passthrough is enabled")
    temp_console.print("  3. nvidia-smi works in WSL2 terminal")

    log_msg = _get_log_file_message()
    if log_msg:
        temp_console.print(log_msg, style="dim")
    temp_console.print()
    temp_console.print("Hardware validation: FAILED", style="bold red")

    return buffer.getvalue()


def format_model_error(error_message: str) -> str:
    """Format model loading error output as string.

    Args:
        error_message: Error message to display

    Returns:
        Formatted error output string
    """
    buffer = StringIO()
    temp_console = Console(file=buffer, force_terminal=True)

    temp_console.print("\n✗ Model loading failed", style="bold red")
    temp_console.print(f"Error: {error_message}", style="red")

    log_msg = _get_log_file_message()
    if log_msg:
        temp_console.print(log_msg, style="dim")
    temp_console.print()
    temp_console.print("Model load: FAILED", style="bold red")

    return buffer.getvalue()


def format_inference_error(error_message: str) -> str:
    """Format inference error output as string.

    Args:
        error_message: Error message to display

    Returns:
        Formatted error output string
    """
    buffer = StringIO()
    temp_console = Console(file=buffer, force_terminal=True)

    temp_console.print("\n✗ Inference failed", style="bold red")
    temp_console.print(f"Error: {error_message}", style="red")

    log_msg = _get_log_file_message()
    if log_msg:
        temp_console.print(log_msg, style="dim")
    temp_console.print()
    temp_console.print("Inference test: FAILED", style="bold red")

    return buffer.getvalue()


def format_gpu_detection_success(gpu: GPUDevice) -> None:
    """
    Format and display successful GPU detection.

    Args:
        gpu: Detected GPU device
    """
    console.print("✓ CUDA Available: Yes", style="bold green")
    console.print(f"✓ GPU Detected: {gpu.name}", style="bold green")
    console.print(f"  - VRAM Total: {gpu.vram_total_mb} MB")
    console.print(f"  - VRAM Free: {gpu.vram_free_mb} MB")
    console.print(f"  - CUDA Version: {gpu.cuda_version}")
    console.print(f"  - Driver Version: {gpu.driver_version}")
    console.print(f"  - Compute Capability: {gpu.compute_capability}")
    console.print(f"  - Device ID: {gpu.device_id}")
    console.print()
    console.print("Hardware validation: PASSED", style="bold green")


def format_gpu_detection_failure(error: str, log_file: Path | None = None) -> None:
    """
    Format and display GPU detection failure.

    Args:
        error: Error message
        log_file: Path to log file for debugging
    """
    console.print("✗ CUDA Available: No", style="bold red")
    console.print("✗ GPU Detected: None", style="bold red")
    console.print()
    console.print(f"Error: {error}", style="red")
    console.print()
    console.print("Recommendation: Ensure:", style="yellow")
    console.print("  1. NVIDIA GPU drivers are installed on Windows host (WSL2)")
    console.print("  2. WSL2 GPU passthrough is enabled")
    console.print("  3. nvidia-smi works in WSL2 terminal")

    if log_file:
        console.print(f"\nLog file: {log_file}", style="dim")

    console.print()
    console.print("Hardware validation: FAILED", style="bold red")


def format_model_loading_progress() -> Progress:
    """
    Create progress indicator for model loading.

    Returns:
        Progress instance for model loading
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    )


def format_model_info(model: LLMModel) -> str:
    """Format model information after successful load.

    Args:
        model: Loaded LLM model

    Returns:
        Formatted model info string
    """
    buffer = StringIO()
    temp_console = Console(file=buffer, force_terminal=True, legacy_windows=False)

    temp_console.print("\n[bold cyan]Model Information:[/bold cyan]")
    temp_console.print(f"  Repository: [bold]{model.repo_id}[/bold]")
    temp_console.print(f"  Filename: {model.filename}")
    temp_console.print(f"  File Size: {model.file_size_mb:,.0f} MB")
    temp_console.print(f"  Parameters: {model.parameter_count / 1e9:.2f}B")
    temp_console.print(f"  Quantization: {model.quantization_type}")
    temp_console.print(f"  Context Length: {model.context_length:,} tokens")
    if model.is_loaded:
        temp_console.print(f"  VRAM Usage: {model.vram_usage_mb:,.0f} MB")
        temp_console.print("  Status: [green]Loaded[/green]")
    else:
        temp_console.print("  Status: [yellow]Downloaded[/yellow]")

    return buffer.getvalue()


def format_model_loaded_success(model: LLMModel, gpu_free_vram_mb: int) -> None:
    """
    Format and display successful model loading.

    Args:
        model: Loaded model
        gpu_free_vram_mb: Free VRAM remaining after load
    """
    console.print("✓ Model loaded successfully", style="bold green")
    console.print(f"  - Parameters: {model.parameter_count / 1e9:.2f}B")
    console.print(f"  - Quantization: {model.quantization_type}")
    console.print(f"  - Context Length: {model.context_length} tokens")
    console.print(f"  - VRAM Usage: {model.vram_usage_mb} MB")
    console.print(f"  - Free VRAM Remaining: {gpu_free_vram_mb} MB")
    console.print()
    console.print("Model load: PASSED", style="bold green")


def format_model_loading_failure(error: str, log_file: Path | None = None) -> None:
    """
    Format and display model loading failure.

    Args:
        error: Error message
        log_file: Path to log file for debugging
    """
    console.print("✗ Model load failed", style="bold red")
    console.print()
    console.print(f"Error: {error}", style="red")

    if log_file:
        console.print(f"Log file: {log_file}", style="dim")

    console.print()
    console.print("Model load: FAILED", style="bold red")


def format_inference_result(result: InferenceResult) -> None:
    """
    Format and display inference result.

    Args:
        result: Inference result
    """
    if not result.success:
        console.print("✗ Inference failed", style="bold red")
        console.print()
        console.print(f"Error: {result.error_message}", style="red")
        console.print()
        console.print("Inference test: FAILED", style="bold red")
        return

    # Display response
    console.print("─" * 60)
    console.print("Response:", style="bold")
    console.print(result.response)
    console.print("─" * 60)
    console.print()

    console.print("✓ Inference completed successfully", style="bold green")
    console.print()

    # Performance metrics table
    table = Table(title="Performance Metrics", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Tokens Generated", str(result.tokens_generated))
    table.add_row("Time to First Token", f"{result.time_to_first_token_sec:.2f} seconds")
    table.add_row("Total Time", f"{result.total_time_sec:.2f} seconds")
    table.add_row("Throughput", f"{result.tokens_per_second:.1f} tokens/second")

    if result.gpu_utilization_percent is not None:
        table.add_row("GPU Utilization", f"{result.gpu_utilization_percent:.0f}% (peak)")

    if result.vram_peak_mb is not None:
        table.add_row("VRAM Peak", f"{result.vram_peak_mb} MB")

    console.print(table)
    console.print()
    console.print("Inference test: PASSED", style="bold green")


def format_validation_summary(steps: dict) -> str:
    """Format validation summary for validate-all command.

    Args:
        steps: Dictionary with step results, format:
            {
                "gpu_detection": {"passed": bool, "error": str|None},
                "model_loading": {"passed": bool, "error": str|None},
                "inference_test": {"passed": bool, "error": str|None}
            }

    Returns:
        Formatted summary string
    """
    buffer = StringIO()
    temp_console = Console(file=buffer, force_terminal=True)

    temp_console.print("═" * 60)
    temp_console.print("Validation Summary:", style="bold")
    temp_console.print("═" * 60)

    # Format each step
    step_labels = {
        "gpu_detection": "GPU Detection",
        "model_loading": "Model Loading",
        "inference_test": "Inference Test",
    }

    for step_key, label in step_labels.items():
        if step_key in steps:
            passed = steps[step_key]["passed"]
            status_mark = "✓" if passed else "✗"
            status_text = "PASSED" if passed else "FAILED"
            status_style = "green" if passed else "red"

            temp_console.print(
                f"{status_mark} {label}: {status_text}",
                style=status_style,
            )

    temp_console.print()

    # Overall status
    all_passed = all(step["passed"] for step in steps.values())
    overall_status = "SUCCESS" if all_passed else "FAILED"
    overall_style = "bold green" if all_passed else "bold red"

    temp_console.print(f"Overall Status: {overall_status}", style=overall_style)

    if all_passed:
        temp_console.print("\nSystem ready for LLM development.", style="green")

    temp_console.print("═" * 60)

    return buffer.getvalue()


def format_error(title: str, message: str) -> str:
    """Format error message with title.

    Args:
        title: Error title
        message: Error message

    Returns:
        Formatted error string
    """
    buffer = StringIO()
    temp_console = Console(file=buffer, force_terminal=True)

    temp_console.print(f"\n[bold red]✗ {title}[/bold red]")
    temp_console.print(f"[red]{message}[/red]\n")

    return buffer.getvalue()
