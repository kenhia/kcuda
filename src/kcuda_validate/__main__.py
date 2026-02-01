"""CLI entry point for kcuda-validate."""

import atexit
import contextlib
import os
import signal
import sys
from pathlib import Path

import click

from kcuda_validate import __version__
from kcuda_validate.cli.detect import detect
from kcuda_validate.cli.infer import infer
from kcuda_validate.cli.load import load
from kcuda_validate.cli.validate_all import validate_all
from kcuda_validate.lib.logger import setup_logger

# Default log configuration from environment
DEFAULT_LOG_DIR = Path(os.getenv("KCUDA_LOG_DIR", Path.home() / ".cache" / "kcuda" / "logs"))
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "kcuda-validate.log"
DEFAULT_LOG_LEVEL = os.getenv("KCUDA_LOG_LEVEL", "INFO")

# Track active model loader for cleanup
_active_loader = None


def _cleanup_on_exit():
    """Clean up GPU resources on exit."""
    global _active_loader
    if _active_loader is not None:
        with contextlib.suppress(Exception):
            _active_loader.cleanup()


def _signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    _cleanup_on_exit()
    sys.exit(130)  # Standard exit code for SIGINT


# Register cleanup handlers
atexit.register(_cleanup_on_exit)
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


@click.group(invoke_without_command=True)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output (DEBUG level logging)",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress informational output (ERROR only)",
)
@click.option(
    "--log-file",
    type=click.Path(path_type=Path),
    default=DEFAULT_LOG_FILE,
    help=f"Write logs to specified file [default: {DEFAULT_LOG_FILE}]",
)
@click.option(
    "--no-log-file",
    is_flag=True,
    help="Disable file logging (stdout/stderr only)",
)
@click.option(
    "--version",
    is_flag=True,
    help="Show version and exit",
)
@click.pass_context
def cli(
    ctx: click.Context,
    verbose: bool,
    quiet: bool,
    log_file: Path,
    no_log_file: bool,
    version: bool,
) -> None:
    """CUDA LLM Hardware Validation Tool for WSL2."""
    if version:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as get_version

        # Show package version and key dependency versions
        click.echo(f"kcuda-validate version {__version__}")
        click.echo("")
        click.echo("Dependencies:")

        # Show dependency versions using importlib.metadata
        # This is more reliable than checking __version__ attributes
        deps = [
            ("torch", "torch"),
            ("llama-cpp-python", "llama-cpp-python"),
            ("nvidia-ml-py", "nvidia-ml-py"),
            ("huggingface-hub", "huggingface-hub"),
        ]

        for display_name, package_name in deps:
            try:
                pkg_version = get_version(package_name)
                click.echo(f"  {display_name}: {pkg_version}")
            except PackageNotFoundError:
                click.echo(f"  {display_name}: not installed")

        ctx.exit(0)

    # If invoked without command, show help
    if ctx.invoked_subcommand is None and not version:
        click.echo(ctx.get_help())
        ctx.exit(0)

    # Determine log level
    if verbose:
        log_level = "DEBUG"
    elif quiet:
        log_level = "ERROR"
    else:
        log_level = DEFAULT_LOG_LEVEL

    # Setup logger
    logger = setup_logger(
        name="kcuda_validate",
        log_level=log_level,
        log_file=log_file if not no_log_file else None,
        enable_file_logging=not no_log_file,
    )

    # Store in context for commands
    ctx.ensure_object(dict)
    ctx.obj["logger"] = logger
    ctx.obj["log_file"] = log_file if not no_log_file else None

    logger.debug(f"Starting kcuda-validate v{__version__}")
    logger.debug(f"Log level: {log_level}")
    if not no_log_file:
        logger.debug(f"Log file: {log_file}")


# Register commands
cli.add_command(detect)
cli.add_command(infer)
cli.add_command(load)
cli.add_command(validate_all)


def main() -> None:
    """Main entry point."""
    try:
        cli(obj={})
    except Exception as e:
        click.echo(f"Fatal error: {e}", err=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
