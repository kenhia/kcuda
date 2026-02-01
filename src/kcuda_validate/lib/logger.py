"""
Logging infrastructure for kcuda-validate.

Provides file-based logging with rotation for debugging CUDA/hardware issues.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Default log settings
DEFAULT_LOG_DIR = Path.home() / ".cache" / "kcuda" / "logs"
DEFAULT_LOG_FILE = "kcuda-validate.log"
MAX_BYTES = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 5

# Global to track active log file path
_active_log_file: Path | None = None


def get_log_file_path() -> Path | None:
    """Get the current log file path.

    Returns:
        Path to the active log file, or None if file logging is disabled
    """
    return _active_log_file


def setup_logger(
    name: str = "kcuda_validate",
    log_level: str = "INFO",
    log_file: Path | None = None,
    enable_file_logging: bool = True,
) -> logging.Logger:
    """
    Setup logger with console and optional file output.

    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (defaults to ~/.cache/kcuda/logs/kcuda-validate.log)
        enable_file_logging: Whether to log to file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)  # Only warnings/errors to console by default
    console_format = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (if enabled)
    if enable_file_logging:
        global _active_log_file
        if log_file is None:
            log_file = DEFAULT_LOG_DIR / DEFAULT_LOG_FILE

        _active_log_file = log_file

        # Create log directory
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(log_file, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT)
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")

    return logger


def get_logger(name: str = "kcuda_validate") -> logging.Logger:
    """Get or create logger instance."""
    return logging.getLogger(name)
