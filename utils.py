"""
Utility functions for AI inference energy profiling.

This module provides common utility functions used across the profiling
infrastructure, including logging setup, file operations, and system utilities.
"""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging configuration for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path. If None, logs to console only.
        log_format: Custom log format string

    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create logger
    logger = logging.getLogger("ai_inference_energy")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def ensure_directory(directory_path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory_path: Path to the directory to create
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def run_command(
    command: List[str],
    timeout: Optional[int] = None,
    capture_output: bool = True,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """
    Run a system command with proper error handling.

    Args:
        command: Command and arguments as a list
        timeout: Optional timeout in seconds
        capture_output: Whether to capture stdout and stderr
        check: Whether to raise an exception on non-zero exit code

    Returns:
        CompletedProcess instance with result information

    Raises:
        subprocess.CalledProcessError: If command fails and check=True
        subprocess.TimeoutExpired: If command times out
    """
    try:
        # Python 3.6 compatibility: use stdout/stderr instead of capture_output
        # and universal_newlines instead of text
        if capture_output:
            result = subprocess.run(
                command,
                timeout=timeout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,  # Python 3.6 compatible (text=True in 3.7+)
                check=check,
            )
        else:
            result = subprocess.run(
                command,
                timeout=timeout,
                universal_newlines=True,
                check=check,  # Python 3.6 compatible (text=True in 3.7+)
            )
        return result
    except subprocess.CalledProcessError as e:
        logger = logging.getLogger("ai_inference_energy")
        logger.error(f"Command failed: {' '.join(command)}")
        logger.error(f"Exit code: {e.returncode}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        raise
    except subprocess.TimeoutExpired as e:
        logger = logging.getLogger("ai_inference_energy")
        logger.error(f"Command timed out: {' '.join(command)}")
        raise


def validate_gpu_available() -> bool:
    """
    Check if NVIDIA GPU is available and accessible.

    Returns:
        True if GPU is available, False otherwise
    """
    try:
        result = run_command(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"]
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def validate_dcgmi_available() -> bool:
    """
    Check if DCGMI (Data Center GPU Manager Interface) is available.

    Returns:
        True if DCGMI is available, False otherwise
    """
    try:
        result = run_command(["dcgmi", "--version"])
        return result.returncode == 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def parse_csv_line(line: str, delimiter: str = ",") -> List[str]:
    """
    Parse a CSV line into fields.

    Args:
        line: CSV line to parse
        delimiter: Field delimiter

    Returns:
        List of parsed fields
    """
    return [field.strip() for field in line.split(delimiter)]


def get_timestamp() -> str:
    """
    Get current timestamp as a formatted string.

    Returns:
        Timestamp string in YYYY-MM-DD_HH-MM-SS format
    """
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def clean_filename(filename: str) -> str:
    """
    Clean a filename by removing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Cleaned filename safe for filesystem use
    """
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")
    return filename


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.2f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours}h {remaining_minutes}m {remaining_seconds:.2f}s"
