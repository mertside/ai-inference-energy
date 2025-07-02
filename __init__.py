"""
AI Inference Energy Profiling Framework

A comprehensive framework for studying energy-efficient GPU frequency selection
for AI inference workloads on NVIDIA A100 and H100 GPUs.

Author: Mert Side
"""

__version__ = "1.0.0"
__author__ = "Mert Side"
__email__ = "mert.side@ttu.edu"

# Import main configuration and utilities
try:
    from .config import gpu_config, model_config, profiling_config, system_config
    from .utils import (
        setup_logging,
        validate_gpu_available,
        validate_dcgmi_available,
        ensure_directory,
        run_command
    )
except ImportError:
    # Handle relative imports when running as scripts
    pass

__all__ = [
    "gpu_config",
    "model_config", 
    "profiling_config",
    "system_config",
    "setup_logging",
    "validate_gpu_available",
    "validate_dcgmi_available",
    "ensure_directory",
    "run_command"
]
