"""
Configuration module for AI inference energy profiling.

This module contains configuration constants and settings used across
the energy profiling infrastructure for AI inference workloads.
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class GPUConfig:
    """Configuration for GPU frequency settings and architecture."""
    
    # A100 GPU configuration
    A100_MEMORY_FREQ: int = 1215  # MHz
    A100_DEFAULT_CORE_FREQ: int = 1410  # MHz
    
    # V100 GPU configuration (commented for reference)
    # V100_MEMORY_FREQ: int = 877  # MHz
    # V100_DEFAULT_CORE_FREQ: int = 1380  # MHz
    
    # Available core frequencies for A100 (MHz)
    A100_CORE_FREQUENCIES: List[int] = [
        1410, 1395, 1380, 1365, 1350, 1335, 1320, 1305, 1290, 1275,
        1260, 1245, 1230, 1215, 1200, 1185, 1170, 1155, 1140, 1125,
        1110, 1095, 1080, 1065, 1050, 1035, 1020, 1005, 990, 975,
        960, 945, 930, 915, 900, 885, 870, 855, 840, 825,
        810, 795, 780, 765, 750, 735, 720, 705, 690, 675,
        660, 645, 630, 615, 600, 585, 570, 555, 540, 525, 510
    ]


@dataclass
class ProfilingConfig:
    """Configuration for power and performance profiling."""
    
    # DCGMI monitoring fields
    DCGMI_FIELDS: List[int] = [
        1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010,
        203, 204, 210, 211, 155, 156, 110
    ]
    
    # Profiling intervals
    DEFAULT_INTERVAL_MS: int = 50  # milliseconds
    SLEEP_INTERVAL_SEC: int = 1  # seconds between runs
    
    # Default number of runs for each frequency
    DEFAULT_NUM_RUNS: int = 2
    
    # Output file configurations
    TEMP_OUTPUT_FILE: str = "changeme"
    RESULTS_DIR: str = "results"
    OUTPUT_FORMAT: str = "csv"


@dataclass
class ModelConfig:
    """Configuration for AI models used in profiling."""
    
    # LLaMA model configuration
    LLAMA_MODEL_NAME: str = "huggyllama/llama-7b"
    LLAMA_TORCH_DTYPE: str = "float16"
    LLAMA_DEFAULT_PROMPT: str = "Plants create energy through a process known as"
    
    # Stable Diffusion model configuration
    STABLE_DIFFUSION_MODEL_NAME: str = "CompVis/stable-diffusion-v1-4"
    STABLE_DIFFUSION_DEFAULT_PROMPT: str = "a photo of an astronaut riding a horse on mars"
    STABLE_DIFFUSION_OUTPUT_FILE: str = "astronaut_rides_horse.png"


@dataclass
class SystemConfig:
    """System-level configuration settings."""
    
    # GPU architecture
    DEFAULT_ARCH: str = "GA100"  # A100 architecture
    PROFILING_MODE: str = "dvfs"
    
    # Paths and directories
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    SCRIPTS_DIR: str = os.path.join(BASE_DIR, "scripts")
    APPS_DIR: str = os.path.join(BASE_DIR, "apps")
    
    # SLURM configuration
    SLURM_PARTITION: str = "toreador"
    SLURM_NODES: int = 1
    SLURM_TASKS_PER_NODE: int = 16
    SLURM_GPUS_PER_NODE: int = 1


# Global configuration instances
gpu_config = GPUConfig()
profiling_config = ProfilingConfig()
model_config = ModelConfig()
system_config = SystemConfig()
