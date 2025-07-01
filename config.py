"""
Configuration module for AI inference energy profiling.

This module contains configuration constants and settings used across
the energy profiling infrastructure for AI inference workloads.

Compatible with Python 3.6+
"""

import os

# GPU Configuration Constants
class GPUConfig:
    """Configuration for GPU frequency settings and architecture."""
    
    # A100 GPU configuration
    A100_MEMORY_FREQ = 1215  # MHz
    A100_DEFAULT_CORE_FREQ = 1410  # MHz
    
    # V100 GPU configuration
    V100_MEMORY_FREQ = 877  # MHz
    V100_DEFAULT_CORE_FREQ = 1380  # MHz
    
    # Available core frequencies for A100 (MHz)
    A100_CORE_FREQUENCIES = [
        1410, 1395, 1380, 1365, 1350, 1335, 1320, 1305, 1290, 1275,
        1260, 1245, 1230, 1215, 1200, 1185, 1170, 1155, 1140, 1125,
        1110, 1095, 1080, 1065, 1050, 1035, 1020, 1005, 990, 975,
        960, 945, 930, 915, 900, 885, 870, 855, 840, 825,
        810, 795, 780, 765, 750, 735, 720, 705, 690, 675,
        660, 645, 630, 615, 600, 585, 570, 555, 540, 525, 510
    ]
    
    # Available core frequencies for V100 (MHz)
    V100_CORE_FREQUENCIES = [
        1380, 1372, 1365, 1357, 1350, 1342, 1335, 1327, 1320, 1312, 1305, 1297, 1290, 1282, 1275, 1267, 
        1260, 1252, 1245, 1237, 1230, 1222, 1215, 1207, 1200, 1192, 1185, 1177, 1170, 1162, 1155, 1147, 
        1140, 1132, 1125, 1117, 1110, 1102, 1095, 1087, 1080, 1072, 1065, 1057, 1050, 1042, 1035, 1027, 
        1020, 1012, 1005, 997, 990, 982, 975, 967, 960, 952, 945, 937, 930, 922, 915, 907, 900, 892, 885, 877, 
        870, 862, 855, 847, 840, 832, 825, 817, 810, 802, 795, 787, 780, 772, 765, 757, 750, 742, 735, 727, 
        720, 712, 705, 697, 690, 682, 675, 667, 660, 652, 645, 637, 630, 622, 615, 607, 600, 592, 585, 577, 
        570, 562, 555, 547, 540, 532, 525, 517, 510, 502, 495, 487, 480, 472, 465, 457, 450, 442, 435, 427, 
        420, 412, 405
    ]


class ProfilingConfig:
    """Configuration for power and performance profiling."""
    
    # DCGMI monitoring fields
    DCGMI_FIELDS = [
        1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010,
        203, 204, 210, 211, 155, 156, 110
    ]
    
    # Profiling intervals
    DEFAULT_INTERVAL_MS = 50  # milliseconds
    SLEEP_INTERVAL_SEC = 1  # seconds between runs
    
    # Default number of runs for each frequency
    DEFAULT_NUM_RUNS = 2
    
    # Output file configurations
    TEMP_OUTPUT_FILE = "changeme"
    RESULTS_DIR = "results"
    OUTPUT_FORMAT = "csv"


class ModelConfig:
    """Configuration for AI models used in profiling."""
    
    # LLaMA model configuration
    LLAMA_MODEL_NAME = "huggyllama/llama-7b"
    LLAMA_TORCH_DTYPE = "float16"
    LLAMA_DEFAULT_PROMPT = "Plants create energy through a process known as"
    
    # Stable Diffusion model configuration
    STABLE_DIFFUSION_MODEL_NAME = "CompVis/stable-diffusion-v1-4"
    STABLE_DIFFUSION_DEFAULT_PROMPT = "a photo of an astronaut riding a horse on mars"
    STABLE_DIFFUSION_OUTPUT_FILE = "astronaut_rides_horse.png"


class SystemConfig:
    """System-level configuration settings."""
    
    # GPU architecture
    DEFAULT_ARCH = "GA100"  # A100 architecture
    PROFILING_MODE = "dvfs"
    
    # Paths and directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
    APPS_DIR = os.path.join(BASE_DIR, "apps")
    
    # SLURM configuration
    SLURM_PARTITION = "toreador"
    SLURM_NODES = 1
    SLURM_TASKS_PER_NODE = 16
    SLURM_GPUS_PER_NODE = 1


# Global configuration instances
gpu_config = GPUConfig()
profiling_config = ProfilingConfig()
model_config = ModelConfig()
system_config = SystemConfig()
"""
Configuration module for AI inference energy profiling.

This module contains configuration constants and settings used across
the energy profiling infrastructure for AI inference workloads.

Compatible with Python 3.6+
"""

import os

# GPU Configuration Constants
class GPUConfig:
    """Configuration for GPU frequency settings and architecture."""
    
    # A100 GPU configuration
    A100_MEMORY_FREQ = 1215  # MHz
    A100_DEFAULT_CORE_FREQ = 1410  # MHz
    
    # V100 GPU configuration
    V100_MEMORY_FREQ = 877  # MHz
    V100_DEFAULT_CORE_FREQ = 1380  # MHz
    
    # Available core frequencies for A100 (MHz)
    A100_CORE_FREQUENCIES = [
        1410, 1395, 1380, 1365, 1350, 1335, 1320, 1305, 1290, 1275,
        1260, 1245, 1230, 1215, 1200, 1185, 1170, 1155, 1140, 1125,
        1110, 1095, 1080, 1065, 1050, 1035, 1020, 1005, 990, 975,
        960, 945, 930, 915, 900, 885, 870, 855, 840, 825,
        810, 795, 780, 765, 750, 735, 720, 705, 690, 675,
        660, 645, 630, 615, 600, 585, 570, 555, 540, 525, 510
    ]
    
    # Available core frequencies for V100 (MHz)
    V100_CORE_FREQUENCIES = [
        1380, 1372, 1365, 1357, 1350, 1342, 1335, 1327, 1320, 1312, 1305, 1297, 1290, 1282, 1275, 1267, 
        1260, 1252, 1245, 1237, 1230, 1222, 1215, 1207, 1200, 1192, 1185, 1177, 1170, 1162, 1155, 1147, 
        1140, 1132, 1125, 1117, 1110, 1102, 1095, 1087, 1080, 1072, 1065, 1057, 1050, 1042, 1035, 1027, 
        1020, 1012, 1005, 997, 990, 982, 975, 967, 960, 952, 945, 937, 930, 922, 915, 907, 900, 892, 885, 877, 
        870, 862, 855, 847, 840, 832, 825, 817, 810, 802, 795, 787, 780, 772, 765, 757, 750, 742, 735, 727, 
        720, 712, 705, 697, 690, 682, 675, 667, 660, 652, 645, 637, 630, 622, 615, 607, 600, 592, 585, 577, 
        570, 562, 555, 547, 540, 532, 525, 517, 510, 502, 495, 487, 480, 472, 465, 457, 450, 442, 435, 427, 
        420, 412, 405
    ]


class ProfilingConfig:
    """Configuration for power and performance profiling."""
    
    # DCGMI monitoring fields
    DCGMI_FIELDS = [
        1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010,
        203, 204, 210, 211, 155, 156, 110
    ]
    
    # Profiling intervals
    DEFAULT_INTERVAL_MS = 50  # milliseconds
    SLEEP_INTERVAL_SEC = 1  # seconds between runs
    
    # Default number of runs for each frequency
    DEFAULT_NUM_RUNS = 2
    
    # Output file configurations
    TEMP_OUTPUT_FILE = "changeme"
    RESULTS_DIR = "results"
    OUTPUT_FORMAT = "csv"


class ModelConfig:
    """Configuration for AI models used in profiling."""
    
    # LLaMA model configuration
    LLAMA_MODEL_NAME = "huggyllama/llama-7b"
    LLAMA_TORCH_DTYPE = "float16"
    LLAMA_DEFAULT_PROMPT = "Plants create energy through a process known as"
    
    # Stable Diffusion model configuration
    STABLE_DIFFUSION_MODEL_NAME = "CompVis/stable-diffusion-v1-4"
    STABLE_DIFFUSION_DEFAULT_PROMPT = "a photo of an astronaut riding a horse on mars"
    STABLE_DIFFUSION_OUTPUT_FILE = "astronaut_rides_horse.png"


class SystemConfig:
    """System-level configuration settings."""
    
    # GPU architecture
    DEFAULT_ARCH = "GA100"  # A100 architecture
    PROFILING_MODE = "dvfs"
    
    # Paths and directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
    APPS_DIR = os.path.join(BASE_DIR, "apps")
    
    # SLURM configuration
    SLURM_PARTITION = "toreador"
    SLURM_NODES = 1
    SLURM_TASKS_PER_NODE = 16
    SLURM_GPUS_PER_NODE = 1


# Global configuration instances
gpu_config = GPUConfig()
profiling_config = ProfilingConfig()
model_config = ModelConfig()
system_config = SystemConfig()