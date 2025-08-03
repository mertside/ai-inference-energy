"""
Configuration module for AI inference energy profiling.

This module contains configuration constants and settings used across
the energy profiling infrastructure for AI inference workloads.

Compatible with Python 3.8+
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

    # H100 GPU configuration
    H100_MEMORY_FREQ = 2619  # MHz (maximum H100 memory frequency)
    H100_DEFAULT_CORE_FREQ = 1785  # MHz (maximum frequency)

    # Available core frequencies for A100 (MHz)
    A100_CORE_FREQUENCIES = [
        1410,
        1395,
        1380,
        1365,
        1350,
        1335,
        1320,
        1305,
        1290,
        1275,
        1260,
        1245,
        1230,
        1215,
        1200,
        1185,
        1170,
        1155,
        1140,
        1125,
        1110,
        1095,
        1080,
        1065,
        1050,
        1035,
        1020,
        1005,
        990,
        975,
        960,
        945,
        930,
        915,
        900,
        885,
        870,
        855,
        840,
        825,
        810,
        795,
        780,
        765,
        750,
        735,
        720,
        705,
        690,
        675,
        660,
        645,
        630,
        615,
        600,
        585,
        570,
        555,
        540,
        525,
        510,
    ]

    # Available core frequencies for V100 (MHz) - updated with actual hardware data (≥510 MHz only)
    V100_CORE_FREQUENCIES = [
        1380,
        1372,
        1365,
        1357,
        1350,
        1342,
        1335,
        1327,
        1320,
        1312,
        1305,
        1297,
        1290,
        1282,
        1275,
        1267,
        1260,
        1252,
        1245,
        1237,
        1230,
        1222,
        1215,
        1207,
        1200,
        1192,
        1185,
        1177,
        1170,
        1162,
        1155,
        1147,
        1140,
        1132,
        1125,
        1117,
        1110,
        1102,
        1095,
        1087,
        1080,
        1072,
        1065,
        1057,
        1050,
        1042,
        1035,
        1027,
        1020,
        1012,
        1005,
        997,
        990,
        982,
        975,
        967,
        960,
        952,
        945,
        937,
        930,
        922,
        915,
        907,
        900,
        892,
        885,
        877,
        870,
        862,
        855,
        847,
        840,
        832,
        825,
        817,
        810,
        802,
        795,
        787,
        780,
        772,
        765,
        757,
        750,
        742,
        735,
        727,
        720,
        712,
        705,
        697,
        690,
        682,
        675,
        667,
        660,
        652,
        645,
        637,
        630,
        622,
        615,
        607,
        600,
        592,
        585,
        577,
        570,
        562,
        555,
        547,
        540,
        532,
        525,
        517,
        510,
    ]

    # Available core frequencies for H100 (MHz)
    # Based on nvidia-smi output: 510-1785 MHz in 15-MHz steps (86 frequencies)
    H100_CORE_FREQUENCIES = [
        1785,
        1770,
        1755,
        1740,
        1725,
        1710,
        1695,
        1680,
        1665,
        1650,
        1635,
        1620,
        1605,
        1590,
        1575,
        1560,
        1545,
        1530,
        1515,
        1500,
        1485,
        1470,
        1455,
        1440,
        1425,
        1410,
        1395,
        1380,
        1365,
        1350,
        1335,
        1320,
        1305,
        1290,
        1275,
        1260,
        1245,
        1230,
        1215,
        1200,
        1185,
        1170,
        1155,
        1140,
        1125,
        1110,
        1095,
        1080,
        1065,
        1050,
        1035,
        1020,
        1005,
        990,
        975,
        960,
        945,
        930,
        915,
        900,
        885,
        870,
        855,
        840,
        825,
        810,
        795,
        780,
        765,
        750,
        735,
        720,
        705,
        690,
        675,
        660,
        645,
        630,
        615,
        600,
        585,
        570,
        555,
        540,
        525,
        510,
        495,
        480,
        465,
        450,
        435,
        420,
        405,
        390,
        375,
        360,
        345,
        330,
        315,
        300,
        285,
        270,
        255,
        240,
        225,
        210,
    ]


class ProfilingConfig:
    """Configuration for power and performance profiling."""

    # DCGMI monitoring fields
    # This matches the field configuration actually used by profile.py
    # DCGMI one-liner: dcgmi dmon -d 50 -e 52,50,155,160,156,150,140,203,204,250,251,252,100,101,110,111,190,1001,1002,1003,1004,1005,1006,1007,1008 -c 1
    DCGMI_FIELDS = [
        # Basic device information
        52,          # DCGM_FI_DEV_NVML_INDEX - GPU device index
        50,          # DCGM_FI_DEV_NAME - GPU device name
        
        # Power metrics
        155,         # DCGM_FI_DEV_POWER_USAGE - Current power draw (W)
        160,         # DCGM_FI_DEV_POWER_MGMT_LIMIT - Power management limit (W)
        156,         # DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION - Total energy consumption (mJ)
        
        # Temperature metrics
        150,         # DCGM_FI_DEV_GPU_TEMP - GPU temperature (C)
        140,         # DCGM_FI_DEV_MEMORY_TEMP - Memory (HBM) temperature (C)
        
        # Utilization metrics
        203,         # DCGM_FI_DEV_GPU_UTIL - GPU utilization (coarse) (%)
        204,         # DCGM_FI_DEV_MEM_COPY_UTIL - Memory copy utilization (≈ util.mem) (%)
        
        # Memory metrics
        250,         # DCGM_FI_DEV_FB_TOTAL - Total framebuffer memory (MB)
        251,         # DCGM_FI_DEV_FB_FREE - Free framebuffer memory (MB)
        252,         # DCGM_FI_DEV_FB_USED - Used framebuffer memory (MB)
        
        # Clock frequencies
        100,         # DCGM_FI_DEV_SM_CLOCK - SM clock frequency (MHz)
        101,         # DCGM_FI_DEV_MEM_CLOCK - Memory clock frequency (MHz)
        110,         # DCGM_FI_DEV_APP_SM_CLOCK - Application SM clock (MHz)
        111,         # DCGM_FI_DEV_APP_MEM_CLOCK - Application memory clock (MHz)
        
        # Performance state
        190,         # DCGM_FI_DEV_PSTATE - Performance state (P-state)
        
        # Advanced compute activity metrics (DCGM 2.0+)
        1001,        # DCGM_FI_PROF_GR_ENGINE_ACTIVE - Graphics engine active (%)
        1002,        # DCGM_FI_PROF_SM_ACTIVE - SM active (%)
        1003,        # DCGM_FI_PROF_SM_OCCUPANCY - SM occupancy (%)
        1004,        # DCGM_FI_PROF_PIPE_TENSOR_ACTIVE - Tensor pipe active (%)
        1005,        # DCGM_FI_PROF_DRAM_ACTIVE - DRAM active (%)
        1006,        # DCGM_FI_PROF_PIPE_FP64_ACTIVE - FP64 pipe active (%)
        1007,        # DCGM_FI_PROF_PIPE_FP32_ACTIVE - FP32 pipe active (%)
        1008,        # DCGM_FI_PROF_PIPE_FP16_ACTIVE - FP16 pipe active (%)
    ]

    # Profiling intervals
    DEFAULT_INTERVAL_MS = 50  # milliseconds (supported by both DCGMI and nvidia-smi --loop-ms)
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
    LLAMA_MODEL_NAME = "huggyllama/llama-7b"  # Default model
    LLAMA_TORCH_DTYPE = "float16"
    LLAMA_DEFAULT_PROMPT = "Plants create energy through a process known as"
    
    # LLaMA model variants with their Hugging Face identifiers
    LLAMA_MODELS = {
        "llama-7b": "huggyllama/llama-7b",
        "llama-13b": "huggyllama/llama-13b",
        "llama-30b": "huggyllama/llama-30b", 
        "llama-65b": "huggyllama/llama-65b",
        "llama2-7b": "meta-llama/Llama-2-7b-hf",
        "llama2-13b": "meta-llama/Llama-2-13b-hf",
        "llama2-70b": "meta-llama/Llama-2-70b-hf",
        "code-llama-7b": "codellama/CodeLlama-7b-hf",
        "code-llama-13b": "codellama/CodeLlama-13b-hf",
        "code-llama-34b": "codellama/CodeLlama-34b-hf"
    }
    
    # LLaMA generation parameters for consistent benchmarking
    LLAMA_DEFAULT_PARAMS = {
        "max_new_tokens": 50,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "do_sample": True,
        "repetition_penalty": 1.1
    }
    
    # Benchmark prompts for LLaMA evaluation
    LLAMA_BENCHMARK_PROMPTS = [
        "The future of artificial intelligence is",
        "Climate change is one of the most pressing issues",
        "In the field of renewable energy",
        "Machine learning algorithms have revolutionized",
        "The impact of social media on society",
        "Quantum computing represents a paradigm shift",
        "The exploration of space has always fascinated",
        "Sustainable development goals are essential"
    ]

    # Stable Diffusion model configuration
    STABLE_DIFFUSION_MODEL_NAME = "CompVis/stable-diffusion-v1-4"
    STABLE_DIFFUSION_DEFAULT_PROMPT = "a photo of an astronaut riding a horse on mars"
    STABLE_DIFFUSION_OUTPUT_FILE = "astronaut_rides_horse.png"

    # Whisper model configuration
    WHISPER_DEFAULT_MODEL = "base"
    WHISPER_TORCH_DTYPE = "float16"
    WHISPER_DEFAULT_LANGUAGE = "en"
    WHISPER_SAMPLE_RATE = 16000
    WHISPER_DEFAULT_DURATION = 30  # seconds
    
    # Whisper model variants with their Hugging Face identifiers
    WHISPER_MODELS = {
        "tiny": "openai/whisper-tiny",
        "base": "openai/whisper-base", 
        "small": "openai/whisper-small",
        "medium": "openai/whisper-medium",
        "large": "openai/whisper-large-v2",
        "large-v3": "openai/whisper-large-v3"
    }
    
    # Whisper benchmark parameters for consistent energy profiling
    WHISPER_DEFAULT_PARAMS = {
        "num_samples": 3,
        "use_dataset": False,
        "language": "en",
        "return_timestamps": False
    }
    
    # Benchmark audio configurations for Whisper evaluation
    WHISPER_BENCHMARK_CONFIGS = [
        {"duration": 5.0, "complexity": "simple"},
        {"duration": 15.0, "complexity": "medium"},
        {"duration": 30.0, "complexity": "complex"},
        {"duration": 60.0, "complexity": "long_form"}
    ]


class SystemConfig:
    """System-level configuration settings."""

    # GPU architecture
    DEFAULT_ARCH = "GA100"  # A100 architecture
    PROFILING_MODE = "dvfs"

    # Paths and directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
    APPS_DIR = os.path.join(BASE_DIR, "apps")
    TEMP_IMAGES_DIR = os.path.join(BASE_DIR, "temp", "images")

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
