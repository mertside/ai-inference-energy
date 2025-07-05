"""
Hardware Abstraction Layer (HAL) for AI Inference Energy Framework

This package provides hardware abstraction capabilities for unified GPU interaction
across different NVIDIA GPU architectures and monitoring tools.

Current Implementation:
- GPU Info Module: Comprehensive GPU specifications and capabilities

Future Implementation:
- Power Monitoring Module: Unified power measurement interface
- Performance Counters Module: Standardized performance metric collection  
- Device Manager Module: Multi-GPU coordination and management

Usage:
    from hardware.gpu_info import GPUSpecifications, get_gpu_info
    
    # Get GPU specifications
    gpu_info = get_gpu_info('V100')
    frequencies = gpu_info.get_available_frequencies()
    specs = gpu_info.get_summary()
"""

# Core GPU information functionality
from .gpu_info import (
    ComputeSpecification,
    FrequencySpecification,
    GPUArchitecture,
    GPUSpecifications,
    MemorySpecification,
    PowerSpecification,
    ThermalSpecification,
    compare_gpus,
    get_gpu_info,
    get_supported_gpus,
    validate_gpu_configuration,
)

# Module metadata
__version__ = "1.0.0"
__author__ = "AI Inference Energy Profiling Framework"
__description__ = "Hardware Abstraction Layer for GPU Energy Profiling"

# Define what gets exported when using "from hardware import *"
__all__ = [
    # Core classes
    "GPUSpecifications",
    "GPUArchitecture",
    "FrequencySpecification",
    "MemorySpecification",
    "ComputeSpecification",
    "PowerSpecification",
    "ThermalSpecification",
    # Convenience functions
    "get_gpu_info",
    "get_supported_gpus",
    "compare_gpus",
    "validate_gpu_configuration",
]


def get_module_info():
    """
    Get information about the hardware module capabilities.

    Returns:
        Dictionary with module information and available features
    """
    return {
        "version": __version__,
        "description": __description__,
        "implemented_modules": ["gpu_info"],
        "planned_modules": ["power_monitoring", "performance_counters", "device_manager"],
        "supported_gpus": get_supported_gpus(),
        "total_gpu_specifications": len(get_supported_gpus()),
        "example_usage": {
            "get_specs": 'gpu_info = get_gpu_info("V100"); specs = gpu_info.get_summary()',
            "validate_frequency": 'gpu_info = get_gpu_info("A100"); valid = gpu_info.validate_frequency(1200)',
            "compare_gpus": 'comparison = compare_gpus(["V100", "A100", "H100"])',
        },
    }
