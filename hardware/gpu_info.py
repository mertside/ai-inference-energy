"""
GPU Information and Specifications Module

This module provides a unified interface for accessing GPU specifications, capabilities,
and hardware information across different NVIDIA GPU architectures. It serves as the
foundation for the Hardware Abstraction Layer (HAL) in the AI Inference Energy framework.

Key Features:
- Comprehensive GPU specifications for V100, A100, and H100
- Unified API for accessing GPU capabilities
- Frequency range validation and management
- Memory and compute specifications
- Architecture-specific optimizations
- Power and thermal characteristics

Supported GPUs:
- Tesla V100 (Volta architecture)
- A100 (Ampere architecture) 
- H100 (Hopper architecture)

Usage:
    from hardware.gpu_info import GPUSpecifications, get_gpu_info
    
    # Get GPU specifications
    gpu_info = GPUSpecifications('V100')
    freq_range = gpu_info.get_frequency_range()
    memory_specs = gpu_info.get_memory_specifications()
    
    # Quick lookup
    specs = get_gpu_info('A100')
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class GPUArchitecture(Enum):
    """GPU architecture enumeration."""

    VOLTA = "Volta"
    AMPERE = "Ampere"
    HOPPER = "Hopper"
    UNKNOWN = "Unknown"


@dataclass
class FrequencySpecification:
    """GPU frequency specification."""

    min_freq: int  # MHz
    max_freq: int  # MHz
    count: int
    step_size: Optional[int] = None  # MHz, if regular stepping
    frequencies: Optional[List[int]] = None  # Exact frequency list


@dataclass
class MemorySpecification:
    """GPU memory specification."""

    size_gb: int
    type: str  # HBM2, HBM2e, HBM3
    bandwidth_gb_s: float
    frequency_mhz: int
    bus_width: int


@dataclass
class ComputeSpecification:
    """GPU compute specification."""

    sm_count: int
    cuda_cores: int
    tensor_cores: bool
    rt_cores: Optional[int] = None
    compute_capability: str = "7.0"
    fp16_performance: Optional[float] = None  # TFLOPS
    fp32_performance: Optional[float] = None  # TFLOPS
    tensor_performance: Optional[float] = None  # TFLOPS


@dataclass
class PowerSpecification:
    """GPU power specification."""

    tdp_watts: int
    min_power_watts: int
    max_power_watts: int
    power_connectors: List[str]
    power_efficiency: Optional[float] = None  # GFLOPS/W


@dataclass
class ThermalSpecification:
    """GPU thermal specification."""

    max_temp_c: int
    throttle_temp_c: int
    idle_temp_c: int
    cooling_solution: str


class GPUSpecifications:
    """
    Comprehensive GPU specifications and capabilities manager.

    Provides unified access to GPU hardware specifications, frequency ranges,
    memory characteristics, and architecture-specific features.
    """

    def __init__(self, gpu_type: str):
        """
        Initialize GPU specifications for the specified GPU type.

        Args:
            gpu_type: GPU type string ('V100', 'A100', 'H100')

        Raises:
            ValueError: If GPU type is not supported
        """
        self.gpu_type = gpu_type.upper()
        if self.gpu_type not in self._get_supported_gpus():
            raise ValueError(f"Unsupported GPU type: {gpu_type}. " f"Supported: {', '.join(self._get_supported_gpus())}")

        self.specifications = self._load_specifications()
        logger.info(f"GPU specifications loaded for {self.gpu_type}")

    @staticmethod
    def _get_supported_gpus() -> List[str]:
        """Get list of supported GPU types."""
        return ["V100", "A100", "H100"]

    def _load_specifications(self) -> Dict[str, Any]:
        """Load comprehensive specifications for the GPU type."""
        return GPU_SPECIFICATIONS[self.gpu_type]

    def get_architecture(self) -> GPUArchitecture:
        """Get GPU architecture."""
        return self.specifications["architecture"]

    def get_frequency_specification(self) -> FrequencySpecification:
        """Get frequency specification with range and available frequencies."""
        freq_spec = self.specifications["frequency"]
        return FrequencySpecification(
            min_freq=freq_spec["min_freq"],
            max_freq=freq_spec["max_freq"],
            count=freq_spec["count"],
            step_size=freq_spec.get("step_size"),
            frequencies=freq_spec.get("frequencies"),
        )

    def get_frequency_range(self) -> Tuple[int, int]:
        """Get frequency range as (min, max) tuple in MHz."""
        freq_spec = self.get_frequency_specification()
        return (freq_spec.min_freq, freq_spec.max_freq)

    def get_available_frequencies(self) -> List[int]:
        """Get list of all available frequencies in MHz."""
        freq_spec = self.get_frequency_specification()
        if freq_spec.frequencies:
            return freq_spec.frequencies.copy()

        # Generate frequency list if not explicitly provided
        frequencies = []
        if freq_spec.step_size:
            freq = freq_spec.min_freq
            while freq <= freq_spec.max_freq:
                frequencies.append(freq)
                freq += freq_spec.step_size
        else:
            # Use count to determine approximate step size
            step = (freq_spec.max_freq - freq_spec.min_freq) / (freq_spec.count - 1)
            for i in range(freq_spec.count):
                freq = freq_spec.min_freq + int(i * step)
                frequencies.append(freq)

        return frequencies

    def get_memory_specification(self) -> MemorySpecification:
        """Get memory specification."""
        mem_spec = self.specifications["memory"]
        return MemorySpecification(
            size_gb=mem_spec["size_gb"],
            type=mem_spec["type"],
            bandwidth_gb_s=mem_spec["bandwidth_gb_s"],
            frequency_mhz=mem_spec["frequency_mhz"],
            bus_width=mem_spec["bus_width"],
        )

    def get_compute_specification(self) -> ComputeSpecification:
        """Get compute specification."""
        compute_spec = self.specifications["compute"]
        return ComputeSpecification(
            sm_count=compute_spec["sm_count"],
            cuda_cores=compute_spec["cuda_cores"],
            tensor_cores=compute_spec["tensor_cores"],
            rt_cores=compute_spec.get("rt_cores"),
            compute_capability=compute_spec["compute_capability"],
            fp16_performance=compute_spec.get("fp16_performance"),
            fp32_performance=compute_spec.get("fp32_performance"),
            tensor_performance=compute_spec.get("tensor_performance"),
        )

    def get_power_specification(self) -> PowerSpecification:
        """Get power specification."""
        power_spec = self.specifications["power"]
        return PowerSpecification(
            tdp_watts=power_spec["tdp_watts"],
            min_power_watts=power_spec["min_power_watts"],
            max_power_watts=power_spec["max_power_watts"],
            power_connectors=power_spec["power_connectors"],
            power_efficiency=power_spec.get("power_efficiency"),
        )

    def get_thermal_specification(self) -> ThermalSpecification:
        """Get thermal specification."""
        thermal_spec = self.specifications["thermal"]
        return ThermalSpecification(
            max_temp_c=thermal_spec["max_temp_c"],
            throttle_temp_c=thermal_spec["throttle_temp_c"],
            idle_temp_c=thermal_spec["idle_temp_c"],
            cooling_solution=thermal_spec["cooling_solution"],
        )

    def validate_frequency(self, frequency: int) -> bool:
        """
        Validate if a frequency is supported by this GPU.

        Args:
            frequency: Frequency in MHz

        Returns:
            True if frequency is valid, False otherwise
        """
        available_freqs = self.get_available_frequencies()
        return frequency in available_freqs

    def get_closest_frequency(self, target_frequency: int) -> int:
        """
        Get the closest supported frequency to the target.

        Args:
            target_frequency: Target frequency in MHz

        Returns:
            Closest supported frequency in MHz
        """
        available_freqs = self.get_available_frequencies()
        return min(available_freqs, key=lambda x: abs(x - target_frequency))

    def get_fgcs_compatible_frequencies(self) -> List[int]:
        """
        Get frequencies compatible with FGCS 2023 methodology.

        Returns subset of frequencies that are well-tested and validated
        for FGCS power modeling.
        """
        fgcs_config = self.specifications.get("fgcs", {})
        if "validated_frequencies" in fgcs_config:
            return fgcs_config["validated_frequencies"].copy()

        # Fallback: return all frequencies
        return self.get_available_frequencies()

    def get_optimal_frequency_for_workload(self, workload_type: str) -> int:
        """
        Get recommended frequency for specific workload types.

        Args:
            workload_type: Type of workload ('inference', 'training', 'compute', 'memory_bound')

        Returns:
            Recommended frequency in MHz
        """
        workload_config = self.specifications.get("workload_recommendations", {})
        if workload_type in workload_config:
            return workload_config[workload_type]

        # Default to maximum frequency
        return self.get_frequency_range()[1]

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of GPU specifications."""
        freq_spec = self.get_frequency_specification()
        memory_spec = self.get_memory_specification()
        compute_spec = self.get_compute_specification()
        power_spec = self.get_power_specification()

        return {
            "gpu_type": self.gpu_type,
            "architecture": self.get_architecture().value,
            "frequency_range": f"{freq_spec.min_freq}-{freq_spec.max_freq} MHz",
            "frequency_count": freq_spec.count,
            "memory_size": f"{memory_spec.size_gb} GB {memory_spec.type}",
            "memory_bandwidth": f"{memory_spec.bandwidth_gb_s} GB/s",
            "sm_count": compute_spec.sm_count,
            "cuda_cores": compute_spec.cuda_cores,
            "tensor_cores": compute_spec.tensor_cores,
            "compute_capability": compute_spec.compute_capability,
            "tdp": f"{power_spec.tdp_watts}W",
            "supported_frequencies": len(self.get_available_frequencies()),
        }


# ============================================================================
# GPU Specifications Database
# ============================================================================

GPU_SPECIFICATIONS = {
    "V100": {
        "architecture": GPUArchitecture.VOLTA,
        "full_name": "Tesla V100-PCIE-32GB",  # Updated based on nvidia-smi output
        "release_year": 2017,
        "frequency": {
            "min_freq": 510,  # Updated: excluding frequencies below 510 MHz
            "max_freq": 1380,
            "count": 117,  # Updated count for frequencies ≥510 MHz
            "step_size": None,  # Irregular stepping
            "frequencies": [
                # Updated with actual nvidia-smi supported clocks data (≥510 MHz only)
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
            ],
        },
        "memory": {"size_gb": 32, "type": "HBM2", "bandwidth_gb_s": 900, "frequency_mhz": 877, "bus_width": 4096},
        "compute": {
            "sm_count": 80,
            "cuda_cores": 5120,
            "tensor_cores": True,
            "rt_cores": None,
            "compute_capability": "7.0",
            "fp16_performance": 125.0,  # TFLOPS
            "fp32_performance": 15.7,  # TFLOPS
            "tensor_performance": 500.0,  # TOPS (INT8)
        },
        "power": {
            "tdp_watts": 300,
            "min_power_watts": 100,
            "max_power_watts": 350,
            "power_connectors": ["SXM2"],
            "power_efficiency": 52.3,  # GFLOPS/W
        },
        "thermal": {"max_temp_c": 89, "throttle_temp_c": 87, "idle_temp_c": 35, "cooling_solution": "Passive (SXM2)"},
        "fgcs": {
            "validated_frequencies": [1380, 1200, 1000, 800, 600, 510],  # Updated: replaced 405 with 510
            "baseline_frequency": 1380,
            "memory_frequency": 877,
        },
        "workload_recommendations": {"inference": 1200, "training": 1380, "compute": 1300, "memory_bound": 1000},
    },
    "A100": {
        "architecture": GPUArchitecture.AMPERE,
        "full_name": "NVIDIA A100-PCIE-40GB",  # Updated based on nvidia-smi output
        "release_year": 2020,
        "frequency": {
            "min_freq": 510,  # Updated: excluding frequencies below 510 MHz
            "max_freq": 1410,
            "count": 61,  # Count of frequencies ≥510 MHz
            "step_size": 15,  # Regular 15 MHz stepping
            "frequencies": [
                # Updated with actual nvidia-smi supported clocks data (≥510 MHz only)
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
            ],
        },
        "memory": {
            "size_gb": 40,  # Updated: nvidia-smi shows 40536MiB = ~40GB
            "type": "HBM2e",
            "bandwidth_gb_s": 1935,
            "frequency_mhz": 1215,  # Confirmed from nvidia-smi output
            "bus_width": 5120,
        },
        "compute": {
            "sm_count": 108,
            "cuda_cores": 6912,
            "tensor_cores": True,
            "rt_cores": None,
            "compute_capability": "8.0",
            "fp16_performance": 312.0,  # TFLOPS
            "fp32_performance": 19.5,  # TFLOPS
            "tensor_performance": 1248.0,  # TOPS (INT8)
        },
        "power": {
            "tdp_watts": 400,
            "min_power_watts": 150,
            "max_power_watts": 450,
            "power_connectors": ["SXM4"],
            "power_efficiency": 78.0,  # GFLOPS/W
        },
        "thermal": {"max_temp_c": 93, "throttle_temp_c": 90, "idle_temp_c": 30, "cooling_solution": "Passive (SXM4)"},
        "fgcs": {
            "validated_frequencies": [1410, 1200, 1000, 800, 600, 510],
            "baseline_frequency": 1410,
            "memory_frequency": 1215,
        },
        "workload_recommendations": {"inference": 1275, "training": 1410, "compute": 1350, "memory_bound": 1100},
    },
    "H100": {
        "architecture": GPUArchitecture.HOPPER,
        "full_name": "NVIDIA H100 NVL",  # Updated based on nvidia-smi output
        "release_year": 2022,
        "frequency": {
            "min_freq": 510,  # Updated: excluding frequencies below 510 MHz
            "max_freq": 1785,  # Updated: actual max frequency from nvidia-smi
            "count": 86,  # Updated count for frequencies ≥510 MHz (1785 down to 510 in 15 MHz steps)
            "step_size": 15,  # Regular 15 MHz stepping
            "frequencies": [
                # Updated with actual nvidia-smi supported clocks data (≥510 MHz only)
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
            ],
        },
        "memory": {
            "size_gb": 95,  # Updated: nvidia-smi shows 95830MiB = ~95GB
            "type": "HBM3",
            "bandwidth_gb_s": 3350,
            "frequency_mhz": 1593,  # Updated: using 1593 MHz as shown in nvidia-smi
            "bus_width": 5120,
        },
        "compute": {
            "sm_count": 132,
            "cuda_cores": 16896,
            "tensor_cores": True,
            "rt_cores": None,
            "compute_capability": "9.0",
            "fp16_performance": 989.0,  # TFLOPS
            "fp32_performance": 67.0,  # TFLOPS
            "tensor_performance": 3958.0,  # TOPS (INT8)
        },
        "power": {
            "tdp_watts": 700,
            "min_power_watts": 200,
            "max_power_watts": 750,
            "power_connectors": ["SXM5"],
            "power_efficiency": 95.7,  # GFLOPS/W
        },
        "thermal": {"max_temp_c": 89, "throttle_temp_c": 87, "idle_temp_c": 28, "cooling_solution": "Passive (SXM5)"},
        "fgcs": {
            "validated_frequencies": [
                1785,
                1500,
                1200,
                1000,
                800,
                600,
                510,
            ],  # Updated: replaced 400, 210 with 510, updated max to 1785
            "baseline_frequency": 1785,  # Updated: using actual max frequency
            "memory_frequency": 1593,
        },
        "workload_recommendations": {
            "inference": 1500,
            "training": 1785,  # Updated: using actual max frequency
            "compute": 1650,
            "memory_bound": 1200,
        },
    },
}


# ============================================================================
# Convenience Functions
# ============================================================================


def get_gpu_info(gpu_type: str) -> GPUSpecifications:
    """
    Convenience function to get GPU specifications.

    Args:
        gpu_type: GPU type string ('V100', 'A100', 'H100')

    Returns:
        GPUSpecifications instance
    """
    return GPUSpecifications(gpu_type)


def get_supported_gpus() -> List[str]:
    """Get list of all supported GPU types."""
    return GPUSpecifications._get_supported_gpus()


def compare_gpus(gpu_types: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Compare specifications across multiple GPU types.

    Args:
        gpu_types: List of GPU type strings

    Returns:
        Dictionary with comparison data
    """
    comparison = {}

    for gpu_type in gpu_types:
        try:
            gpu_info = GPUSpecifications(gpu_type)
            comparison[gpu_type] = gpu_info.get_summary()
        except ValueError as e:
            logger.warning(f"Could not load specifications for {gpu_type}: {e}")
            comparison[gpu_type] = {"error": str(e)}

    return comparison


def validate_gpu_configuration(gpu_type: str, frequency: int, memory_freq: Optional[int] = None) -> Dict[str, bool]:
    """
    Validate a GPU configuration.

    Args:
        gpu_type: GPU type string
        frequency: Core frequency in MHz
        memory_freq: Memory frequency in MHz (optional)

    Returns:
        Dictionary with validation results
    """
    try:
        gpu_info = GPUSpecifications(gpu_type)

        results = {
            "gpu_type_valid": True,
            "frequency_valid": gpu_info.validate_frequency(frequency),
            "memory_freq_valid": True,
        }

        if memory_freq:
            memory_spec = gpu_info.get_memory_specification()
            results["memory_freq_valid"] = memory_freq == memory_spec.frequency_mhz

        results["overall_valid"] = all(results.values())

        return results

    except ValueError:
        return {"gpu_type_valid": False, "frequency_valid": False, "memory_freq_valid": False, "overall_valid": False}


if __name__ == "__main__":
    # Example usage and testing
    print("GPU Information Module - Example Usage")
    print("=" * 50)

    for gpu_type in get_supported_gpus():
        print(f"\n{gpu_type} Specifications:")
        print("-" * 30)

        gpu_info = get_gpu_info(gpu_type)
        summary = gpu_info.get_summary()

        for key, value in summary.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")

        # Show some available frequencies
        frequencies = gpu_info.get_available_frequencies()
        print(f"  Sample Frequencies: {frequencies[:5]}...{frequencies[-5:]} ({len(frequencies)} total)")

    # Comparison example
    print(f"\n\nGPU Comparison:")
    print("-" * 30)
    comparison = compare_gpus(["V100", "A100", "H100"])

    for gpu, specs in comparison.items():
        if "error" not in specs:
            print(f"{gpu}: {specs['frequency_count']} frequencies, " f"{specs['memory_size']}, {specs['tdp']}")
