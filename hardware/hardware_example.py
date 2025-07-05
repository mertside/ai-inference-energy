#!/usr/bin/env python3
"""
Hardware GPU Info Module - Example Usage

This script demonstrates the capabilities of the GPU Info Module,
which provides comprehensive specifications and validation for
NVIDIA GPU architectures used in AI inference energy profiling.

Usage:
    python hardware_example.py [--gpu-type V100|A100|H100] [--compare-all]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hardware import (
    compare_gpus,
    get_gpu_info,
    get_module_info,
    get_supported_gpus,
    validate_gpu_configuration,
)


def demonstrate_gpu_specifications(gpu_type: str):
    """Demonstrate comprehensive GPU specifications access."""
    print(f"\n{'='*60}")
    print(f"📊 {gpu_type} GPU Specifications")
    print(f"{'='*60}")

    try:
        gpu_info = get_gpu_info(gpu_type)

        # Basic information
        print(f"Architecture: {gpu_info.get_architecture().value}")
        print(f"Full Name: {gpu_info.specifications['full_name']}")
        print(f"Release Year: {gpu_info.specifications['release_year']}")

        # Frequency specifications
        freq_spec = gpu_info.get_frequency_specification()
        print(f"\n🔧 Frequency Specifications:")
        print(f"  Range: {freq_spec.min_freq} - {freq_spec.max_freq} MHz")
        print(f"  Total Frequencies: {freq_spec.count}")
        if freq_spec.step_size:
            print(f"  Step Size: {freq_spec.step_size} MHz")

        # Show sample frequencies
        frequencies = gpu_info.get_available_frequencies()
        print(f"  Sample Frequencies: {frequencies[:5]} ... {frequencies[-5:]}")

        # Memory specifications
        memory_spec = gpu_info.get_memory_specification()
        print(f"\n💾 Memory Specifications:")
        print(f"  Size: {memory_spec.size_gb} GB {memory_spec.type}")
        print(f"  Bandwidth: {memory_spec.bandwidth_gb_s} GB/s")
        print(f"  Frequency: {memory_spec.frequency_mhz} MHz")
        print(f"  Bus Width: {memory_spec.bus_width} bits")

        # Compute specifications
        compute_spec = gpu_info.get_compute_specification()
        print(f"\n🖥️  Compute Specifications:")
        print(f"  SM Count: {compute_spec.sm_count}")
        print(f"  CUDA Cores: {compute_spec.cuda_cores}")
        print(f"  Tensor Cores: {'Yes' if compute_spec.tensor_cores else 'No'}")
        print(f"  Compute Capability: {compute_spec.compute_capability}")
        if compute_spec.fp32_performance:
            print(f"  FP32 Performance: {compute_spec.fp32_performance} TFLOPS")
        if compute_spec.tensor_performance:
            print(f"  Tensor Performance: {compute_spec.tensor_performance} TOPS")

        # Power specifications
        power_spec = gpu_info.get_power_specification()
        print(f"\n⚡ Power Specifications:")
        print(f"  TDP: {power_spec.tdp_watts}W")
        print(
            f"  Power Range: {power_spec.min_power_watts}W - {power_spec.max_power_watts}W"
        )
        print(f"  Power Connectors: {', '.join(power_spec.power_connectors)}")
        if power_spec.power_efficiency:
            print(f"  Power Efficiency: {power_spec.power_efficiency} GFLOPS/W")

        # Thermal specifications
        thermal_spec = gpu_info.get_thermal_specification()
        print(f"\n🌡️  Thermal Specifications:")
        print(f"  Max Temperature: {thermal_spec.max_temp_c}°C")
        print(f"  Throttle Temperature: {thermal_spec.throttle_temp_c}°C")
        print(f"  Idle Temperature: {thermal_spec.idle_temp_c}°C")
        print(f"  Cooling Solution: {thermal_spec.cooling_solution}")

        # FGCS compatibility
        fgcs_freqs = gpu_info.get_fgcs_compatible_frequencies()
        print(f"\n🔬 FGCS Compatibility:")
        print(f"  Validated Frequencies: {fgcs_freqs}")
        print(
            f"  Baseline Frequency: {gpu_info.specifications['fgcs']['baseline_frequency']} MHz"
        )

        # Workload recommendations
        print(f"\n🎯 Workload Recommendations:")
        workload_recs = gpu_info.specifications.get("workload_recommendations", {})
        for workload, freq in workload_recs.items():
            print(f"  {workload.title()}: {freq} MHz")

    except ValueError as e:
        print(f"❌ Error: {e}")


def demonstrate_frequency_validation(gpu_type: str):
    """Demonstrate frequency validation capabilities."""
    print(f"\n{'='*60}")
    print(f"🔍 {gpu_type} Frequency Validation")
    print(f"{'='*60}")

    try:
        gpu_info = get_gpu_info(gpu_type)

        # Test various frequencies
        test_frequencies = [800, 1000, 1200, 1400, 1600, 2000]

        print("Frequency Validation Results:")
        for freq in test_frequencies:
            is_valid = gpu_info.validate_frequency(freq)
            closest = gpu_info.get_closest_frequency(freq)
            status = "✅ Valid" if is_valid else f"❌ Invalid (closest: {closest} MHz)"
            print(f"  {freq} MHz: {status}")

        # Test configuration validation
        print(f"\nConfiguration Validation:")
        memory_freq = gpu_info.get_memory_specification().frequency_mhz

        test_configs = [
            (1200, memory_freq),
            (1500, memory_freq),
            (2000, memory_freq),
            (1200, 1000),
        ]  # Wrong memory frequency

        for core_freq, mem_freq in test_configs:
            validation = validate_gpu_configuration(gpu_type, core_freq, mem_freq)
            status = "✅ Valid" if validation["overall_valid"] else "❌ Invalid"
            print(f"  Core: {core_freq} MHz, Memory: {mem_freq} MHz - {status}")
            if not validation["overall_valid"]:
                issues = [
                    k for k, v in validation.items() if not v and k != "overall_valid"
                ]
                print(f"    Issues: {', '.join(issues)}")

    except ValueError as e:
        print(f"❌ Error: {e}")


def demonstrate_gpu_comparison():
    """Demonstrate GPU comparison capabilities."""
    print(f"\n{'='*60}")
    print(f"⚖️  GPU Architecture Comparison")
    print(f"{'='*60}")

    supported_gpus = get_supported_gpus()
    comparison = compare_gpus(supported_gpus)

    # Create comparison table
    print(f"{'Metric':<25} {'V100':<15} {'A100':<15} {'H100':<15}")
    print("-" * 70)

    metrics = [
        "architecture",
        "frequency_count",
        "memory_size",
        "sm_count",
        "cuda_cores",
        "tdp",
    ]

    for metric in metrics:
        values = []
        for gpu in ["V100", "A100", "H100"]:
            if gpu in comparison and "error" not in comparison[gpu]:
                value = comparison[gpu].get(metric, "N/A")
                values.append(str(value))
            else:
                values.append("N/A")

        print(
            f"{metric.replace('_', ' ').title():<25} {values[0]:<15} {values[1]:<15} {values[2]:<15}"
        )

    # Performance evolution
    print(f"\n📈 Performance Evolution:")
    for gpu in ["V100", "A100", "H100"]:
        if gpu in comparison and "error" not in comparison[gpu]:
            specs = comparison[gpu]
            print(
                f"  {gpu}: {specs['cuda_cores']} CUDA cores, "
                f"{specs['memory_size']}, {specs['tdp']}"
            )


def demonstrate_integration_examples():
    """Demonstrate integration with existing framework."""
    print(f"\n{'='*60}")
    print(f"🔗 Framework Integration Examples")
    print(f"{'='*60}")

    # Example 1: FGCS frequency selection
    print("1. FGCS Frequency Selection:")
    v100_info = get_gpu_info("V100")
    fgcs_frequencies = v100_info.get_fgcs_compatible_frequencies()
    print(f"   FGCS validated frequencies for V100: {fgcs_frequencies}")

    # Example 2: Workload-specific optimization
    print("\n2. Workload-Specific Frequency Recommendations:")
    for gpu_type in get_supported_gpus():
        gpu_info = get_gpu_info(gpu_type)
        inference_freq = gpu_info.get_optimal_frequency_for_workload("inference")
        training_freq = gpu_info.get_optimal_frequency_for_workload("training")
        print(
            f"   {gpu_type}: Inference={inference_freq} MHz, Training={training_freq} MHz"
        )

    # Example 3: Configuration validation pipeline
    print("\n3. Configuration Validation Pipeline:")
    print("   Example validation workflow:")
    print("   ├── Validate GPU type")
    print("   ├── Validate frequency range")
    print("   ├── Validate memory configuration")
    print("   └── Generate optimized configuration")

    # Example 4: Integration code snippet
    print("\n4. Integration Code Example:")
    print(
        """
   # Integration with EDP analysis
   from hardware import get_gpu_info
   from edp_analysis import EDPCalculator
   
   gpu_info = get_gpu_info('A100')
   frequencies = gpu_info.get_fgcs_compatible_frequencies()
   
   calculator = EDPCalculator()
   for freq in frequencies:
       if gpu_info.validate_frequency(freq):
           # Run EDP analysis with validated frequency
           pass
   """
    )


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Hardware GPU Info Module Example")
    parser.add_argument(
        "--gpu-type",
        choices=["V100", "A100", "H100"],
        default="V100",
        help="GPU type to demonstrate",
    )
    parser.add_argument(
        "--compare-all", action="store_true", help="Show comparison of all GPU types"
    )
    parser.add_argument(
        "--integration", action="store_true", help="Show framework integration examples"
    )

    args = parser.parse_args()

    print("🔧 Hardware GPU Info Module - Example Usage")
    print(f"Demonstrating GPU hardware abstraction capabilities")

    # Show module information
    module_info = get_module_info()
    print(f"\nModule Version: {module_info['version']}")
    print(f"Supported GPUs: {', '.join(module_info['supported_gpus'])}")

    # Demonstrate specific GPU
    if not args.compare_all:
        demonstrate_gpu_specifications(args.gpu_type)
        demonstrate_frequency_validation(args.gpu_type)

    # Show comparison
    if args.compare_all:
        demonstrate_gpu_comparison()

    # Show integration examples
    if args.integration:
        demonstrate_integration_examples()

    print(f"\n✅ Hardware GPU Info Module demonstration completed!")
    print(f"💡 Try: python hardware_example.py --compare-all --integration")


if __name__ == "__main__":
    main()
