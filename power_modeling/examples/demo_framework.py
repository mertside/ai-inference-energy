#!/usr/bin/env python3
"""
Power Modeling Framework Example

This script demonstrates the complete usage of the FGCS power modeling framework,
including data loading, model training, power prediction, and EDP optimization.

Run this script to see the framework in action with sample data.
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from power_modeling import FGCSPowerModelingFramework, analyze_application

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_sample_data():
    """Generate sample profiling data for demonstration."""
    logger.info("Generating sample profiling data...")

    # Simulate realistic profiling data
    np.random.seed(42)
    n_samples = 100

    # Generate realistic FP activity, DRAM activity, and frequency data
    fp_activity = np.random.uniform(0.1, 0.8, n_samples)
    dram_activity = np.random.uniform(0.05, 0.4, n_samples)
    frequencies = np.random.choice(range(800, 1401, 50), n_samples)

    # Generate realistic power consumption based on activities and frequency
    # Using a simplified power model for demonstration
    power = (
        100  # Base power
        + 200 * fp_activity  # FP contribution
        + 80 * dram_activity  # DRAM contribution
        + 0.05 * frequencies  # Frequency contribution
        + np.random.normal(0, 10, n_samples)  # Noise
    )

    # Create DataFrame
    data = pd.DataFrame(
        {
            "app_name": ["SampleApp"] * n_samples,
            "fp_activity": fp_activity,
            "dram_activity": dram_activity,
            "sm_clock": frequencies,
            "power": power,
        }
    )

    return data


def generate_performance_data():
    """Generate sample performance data."""
    logger.info("Generating sample performance data...")

    # Simulate performance data
    frequencies = [800, 900, 1000, 1100, 1200, 1300, 1400]

    # Higher frequency = better performance (lower runtime)
    runtimes = [1.5, 1.4, 1.3, 1.2, 1.1, 1.05, 1.0]
    throughputs = [1 / rt for rt in runtimes]  # Inverse of runtime

    performance_data = pd.DataFrame(
        {
            "app_name": ["SampleApp"] * len(frequencies),
            "frequency": frequencies,
            "runtime": runtimes,
            "throughput": throughputs,
        }
    )

    return performance_data


def demo_quick_analysis():
    """Demonstrate quick analysis functionality."""
    logger.info("=== Quick Analysis Demo ===")

    # Generate sample data
    sample_data = generate_sample_data()
    performance_data = generate_performance_data()

    # Save to temporary files
    data_dir = Path("temp_demo_data")
    data_dir.mkdir(exist_ok=True)

    profiling_file = data_dir / "profiling.csv"
    performance_file = data_dir / "performance.csv"

    sample_data.to_csv(profiling_file, index=False)
    performance_data.to_csv(performance_file, index=False)

    # Run quick analysis
    results = analyze_application(
        profiling_file=str(profiling_file),
        performance_file=str(performance_file),
        app_name="SampleApp",
        gpu_type="V100",
        output_dir=str(data_dir / "results"),
    )

    # Display results
    print("\n📊 Quick Analysis Results:")
    print(f"  Optimal Frequency: {results['summary']['optimal_frequency']}")
    print(f"  Energy Savings: {results['summary']['energy_savings']}")
    print(f"  Performance Impact: {results['summary']['performance_impact']}")
    print(f"  Recommendation: {results['summary']['recommendation']}")

    # Cleanup
    import shutil

    shutil.rmtree(data_dir)

    return results


def demo_full_framework():
    """Demonstrate full framework capabilities."""
    logger.info("=== Full Framework Demo ===")

    # Generate sample data
    sample_data = generate_sample_data()

    # Initialize framework
    framework = FGCSPowerModelingFramework(
        model_types=["fgcs_original", "polynomial_deg2", "random_forest_enhanced"],
        gpu_type="V100",
    )

    # Train models
    print("\n🤖 Training Models...")
    training_results = framework.train_models(
        sample_data, target_column="power", test_size=0.2
    )

    print(f"  Best model: {training_results['best_model'][0]}")
    print(f"  R² score: {training_results['best_model'][2]:.4f}")
    print(f"  MAE: {training_results['best_model'][3]:.2f} W")

    # Predict power across frequency range
    print("\n⚡ Power Prediction Sweep...")
    power_sweep = framework.predict_power_sweep(
        fp_activity=0.3,
        dram_activity=0.15,
        frequencies=[800, 900, 1000, 1100, 1200, 1300, 1400],
    )

    print("  Frequency (MHz) | Power (W)")
    print("  ----------------|----------")
    for _, row in power_sweep.iterrows():
        print(f"  {row['frequency']:13} | {row['power']:8.1f}")

    # Optimize application
    print("\n🎯 EDP Optimization...")
    optimization_results = framework.optimize_application(
        fp_activity=0.3, dram_activity=0.15, baseline_runtime=1.0, app_name="SampleApp"
    )

    edp_opt = optimization_results["edp_optimal"]
    ed2p_opt = optimization_results["ed2p_optimal"]

    print(f"  EDP Optimal:")
    print(f"    Frequency: {edp_opt['frequency']} MHz")
    print(f"    Power: {edp_opt['power']:.1f} W")
    print(f"    Runtime: {edp_opt['runtime']:.2f} s")
    print(f"    EDP: {edp_opt['edp']:.2f}")

    print(f"  ED²P Optimal:")
    print(f"    Frequency: {ed2p_opt['frequency']} MHz")
    print(f"    Power: {ed2p_opt['power']:.1f} W")
    print(f"    Runtime: {ed2p_opt['runtime']:.2f} s")
    print(f"    ED²P: {ed2p_opt['ed2p']:.2f}")

    return framework, optimization_results


def demo_model_comparison():
    """Demonstrate model comparison capabilities."""
    logger.info("=== Model Comparison Demo ===")

    # Generate sample data
    sample_data = generate_sample_data()

    # Test different model types
    model_types = [
        ["fgcs_original"],
        ["polynomial_deg2"],
        ["random_forest_enhanced"],
        ["fgcs_original", "polynomial_deg2", "random_forest_enhanced"],
    ]

    print("\n📈 Model Performance Comparison:")
    print("  Model Type                | R² Score | MAE (W)")
    print("  --------------------------|----------|--------")

    for models in model_types:
        framework = FGCSPowerModelingFramework(model_types=models)
        results = framework.train_models(sample_data, target_column="power")

        best_model = results["best_model"]
        model_name = best_model[0] if len(models) == 1 else "Ensemble"
        r2_score = best_model[2]
        mae = best_model[3]

        print(f"  {model_name:25} | {r2_score:8.4f} | {mae:6.2f}")


def demo_gpu_comparison():
    """Demonstrate GPU-specific configurations."""
    logger.info("=== GPU Configuration Demo ===")

    print("\n🔧 GPU-Specific Frequency Ranges:")

    for gpu_type in ["V100", "A100", "H100"]:
        framework = FGCSPowerModelingFramework(gpu_type=gpu_type)
        freq_range = framework.frequency_configs[gpu_type]

        print(
            f"  {gpu_type}: {min(freq_range)}-{max(freq_range)} MHz "
            f"({len(freq_range)} frequencies)"
        )


def main():
    """Run all demonstrations."""
    print("🚀 Power Modeling Framework Demo")
    print("=" * 40)

    try:
        # Run demonstrations
        demo_quick_analysis()
        demo_full_framework()
        demo_model_comparison()
        demo_gpu_comparison()

        print("\n✅ All demonstrations completed successfully!")
        print("\nNext steps:")
        print("1. Replace sample data with your actual profiling data")
        print("2. Adjust model types based on your requirements")
        print("3. Customize frequency ranges for your GPU")
        print("4. Integrate with your existing profiling pipeline")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
