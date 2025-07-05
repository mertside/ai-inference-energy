#!/usr/bin/env python3
"""
Enhanced EDP Analysis Example

This script demonstrates the enhanced EDP analysis capabilities inspired by the
FGCS 2023 methodology, including feature selection, energy profiling,
performance analysis, and optimization.

Usage:
    python enhanced_edp_example.py [--gpu-type V100] [--synthetic-data]
"""

import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def create_synthetic_profiling_data():
    """Create realistic synthetic profiling data for demonstration."""
    try:
        import numpy as np
        import pandas as pd

        # Set random seed for reproducibility
        np.random.seed(42)

        # Generate data for multiple frequencies
        frequencies = [800, 900, 1000, 1100, 1200, 1300, 1400]
        n_runs_per_freq = 5

        data = []
        for freq in frequencies:
            for run in range(n_runs_per_freq):
                # Simulate realistic GPU behavior
                # Higher frequency -> higher power but lower execution time
                base_power = 150 + (freq - 800) * 0.08  # Base power scaling
                power_noise = np.random.normal(0, 10)  # Measurement noise
                power = max(100, base_power + power_noise)  # Ensure positive power

                # Execution time inversely related to frequency (with some noise)
                base_time = 2.0 * (1400 / freq)  # Inverse scaling
                time_noise = np.random.normal(0, 0.1)
                execution_time = max(
                    0.5, base_time + time_noise
                )  # Ensure positive time

                # Simulate FP and DRAM activity
                fp_activity = np.random.uniform(0.2, 0.6)
                dram_activity = np.random.uniform(0.1, 0.3)

                # Calculate derived metrics
                energy = power * execution_time

                data.append(
                    {
                        "frequency": freq,
                        "sm_clock": freq,
                        "power": power,
                        "execution_time": execution_time,
                        "energy": energy,
                        "fp_activity": fp_activity,
                        "dram_activity": dram_activity,
                        "run_id": run,
                        "gpu_utilization": np.random.uniform(80, 95),
                        "memory_utilization": np.random.uniform(60, 85),
                    }
                )

        df = pd.DataFrame(data)
        logger.info(
            f"Generated synthetic data: {len(df)} samples across {len(frequencies)} frequencies"
        )
        return df

    except ImportError as e:
        logger.error(f"Required packages not available: {e}")
        return None


def demo_feature_selection_and_engineering():
    """Demonstrate feature selection and engineering capabilities."""
    logger.info("\n=== Feature Selection and Engineering Demo ===")

    try:
        from edp_analysis.feature_selection import create_optimized_feature_set

        # Create synthetic data
        df = create_synthetic_profiling_data()
        if df is None:
            return False

        logger.info(f"Input data shape: {df.shape}")
        logger.info(f"Input columns: {list(df.columns)}")

        # Apply feature engineering and selection
        df_optimized, analysis_results = create_optimized_feature_set(
            df, gpu_type="V100", target_col="power", max_features=8
        )

        logger.info(f"Optimized data shape: {df_optimized.shape}")
        logger.info(
            f"Selected features: {analysis_results['final_feature_set']['features']}"
        )

        # Display feature analysis
        if "validation" in analysis_results:
            validation = analysis_results["validation"]
            logger.info(f"FGCS core features found: {validation['fgcs_core_features']}")
            if validation["missing_fgcs_features"]:
                logger.info(
                    f"Missing FGCS features: {validation['missing_fgcs_features']}"
                )

        if "selection" in analysis_results:
            selection = analysis_results["selection"]
            logger.info(f"Feature selection method: {selection['selection_method']}")
            logger.info(f"Number of features selected: {selection['num_selected']}")

        return True

    except ImportError:
        logger.warning("Feature selection module not available")
        return False
    except Exception as e:
        logger.error(f"Feature selection demo failed: {e}")
        return False


def demo_enhanced_edp_calculation():
    """Demonstrate enhanced EDP calculation with feature integration."""
    logger.info("\n=== Enhanced EDP Calculation Demo ===")

    try:
        from edp_analysis.edp_calculator import (
            analyze_feature_importance_for_edp,
            calculate_edp_with_features,
        )

        # Create synthetic data
        df = create_synthetic_profiling_data()
        if df is None:
            return False

        # Enhanced EDP calculation with feature selection
        edp_results = calculate_edp_with_features(
            df,
            energy_col="energy",
            delay_col="execution_time",
            use_feature_selection=True,
            gpu_type="V100",
        )

        # Display EDP analysis results
        if "edp_analysis" in edp_results:
            edp_analysis = edp_results["edp_analysis"]
            if "statistics" in edp_analysis:
                stats = edp_analysis["statistics"]
                logger.info(f"EDP Statistics:")
                logger.info(f"  Mean EDP: {stats['mean_edp']:.2f}")
                logger.info(f"  Min EDP: {stats['min_edp']:.2f}")
                logger.info(f"  Max EDP: {stats['max_edp']:.2f}")
                logger.info(f"  EDP Range: {stats['max_edp'] - stats['min_edp']:.2f}")

        if "optimization_results" in edp_results:
            opt_results = edp_results["optimization_results"]
            if "edp_optimal" in opt_results:
                edp_opt = opt_results["edp_optimal"]
                logger.info(f"EDP Optimal Configuration:")
                logger.info(f"  Frequency: {edp_opt['frequency']} MHz")
                logger.info(f"  Energy: {edp_opt['energy']:.2f} J")
                logger.info(f"  Time: {edp_opt['time']:.2f} s")

        # Feature importance analysis
        importance_results = analyze_feature_importance_for_edp(
            df, target_metrics=["energy", "execution_time"], gpu_type="V100"
        )

        if "feature_importance" in importance_results:
            for target, analysis in importance_results["feature_importance"].items():
                logger.info(
                    f"Top features for {target}: {analysis['selected_features'][:3]}"
                )

        return True

    except Exception as e:
        logger.error(f"Enhanced EDP calculation demo failed: {e}")
        return False


def demo_energy_profiling():
    """Demonstrate energy profiling capabilities."""
    logger.info("\n=== Energy Profiling Demo ===")

    try:
        from edp_analysis.energy_profiler import EnergyProfiler

        # Create profiler
        profiler = EnergyProfiler(power_units="watts", time_units="seconds")

        # Create synthetic data
        df = create_synthetic_profiling_data()
        if df is None:
            return False

        # FGCS-compatible energy calculation
        df_energy = profiler.calculate_fgcs_compatible_energy(
            df, power_col="power", time_col="execution_time", frequency_col="frequency"
        )

        logger.info(
            f"Added FGCS-compatible energy columns: {[col for col in df_energy.columns if col.startswith('n_') or col.startswith('predicted_')]}"
        )

        # Energy efficiency analysis
        efficiency_results = profiler.analyze_energy_efficiency_across_frequencies(
            df, power_col="power", time_col="execution_time", frequency_col="frequency"
        )

        if "efficiency_metrics" in efficiency_results:
            metrics = efficiency_results["efficiency_metrics"]
            logger.info(f"Energy Efficiency Analysis:")
            logger.info(
                f"  Min energy frequency: {metrics['min_energy_frequency']} MHz ({metrics['min_energy_value']:.2f} J)"
            )
            logger.info(
                f"  Min time frequency: {metrics['min_time_frequency']} MHz ({metrics['min_time_value']:.2f} s)"
            )
            logger.info(
                f"  Energy range: {metrics['energy_range'][0]:.2f} - {metrics['energy_range'][1]:.2f} J"
            )

        if "optimization_insights" in efficiency_results:
            insights = efficiency_results["optimization_insights"]
            logger.info(f"Optimization Insights:")
            logger.info(
                f"  Max energy savings: {insights['max_energy_savings_percent']:.1f}%"
            )
            logger.info(
                f"  Max time penalty: {insights['max_time_penalty_percent']:.1f}%"
            )

        # Measurement validation
        validation_results = profiler.validate_energy_measurements(
            df, power_col="power", time_col="execution_time", tolerance=0.1
        )

        if "data_quality" in validation_results:
            quality = validation_results["data_quality"]
            logger.info(f"Data Quality:")
            logger.info(f"  Total samples: {quality['total_samples']}")
            logger.info(f"  Data completeness: {quality['data_completeness']:.1%}")

        return True

    except Exception as e:
        logger.error(f"Energy profiling demo failed: {e}")
        return False


def demo_performance_profiling():
    """Demonstrate performance profiling capabilities."""
    logger.info("\n=== Performance Profiling Demo ===")

    try:
        from edp_analysis.performance_profiler import PerformanceProfiler

        # Create profiler
        profiler = PerformanceProfiler(time_units="seconds")

        # Create synthetic data
        df = create_synthetic_profiling_data()
        if df is None:
            return False

        # FGCS performance model analysis
        fgcs_results = profiler.analyze_fgcs_performance_model(
            df,
            fp_activity=0.3,
            baseline_time=None,
            time_col="execution_time",
            frequency_col="frequency",
        )

        if "accuracy_metrics" in fgcs_results and fgcs_results["accuracy_metrics"]:
            accuracy = fgcs_results["accuracy_metrics"]
            logger.info(f"FGCS Model Accuracy:")
            logger.info(f"  MAE: {accuracy['mae']:.3f} seconds")
            logger.info(f"  MAPE: {accuracy['mape']:.1f}%")
            logger.info(
                f"  Valid predictions: {accuracy['valid_predictions']}/{accuracy['total_predictions']}"
            )

        if "performance_insights" in fgcs_results:
            insights = fgcs_results["performance_insights"]
            if insights:  # Check if insights dict is not empty
                logger.info(f"Performance Insights:")
                logger.info(
                    f"  Fastest frequency: {insights.get('fastest_frequency', 'N/A')} MHz"
                )
                logger.info(
                    f"  Performance improvement potential: {insights.get('performance_improvement_potential', 0):.1f}%"
                )

        # Throughput analysis
        throughput_results = profiler.calculate_throughput_metrics(
            df, time_col="execution_time", frequency_col="frequency", workload_size=1000
        )

        if "basic_metrics" in throughput_results:
            metrics = throughput_results["basic_metrics"]
            logger.info(f"Throughput Metrics:")
            logger.info(f"  Mean throughput: {metrics['mean_throughput']:.2f} ops/sec")
            logger.info(f"  Throughput range: {metrics['throughput_range']:.2f}")
            logger.info(
                f"  Coefficient of variation: {metrics['coefficient_of_variation']:.3f}"
            )

        return True

    except Exception as e:
        logger.error(f"Performance profiling demo failed: {e}")
        return False


def demo_optimization_analysis():
    """Demonstrate optimization analysis capabilities."""
    logger.info("\n=== Optimization Analysis Demo ===")

    try:
        from edp_analysis.optimization_analyzer import (
            MultiObjectiveOptimizer,
            OptimizationResult,
        )

        # Create synthetic data
        df = create_synthetic_profiling_data()
        if df is None:
            return False

        # Multi-objective optimization
        optimizer = MultiObjectiveOptimizer()

        # Prepare data for optimization
        energy_values = df["energy"].values
        time_values = df["execution_time"].values
        frequencies = df["frequency"].values

        # Find Pareto optimal solutions
        pareto_results = optimizer.find_pareto_optimal_solutions(
            energy_values, time_values, frequencies
        )

        logger.info(f"Pareto Analysis:")
        logger.info(f"  Total configurations: {len(df)}")
        logger.info(f"  Pareto optimal solutions: {len(pareto_results)}")

        if pareto_results:
            # Show first few Pareto optimal points
            for i, result in enumerate(pareto_results[:3]):
                logger.info(
                    f"  Pareto point {i+1}: {result.frequency} MHz, "
                    f"E={result.energy:.2f}J, T={result.execution_time:.2f}s"
                )

        return True

    except Exception as e:
        logger.error(f"Optimization analysis demo failed: {e}")
        return False


def main():
    """Run all enhanced EDP analysis demonstrations."""
    logger.info("Starting Enhanced EDP Analysis Framework Demo")
    logger.info("=" * 60)

    demos = [
        ("Feature Selection and Engineering", demo_feature_selection_and_engineering),
        ("Enhanced EDP Calculation", demo_enhanced_edp_calculation),
        ("Energy Profiling", demo_energy_profiling),
        ("Performance Profiling", demo_performance_profiling),
        ("Optimization Analysis", demo_optimization_analysis),
    ]

    results = {}
    for demo_name, demo_func in demos:
        try:
            success = demo_func()
            results[demo_name] = success
            if success:
                logger.info(f"✓ {demo_name} completed successfully")
            else:
                logger.warning(f"⚠ {demo_name} completed with issues")
        except Exception as e:
            logger.error(f"✗ {demo_name} failed: {e}")
            results[demo_name] = False

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DEMO SUMMARY")
    logger.info("=" * 60)

    successful_demos = sum(results.values())
    total_demos = len(results)

    for demo_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status}: {demo_name}")

    logger.info(f"\nOverall: {successful_demos}/{total_demos} demos successful")

    if successful_demos == total_demos:
        logger.info("🎉 All enhanced EDP analysis features working correctly!")
        return True
    else:
        logger.warning(
            f"⚠ {total_demos - successful_demos} demos failed - check dependencies and setup"
        )
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced EDP Analysis Demo")
    parser.add_argument(
        "--gpu-type",
        default="V100",
        choices=["V100", "A100", "H100"],
        help="GPU type for analysis",
    )
    parser.add_argument(
        "--synthetic-data",
        action="store_true",
        help="Use synthetic data for demonstration",
    )

    args = parser.parse_args()

    success = main()
    sys.exit(0 if success else 1)
