#!/usr/bin/env python3
"""
Enhanced Visualization Demo for EDP Analysis

This script demonstrates the enhanced visualization capabilities of the EDP analysis module,
showcasing the integration with FGCS 2023 methodology, feature selection, and comprehensive
energy-performance analysis.

Features demonstrated:
- Feature importance visualization
- FGCS model validation plots
- Comprehensive EDP dashboards
- Energy efficiency analysis
- Performance breakdown visualization
- Multi-objective optimization results

Usage:
    python enhanced_visualization_demo.py [--save-plots] [--output-dir plots/]
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def create_synthetic_data_for_visualization():
    """Create comprehensive synthetic data for visualization demo."""
    try:
        import numpy as np
        import pandas as pd

        # Set random seed for reproducibility
        np.random.seed(42)

        # Generate data for multiple frequencies
        frequencies = np.array([800, 900, 1000, 1100, 1200, 1300, 1400])
        n_runs_per_freq = 5

        data = []
        for freq in frequencies:
            for run in range(n_runs_per_freq):
                # Simulate realistic GPU behavior
                base_power = 150 + (freq - 800) * 0.08
                power_noise = np.random.normal(0, 8)
                power = max(100, base_power + power_noise)

                # Execution time inversely related to frequency
                base_time = 2.0 * (1400 / freq)
                time_noise = np.random.normal(0, 0.08)
                execution_time = max(0.5, base_time + time_noise)

                # Energy calculation
                energy = power * execution_time

                # FGCS-style features
                fp_activity = 0.2 + (freq - 800) / 6000 * 0.3 + np.random.normal(0, 0.02)
                dram_activity = 0.1 + (freq - 800) / 6000 * 0.1 + np.random.normal(0, 0.01)

                # Throughput calculation
                throughput = 1000 / execution_time  # operations per second

                data.append(
                    {
                        "frequency": freq,
                        "power": power,
                        "execution_time": execution_time,
                        "energy": energy,
                        "energy_joules": energy,
                        "fp_activity": fp_activity,
                        "dram_activity": dram_activity,
                        "throughput": throughput,
                        "run": run,
                    }
                )

        df = pd.DataFrame(data)

        # Add predicted values for validation plots
        df["predicted_power"] = df["power"] + np.random.normal(0, 5, len(df))
        df["predicted_runtime"] = df["execution_time"] + np.random.normal(0, 0.05, len(df))

        logger.info(f"Created synthetic dataset with {len(df)} data points")
        return df

    except ImportError as e:
        logger.error(f"Required packages not available: {e}")
        return None


def create_feature_importance_data():
    """Create synthetic feature importance data."""
    try:
        import numpy as np

        # Simulate feature importance scores from FGCS methodology
        features = [
            "fp_activity",
            "dram_activity",
            "sm_clock",
            "frequency_squared",
            "fp_dram_interaction",
            "log_frequency",
            "power_density",
            "compute_intensity",
            "memory_bandwidth_util",
            "cache_hit_rate",
            "instruction_throughput",
            "memory_latency",
            "thermal_state",
            "voltage_level",
            "workload_complexity",
        ]

        # Generate realistic importance scores
        np.random.seed(42)
        importance_scores = np.random.exponential(0.1, len(features))
        importance_scores = importance_scores / importance_scores.sum()  # Normalize

        feature_importance = dict(zip(features, importance_scores))

        logger.info(f"Created feature importance data with {len(features)} features")
        return feature_importance

    except ImportError as e:
        logger.error(f"Required packages not available: {e}")
        return None


def create_optimization_results(df):
    """Create synthetic optimization results."""
    try:
        import numpy as np

        # Calculate EDP and ED²P for each frequency
        edp_values = df["energy"] * df["execution_time"]
        ed2p_values = df["energy"] * (df["execution_time"] ** 2)

        # Find optimal points
        avg_df = df.groupby("frequency").agg({"energy": "mean", "execution_time": "mean", "power": "mean"}).reset_index()

        avg_edp = avg_df["energy"] * avg_df["execution_time"]
        avg_ed2p = avg_df["energy"] * (avg_df["execution_time"] ** 2)

        edp_optimal_idx = avg_edp.idxmin()
        ed2p_optimal_idx = avg_ed2p.idxmin()

        optimization_results = {
            "edp_optimal": {
                "frequency": int(avg_df.loc[edp_optimal_idx, "frequency"]),
                "energy": float(avg_df.loc[edp_optimal_idx, "energy"]),
                "runtime": float(avg_df.loc[edp_optimal_idx, "execution_time"]),
                "power": float(avg_df.loc[edp_optimal_idx, "power"]),
                "edp": float(avg_edp.iloc[edp_optimal_idx]),
            },
            "ed2p_optimal": {
                "frequency": int(avg_df.loc[ed2p_optimal_idx, "frequency"]),
                "energy": float(avg_df.loc[ed2p_optimal_idx, "energy"]),
                "runtime": float(avg_df.loc[ed2p_optimal_idx, "execution_time"]),
                "power": float(avg_df.loc[ed2p_optimal_idx, "power"]),
                "ed2p": float(avg_ed2p.iloc[ed2p_optimal_idx]),
            },
            "frequency_sweep_data": {
                "sm_app_clock": avg_df["frequency"].tolist(),
                "predicted_n_to_r_energy": avg_df["energy"].tolist(),
                "predicted_n_to_r_run_time": avg_df["execution_time"].tolist(),
                "predicted_n_to_r_power_usage": avg_df["power"].tolist(),
            },
        }

        logger.info("Created optimization results")
        return optimization_results

    except Exception as e:
        logger.error(f"Error creating optimization results: {e}")
        return None


def demonstrate_edp_visualizations(df, optimization_results, feature_importance, output_dir):
    """Demonstrate EDP visualization capabilities."""
    try:
        from edp_analysis.visualization.edp_plots import EDPPlotter

        plotter = EDPPlotter(style="seaborn-v0_8", figsize=(12, 8))
        logger.info("Demonstrating EDP visualizations...")

        # 1. Basic EDP vs Frequency plot
        fig1 = plotter.plot_edp_vs_frequency(
            df=df.groupby("frequency").mean().reset_index(),
            optimal_freq=optimization_results["edp_optimal"]["frequency"],
            title="EDP Analysis - Synthetic GPU Application",
            save_path=output_dir / "edp_vs_frequency.png" if output_dir else None,
        )

        # 2. Energy-Delay Trade-off plot
        fig2 = plotter.plot_energy_delay_tradeoff(
            df=df,
            optimal_points=optimization_results,
            title="Energy-Delay Trade-off Analysis",
            save_path=output_dir / "energy_delay_tradeoff.png" if output_dir else None,
        )

        # 3. Feature importance plot
        if feature_importance:
            fig3 = plotter.plot_feature_importance_for_edp(
                feature_importance=feature_importance,
                title="Feature Importance for EDP Optimization",
                save_path=output_dir / "feature_importance.png" if output_dir else None,
            )

        # 4. FGCS optimization results
        fig4 = plotter.plot_fgcs_optimization_results(
            optimization_results=optimization_results,
            title="FGCS EDP Optimization Results",
            save_path=output_dir / "fgcs_optimization.png" if output_dir else None,
        )

        # 5. Comprehensive dashboard
        fig5 = plotter.create_comprehensive_edp_dashboard(
            profiling_data=df,
            optimization_results=optimization_results,
            feature_importance=feature_importance,
            app_name="Synthetic GPU Application",
            save_path=output_dir / "comprehensive_dashboard.png" if output_dir else None,
        )

        logger.info("✅ EDP visualizations completed successfully")
        return True

    except ImportError:
        logger.warning("Matplotlib not available, skipping EDP visualizations")
        return False
    except Exception as e:
        logger.error(f"Error in EDP visualizations: {e}")
        return False


def demonstrate_power_visualizations(df, output_dir):
    """Demonstrate power visualization capabilities."""
    try:
        from edp_analysis.visualization.power_plots import PowerPlotter

        plotter = PowerPlotter(style="seaborn-v0_8", figsize=(12, 8))
        logger.info("Demonstrating power visualizations...")

        # 1. Power vs Frequency
        fig1 = plotter.plot_power_vs_frequency(
            df=df.groupby("frequency").mean().reset_index(),
            app_name="Synthetic GPU Application",
            title="Power Consumption Analysis",
            save_path=output_dir / "power_vs_frequency.png" if output_dir else None,
        )

        # 2. FGCS Power Validation
        fig2 = plotter.plot_fgcs_power_validation(
            df=df,
            app_name="Synthetic GPU Application",
            title="FGCS Power Model Validation",
            save_path=output_dir / "power_validation.png" if output_dir else None,
        )

        # 3. Power Breakdown Analysis
        fig3 = plotter.plot_power_breakdown_analysis(
            df=df,
            app_name="Synthetic GPU Application",
            title="Power Breakdown Analysis",
            save_path=output_dir / "power_breakdown.png" if output_dir else None,
        )

        # 4. Energy Efficiency Analysis
        fig4 = plotter.plot_energy_efficiency_analysis(
            df=df,
            app_name="Synthetic GPU Application",
            title="Energy Efficiency Analysis",
            save_path=output_dir / "energy_efficiency.png" if output_dir else None,
        )

        logger.info("✅ Power visualizations completed successfully")
        return True

    except ImportError:
        logger.warning("Matplotlib not available, skipping power visualizations")
        return False
    except Exception as e:
        logger.error(f"Error in power visualizations: {e}")
        return False


def demonstrate_performance_visualizations(df, output_dir):
    """Demonstrate performance visualization capabilities."""
    try:
        from edp_analysis.visualization.performance_plots import PerformancePlotter

        plotter = PerformancePlotter(style="seaborn-v0_8", figsize=(12, 8))
        logger.info("Demonstrating performance visualizations...")

        # 1. Execution Time vs Frequency
        fig1 = plotter.plot_execution_time_vs_frequency(
            df=df.groupby("frequency").mean().reset_index(),
            app_name="Synthetic GPU Application",
            title="Performance Analysis",
            save_path=output_dir / "performance_vs_frequency.png" if output_dir else None,
        )

        # 2. FGCS Performance Validation
        fig2 = plotter.plot_fgcs_performance_validation(
            df=df,
            app_name="Synthetic GPU Application",
            title="FGCS Performance Model Validation",
            save_path=output_dir / "performance_validation.png" if output_dir else None,
        )

        # 3. Throughput Analysis
        fig3 = plotter.plot_throughput_analysis(
            df=df,
            app_name="Synthetic GPU Application",
            title="Throughput and Latency Analysis",
            save_path=output_dir / "throughput_analysis.png" if output_dir else None,
        )

        # 4. Performance Breakdown
        fig4 = plotter.plot_performance_breakdown(
            df=df,
            app_name="Synthetic GPU Application",
            title="Performance Breakdown Analysis",
            save_path=output_dir / "performance_breakdown.png" if output_dir else None,
        )

        logger.info("✅ Performance visualizations completed successfully")
        return True

    except ImportError:
        logger.warning("Matplotlib not available, skipping performance visualizations")
        return False
    except Exception as e:
        logger.error(f"Error in performance visualizations: {e}")
        return False


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Enhanced EDP Visualization Demo")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to files")
    parser.add_argument("--output-dir", type=str, default="plots", help="Output directory for saved plots")

    args = parser.parse_args()

    # Setup output directory
    output_dir = None
    if args.save_plots:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        logger.info(f"Plots will be saved to: {output_dir}")

    logger.info("🚀 Starting Enhanced EDP Visualization Demo")

    # Create synthetic data
    logger.info("📊 Creating synthetic data...")
    df = create_synthetic_data_for_visualization()
    if df is None:
        logger.error("Failed to create synthetic data")
        return 1

    feature_importance = create_feature_importance_data()
    optimization_results = create_optimization_results(df)

    if optimization_results is None:
        logger.error("Failed to create optimization results")
        return 1

    # Demonstrate visualizations
    success_count = 0

    # EDP Visualizations
    if demonstrate_edp_visualizations(df, optimization_results, feature_importance, output_dir):
        success_count += 1

    # Power Visualizations
    if demonstrate_power_visualizations(df, output_dir):
        success_count += 1

    # Performance Visualizations
    if demonstrate_performance_visualizations(df, output_dir):
        success_count += 1

    # Summary
    logger.info(f"🎉 Demo completed! Successfully demonstrated {success_count}/3 visualization categories")

    if args.save_plots and success_count > 0:
        logger.info(f"📁 All plots saved to: {output_dir}")
        logger.info("📋 Generated files:")
        for plot_file in sorted(output_dir.glob("*.png")):
            logger.info(f"  - {plot_file.name}")

    if success_count == 0:
        logger.warning("⚠️  No visualizations were generated. Install matplotlib and seaborn to see plots.")
        logger.info("Run: pip install matplotlib seaborn")

    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
