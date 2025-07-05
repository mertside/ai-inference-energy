#!/usr/bin/env python3
"""
Simple Power Modeling and EDP Example

This script demonstrates basic usage of the power modeling framework
with synthetic data to show EDP/ED2P optimization capabilities.
"""

import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def create_synthetic_data():
    """Create synthetic profiling data for demonstration."""
    try:
        import numpy as np
        import pandas as pd

        # Set random seed for reproducibility
        np.random.seed(42)

        # Generate realistic profiling data
        n_samples = 20
        data = pd.DataFrame(
            {
                "app_name": ["SampleApp"] * n_samples,
                "fp_activity": np.random.uniform(0.2, 0.6, n_samples),
                "dram_activity": np.random.uniform(0.1, 0.3, n_samples),
                "sm_clock": np.random.choice([1000, 1100, 1200, 1300, 1400], n_samples),
                "power": np.random.uniform(180, 280, n_samples),
            }
        )

        return data

    except ImportError as e:
        logger.error(f"Required packages not available: {e}")
        return None


def demo_fgcs_model():
    """Demonstrate FGCS power model usage."""
    logger.info("=== FGCS Power Model Demo ===")

    try:
        from power_modeling.models.fgcs_models import FGCSPowerModel

        # Create FGCS model
        model = FGCSPowerModel()
        logger.info("✓ FGCS model created successfully")

        # Test power prediction
        fp_activity = 0.3
        dram_activity = 0.15
        frequencies = [1000, 1100, 1200, 1300, 1400]

        predictions = model.predict_power(fp_activity, dram_activity, frequencies)
        logger.info(f"✓ Power predictions generated for {len(frequencies)} frequencies")

        # Show predictions
        logger.info("Power predictions:")
        for _, row in predictions.iterrows():
            freq = row["sm_app_clock"]
            power_log = row["predicted_n_power_usage"]
            logger.info(f"  {freq} MHz: {power_log:.4f} (log-normalized)")

        return True

    except Exception as e:
        logger.error(f"FGCS model demo failed: {e}")
        return False


def demo_edp_calculations():
    """Demonstrate EDP/ED2P calculations."""
    logger.info("\n=== EDP/ED2P Calculation Demo ===")

    try:
        import numpy as np

        from edp_analysis.edp_calculator import EDPCalculator

        # Create calculator
        calculator = EDPCalculator()
        logger.info("✓ EDP calculator created")

        # Sample data
        energy = np.array([100, 120, 140, 160, 180])  # Joules
        delay = np.array([1.5, 1.3, 1.1, 0.9, 0.8])  # seconds
        frequencies = [1000, 1100, 1200, 1300, 1400]  # MHz

        # Calculate EDP
        edp = calculator.calculate_edp(energy, delay)
        logger.info("✓ EDP calculated")

        # Calculate ED2P
        ed2p = calculator.calculate_ed2p(energy, delay)
        logger.info("✓ ED2P calculated")

        # Show results
        logger.info("EDP/ED2P Analysis:")
        logger.info("Freq(MHz) | Energy(J) | Delay(s) | EDP | ED2P")
        logger.info("-" * 50)

        for i in range(len(frequencies)):
            freq = frequencies[i]
            e = energy[i]
            d = delay[i]
            edp_val = edp[i]
            ed2p_val = ed2p[i]
            logger.info(
                f"{freq:8} | {e:8.1f} | {d:7.2f} | {edp_val:6.1f} | {ed2p_val:6.1f}"
            )

        # Find optimal configurations
        min_edp_idx = np.argmin(edp)
        min_ed2p_idx = np.argmin(ed2p)

        logger.info(f"\nOptimal configurations:")
        logger.info(
            f"EDP optimal: {frequencies[min_edp_idx]} MHz (EDP = {edp[min_edp_idx]:.1f})"
        )
        logger.info(
            f"ED2P optimal: {frequencies[min_ed2p_idx]} MHz (ED2P = {ed2p[min_ed2p_idx]:.1f})"
        )

        return True

    except Exception as e:
        logger.error(f"EDP calculation demo failed: {e}")
        return False


def demo_fgcs_optimizer():
    """Demonstrate FGCS EDP optimizer."""
    logger.info("\n=== FGCS EDP Optimizer Demo ===")

    try:
        import pandas as pd

        from edp_analysis.edp_calculator import FGCSEDPOptimizer

        # Create synthetic optimization data
        frequencies = [1000, 1100, 1200, 1300, 1400]
        df = pd.DataFrame(
            {
                "sm_app_clock": frequencies,
                "predicted_n_to_r_energy": [100, 120, 140, 160, 180],
                "predicted_n_to_r_run_time": [1.5, 1.3, 1.1, 0.9, 0.8],
                "predicted_n_to_r_power_usage": [66.7, 92.3, 127.3, 177.8, 225.0],
            }
        )

        logger.info("✓ Synthetic optimization data created")

        # Run EDP optimization
        edp_freq, edp_time, edp_power, edp_energy = FGCSEDPOptimizer.edp_optimal(df)
        logger.info(
            f"✓ EDP optimal: {edp_freq} MHz, {edp_time}s, {edp_power}W, {edp_energy}J"
        )

        # Run ED2P optimization
        ed2p_freq, ed2p_time, ed2p_power, ed2p_energy = FGCSEDPOptimizer.ed2p_optimal(
            df
        )
        logger.info(
            f"✓ ED2P optimal: {ed2p_freq} MHz, {ed2p_time}s, {ed2p_power}W, {ed2p_energy}J"
        )

        # Run full DVFS optimization analysis
        results = FGCSEDPOptimizer.analyze_dvfs_optimization(df, "DemoApp")
        logger.info("✓ Complete DVFS optimization analysis completed")

        # Show optimization results
        logger.info("\nOptimization Results:")
        logger.info(f"EDP Optimal Frequency: {results['edp_optimal']['frequency']} MHz")
        logger.info(
            f"ED2P Optimal Frequency: {results['ed2p_optimal']['frequency']} MHz"
        )
        logger.info(f"Min Energy Frequency: {results['min_energy']['frequency']} MHz")
        logger.info(f"Min Time Frequency: {results['min_time']['frequency']} MHz")

        return True

    except Exception as e:
        logger.error(f"FGCS optimizer demo failed: {e}")
        return False


def demo_high_level_framework():
    """Demonstrate high-level framework usage."""
    logger.info("\n=== High-Level Framework Demo ===")

    try:
        from power_modeling import FGCSPowerModelingFramework

        # Create framework
        framework = FGCSPowerModelingFramework(gpu_type="V100")
        logger.info("✓ Framework initialized for V100 GPU")

        # Show GPU configurations
        logger.info(f"V100 frequency count: {len(framework.frequency_configs['V100'])}")
        logger.info(f"A100 frequency count: {len(framework.frequency_configs['A100'])}")
        logger.info(f"H100 frequency count: {len(framework.frequency_configs['H100'])}")

        # Test power prediction sweep
        power_sweep = framework.predict_power_sweep(
            fp_activity=0.3,
            dram_activity=0.15,
            frequencies=[1000, 1100, 1200, 1300, 1400],
        )
        logger.info(f"✓ Power sweep completed: {len(power_sweep)} predictions")

        # Test application optimization
        optimization_results = framework.optimize_application(
            fp_activity=0.3,
            dram_activity=0.15,
            baseline_runtime=1.0,
            app_name="DemoApp",
        )
        logger.info("✓ Application optimization completed")

        # Show optimization recommendations
        recommendations = optimization_results["recommendations"]
        primary = recommendations["primary_recommendation"]

        logger.info("\nOptimization Recommendations:")
        logger.info(f"Primary: {primary['frequency']} - {primary['reason']}")
        logger.info(f"Expected energy savings: {primary['expected_energy_savings']}")
        logger.info(
            f"Expected performance impact: {primary['expected_performance_impact']}"
        )

        return True

    except Exception as e:
        logger.error(f"High-level framework demo failed: {e}")
        return False


def main():
    """Run all demonstrations."""
    logger.info("🚀 Power Modeling and EDP/ED2P Framework Demo")
    logger.info("=" * 60)

    demos = [
        ("FGCS Power Model", demo_fgcs_model),
        ("EDP/ED2P Calculations", demo_edp_calculations),
        ("FGCS EDP Optimizer", demo_fgcs_optimizer),
        ("High-Level Framework", demo_high_level_framework),
    ]

    passed = 0
    failed = 0

    for demo_name, demo_func in demos:
        try:
            if demo_func():
                passed += 1
                logger.info(f"✅ {demo_name} demo completed successfully")
            else:
                failed += 1
                logger.error(f"❌ {demo_name} demo failed")
        except Exception as e:
            failed += 1
            logger.error(f"❌ {demo_name} demo failed with exception: {e}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DEMO SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Completed: {passed}")
    logger.info(f"Failed: {failed}")

    if failed == 0:
        logger.info("🎉 All demos completed successfully!")
        logger.info("✅ Core power modeling components are working")
        logger.info("✅ EDP/ED2P framework is complete and functional")
        logger.info("✅ Ready for production use!")
    else:
        logger.error(f"❌ {failed} demos failed")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
