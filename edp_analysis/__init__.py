"""
EDP Analysis Module

Enhanced Energy-Delay Product analysis framework inspired by FGCS 2023 methodology.
Provides comprehensive tools for energy-performance optimization in GPU applications.

Core Components:
- EDPCalculator: Core EDP = Energy * Delay calculation
- EnergyProfiler: Energy consumption measurement and analysis
- PerformanceProfiler: Execution time measurement and performance analysis
- OptimizationAnalyzer: Multi-objective optimization and Pareto analysis
- Feature Selection: FGCS-inspired feature engineering and selection
- Visualization: EDP plots, heatmaps, and trade-off analysis

Quick Usage:
    from edp_analysis import EDPCalculator, EnergyProfiler
    
    calculator = EDPCalculator()
    edp = calculator.calculate_edp(energy, delay)
    
    profiler = EnergyProfiler()
    energy_metrics = profiler.calculate_energy_from_power_time(power, time)
"""

# Core EDP calculation functionality
from .edp_calculator import (
    DVFSOptimizationPipeline,
    EDPCalculator,
    FGCSEDPOptimizer,
    analyze_feature_importance_for_edp,
    calculate_edp_with_features,
    calculate_energy_from_power_time,
    normalize_metrics,
)

# Energy profiling
from .energy_profiler import EnergyProfiler

# Optimization analysis
from .optimization_analyzer import MultiObjectiveOptimizer, OptimizationRecommendation, OptimizationResult

# Performance profiling
from .performance_profiler import PerformanceProfiler

# Feature selection (conditional import - may not be available in all environments)
try:
    from .feature_selection import EDPFeatureSelector, FGCSFeatureEngineering, create_optimized_feature_set

    FEATURE_SELECTION_AVAILABLE = True
except ImportError:
    FEATURE_SELECTION_AVAILABLE = False

# Visualization (conditional import)
try:
    from .visualization import create_edp_heatmap, plot_edp_vs_frequency, plot_energy_delay_tradeoff, plot_pareto_frontier

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Module metadata
__version__ = "1.0.0"
__author__ = "AI Inference Energy Profiling Framework"
__description__ = "Enhanced EDP analysis with FGCS methodology integration"

# Export lists for explicit imports
__all__ = [
    # Core EDP functionality
    "EDPCalculator",
    "FGCSEDPOptimizer",
    "DVFSOptimizationPipeline",
    "calculate_energy_from_power_time",
    "normalize_metrics",
    "calculate_edp_with_features",
    "analyze_feature_importance_for_edp",
    # Profiling
    "EnergyProfiler",
    "PerformanceProfiler",
    # Optimization
    "MultiObjectiveOptimizer",
    "OptimizationResult",
    "OptimizationRecommendation",
    # Feature availability flags
    "FEATURE_SELECTION_AVAILABLE",
    "VISUALIZATION_AVAILABLE",
]

# Add feature selection exports if available
if FEATURE_SELECTION_AVAILABLE:
    __all__.extend(["FGCSFeatureEngineering", "EDPFeatureSelector", "create_optimized_feature_set"])

# Add visualization exports if available
if VISUALIZATION_AVAILABLE:
    __all__.extend(["plot_edp_vs_frequency", "plot_pareto_frontier", "plot_energy_delay_tradeoff", "create_edp_heatmap"])


def get_module_info():
    """
    Get information about the EDP analysis module capabilities.

    Returns:
        Dictionary with module information and available features
    """
    return {
        "version": __version__,
        "description": __description__,
        "core_components": ["EDPCalculator", "EnergyProfiler", "PerformanceProfiler", "OptimizationAnalyzer"],
        "feature_selection_available": FEATURE_SELECTION_AVAILABLE,
        "visualization_available": VISUALIZATION_AVAILABLE,
        "fgcs_compatibility": True,
        "supported_gpus": ["V100", "A100", "H100"],
        "optimization_methods": ["EDP", "ED2P", "Pareto", "Multi-objective"],
        "example_usage": {
            "basic_edp": "calculator = EDPCalculator(); edp = calculator.calculate_edp(energy, delay)",
            "energy_profiling": "profiler = EnergyProfiler(); metrics = profiler.calculate_energy_from_power_time(power, time)",
            "feature_selection": (
                'df_opt, analysis = create_optimized_feature_set(df, gpu_type="V100")'
                if FEATURE_SELECTION_AVAILABLE
                else "Not available"
            ),
        },
    }


def quick_edp_analysis(power_data, time_data, frequencies=None, gpu_type="V100"):
    """
    Quick EDP analysis function for simple use cases.

    Args:
        power_data: Array-like power measurements (Watts)
        time_data: Array-like execution time measurements (seconds)
        frequencies: Optional frequency data for optimization
        gpu_type: GPU type for FGCS compatibility

    Returns:
        Dictionary with EDP analysis results
    """
    try:
        import numpy as np

        # Calculate basic EDP metrics
        calculator = EDPCalculator()

        energy_data = np.array(power_data) * np.array(time_data)
        edp_values = calculator.calculate_edp(energy_data, time_data)
        ed2p_values = calculator.calculate_ed2p(energy_data, time_data)

        results = {
            "edp_values": edp_values,
            "ed2p_values": ed2p_values,
            "min_edp": np.min(edp_values),
            "min_ed2p": np.min(ed2p_values),
            "energy_range": (np.min(energy_data), np.max(energy_data)),
            "time_range": (np.min(time_data), np.max(time_data)),
        }

        if frequencies is not None:
            min_edp_idx = np.argmin(edp_values)
            min_ed2p_idx = np.argmin(ed2p_values)

            results["optimal_frequencies"] = {
                "edp_optimal": frequencies[min_edp_idx],
                "ed2p_optimal": frequencies[min_ed2p_idx],
            }

        return results

    except ImportError:
        return {"error": "NumPy not available for quick analysis"}
    except Exception as e:
        return {"error": f"Quick analysis failed: {e}"}


# Module initialization message
import logging

logger = logging.getLogger(__name__)
logger.info(f"EDP Analysis Module v{__version__} loaded")
if FEATURE_SELECTION_AVAILABLE:
    logger.info("✓ Feature selection capabilities available")
else:
    logger.info("⚠ Feature selection capabilities not available (sklearn dependencies missing)")
if VISUALIZATION_AVAILABLE:
    logger.info("✓ Visualization capabilities available")
else:
    logger.info("⚠ Visualization capabilities not available (matplotlib dependencies missing)")
