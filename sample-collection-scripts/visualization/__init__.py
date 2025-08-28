"""
Visualization Package for EDP Analysis

This package provides comprehensive visualization capabilities for the AI Inference Energy
Profiling Framework, including EDP analysis, power consumption, and performance visualization.

Enhanced with FGCS 2023 methodology integration:
- Feature importance visualization
- FGCS model validation plots
- Comprehensive EDP dashboards
- Energy efficiency analysis
- Performance breakdown visualization

Modules:
- edp_plots: Energy-Delay Product visualization with feature importance
- power_plots: Power consumption analysis and FGCS validation plots
- performance_plots: Performance analysis and throughput visualization

The visualization modules handle conditional imports for matplotlib, seaborn, and other
plotting libraries, providing graceful degradation when dependencies are not available.
"""

from .data_preprocessor import ProfilingDataPreprocessor, create_synthetic_profiling_data
from .edp_plots import EDPPlotter, create_optimization_summary_plot, plot_edp_heatmap
from .performance_plots import PerformancePlotter, plot_performance_heatmap
from .power_plots import PowerPlotter, plot_power_histogram

__all__ = [
    # Core plotter classes
    "EDPPlotter",
    "PowerPlotter",
    "PerformancePlotter",
    # Data preprocessing
    "ProfilingDataPreprocessor",
    "create_synthetic_profiling_data",
    # Convenience functions
    "plot_edp_heatmap",
    "plot_power_histogram",
    "plot_performance_heatmap",
    "create_optimization_summary_plot",
]

__version__ = "1.1.0"  # Updated for enhanced FGCS integration
