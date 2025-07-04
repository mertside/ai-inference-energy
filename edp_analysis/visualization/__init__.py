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

from .edp_plots import (
    EDPPlotter, 
    plot_edp_heatmap, 
    create_optimization_summary_plot
)
from .power_plots import (
    PowerPlotter, 
    plot_power_histogram
)
from .performance_plots import (
    PerformancePlotter, 
    plot_performance_heatmap
)

__all__ = [
    # Core plotter classes
    'EDPPlotter',
    'PowerPlotter', 
    'PerformancePlotter',
    
    # Convenience functions
    'plot_edp_heatmap',
    'plot_power_histogram',
    'plot_performance_heatmap',
    'create_optimization_summary_plot'
]

__version__ = "1.1.0"  # Updated for enhanced FGCS integration
