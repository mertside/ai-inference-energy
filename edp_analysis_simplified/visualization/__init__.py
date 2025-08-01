"""
Visualization Components for GPU Frequency Optimization

This package provides plotting and visualization capabilities for
frequency optimization analysis.
"""

from .plots import (
    create_pareto_plot,
    create_frequency_bar_plot,
    create_efficiency_ratio_plot,
    create_summary_dashboard,
    generate_all_plots
)

__all__ = [
    'create_pareto_plot',
    'create_frequency_bar_plot', 
    'create_efficiency_ratio_plot',
    'create_summary_dashboard',
    'generate_all_plots'
]
