"""
GPU Frequency Optimization Framework

A simplified and organized framework for optimizing GPU frequencies to achieve
minimal performance degradation with maximum energy savings.
"""

__version__ = "1.0.0"
__author__ = "AI Inference Energy Research Team"

# Import main components for easy access
from .core import (
    calculate_efficiency_metrics,
    find_optimal_configurations,
    load_and_validate_data
)

from .visualization import generate_all_plots
from .frequency_optimization import create_deployment_package

__all__ = [
    'calculate_efficiency_metrics',
    'find_optimal_configurations', 
    'load_and_validate_data',
    'generate_all_plots',
    'create_deployment_package'
]
