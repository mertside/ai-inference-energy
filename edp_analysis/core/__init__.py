"""
Core Module for EDP Analysis Framework

This module provides the core functionality for GPU frequency optimization
and energy-delay product analysis.

Author: Mert Side
"""

from .data_loader import ProfilingDataLoader, load_profiling_data, validate_profiling_data
from .energy_calculator import EnergyCalculator, EDPResult, FGCSEDPOptimizer, quick_energy_analysis
from .frequency_optimizer import FrequencyOptimizer, quick_frequency_optimization, optimize_all_frequencies
from .performance_analyzer import PerformanceAnalyzer, PerformanceMetrics, PerformanceConstraintManager, quick_performance_analysis
from .utils import (
    setup_logging, load_config, save_config, save_results, load_results,
    validate_gpu_name, validate_application_name, create_configuration_key,
    format_percentage, format_frequency, format_time, format_energy, format_power,
    create_summary_table, create_deployment_summary, get_framework_version
)

__version__ = "2.0.0-refactored"

# High-level API functions for common use cases
def analyze_configuration(data_path: str, 
                         gpu: str, 
                         application: str,
                         config_path: str = None) -> dict:
    """
    High-level function to analyze a single GPU-application configuration.
    
    Args:
        data_path: Path to aggregated profiling data
        gpu: GPU type (e.g., 'A100', 'V100')
        application: Application name (e.g., 'LLAMA', 'STABLEDIFFUSION')
        config_path: Optional path to configuration file
        
    Returns:
        Dictionary with complete analysis results
    """
    # Load data
    data, metadata = load_profiling_data(data_path, config_path)
    
    # Run optimization
    result = quick_frequency_optimization(data, gpu, application, config_path)
    
    # Add metadata
    result['data_metadata'] = metadata
    result['analysis_timestamp'] = pd.Timestamp.now().isoformat()
    
    return result


def optimize_all_configurations(data_path: str, 
                               config_path: str = None,
                               output_path: str = None) -> dict:
    """
    High-level function to optimize all configurations in a dataset.
    
    Args:
        data_path: Path to aggregated profiling data
        config_path: Optional path to configuration file
        output_path: Optional path to save results
        
    Returns:
        Dictionary with complete optimization results
    """
    # Load data
    data, metadata = load_profiling_data(data_path, config_path)
    
    # Run optimization
    results = optimize_all_frequencies(data, config_path, output_path)
    
    # Add metadata
    results['data_metadata'] = metadata
    
    return results


def quick_analysis(data_path: str, 
                  gpu: str = None, 
                  application: str = None,
                  save_results_to: str = None) -> dict:
    """
    Quick analysis function for rapid insights.
    
    Args:
        data_path: Path to aggregated profiling data
        gpu: Optional GPU filter
        application: Optional application filter
        save_results_to: Optional path to save results
        
    Returns:
        Dictionary with analysis results
    """
    # Load data
    data, metadata = load_profiling_data(data_path)
    
    # Filter data if specified
    if gpu is not None:
        data = data[data['gpu'] == validate_gpu_name(gpu)]
    if application is not None:
        data = data[data['application'] == validate_application_name(application)]
    
    if len(data) == 0:
        return {'error': 'No data found matching filters'}
    
    # Run analysis
    if gpu is not None and application is not None:
        # Single configuration analysis
        result = analyze_configuration(data_path, gpu, application)
    else:
        # Multi-configuration analysis
        result = optimize_all_configurations(data_path)
    
    # Save results if requested
    if save_results_to is not None:
        save_results(result, save_results_to)
    
    return result


# Convenience imports for backward compatibility
import pandas as pd

__all__ = [
    # Core classes
    'ProfilingDataLoader',
    'EnergyCalculator', 
    'FrequencyOptimizer',
    'PerformanceAnalyzer',
    'PerformanceConstraintManager',
    
    # Data structures
    'EDPResult',
    'PerformanceMetrics',
    
    # High-level functions
    'analyze_configuration',
    'optimize_all_configurations', 
    'quick_analysis',
    
    # Utility functions
    'load_profiling_data',
    'quick_energy_analysis',
    'quick_frequency_optimization',
    'quick_performance_analysis',
    
    # Helper functions
    'setup_logging',
    'load_config',
    'save_config',
    'save_results',
    'load_results',
    'validate_gpu_name',
    'validate_application_name',
    'create_configuration_key',
    'format_percentage',
    'format_frequency',
    'format_time',
    'format_energy',
    'format_power',
    'create_summary_table',
    'create_deployment_summary',
    'get_framework_version',
    
    # Version
    '__version__'
]
