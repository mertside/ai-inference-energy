"""
Core Components for GPU Frequency Optimization

This package contains the fundamental analysis functions and utilities
for GPU frequency optimization.
"""

from .analysis import (
    calculate_efficiency_metrics,
    categorize_performance_impact,
    find_optimal_configurations,
    validate_data_format,
    load_and_validate_data,
    calculate_summary_statistics
)

from .utils import (
    create_output_directory,
    save_json_results,
    format_percentage,
    format_frequency,
    format_efficiency_ratio,
    print_section_header,
    print_subsection_header,
    print_success,
    print_info,
    print_warning,
    print_error,
    get_memory_frequency,
    validate_frequency_range
)

__all__ = [
    # Analysis functions
    'calculate_efficiency_metrics',
    'categorize_performance_impact', 
    'find_optimal_configurations',
    'validate_data_format',
    'load_and_validate_data',
    'calculate_summary_statistics',
    
    # Utility functions
    'create_output_directory',
    'save_json_results',
    'format_percentage',
    'format_frequency', 
    'format_efficiency_ratio',
    'print_section_header',
    'print_subsection_header',
    'print_success',
    'print_info',
    'print_warning',
    'print_error',
    'get_memory_frequency',
    'validate_frequency_range'
]
