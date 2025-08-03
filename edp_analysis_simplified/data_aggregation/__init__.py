"""
Data Aggregation Components for GPU Frequency Optimization

This package provides data loading, aggregation, and preprocessing
capabilities for profiling data.
"""

from .aggregator import (
    load_raw_profiling_data,
    parse_space_separated_profiling_files,
    aggregate_profiling_metrics,
    standardize_column_names,
    clean_profiling_data,
    create_aggregated_dataset
)

__all__ = [
    'load_raw_profiling_data',
    'parse_space_separated_profiling_files',
    'aggregate_profiling_metrics', 
    'standardize_column_names',
    'clean_profiling_data',
    'create_aggregated_dataset'
]
