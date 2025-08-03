"""
Data aggregation and preprocessing functions for GPU frequency optimization.
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def load_raw_profiling_data(data_dir: str, pattern: str = "*.csv") -> pd.DataFrame:
    """
    Load raw profiling data from multiple CSV files.
    
    Args:
        data_dir: Directory containing profiling data files
        pattern: File pattern to match (default: "*.csv")
        
    Returns:
        Combined DataFrame with all profiling data
    """
    data_files = glob.glob(os.path.join(data_dir, pattern))
    
    if not data_files:
        raise FileNotFoundError(f"No files found matching pattern {pattern} in {data_dir}")
    
    dataframes = []
    
    for file_path in data_files:
        try:
            df = pd.read_csv(file_path)
            # Add source file information
            df['source_file'] = os.path.basename(file_path)
            dataframes.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
    
    if not dataframes:
        raise ValueError("No valid data files were loaded")
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df


def parse_space_separated_profiling_files(data_dir: str) -> pd.DataFrame:
    """
    Parse space-separated profiling files commonly found in GPU profiling output.
    
    Args:
        data_dir: Directory containing space-separated profiling files
        
    Returns:
        DataFrame with parsed profiling data
    """
    all_data = []
    
    for file_path in glob.glob(os.path.join(data_dir, "*.txt")):
        try:
            # Read space-separated file
            df = pd.read_csv(file_path, sep=r'\s+', engine='python')
            
            # Extract metadata from filename if possible
            filename = os.path.basename(file_path)
            # Assume filename format: gpu_app_frequency_run.txt
            parts = filename.replace('.txt', '').split('_')
            if len(parts) >= 3:
                df['gpu'] = parts[0]
                df['application'] = parts[1]
                df['frequency'] = int(parts[2]) if parts[2].isdigit() else None
            
            df['source_file'] = filename
            all_data.append(df)
            
        except Exception as e:
            print(f"Warning: Failed to parse {file_path}: {e}")
    
    if not all_data:
        raise ValueError("No valid profiling files were parsed")
    
    return pd.concat(all_data, ignore_index=True)


def aggregate_profiling_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate profiling metrics by configuration.
    
    Args:
        df: Raw profiling DataFrame
        
    Returns:
        Aggregated DataFrame with summary statistics
    """
    # Define aggregation functions for different metrics
    agg_functions = {
        'execution_time': ['mean', 'std', 'min', 'max'],
        'power': ['mean', 'std', 'min', 'max'],
        'energy': ['mean', 'std', 'sum'],
        'temperature': ['mean', 'max'],
        'gpu_utilization': ['mean', 'std'],
        'memory_utilization': ['mean', 'std']
    }
    
    # Group by configuration identifiers
    groupby_cols = ['gpu', 'application', 'frequency']
    available_cols = [col for col in groupby_cols if col in df.columns]
    
    if not available_cols:
        raise ValueError("No grouping columns found in data")
    
    # Aggregate available metrics
    result_dfs = []
    
    for metric, funcs in agg_functions.items():
        if metric in df.columns:
            grouped = df.groupby(available_cols)[metric].agg(funcs)
            grouped.columns = [f"{metric}_{func}" for func in funcs]
            result_dfs.append(grouped)
    
    if not result_dfs:
        raise ValueError("No aggregatable metrics found in data")
    
    # Combine all aggregated metrics
    aggregated_df = pd.concat(result_dfs, axis=1)
    aggregated_df = aggregated_df.reset_index()
    
    # Calculate derived metrics
    if 'power_mean' in aggregated_df.columns and 'execution_time_mean' in aggregated_df.columns:
        aggregated_df['energy_calculated'] = (aggregated_df['power_mean'] * 
                                            aggregated_df['execution_time_mean'])
    
    return aggregated_df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names for consistency across the framework.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with standardized column names
    """
    # Define column name mappings
    column_mappings = {
        # Power metrics
        'power_mean': 'avg_power',
        'power_std': 'std_power',
        'power_min': 'min_power',
        'power_max': 'max_power',
        
        # Time metrics
        'execution_time_mean': 'execution_time',
        'execution_time_std': 'std_execution_time',
        
        # Utilization metrics
        'gpu_utilization_mean': 'avg_gputl',
        'gpu_utilization_std': 'std_gputl',
        'memory_utilization_mean': 'avg_mcutl',
        'memory_utilization_std': 'std_mcutl',
        
        # Temperature metrics
        'temperature_mean': 'avg_temp',
        'temperature_max': 'max_temp',
        
        # Frequency metrics
        'frequency': 'frequency',
        'memory_frequency': 'avg_mmclk'
    }
    
    # Apply mappings
    df_renamed = df.rename(columns=column_mappings)
    
    # Ensure required columns exist
    required_columns = ['frequency', 'execution_time', 'avg_power', 'gpu', 'application']
    missing_required = [col for col in required_columns if col not in df_renamed.columns]
    
    if missing_required:
        print(f"Warning: Missing required columns after standardization: {missing_required}")
    
    return df_renamed


def clean_profiling_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean profiling data by removing outliers and invalid values.
    
    Args:
        df: Input profiling DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Remove rows with missing critical values
    critical_columns = ['frequency', 'execution_time', 'avg_power']
    available_critical = [col for col in critical_columns if col in df_clean.columns]
    
    if available_critical:
        df_clean = df_clean.dropna(subset=available_critical)
    
    # Remove obviously invalid values
    if 'frequency' in df_clean.columns:
        df_clean = df_clean[df_clean['frequency'] > 0]
    
    if 'execution_time' in df_clean.columns:
        df_clean = df_clean[df_clean['execution_time'] > 0]
    
    if 'avg_power' in df_clean.columns:
        df_clean = df_clean[df_clean['avg_power'] > 0]
    
    # Remove outliers using IQR method for execution time and power
    for column in ['execution_time', 'avg_power']:
        if column in df_clean.columns:
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[column] >= lower_bound) & 
                               (df_clean[column] <= upper_bound)]
    
    return df_clean


def create_aggregated_dataset(raw_data_dir: str, 
                            output_file: str,
                            file_pattern: str = "*.csv") -> str:
    """
    Create aggregated dataset from raw profiling data.
    
    Args:
        raw_data_dir: Directory containing raw profiling data
        output_file: Path for output aggregated CSV file
        file_pattern: Pattern to match data files
        
    Returns:
        Path to created aggregated file
    """
    print(f"Loading raw data from {raw_data_dir}...")
    
    # Load raw data
    if file_pattern.endswith('.txt'):
        df_raw = parse_space_separated_profiling_files(raw_data_dir)
    else:
        df_raw = load_raw_profiling_data(raw_data_dir, file_pattern)
    
    print(f"Loaded {len(df_raw)} raw data points")
    
    # Aggregate metrics
    print("Aggregating profiling metrics...")
    df_aggregated = aggregate_profiling_metrics(df_raw)
    print(f"Created {len(df_aggregated)} aggregated configurations")
    
    # Standardize column names
    print("Standardizing column names...")
    df_standardized = standardize_column_names(df_aggregated)
    
    # Clean data
    print("Cleaning data...")
    df_clean = clean_profiling_data(df_standardized)
    print(f"Final dataset: {len(df_clean)} clean configurations")
    
    # Save aggregated dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_clean.to_csv(output_file, index=False)
    print(f"Saved aggregated dataset to {output_file}")
    
    return output_file
