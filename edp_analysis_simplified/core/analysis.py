"""
Core Analysis Functions for GPU Frequency Optimization

This module contains the fundamental analysis functions used across
the frequency optimization framework.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


def calculate_efficiency_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate efficiency metrics for frequency optimization analysis.
    
    Args:
        df: DataFrame with columns 'frequency', 'execution_time', 'avg_power', 'gpu', 'application'
        
    Returns:
        DataFrame with efficiency metrics added
    """
    # Group by configuration to find baseline (highest frequency) for each
    baseline_df = df.loc[df.groupby(['gpu', 'application'])['frequency'].idxmax()]
    baseline_dict = {}
    
    for _, baseline in baseline_df.iterrows():
        key = f"{baseline['gpu']}_{baseline['application']}"
        baseline_dict[key] = {
            'power': baseline['avg_power'],
            'execution_time': baseline['execution_time']
        }
    
    # Calculate metrics for each frequency point
    efficiency_data = []
    
    for _, row in df.iterrows():
        config_key = f"{row['gpu']}_{row['application']}"
        config_name = f"{row['gpu']}+{row['application']}"
        
        if config_key in baseline_dict:
            baseline = baseline_dict[config_key]
            
            # Calculate performance penalty (positive = slower)
            perf_penalty = ((row['execution_time'] - baseline['execution_time']) / 
                           baseline['execution_time']) * 100
            
            # Calculate energy savings (positive = less energy)
            energy_current = row['avg_power'] * row['execution_time']
            energy_baseline = baseline['power'] * baseline['execution_time']
            energy_savings = ((energy_baseline - energy_current) / energy_baseline) * 100
            
            # Calculate efficiency ratio (energy savings per % performance loss)
            if abs(perf_penalty) > 0.01:  # Avoid division by very small numbers
                efficiency_ratio = max(0, energy_savings) / max(0.01, abs(perf_penalty))
            else:
                # Performance improved or negligible change
                efficiency_ratio = max(0, energy_savings) * 100  # Very high efficiency
            
            efficiency_data.append({
                'config': config_name,
                'gpu': row['gpu'],
                'application': row['application'],
                'gpu_frequency': row['frequency'],
                'mem_frequency': row.get('avg_mmclk', 1215),
                'power': row['avg_power'],
                'execution_time': row['execution_time'],
                'performance_penalty': perf_penalty,
                'energy_savings': energy_savings,
                'efficiency_ratio': efficiency_ratio
            })
    
    return pd.DataFrame(efficiency_data)


def categorize_performance_impact(performance_penalty: float) -> str:
    """
    Categorize performance impact based on penalty percentage.
    
    Args:
        performance_penalty: Performance penalty as percentage
        
    Returns:
        Category string
    """
    abs_penalty = abs(performance_penalty)
    
    if abs_penalty <= 2:
        return 'Minimal Impact'
    elif abs_penalty <= 5:
        return 'Low Impact'
    elif abs_penalty <= 10:
        return 'Moderate Impact'
    elif abs_penalty <= 15:
        return 'High Impact'
    else:
        return 'Extreme Impact'


def find_optimal_configurations(efficiency_df: pd.DataFrame, 
                               max_degradation: float = 15.0,
                               min_efficiency: float = 2.0) -> pd.DataFrame:
    """
    Find optimal frequency configurations based on efficiency criteria.
    
    Args:
        efficiency_df: DataFrame with efficiency metrics
        max_degradation: Maximum acceptable performance degradation (%)
        min_efficiency: Minimum energy efficiency ratio
        
    Returns:
        DataFrame with optimal configurations
    """
    # Filter by performance criteria
    filtered_df = efficiency_df[
        (efficiency_df['performance_penalty'] <= max_degradation) &
        (efficiency_df['energy_savings'] > 0) &
        (efficiency_df['efficiency_ratio'] >= min_efficiency)
    ].copy()
    
    # Group by config and find best frequency for each
    optimal_configs = []
    
    for config in filtered_df['config'].unique():
        config_data = filtered_df[filtered_df['config'] == config]
        
        # Sort by efficiency ratio (descending) and performance penalty (ascending)
        best_config = config_data.sort_values(
            ['efficiency_ratio', 'performance_penalty'], 
            ascending=[False, True]
        ).iloc[0]
        
        # Categorize performance impact
        category = categorize_performance_impact(best_config['performance_penalty'])
        
        optimal_configs.append({
            'config': best_config['config'],
            'gpu': best_config['gpu'],
            'application': best_config['application'],
            'optimal_frequency': best_config['gpu_frequency'],
            'performance_penalty': best_config['performance_penalty'],
            'energy_savings': best_config['energy_savings'],
            'efficiency_ratio': best_config['efficiency_ratio'],
            'category': category
        })
    
    return pd.DataFrame(optimal_configs)


def validate_data_format(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that the input DataFrame has required columns and format.
    
    Args:
        df: Input DataFrame to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    required_columns = ['frequency', 'execution_time', 'avg_power', 'gpu', 'application']
    errors = []
    
    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    
    # Check data types and values
    if 'frequency' in df.columns:
        if not df['frequency'].dtype in ['int64', 'float64']:
            errors.append("'frequency' column must be numeric")
        if (df['frequency'] <= 0).any():
            errors.append("'frequency' must be positive")
    
    if 'execution_time' in df.columns:
        if not df['execution_time'].dtype in ['int64', 'float64']:
            errors.append("'execution_time' column must be numeric")
        if (df['execution_time'] <= 0).any():
            errors.append("'execution_time' must be positive")
    
    if 'avg_power' in df.columns:
        if not df['avg_power'].dtype in ['int64', 'float64']:
            errors.append("'avg_power' column must be numeric")
        if (df['avg_power'] <= 0).any():
            errors.append("'avg_power' must be positive")
    
    return len(errors) == 0, errors


def load_and_validate_data(file_path: str, 
                          gpu_filter: Optional[str] = None,
                          app_filter: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load data from CSV file and apply validation and filtering.
    
    Args:
        file_path: Path to CSV file
        gpu_filter: Optional GPU type filter ('A100', 'V100')
        app_filter: Optional application filter
        
    Returns:
        Tuple of (processed_dataframe, info_messages)
    """
    info_messages = []
    
    try:
        df = pd.read_csv(file_path)
        info_messages.append(f"Loaded {len(df)} raw configurations from {file_path}")
        
        # Validate data format
        is_valid, errors = validate_data_format(df)
        if not is_valid:
            raise ValueError(f"Data validation failed: {errors}")
        
        # Apply filters
        original_count = len(df)
        
        if gpu_filter:
            df = df[df['gpu'].str.upper() == gpu_filter.upper()]
            info_messages.append(f"Filtered to {gpu_filter}: {len(df)} configurations")
        
        if app_filter:
            df = df[df['application'].str.upper() == app_filter.upper()]
            info_messages.append(f"Filtered to {app_filter}: {len(df)} configurations")
        
        if len(df) == 0:
            raise ValueError("No data remaining after filtering")
        
        return df, info_messages
        
    except Exception as e:
        raise Exception(f"Failed to load data: {e}")


def calculate_summary_statistics(optimal_df: pd.DataFrame) -> Dict:
    """
    Calculate summary statistics for optimal configurations.
    
    Args:
        optimal_df: DataFrame with optimal configurations
        
    Returns:
        Dictionary with summary statistics
    """
    if len(optimal_df) == 0:
        return {
            'total_configurations': 0,
            'average_energy_savings': 0.0,
            'average_performance_impact': 0.0,
            'best_efficiency_ratio': 0.0,
            'configurations_by_category': {}
        }
    
    return {
        'total_configurations': len(optimal_df),
        'average_energy_savings': float(optimal_df['energy_savings'].mean()),
        'average_performance_impact': float(optimal_df['performance_penalty'].abs().mean()),
        'best_efficiency_ratio': float(optimal_df['efficiency_ratio'].max()),
        'configurations_by_category': optimal_df['category'].value_counts().to_dict()
    }
