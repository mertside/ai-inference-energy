"""
Advanced optimization algorithms for frequency selection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable


def pareto_frontier_optimization(efficiency_df: pd.DataFrame) -> pd.DataFrame:
    """
    Find Pareto-optimal configurations (non-dominated solutions).
    
    Args:
        efficiency_df: DataFrame with efficiency metrics
        
    Returns:
        DataFrame with Pareto-optimal configurations
    """
    pareto_configs = []
    
    for config in efficiency_df['config'].unique():
        config_data = efficiency_df[efficiency_df['config'] == config].copy()
        
        # For each configuration, find Pareto frontier
        # We want to minimize performance penalty and maximize energy savings
        pareto_points = []
        
        for i, row1 in config_data.iterrows():
            is_dominated = False
            
            for j, row2 in config_data.iterrows():
                if i != j:
                    # Check if row1 is dominated by row2
                    # row2 dominates row1 if:
                    # - row2 has better or equal performance penalty (lower is better)
                    # - row2 has better or equal energy savings (higher is better)
                    # - row2 is better in at least one dimension
                    
                    perf1, energy1 = row1['performance_penalty'], row1['energy_savings']
                    perf2, energy2 = row2['performance_penalty'], row2['energy_savings']
                    
                    if (perf2 <= perf1 and energy2 >= energy1 and 
                        (perf2 < perf1 or energy2 > energy1)):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_points.append(i)
        
        # Add Pareto-optimal points for this configuration
        pareto_configs.extend(pareto_points)
    
    return efficiency_df.loc[pareto_configs].reset_index(drop=True)


def multi_objective_optimization(efficiency_df: pd.DataFrame,
                               weights: Dict[str, float] = None) -> pd.DataFrame:
    """
    Multi-objective optimization using weighted scoring.
    
    Args:
        efficiency_df: DataFrame with efficiency metrics
        weights: Dictionary with weights for different objectives
        
    Returns:
        DataFrame with optimized configurations
    """
    if weights is None:
        weights = {
            'energy_savings': 0.6,      # 60% weight on energy savings
            'performance_penalty': -0.3, # 30% penalty for performance loss (negative weight)
            'efficiency_ratio': 0.1      # 10% weight on efficiency ratio
        }
    
    df_scored = efficiency_df.copy()
    
    # Normalize metrics to 0-1 scale for fair comparison
    for metric in ['energy_savings', 'performance_penalty', 'efficiency_ratio']:
        if metric in df_scored.columns:
            min_val = df_scored[metric].min()
            max_val = df_scored[metric].max()
            if max_val > min_val:
                df_scored[f'{metric}_normalized'] = ((df_scored[metric] - min_val) / 
                                                   (max_val - min_val))
            else:
                df_scored[f'{metric}_normalized'] = 0.5  # Middle value if no variation
    
    # Calculate weighted score
    df_scored['optimization_score'] = 0
    
    for metric, weight in weights.items():
        normalized_col = f'{metric}_normalized'
        if normalized_col in df_scored.columns:
            df_scored['optimization_score'] += weight * df_scored[normalized_col]
    
    # Find best configuration for each GPU+App combination
    optimal_configs = []
    
    for config in df_scored['config'].unique():
        config_data = df_scored[df_scored['config'] == config]
        best_config = config_data.loc[config_data['optimization_score'].idxmax()]
        optimal_configs.append(best_config)
    
    return pd.DataFrame(optimal_configs).reset_index(drop=True)


def constraint_based_optimization(efficiency_df: pd.DataFrame,
                                constraints: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """
    Optimization with hard constraints on performance and efficiency.
    
    Args:
        efficiency_df: DataFrame with efficiency metrics
        constraints: Dictionary with (min, max) constraints for metrics
        
    Returns:
        DataFrame with constrained optimal configurations
    """
    df_filtered = efficiency_df.copy()
    
    # Apply constraints
    for metric, (min_val, max_val) in constraints.items():
        if metric in df_filtered.columns:
            df_filtered = df_filtered[
                (df_filtered[metric] >= min_val) & 
                (df_filtered[metric] <= max_val)
            ]
    
    if len(df_filtered) == 0:
        return pd.DataFrame()  # No configurations meet constraints
    
    # Among feasible solutions, optimize for best efficiency ratio
    optimal_configs = []
    
    for config in df_filtered['config'].unique():
        config_data = df_filtered[df_filtered['config'] == config]
        if len(config_data) > 0:
            # Sort by efficiency ratio (descending) and performance penalty (ascending)
            best_config = config_data.sort_values(
                ['efficiency_ratio', 'performance_penalty'],
                ascending=[False, True]
            ).iloc[0]
            optimal_configs.append(best_config)
    
    return pd.DataFrame(optimal_configs).reset_index(drop=True)


def adaptive_optimization(efficiency_df: pd.DataFrame,
                        target_energy_savings: float = 20.0,
                        max_performance_degradation: float = 15.0) -> pd.DataFrame:
    """
    Adaptive optimization that adjusts constraints based on available data.
    
    Args:
        efficiency_df: DataFrame with efficiency metrics
        target_energy_savings: Target energy savings percentage
        max_performance_degradation: Maximum allowed performance degradation
        
    Returns:
        DataFrame with adaptively optimized configurations
    """
    optimal_configs = []
    
    for config in efficiency_df['config'].unique():
        config_data = efficiency_df[efficiency_df['config'] == config]
        
        # First, try to meet target energy savings with performance constraint
        candidates = config_data[
            (config_data['energy_savings'] >= target_energy_savings) &
            (config_data['performance_penalty'] <= max_performance_degradation)
        ]
        
        if len(candidates) > 0:
            # Found configurations meeting both criteria
            best = candidates.sort_values('performance_penalty').iloc[0]
        else:
            # Relax constraints adaptively
            # Try relaxing energy savings target first
            for energy_threshold in [target_energy_savings * 0.8, 
                                   target_energy_savings * 0.6,
                                   target_energy_savings * 0.4]:
                candidates = config_data[
                    (config_data['energy_savings'] >= energy_threshold) &
                    (config_data['performance_penalty'] <= max_performance_degradation)
                ]
                if len(candidates) > 0:
                    best = candidates.sort_values('performance_penalty').iloc[0]
                    break
            else:
                # If still no solution, relax performance constraint
                candidates = config_data[config_data['energy_savings'] > 0]
                if len(candidates) > 0:
                    best = candidates.sort_values('efficiency_ratio', ascending=False).iloc[0]
                else:
                    continue  # Skip this configuration
        
        optimal_configs.append(best)
    
    return pd.DataFrame(optimal_configs).reset_index(drop=True)


def custom_optimization(efficiency_df: pd.DataFrame,
                       objective_function: Callable[[pd.Series], float]) -> pd.DataFrame:
    """
    Custom optimization using user-defined objective function.
    
    Args:
        efficiency_df: DataFrame with efficiency metrics
        objective_function: Function that takes a row and returns a score
        
    Returns:
        DataFrame with custom optimized configurations
    """
    df_scored = efficiency_df.copy()
    df_scored['custom_score'] = df_scored.apply(objective_function, axis=1)
    
    optimal_configs = []
    
    for config in df_scored['config'].unique():
        config_data = df_scored[df_scored['config'] == config]
        best_config = config_data.loc[config_data['custom_score'].idxmax()]
        optimal_configs.append(best_config)
    
    return pd.DataFrame(optimal_configs).reset_index(drop=True)
