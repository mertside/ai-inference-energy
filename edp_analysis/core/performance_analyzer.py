"""
Performance Analysis Module

This module provides comprehensive performance analysis and constraint management
for GPU frequency optimization, consolidating functionality from the original
performance_profiler.py module.

Author: Mert Side
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance analysis results."""
    execution_time: float
    throughput: float
    latency: float
    performance_relative_to_baseline: float
    performance_penalty_percent: float
    frequency_scaling_factor: float


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for GPU applications.
    """
    
    def __init__(self):
        """Initialize performance analyzer."""
        self.baseline_metrics = {}
        self.performance_models = {}
        
    def set_baseline_performance(self, 
                               frequency: float,
                               execution_time: float,
                               throughput: Optional[float] = None) -> None:
        """
        Set baseline performance metrics for comparison.
        
        Args:
            frequency: Baseline frequency in MHz
            execution_time: Baseline execution time in seconds
            throughput: Optional baseline throughput metric
        """
        if throughput is None:
            throughput = 1.0 / execution_time  # Inverse of execution time
        
        self.baseline_metrics = {
            'frequency': frequency,
            'execution_time': execution_time,
            'throughput': throughput,
            'latency': execution_time  # For single inference tasks
        }
        
        logger.info(f"Set baseline performance: {frequency} MHz, {execution_time:.3f}s")
    
    def calculate_performance_metrics(self, 
                                    frequency: float,
                                    execution_time: float,
                                    throughput: Optional[float] = None) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            frequency: Current frequency in MHz
            execution_time: Current execution time in seconds
            throughput: Optional throughput metric
            
        Returns:
            PerformanceMetrics object with calculated values
        """
        if not self.baseline_metrics:
            raise ValueError("Baseline metrics not set. Call set_baseline_performance() first.")
        
        if throughput is None:
            throughput = 1.0 / execution_time
        
        # Calculate relative performance
        baseline_throughput = self.baseline_metrics['throughput']
        performance_relative = throughput / baseline_throughput
        
        # Calculate performance penalty (positive = slower, negative = faster)
        baseline_time = self.baseline_metrics['execution_time']
        performance_penalty = (execution_time - baseline_time) / baseline_time * 100
        
        # Calculate frequency scaling factor
        baseline_freq = self.baseline_metrics['frequency']
        freq_scaling = frequency / baseline_freq
        
        return PerformanceMetrics(
            execution_time=execution_time,
            throughput=throughput,
            latency=execution_time,
            performance_relative_to_baseline=performance_relative,
            performance_penalty_percent=performance_penalty,
            frequency_scaling_factor=freq_scaling
        )
    
    def validate_performance_constraint(self, 
                                      execution_time: float,
                                      max_penalty_percent: float) -> bool:
        """
        Validate that performance meets the specified constraint.
        
        Args:
            execution_time: Current execution time in seconds
            max_penalty_percent: Maximum allowable performance penalty
            
        Returns:
            True if constraint is satisfied
        """
        if not self.baseline_metrics:
            raise ValueError("Baseline metrics not set. Call set_baseline_performance() first.")
        
        baseline_time = self.baseline_metrics['execution_time']
        penalty = (execution_time - baseline_time) / baseline_time * 100
        
        return penalty <= max_penalty_percent
    
    def get_max_allowable_execution_time(self, max_penalty_percent: float) -> float:
        """
        Get maximum allowable execution time for given performance constraint.
        
        Args:
            max_penalty_percent: Maximum allowable performance penalty
            
        Returns:
            Maximum allowable execution time in seconds
        """
        if not self.baseline_metrics:
            raise ValueError("Baseline metrics not set. Call set_baseline_performance() first.")
        
        baseline_time = self.baseline_metrics['execution_time']
        return baseline_time * (1 + max_penalty_percent / 100)
    
    def analyze_frequency_scaling(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze performance scaling with frequency.
        
        Args:
            data: DataFrame with frequency and execution_time columns
            
        Returns:
            Dictionary with scaling analysis results
        """
        # Sort by frequency
        data_sorted = data.sort_values('frequency')
        
        # Calculate scaling metrics
        freq_range = data_sorted['frequency'].max() - data_sorted['frequency'].min()
        time_range = data_sorted['execution_time'].max() - data_sorted['execution_time'].min()
        
        # Estimate linear scaling coefficient
        freq_normalized = (data_sorted['frequency'] - data_sorted['frequency'].min()) / freq_range
        time_normalized = (data_sorted['execution_time'] - data_sorted['execution_time'].min()) / time_range
        
        # Correlation between frequency and execution time
        correlation = np.corrcoef(freq_normalized, time_normalized)[0, 1]
        
        # Fit linear model for prediction
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        X = data_sorted['frequency'].values.reshape(-1, 1)
        y = data_sorted['execution_time'].values
        model.fit(X, y)
        
        # Store model for future predictions
        config_key = "default"  # Could be extended to include gpu+app
        self.performance_models[config_key] = model
        
        return {
            'frequency_range_mhz': freq_range,
            'execution_time_range_s': time_range,
            'frequency_time_correlation': correlation,
            'linear_model_score': model.score(X, y),
            'linear_model_coefficients': {
                'slope': model.coef_[0],
                'intercept': model.intercept_
            },
            'performance_scaling_summary': {
                'best_frequency_mhz': data_sorted.loc[data_sorted['execution_time'].idxmin(), 'frequency'],
                'worst_frequency_mhz': data_sorted.loc[data_sorted['execution_time'].idxmax(), 'frequency'],
                'performance_span_percent': (time_range / data_sorted['execution_time'].min()) * 100
            }
        }
    
    def predict_execution_time(self, frequency: float, config_key: str = "default") -> float:
        """
        Predict execution time for a given frequency using fitted model.
        
        Args:
            frequency: Frequency in MHz
            config_key: Configuration key for model selection
            
        Returns:
            Predicted execution time in seconds
        """
        if config_key not in self.performance_models:
            raise ValueError(f"No performance model found for {config_key}")
        
        model = self.performance_models[config_key]
        prediction = model.predict([[frequency]])[0]
        
        return max(prediction, 0.001)  # Ensure positive prediction
    
    def find_optimal_performance_frequency(self, 
                                         data: pd.DataFrame,
                                         optimization_goal: str = "minimize_time") -> Dict[str, Any]:
        """
        Find optimal frequency for pure performance optimization.
        
        Args:
            data: DataFrame with frequency and execution_time columns
            optimization_goal: "minimize_time" or "maximize_throughput"
            
        Returns:
            Dictionary with optimal performance configuration
        """
        if optimization_goal == "minimize_time":
            optimal_idx = data['execution_time'].idxmin()
        elif optimization_goal == "maximize_throughput":
            # Calculate throughput as inverse of execution time
            throughput = 1.0 / data['execution_time']
            optimal_idx = throughput.idxmax()
        else:
            raise ValueError(f"Unknown optimization goal: {optimization_goal}")
        
        optimal_row = data.loc[optimal_idx]
        
        # Calculate metrics if baseline is set
        if self.baseline_metrics:
            metrics = self.calculate_performance_metrics(
                optimal_row['frequency'],
                optimal_row['execution_time']
            )
        else:
            metrics = PerformanceMetrics(
                execution_time=optimal_row['execution_time'],
                throughput=1.0 / optimal_row['execution_time'],
                latency=optimal_row['execution_time'],
                performance_relative_to_baseline=1.0,
                performance_penalty_percent=0.0,
                frequency_scaling_factor=1.0
            )
        
        return {
            'optimal_frequency': optimal_row['frequency'],
            'metrics': metrics,
            'optimization_goal': optimization_goal,
            'data_point': optimal_row.to_dict()
        }
    
    def filter_by_performance_constraint(self, 
                                       data: pd.DataFrame,
                                       max_penalty_percent: float,
                                       baseline_frequency: Optional[float] = None) -> pd.DataFrame:
        """
        Filter data to only include configurations meeting performance constraint.
        
        Args:
            data: DataFrame with frequency and execution_time columns
            max_penalty_percent: Maximum allowable performance penalty
            baseline_frequency: Baseline frequency (uses max if not provided)
            
        Returns:
            Filtered DataFrame
        """
        # Set baseline if not already set
        if not self.baseline_metrics:
            if baseline_frequency is None:
                baseline_frequency = data['frequency'].max()
            
            baseline_row = data[data['frequency'] == baseline_frequency]
            if len(baseline_row) == 0:
                # Use closest frequency if exact match not found
                baseline_frequency = data['frequency'].iloc[np.argmin(np.abs(data['frequency'] - baseline_frequency))]
                baseline_row = data[data['frequency'] == baseline_frequency]
            
            self.set_baseline_performance(
                baseline_frequency,
                baseline_row.iloc[0]['execution_time']
            )
        
        # Calculate maximum allowable execution time
        max_time = self.get_max_allowable_execution_time(max_penalty_percent)
        
        # Filter data
        filtered_data = data[data['execution_time'] <= max_time]
        
        logger.info(f"Performance constraint filter: {len(filtered_data)}/{len(data)} configurations remain")
        
        return filtered_data
    
    def analyze_performance_vs_frequency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive analysis of performance vs frequency relationship.
        
        Args:
            data: DataFrame with frequency and execution_time columns
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        analysis = {}
        
        # Basic statistics
        analysis['frequency_stats'] = data['frequency'].describe().to_dict()
        analysis['execution_time_stats'] = data['execution_time'].describe().to_dict()
        
        # Frequency scaling analysis
        analysis['scaling_analysis'] = self.analyze_frequency_scaling(data)
        
        # Performance optimization
        analysis['performance_optimization'] = self.find_optimal_performance_frequency(data)
        
        # Constraint analysis for different penalty levels
        penalty_levels = [5, 10, 15, 20, 30, 50]
        analysis['constraint_analysis'] = {}
        
        for penalty in penalty_levels:
            try:
                filtered_data = self.filter_by_performance_constraint(data, penalty)
                if len(filtered_data) > 0:
                    analysis['constraint_analysis'][f"{penalty}%"] = {
                        'configurations_remaining': len(filtered_data),
                        'best_frequency': filtered_data.loc[filtered_data['execution_time'].idxmin(), 'frequency'],
                        'frequency_range': {
                            'min': filtered_data['frequency'].min(),
                            'max': filtered_data['frequency'].max()
                        }
                    }
                else:
                    analysis['constraint_analysis'][f"{penalty}%"] = {
                        'configurations_remaining': 0,
                        'note': 'No configurations meet this constraint'
                    }
            except Exception as e:
                analysis['constraint_analysis'][f"{penalty}%"] = {'error': str(e)}
        
        return analysis


class PerformanceConstraintManager:
    """
    Manager for application-specific performance constraints.
    """
    
    def __init__(self, constraints_config: Optional[Dict] = None):
        """Initialize with constraints configuration."""
        if constraints_config is None:
            constraints_config = self._get_default_constraints()
        
        self.constraints = constraints_config
    
    def _get_default_constraints(self) -> Dict[str, float]:
        """Get default performance constraints."""
        return {
            'llama': 0.05,           # 5% max penalty for interactive LLM
            'stable_diffusion': 0.20, # 20% max penalty for image generation
            'vit': 0.20,             # 20% max penalty for vision tasks
            'whisper': 0.15,         # 15% max penalty for speech recognition
            'lstm': 0.10,            # 10% max penalty for lightweight models
            'default': 0.15          # Default constraint
        }
    
    def get_constraint(self, application: str) -> float:
        """Get performance constraint for an application."""
        app_lower = application.lower().replace('+', '')
        
        # Try exact match
        if app_lower in self.constraints:
            return self.constraints[app_lower]
        
        # Try partial matches
        for app_key in self.constraints:
            if app_key in app_lower or app_lower in app_key:
                return self.constraints[app_key]
        
        # Return default
        return self.constraints.get('default', 0.15)
    
    def validate_constraint(self, application: str, penalty_percent: float) -> bool:
        """Validate that penalty meets application constraint."""
        constraint = self.get_constraint(application)
        return abs(penalty_percent) <= constraint * 100
    
    def get_all_constraints(self) -> Dict[str, float]:
        """Get all configured constraints."""
        return self.constraints.copy()


def quick_performance_analysis(data: pd.DataFrame, 
                             max_penalty_percent: float = 20.0) -> Dict:
    """
    Quick performance analysis for a dataset.
    
    Args:
        data: DataFrame with frequency and execution_time columns
        max_penalty_percent: Maximum allowable performance penalty
        
    Returns:
        Dictionary with performance analysis results
    """
    analyzer = PerformanceAnalyzer()
    
    try:
        analysis = analyzer.analyze_performance_vs_frequency(data)
        
        # Add constraint filtering
        filtered_data = analyzer.filter_by_performance_constraint(data, max_penalty_percent)
        analysis['constraint_filtered_data'] = {
            'original_configurations': len(data),
            'configurations_meeting_constraint': len(filtered_data),
            'constraint_penalty_percent': max_penalty_percent
        }
        
        return analysis
    except Exception as e:
        logger.error(f"Quick performance analysis failed: {e}")
        return {'error': str(e)}


def validate_performance_data(data: pd.DataFrame) -> Dict[str, bool]:
    """
    Validate performance data quality.
    
    Args:
        data: DataFrame to validate
        
    Returns:
        Dictionary of validation results
    """
    validation_results = {
        'has_required_columns': all(col in data.columns for col in ['frequency', 'execution_time']),
        'positive_values': (data[['frequency', 'execution_time']] > 0).all().all(),
        'reasonable_ranges': (
            data['frequency'].between(100, 3000).all() and
            data['execution_time'].between(0.001, 10000).all()
        ),
        'sufficient_data_points': len(data) >= 3,
        'frequency_variation': data['frequency'].nunique() >= 2
    }
    
    validation_results['all_validations_passed'] = all(validation_results.values())
    
    return validation_results
