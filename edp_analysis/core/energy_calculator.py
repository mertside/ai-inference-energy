"""
Energy Calculation and EDP Analysis Module

This module provides comprehensive energy and Energy-Delay Product (EDP) calculations
for GPU frequency optimization, consolidating functionality from the original
edp_calculator.py and energy_profiler.py modules.

Author: Mert Side
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EDPResult:
    """Container for EDP optimization results."""
    frequency: float
    execution_time: float
    power: float
    energy: float
    edp: float
    ed2p: float
    energy_savings_percent: float
    performance_penalty_percent: float
    edp_improvement_percent: float


class EnergyCalculator:
    """
    Core energy calculation and EDP analysis engine.
    """
    
    def __init__(self):
        """Initialize energy calculator."""
        self.baseline_metrics = {}
        
    def calculate_energy(self, 
                        power: Union[float, pd.Series, np.ndarray],
                        execution_time: Union[float, pd.Series, np.ndarray]) -> Union[float, pd.Series, np.ndarray]:
        """
        Calculate energy consumption from power and execution time.
        
        Args:
            power: Power consumption in watts
            execution_time: Execution time in seconds
            
        Returns:
            Energy consumption in joules (watt-seconds)
        """
        return power * execution_time
    
    def calculate_edp(self, 
                     energy: Union[float, pd.Series, np.ndarray],
                     execution_time: Union[float, pd.Series, np.ndarray]) -> Union[float, pd.Series, np.ndarray]:
        """
        Calculate Energy-Delay Product (EDP).
        
        Args:
            energy: Energy consumption in joules
            execution_time: Execution time in seconds
            
        Returns:
            EDP value (energy × delay)
        """
        return energy * execution_time
    
    def calculate_ed2p(self, 
                      energy: Union[float, pd.Series, np.ndarray],
                      execution_time: Union[float, pd.Series, np.ndarray]) -> Union[float, pd.Series, np.ndarray]:
        """
        Calculate Energy-Delay² Product (ED²P) for performance-prioritized optimization.
        
        Args:
            energy: Energy consumption in joules
            execution_time: Execution time in seconds
            
        Returns:
            ED²P value (energy × delay²)
        """
        return energy * (execution_time ** 2)
    
    def add_derived_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived energy metrics to a DataFrame.
        
        Args:
            data: DataFrame with power and execution_time columns
            
        Returns:
            DataFrame with additional energy, EDP, and ED²P columns
        """
        data = data.copy()
        
        # Calculate energy if not present
        if 'energy' not in data.columns:
            data['energy'] = self.calculate_energy(data['power'], data['execution_time'])
        
        # Calculate EDP if not present
        if 'edp' not in data.columns:
            data['edp'] = self.calculate_edp(data['energy'], data['execution_time'])
        
        # Calculate ED²P if not present
        if 'ed2p' not in data.columns:
            data['ed2p'] = self.calculate_ed2p(data['energy'], data['execution_time'])
        
        return data
    
    def set_baseline(self, 
                    frequency: float,
                    execution_time: float,
                    power: float,
                    energy: Optional[float] = None) -> None:
        """
        Set baseline metrics for comparison calculations.
        
        Args:
            frequency: Baseline frequency in MHz
            execution_time: Baseline execution time in seconds
            power: Baseline power consumption in watts
            energy: Baseline energy consumption in joules (calculated if not provided)
        """
        if energy is None:
            energy = self.calculate_energy(power, execution_time)
        
        self.baseline_metrics = {
            'frequency': frequency,
            'execution_time': execution_time,
            'power': power,
            'energy': energy,
            'edp': self.calculate_edp(energy, execution_time),
            'ed2p': self.calculate_ed2p(energy, execution_time)
        }
        
        logger.info(f"Set baseline: {frequency} MHz, {execution_time:.2f}s, {power:.1f}W, {energy:.1f}J")
    
    def calculate_improvements(self, 
                             frequency: float,
                             execution_time: float,
                             power: float,
                             energy: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate improvements relative to baseline metrics.
        
        Args:
            frequency: Current frequency in MHz
            execution_time: Current execution time in seconds
            power: Current power consumption in watts
            energy: Current energy consumption in joules (calculated if not provided)
            
        Returns:
            Dictionary of improvement percentages
        """
        if not self.baseline_metrics:
            raise ValueError("Baseline metrics not set. Call set_baseline() first.")
        
        if energy is None:
            energy = self.calculate_energy(power, execution_time)
        
        edp = self.calculate_edp(energy, execution_time)
        ed2p = self.calculate_ed2p(energy, execution_time)
        
        improvements = {
            'frequency_reduction_percent': (self.baseline_metrics['frequency'] - frequency) / self.baseline_metrics['frequency'] * 100,
            'energy_savings_percent': (self.baseline_metrics['energy'] - energy) / self.baseline_metrics['energy'] * 100,
            'power_savings_percent': (self.baseline_metrics['power'] - power) / self.baseline_metrics['power'] * 100,
            'performance_penalty_percent': (execution_time - self.baseline_metrics['execution_time']) / self.baseline_metrics['execution_time'] * 100,
            'edp_improvement_percent': (self.baseline_metrics['edp'] - edp) / self.baseline_metrics['edp'] * 100,
            'ed2p_improvement_percent': (self.baseline_metrics['ed2p'] - ed2p) / self.baseline_metrics['ed2p'] * 100
        }
        
        return improvements
    
    def find_optimal_frequency(self, 
                             data: pd.DataFrame,
                             method: str = "edp",
                             performance_constraint: Optional[float] = None) -> EDPResult:
        """
        Find optimal frequency based on specified optimization method.
        
        Args:
            data: DataFrame with frequency, execution_time, power columns
            method: Optimization method ("edp", "ed2p", "energy", "performance")
            performance_constraint: Maximum allowable performance penalty (0.0-1.0)
            
        Returns:
            EDPResult with optimal configuration
        """
        # Add derived metrics
        data = self.add_derived_metrics(data)
        
        # Apply performance constraint if specified
        if performance_constraint is not None:
            if not self.baseline_metrics:
                # Use maximum frequency as baseline
                max_freq_row = data.loc[data['frequency'].idxmax()]
                self.set_baseline(
                    max_freq_row['frequency'],
                    max_freq_row['execution_time'],
                    max_freq_row['power']
                )
            
            # Filter data based on performance constraint
            baseline_time = self.baseline_metrics['execution_time']
            max_time = baseline_time * (1 + performance_constraint)
            data = data[data['execution_time'] <= max_time]
            
            if len(data) == 0:
                raise ValueError(f"No configurations meet performance constraint of {performance_constraint*100:.1f}%")
        
        # Find optimal configuration based on method
        if method == "edp":
            optimal_idx = data['edp'].idxmin()
        elif method == "ed2p":
            optimal_idx = data['ed2p'].idxmin()
        elif method == "energy":
            optimal_idx = data['energy'].idxmin()
        elif method == "performance":
            optimal_idx = data['execution_time'].idxmin()
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        optimal_row = data.loc[optimal_idx]
        
        # Calculate improvements if baseline is set
        if self.baseline_metrics:
            improvements = self.calculate_improvements(
                optimal_row['frequency'],
                optimal_row['execution_time'],
                optimal_row['power'],
                optimal_row['energy']
            )
        else:
            improvements = {key: 0.0 for key in ['energy_savings_percent', 'performance_penalty_percent', 'edp_improvement_percent']}
        
        return EDPResult(
            frequency=optimal_row['frequency'],
            execution_time=optimal_row['execution_time'],
            power=optimal_row['power'],
            energy=optimal_row['energy'],
            edp=optimal_row['edp'],
            ed2p=optimal_row['ed2p'],
            energy_savings_percent=improvements['energy_savings_percent'],
            performance_penalty_percent=improvements['performance_penalty_percent'],
            edp_improvement_percent=improvements['edp_improvement_percent']
        )
    
    def analyze_configuration(self, 
                            data: pd.DataFrame,
                            gpu: str,
                            application: str,
                            methods: List[str] = ["edp", "ed2p"],
                            performance_constraint: Optional[float] = None) -> Dict[str, EDPResult]:
        """
        Analyze a GPU-application configuration with multiple optimization methods.
        
        Args:
            data: Complete dataset
            gpu: GPU type
            application: Application name
            methods: List of optimization methods to apply
            performance_constraint: Maximum allowable performance penalty
            
        Returns:
            Dictionary mapping method names to EDPResult objects
        """
        # Filter data for specific configuration
        config_data = data[
            (data['gpu'] == gpu) & 
            (data['application'] == application)
        ].copy()
        
        if len(config_data) == 0:
            raise ValueError(f"No data found for {gpu}+{application}")
        
        # Set baseline (maximum frequency)
        max_freq_row = config_data.loc[config_data['frequency'].idxmax()]
        self.set_baseline(
            max_freq_row['frequency'],
            max_freq_row['execution_time'],
            max_freq_row['power']
        )
        
        # Analyze with each method
        results = {}
        for method in methods:
            try:
                results[method] = self.find_optimal_frequency(
                    config_data, 
                    method=method,
                    performance_constraint=performance_constraint
                )
                logger.info(f"{gpu}+{application} - {method}: {results[method].frequency} MHz")
            except Exception as e:
                logger.error(f"Failed to optimize {gpu}+{application} with {method}: {e}")
                
        return results
    
    def validate_energy_calculation(self, 
                                  data: pd.DataFrame,
                                  tolerance: float = 0.01) -> bool:
        """
        Validate energy calculations by checking conservation principles.
        
        Args:
            data: DataFrame with power, execution_time, and energy columns
            tolerance: Relative tolerance for validation
            
        Returns:
            True if validation passes
        """
        if 'energy' not in data.columns:
            logger.warning("No energy column found for validation")
            return False
        
        # Calculate energy from power and time
        calculated_energy = self.calculate_energy(data['power'], data['execution_time'])
        
        # Check relative difference
        relative_diff = np.abs(data['energy'] - calculated_energy) / calculated_energy
        max_diff = relative_diff.max()
        
        if max_diff > tolerance:
            logger.warning(f"Energy validation failed: max relative difference {max_diff:.4f} > {tolerance}")
            return False
        
        logger.info(f"Energy validation passed: max relative difference {max_diff:.4f}")
        return True


class FGCSEDPOptimizer:
    """
    FGCS 2023 paper-inspired EDP optimizer with polynomial modeling.
    """
    
    def __init__(self, polynomial_degree: int = 2):
        """Initialize FGCS optimizer."""
        self.degree = polynomial_degree
        self.models = {}
        
    def fit_polynomial_model(self, 
                           frequencies: np.ndarray,
                           values: np.ndarray,
                           metric_name: str) -> np.ndarray:
        """
        Fit polynomial model to frequency-metric relationship.
        
        Args:
            frequencies: Frequency values
            values: Metric values (power, execution time, etc.)
            metric_name: Name of the metric for logging
            
        Returns:
            Polynomial coefficients
        """
        # Normalize frequencies for better numerical stability
        freq_norm = (frequencies - frequencies.min()) / (frequencies.max() - frequencies.min())
        
        # Fit polynomial
        coeffs = np.polyfit(freq_norm, values, self.degree)
        self.models[metric_name] = {
            'coefficients': coeffs,
            'freq_min': frequencies.min(),
            'freq_max': frequencies.max()
        }
        
        logger.info(f"Fitted {metric_name} polynomial model (degree {self.degree})")
        return coeffs
    
    def predict_metric(self, 
                      frequencies: np.ndarray,
                      metric_name: str) -> np.ndarray:
        """
        Predict metric values using fitted polynomial model.
        
        Args:
            frequencies: Frequency values for prediction
            metric_name: Name of the metric model to use
            
        Returns:
            Predicted metric values
        """
        if metric_name not in self.models:
            raise ValueError(f"No model fitted for {metric_name}")
        
        model = self.models[metric_name]
        
        # Normalize frequencies
        freq_norm = (frequencies - model['freq_min']) / (model['freq_max'] - model['freq_min'])
        
        # Predict using polynomial
        predictions = np.polyval(model['coefficients'], freq_norm)
        return predictions
    
    def optimize_edp_fgcs(self, 
                         data: pd.DataFrame,
                         frequency_range: Optional[Tuple[float, float]] = None) -> EDPResult:
        """
        Optimize EDP using FGCS methodology with polynomial modeling.
        
        Args:
            data: Configuration data
            frequency_range: Optional frequency range for optimization
            
        Returns:
            EDPResult with optimal configuration
        """
        # Fit polynomial models for power and execution time
        frequencies = data['frequency'].values
        
        self.fit_polynomial_model(frequencies, data['power'].values, 'power')
        self.fit_polynomial_model(frequencies, data['execution_time'].values, 'execution_time')
        
        # Define frequency range for optimization
        if frequency_range is None:
            freq_min, freq_max = frequencies.min(), frequencies.max()
        else:
            freq_min, freq_max = frequency_range
        
        # Generate fine-grained frequency grid for optimization
        freq_grid = np.linspace(freq_min, freq_max, 1000)
        
        # Predict metrics
        power_pred = self.predict_metric(freq_grid, 'power')
        time_pred = self.predict_metric(freq_grid, 'execution_time')
        
        # Calculate derived metrics
        energy_pred = power_pred * time_pred
        edp_pred = energy_pred * time_pred
        
        # Find optimal frequency
        optimal_idx = np.argmin(edp_pred)
        optimal_freq = freq_grid[optimal_idx]
        
        return EDPResult(
            frequency=optimal_freq,
            execution_time=time_pred[optimal_idx],
            power=power_pred[optimal_idx],
            energy=energy_pred[optimal_idx],
            edp=edp_pred[optimal_idx],
            ed2p=energy_pred[optimal_idx] * (time_pred[optimal_idx] ** 2),
            energy_savings_percent=0.0,  # Would need baseline for calculation
            performance_penalty_percent=0.0,
            edp_improvement_percent=0.0
        )


def quick_energy_analysis(data: pd.DataFrame, 
                         gpu: str, 
                         application: str,
                         performance_constraint: float = 0.2) -> Dict:
    """
    Quick energy analysis for a specific configuration.
    
    Args:
        data: Complete profiling dataset
        gpu: GPU type
        application: Application name
        performance_constraint: Maximum performance penalty (default 20%)
        
    Returns:
        Dictionary with optimization results
    """
    calculator = EnergyCalculator()
    
    try:
        results = calculator.analyze_configuration(
            data, gpu, application, 
            methods=["edp", "energy"],
            performance_constraint=performance_constraint
        )
        
        return {
            'gpu': gpu,
            'application': application,
            'results': results,
            'baseline': calculator.baseline_metrics
        }
    except Exception as e:
        logger.error(f"Quick analysis failed for {gpu}+{application}: {e}")
        return {'error': str(e)}


def validate_edp_calculations(data: pd.DataFrame) -> Dict[str, bool]:
    """
    Validate EDP calculations in a dataset.
    
    Args:
        data: DataFrame with energy and time columns
        
    Returns:
        Dictionary of validation results
    """
    calculator = EnergyCalculator()
    
    validation_results = {
        'energy_calculation': calculator.validate_energy_calculation(data),
        'positive_values': (data[['power', 'execution_time', 'energy']] > 0).all().all(),
        'reasonable_ranges': (
            data['power'].between(10, 1000).all() and
            data['execution_time'].between(0.1, 10000).all()
        )
    }
    
    return validation_results
