"""
Energy-Delay Product (EDP) Analysis Module

This module implements the EDP calculation and optimization algorithms as described in:
"Energy-efficient DVFS scheduling for mixed-criticality systems"
Future Generation Computer Systems, 2023

Key features:
- EDP calculation (Energy × Delay)
- ED²P calculation (Energy × Delay²) 
- Pareto frontier analysis
- Multi-objective optimization
- Frequency-performance trade-off analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class EDPCalculator:
    """
    Energy-Delay Product calculator for GPU power and performance analysis.
    
    Implements the methodology from the FGCS 2023 paper for finding optimal
    frequency configurations that minimize energy-delay product.
    """
    
    def __init__(self, energy_weight: float = 0.5, delay_weight: float = 0.5):
        """
        Initialize EDP calculator with configurable weights.
        
        Args:
            energy_weight: Weight for energy component (0.0 to 1.0)
            delay_weight: Weight for delay component (0.0 to 1.0)
        """
        if not (0.0 <= energy_weight <= 1.0 and 0.0 <= delay_weight <= 1.0):
            raise ValueError("Weights must be between 0.0 and 1.0")
        
        self.energy_weight = energy_weight
        self.delay_weight = delay_weight
        logger.info(f"EDP Calculator initialized with energy_weight={energy_weight}, delay_weight={delay_weight}")
    
    def calculate_edp(self, energy: Union[float, np.ndarray], delay: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate Energy-Delay Product (EDP).
        
        EDP = Energy × Delay
        
        Args:
            energy: Energy consumption values (Joules)
            delay: Execution time values (seconds)
            
        Returns:
            EDP values
        """
        return energy * delay
    
    def calculate_ed2p(self, energy: Union[float, np.ndarray], delay: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate Energy-Delay² Product (ED²P).
        
        ED²P = Energy × Delay²
        
        Args:
            energy: Energy consumption values (Joules)
            delay: Execution time values (seconds)
            
        Returns:
            ED²P values
        """
        return energy * (delay ** 2)
    
    def calculate_weighted_score(self, energy: Union[float, np.ndarray], delay: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate weighted energy-delay score.
        
        Score = (weight_e × energy) × (weight_d × delay)
        
        Args:
            energy: Energy consumption values
            delay: Execution time values
            
        Returns:
            Weighted score values
        """
        return (self.energy_weight * energy) * (self.delay_weight * delay)
    
    def find_optimal_configuration(self, 
                                   df: pd.DataFrame,
                                   energy_col: str = 'energy',
                                   delay_col: str = 'execution_time',
                                   frequency_col: str = 'frequency',
                                   metric: str = 'edp') -> Dict:
        """
        Find optimal configuration that minimizes the specified metric.
        
        Args:
            df: DataFrame with profiling results
            energy_col: Column name for energy values
            delay_col: Column name for delay/execution time values  
            frequency_col: Column name for frequency values
            metric: Optimization metric ('edp', 'ed2p', 'weighted')
            
        Returns:
            Dictionary with optimal configuration details
        """
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        required_cols = [energy_col, delay_col, frequency_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Calculate the optimization metric
        if metric == 'edp':
            df['score'] = self.calculate_edp(df[energy_col], df[delay_col])
        elif metric == 'ed2p':
            df['score'] = self.calculate_ed2p(df[energy_col], df[delay_col])
        elif metric == 'weighted':
            df['score'] = self.calculate_weighted_score(df[energy_col], df[delay_col])
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'edp', 'ed2p', or 'weighted'")
        
        # Find optimal solution
        optimal_idx = df['score'].idxmin()
        optimal_row = df.loc[optimal_idx]
        
        result = {
            'optimal_frequency': optimal_row[frequency_col],
            'optimal_energy': optimal_row[energy_col],
            'optimal_delay': optimal_row[delay_col],
            'optimal_score': optimal_row['score'],
            'metric_used': metric,
            'improvement_over_max_freq': None,
            'improvement_over_min_freq': None
        }
        
        # Calculate improvements
        max_freq_idx = df[frequency_col].idxmax()
        min_freq_idx = df[frequency_col].idxmin()
        
        if max_freq_idx != optimal_idx:
            max_freq_score = df.loc[max_freq_idx, 'score']
            result['improvement_over_max_freq'] = (max_freq_score - optimal_row['score']) / max_freq_score * 100
        
        if min_freq_idx != optimal_idx:
            min_freq_score = df.loc[min_freq_idx, 'score']
            result['improvement_over_min_freq'] = (min_freq_score - optimal_row['score']) / min_freq_score * 100
        
        logger.info(f"Optimal configuration found: {result['optimal_frequency']} MHz with {metric}={result['optimal_score']:.4f}")
        
        return result
    
    def generate_pareto_frontier(self, 
                                df: pd.DataFrame,
                                energy_col: str = 'energy',
                                delay_col: str = 'execution_time') -> pd.DataFrame:
        """
        Generate Pareto frontier for energy-delay trade-offs.
        
        Args:
            df: DataFrame with profiling results
            energy_col: Column name for energy values
            delay_col: Column name for delay values
            
        Returns:
            DataFrame containing Pareto-optimal points
        """
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        # Sort by energy to process systematically
        sorted_df = df.sort_values(energy_col).reset_index(drop=True)
        pareto_points = []
        
        min_delay = float('inf')
        
        for idx, row in sorted_df.iterrows():
            if row[delay_col] < min_delay:
                min_delay = row[delay_col]
                pareto_points.append(idx)
        
        pareto_df = sorted_df.iloc[pareto_points].copy()
        logger.info(f"Pareto frontier contains {len(pareto_df)} points out of {len(df)} total points")
        
        return pareto_df
    
    def analyze_frequency_sweep(self,
                               df: pd.DataFrame,
                               power_col: str = 'power',
                               time_col: str = 'execution_time',
                               frequency_col: str = 'frequency') -> Dict:
        """
        Analyze complete frequency sweep results and provide comprehensive metrics.
        
        Args:
            df: DataFrame with frequency sweep results
            power_col: Column name for power values (Watts)
            time_col: Column name for execution time (seconds)
            frequency_col: Column name for frequency values (MHz)
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        # Calculate energy if not present
        if 'energy' not in df.columns:
            df = df.copy()
            df['energy'] = df[power_col] * df[time_col]
        
        # Perform all optimizations
        edp_optimal = self.find_optimal_configuration(df, 'energy', time_col, frequency_col, 'edp')
        ed2p_optimal = self.find_optimal_configuration(df, 'energy', time_col, frequency_col, 'ed2p')
        weighted_optimal = self.find_optimal_configuration(df, 'energy', time_col, frequency_col, 'weighted')
        
        # Generate Pareto frontier
        pareto_frontier = self.generate_pareto_frontier(df, 'energy', time_col)
        
        # Calculate additional metrics
        freq_range = df[frequency_col].max() - df[frequency_col].min()
        energy_range = df['energy'].max() - df['energy'].min()
        time_range = df[time_col].max() - df[time_col].min()
        
        analysis_result = {
            'edp_optimization': edp_optimal,
            'ed2p_optimization': ed2p_optimal,
            'weighted_optimization': weighted_optimal,
            'pareto_frontier': pareto_frontier,
            'frequency_range': freq_range,
            'energy_range': energy_range,
            'time_range': time_range,
            'total_configurations': len(df),
            'pareto_points': len(pareto_frontier)
        }
        
        logger.info(f"Frequency sweep analysis completed for {len(df)} configurations")
        
        return analysis_result


def calculate_energy_from_power_time(df: pd.DataFrame, 
                                   power_col: str = 'power', 
                                   time_col: str = 'execution_time',
                                   energy_col: str = 'energy') -> pd.DataFrame:
    """
    Calculate energy consumption from power and time measurements.
    
    Energy (J) = Power (W) × Time (s)
    
    Args:
        df: DataFrame with power and time data
        power_col: Column name for power values (Watts)
        time_col: Column name for execution time (seconds)
        energy_col: Name for the new energy column
        
    Returns:
        DataFrame with added energy column
    """
    df = df.copy()
    df[energy_col] = df[power_col] * df[time_col]
    return df


def normalize_metrics(df: pd.DataFrame, 
                     columns: List[str],
                     method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize metrics for fair comparison in multi-objective optimization.
    
    Args:
        df: DataFrame with metrics to normalize
        columns: List of column names to normalize
        method: Normalization method ('minmax', 'zscore')
        
    Returns:
        DataFrame with normalized columns (suffixed with '_normalized')
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in DataFrame")
        
        if method == 'minmax':
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max != col_min:
                df[f"{col}_normalized"] = (df[col] - col_min) / (col_max - col_min)
            else:
                df[f"{col}_normalized"] = 0.0
        elif method == 'zscore':
            df[f"{col}_normalized"] = (df[col] - df[col].mean()) / df[col].std()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    return df

# ==============================================================================
# FGCS 2023 EDP Optimization Functions
# ==============================================================================

class FGCSEDPOptimizer:
    """
    EDP optimizer implementing the exact methodology from FGCS 2023 paper.
    """
    
    @staticmethod
    def edp_optimal(df: pd.DataFrame, energy_col: str = 'predicted_n_to_r_energy', 
                   time_col: str = 'predicted_n_to_r_run_time',
                   energy_weight: float = 0.5, time_weight: float = 0.5) -> Tuple[int, float, float, float]:
        """
        Find EDP optimal configuration using FGCS 2023 methodology.
        
        Args:
            df: DataFrame with frequency, energy, and time data
            energy_col: Column name for energy values
            time_col: Column name for time values
            energy_weight: Weight for energy component
            time_weight: Weight for time component
            
        Returns:
            Tuple of (optimal_frequency, optimal_time, optimal_power, optimal_energy)
        """
        logger.info("Finding EDP optimal configuration using FGCS methodology")
        
        # Calculate weighted EDP score
        df = df.copy()
        df['score'] = (time_weight * df[time_col]) * (energy_weight * df[energy_col])
        
        # Find minimum score (optimal solution)
        optimal_sol = df.loc[df['score'] == df['score'].min()]
        
        # Extract results
        frequency = int(optimal_sol.iloc[0]['sm_app_clock'])
        time_val = round(optimal_sol.iloc[0]['predicted_n_to_r_run_time'], 2)
        power_val = round(optimal_sol.iloc[0]['predicted_n_to_r_power_usage'], 2)
        energy_val = round(optimal_sol.iloc[0][energy_col], 2)
        
        logger.info(f"EDP Optimal: f={frequency}MHz, t={time_val}s, p={power_val}W, e={energy_val}J")
        
        return frequency, time_val, power_val, energy_val
    
    @staticmethod
    def ed2p_optimal(df: pd.DataFrame, energy_col: str = 'predicted_n_to_r_energy',
                    time_col: str = 'predicted_n_to_r_run_time',
                    energy_weight: float = 0.5, time_weight: float = 0.5) -> Tuple[int, float, float, float]:
        """
        Find ED²P optimal configuration using FGCS 2023 methodology.
        
        Args:
            df: DataFrame with frequency, energy, and time data
            energy_col: Column name for energy values
            time_col: Column name for time values
            energy_weight: Weight for energy component
            time_weight: Weight for time component
            
        Returns:
            Tuple of (optimal_frequency, optimal_time, optimal_power, optimal_energy)
        """
        logger.info("Finding ED²P optimal configuration using FGCS methodology")
        
        # Calculate weighted ED²P score
        df = df.copy()
        df['score'] = (time_weight * (df[time_col] ** 2)) * (energy_weight * df[energy_col])
        
        # Find minimum score (optimal solution)
        optimal_sol = df.loc[df['score'] == df['score'].min()]
        
        # Extract results
        frequency = int(optimal_sol.iloc[0]['sm_app_clock'])
        time_val = round(optimal_sol.iloc[0]['predicted_n_to_r_run_time'], 2)
        power_val = round(optimal_sol.iloc[0]['predicted_n_to_r_power_usage'], 2)
        energy_val = round(optimal_sol.iloc[0][energy_col], 2)
        
        logger.info(f"ED²P Optimal: f={frequency}MHz, t={time_val}s, p={power_val}W, e={energy_val}J")
        
        return frequency, time_val, power_val, energy_val
    
    @staticmethod
    def analyze_dvfs_optimization(df: pd.DataFrame, app_name: str,
                                energy_col: str = 'predicted_n_to_r_energy',
                                time_col: str = 'predicted_n_to_r_run_time') -> Dict:
        """
        Perform complete DVFS optimization analysis as per FGCS 2023.
        
        Args:
            df: DataFrame with prediction results
            app_name: Application name for logging
            energy_col: Column name for energy values
            time_col: Column name for time values
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Performing DVFS optimization analysis for {app_name}")
        
        # Find EDP optimal
        edp_freq, edp_time, edp_power, edp_energy = FGCSEDPOptimizer.edp_optimal(df, energy_col, time_col)
        
        # Find ED²P optimal
        ed2p_freq, ed2p_time, ed2p_power, ed2p_energy = FGCSEDPOptimizer.ed2p_optimal(df, energy_col, time_col)
        
        # Find min energy configuration
        min_energy_idx = df[energy_col].idxmin()
        min_energy_row = df.loc[min_energy_idx]
        min_energy_freq = int(min_energy_row['sm_app_clock'])
        min_energy_val = round(min_energy_row[energy_col], 2)
        
        # Find min time (max performance) configuration
        min_time_idx = df[time_col].idxmin()
        min_time_row = df.loc[min_time_idx]
        min_time_freq = int(min_time_row['sm_app_clock'])
        min_time_val = round(min_time_row[time_col], 2)
        
        # Calculate improvements
        baseline_energy = df[energy_col].max()  # Worst case energy
        baseline_time = df[time_col].max()      # Worst case time
        
        edp_energy_improvement = (baseline_energy - edp_energy) / baseline_energy * 100
        edp_time_improvement = (baseline_time - edp_time) / baseline_time * 100
        
        results = {
            'application': app_name,
            'edp_optimal': {
                'frequency': edp_freq,
                'time': edp_time,
                'power': edp_power,
                'energy': edp_energy,
                'energy_improvement': edp_energy_improvement,
                'time_improvement': edp_time_improvement
            },
            'ed2p_optimal': {
                'frequency': ed2p_freq,
                'time': ed2p_time,
                'power': ed2p_power,
                'energy': ed2p_energy
            },
            'min_energy': {
                'frequency': min_energy_freq,
                'energy': min_energy_val
            },
            'min_time': {
                'frequency': min_time_freq,
                'time': min_time_val
            }
        }
        
        logger.info(f"DVFS optimization complete for {app_name}")
        logger.info(f"EDP optimal: {edp_freq}MHz (E={edp_energy}J, T={edp_time}s)")
        logger.info(f"ED²P optimal: {ed2p_freq}MHz (E={ed2p_energy}J, T={ed2p_time}s)")
        
        return results


class DVFSOptimizationPipeline:
    """
    Complete DVFS optimization pipeline integrating power modeling and EDP analysis.
    """
    
    def __init__(self, power_model, runtime_model=None):
        """
        Initialize optimization pipeline.
        
        Args:
            power_model: Trained power prediction model
            runtime_model: Optional runtime prediction model
        """
        self.power_model = power_model
        self.runtime_model = runtime_model
        self.edp_calculator = EDPCalculator()
        
    def optimize_application(self, fp_activity: float, dram_activity: float,
                           baseline_runtime: float, frequencies: List[int],
                           app_name: str = "Application") -> Dict:
        """
        Complete optimization pipeline for an application.
        
        Args:
            fp_activity: FP operations activity metric
            dram_activity: DRAM activity metric
            baseline_runtime: Baseline execution time
            frequencies: List of frequencies to evaluate
            app_name: Application name for results
            
        Returns:
            Dictionary with optimization results and recommendations
        """
        logger.info(f"Starting optimization pipeline for {app_name}")
        
        # Step 1: Power prediction across frequencies
        if hasattr(self.power_model, 'predict_power'):
            # FGCS model with built-in frequency prediction
            power_df = self.power_model.predict_power(fp_activity, dram_activity, frequencies)
        else:
            # Generic model - create features and predict
            features = pd.DataFrame({
                'fp_activity': [fp_activity] * len(frequencies),
                'dram_activity': [dram_activity] * len(frequencies),
                'sm_clock': frequencies
            })
            power_predictions = self.power_model.predict(features.values)
            power_df = pd.DataFrame({
                'sm_app_clock': frequencies,
                'predicted_power': power_predictions
            })
        
        # Step 2: Runtime prediction (if model available) or use baseline scaling
        if self.runtime_model and hasattr(self.power_model, 'predict_runtime'):
            result_df = self.power_model.predict_runtime(power_df, baseline_runtime, fp_activity)
        else:
            # Simple frequency scaling assumption
            result_df = power_df.copy()
            max_freq = max(frequencies)
            result_df['predicted_n_to_r_run_time'] = baseline_runtime * (max_freq / result_df['sm_app_clock'])
            result_df['predicted_n_to_r_power_usage'] = result_df.get('predicted_power', 
                                                                   result_df.get('predicted_n_to_r_power_usage', 0))
            result_df['predicted_n_to_r_energy'] = (result_df['predicted_n_to_r_run_time'] * 
                                                  result_df['predicted_n_to_r_power_usage'])
        
        # Step 3: EDP optimization
        optimization_results = FGCSEDPOptimizer.analyze_dvfs_optimization(
            result_df, app_name, 'predicted_n_to_r_energy', 'predicted_n_to_r_run_time'
        )
        
        # Step 4: Generate recommendations
        recommendations = self._generate_recommendations(optimization_results, result_df)
        
        final_results = {
            'optimization_results': optimization_results,
            'recommendations': recommendations,
            'frequency_sweep_data': result_df,
            'input_parameters': {
                'fp_activity': fp_activity,
                'dram_activity': dram_activity,
                'baseline_runtime': baseline_runtime,
                'frequencies_evaluated': frequencies
            }
        }
        
        logger.info(f"Optimization pipeline complete for {app_name}")
        return final_results
    
    def _generate_recommendations(self, optimization_results: Dict, sweep_data: pd.DataFrame) -> Dict:
        """Generate practical recommendations based on optimization results."""
        edp_optimal = optimization_results['edp_optimal']
        ed2p_optimal = optimization_results['ed2p_optimal']
        
        recommendations = {
            'primary_recommendation': {
                'frequency': edp_optimal['frequency'],
                'reason': 'EDP optimal - best energy-delay trade-off',
                'expected_energy_savings': f"{edp_optimal.get('energy_improvement', 0):.1f}%",
                'expected_performance_impact': f"{edp_optimal.get('time_improvement', 0):.1f}%"
            },
            'alternative_recommendation': {
                'frequency': ed2p_optimal['frequency'],
                'reason': 'ED²P optimal - prioritizes performance over energy',
                'use_case': 'When response time is critical'
            },
            'energy_conservative': {
                'frequency': optimization_results['min_energy']['frequency'],
                'reason': 'Minimum energy consumption',
                'use_case': 'When energy efficiency is paramount'
            },
            'performance_oriented': {
                'frequency': optimization_results['min_time']['frequency'],
                'reason': 'Maximum performance',
                'use_case': 'When execution time is critical'
            }
        }
        
        return recommendations
