"""
Core EDP Calculator Module

This module implements the fundamental Energy-Delay Product (EDP) calculations as described in:
"Energy-efficient DVFS scheduling for mixed-criticality systems"
Future Generation Computer Systems, 2023

Core functionality:
- EDP calculation (Energy * Delay)
- ED²P calculation (Energy * Delay²) 
- Basic optimization metrics
- Utility functions for energy calculations

For optimization analysis, see optimization_analyzer.py
For energy profiling, see energy_profiler.py  
For performance profiling, see performance_profiler.py
For visualization, see the visualization/ package
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
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
        
        NOTE: This method provides basic optimization. For advanced multi-objective 
        optimization, use the OptimizationAnalyzer class.
        
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
    
    def basic_pareto_analysis(self, 
                             df: pd.DataFrame,
                             energy_col: str = 'energy',
                             delay_col: str = 'execution_time') -> pd.DataFrame:
        """
        Generate basic Pareto frontier for energy-delay trade-offs.
        
        NOTE: For advanced Pareto analysis, use the OptimizationAnalyzer class.
        
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
        logger.info(f"Basic Pareto frontier contains {len(pareto_df)} points out of {len(df)} total points")
        
        return pareto_df


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
        
        # Calculate energy improvement with proper error handling
        if baseline_energy > 0 and not np.isnan(baseline_energy) and not np.isnan(edp_energy):
            edp_energy_improvement = (baseline_energy - edp_energy) / baseline_energy * 100
        else:
            edp_energy_improvement = 0.0
            logger.warning(f"Cannot calculate energy improvement: baseline_energy={baseline_energy}, edp_energy={edp_energy}")
        
        # Calculate time improvement with proper error handling
        if baseline_time > 0 and not np.isnan(baseline_time) and not np.isnan(edp_time):
            edp_time_improvement = (baseline_time - edp_time) / baseline_time * 100
        else:
            edp_time_improvement = 0.0
            logger.warning(f"Cannot calculate time improvement: baseline_time={baseline_time}, edp_time={edp_time}")
        
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

# Enhanced EDP calculation with feature selection integration
def calculate_edp_with_features(df: pd.DataFrame, 
                               energy_col: str = 'energy',
                               delay_col: str = 'execution_time',
                               use_feature_selection: bool = True,
                               gpu_type: str = 'V100') -> Dict[str, Any]:
    """
    Calculate EDP metrics with optional feature selection and FGCS integration.
    
    Args:
        df: DataFrame with profiling data
        energy_col: Energy column name
        delay_col: Delay/execution time column name
        use_feature_selection: Whether to apply feature selection
        gpu_type: GPU type for FGCS feature engineering
        
    Returns:
        Dictionary with EDP results and feature analysis
    """
    logger.info("Calculating EDP with enhanced feature analysis")
    
    results = {
        'edp_analysis': {},
        'feature_analysis': {},
        'optimization_results': {}
    }
    
    # Apply feature selection if requested
    if use_feature_selection:
        try:
            from .feature_selection import create_optimized_feature_set
            df_optimized, feature_analysis = create_optimized_feature_set(
                df, gpu_type=gpu_type, target_col='power', max_features=8
            )
            results['feature_analysis'] = feature_analysis
            df_to_use = df_optimized
            logger.info("Applied feature selection and engineering")
        except ImportError:
            logger.warning("Feature selection module not available, using original data")
            df_to_use = df
    else:
        df_to_use = df
    
    # Basic EDP calculations
    calculator = EDPCalculator()
    
    # Ensure we have energy and delay columns
    if energy_col not in df_to_use.columns and 'power' in df_to_use.columns and delay_col in df_to_use.columns:
        df_to_use = df_to_use.copy()
        df_to_use[energy_col] = df_to_use['power'] * df_to_use[delay_col]
        logger.info("Calculated energy from power × time")
    
    if energy_col in df_to_use.columns and delay_col in df_to_use.columns:
        # Calculate EDP metrics
        edp_values = calculator.calculate_edp(df_to_use[energy_col], df_to_use[delay_col])
        ed2p_values = calculator.calculate_ed2p(df_to_use[energy_col], df_to_use[delay_col])
        
        # Add to DataFrame for analysis
        df_to_use = df_to_use.copy()
        df_to_use['edp'] = edp_values
        df_to_use['ed2p'] = ed2p_values
        
        # Find optimal configurations
        optimal_config = calculator.find_optimal_configuration(
            df_to_use, energy_col, delay_col, 'frequency' if 'frequency' in df_to_use.columns else 'sm_clock'
        )
        
        results['edp_analysis'] = {
            'edp_values': edp_values.tolist() if hasattr(edp_values, 'tolist') else [edp_values],
            'ed2p_values': ed2p_values.tolist() if hasattr(ed2p_values, 'tolist') else [ed2p_values],
            'optimal_config': optimal_config,
            'statistics': {
                'mean_edp': np.mean(edp_values),
                'std_edp': np.std(edp_values),
                'min_edp': np.min(edp_values),
                'max_edp': np.max(edp_values),
                'mean_ed2p': np.mean(ed2p_values),
                'std_ed2p': np.std(ed2p_values),
                'min_ed2p': np.min(ed2p_values),
                'max_ed2p': np.max(ed2p_values)
            }
        }
        
        # Integration with FGCS optimizer if available
        try:
            fgcs_results = FGCSEDPOptimizer.analyze_dvfs_optimization(df_to_use, "Enhanced_Analysis")
            results['optimization_results'] = fgcs_results
            logger.info("Integrated FGCS optimization analysis")
        except Exception as e:
            logger.warning(f"FGCS optimization failed: {e}")
    
    else:
        logger.error(f"Required columns not found: {energy_col}, {delay_col}")
        results['error'] = f"Missing required columns: {energy_col}, {delay_col}"
    
    return results


def analyze_feature_importance_for_edp(df: pd.DataFrame,
                                     target_metrics: List[str] = ['energy', 'execution_time'],
                                     gpu_type: str = 'V100') -> Dict[str, Any]:
    """
    Analyze feature importance specifically for EDP optimization.
    
    Args:
        df: DataFrame with profiling data and features
        target_metrics: Target metrics to analyze (['energy', 'execution_time', 'power'])
        gpu_type: GPU type for FGCS-specific analysis
        
    Returns:
        Feature importance analysis results
    """
    logger.info("Analyzing feature importance for EDP optimization")
    
    results = {
        'feature_importance': {},
        'correlation_analysis': {},
        'fgcs_compatibility': {},
        'recommendations': []
    }
    
    try:
        from .feature_selection import FGCSFeatureEngineering, EDPFeatureSelector
        
        # Apply FGCS feature engineering
        feature_engineer = FGCSFeatureEngineering(gpu_type=gpu_type)
        df_features = feature_engineer.extract_fgcs_features(df)
        df_interactions = feature_engineer.create_interaction_features(df_features)
        
        # Validate FGCS compatibility
        validation_results = feature_engineer.validate_features(df_interactions)
        results['fgcs_compatibility'] = validation_results
        
        # Analyze feature importance for each target metric
        for target in target_metrics:
            if target in df_interactions.columns:
                # Statistical feature selection
                selector = EDPFeatureSelector(selection_method='model_based')
                selected_features = selector.select_features_for_edp(
                    df_interactions, target_col=target, max_features=10
                )
                
                results['feature_importance'][target] = {
                    'selected_features': selected_features,
                    'importance_scores': selector.feature_importance_scores,
                    'analysis': selector.get_feature_analysis()
                }
                
                # Correlation analysis
                numeric_cols = df_interactions.select_dtypes(include=[np.number]).columns.tolist()
                if target in numeric_cols:
                    correlations = df_interactions[numeric_cols].corr()[target].sort_values(
                        key=abs, ascending=False
                    )[1:11]  # Top 10 excluding self-correlation
                    results['correlation_analysis'][target] = correlations.to_dict()
        
        # Generate recommendations
        if len(validation_results['missing_fgcs_features']) > 0:
            results['recommendations'].append(
                f"Consider collecting missing FGCS features: {validation_results['missing_fgcs_features']}"
            )
        
        if validation_results['data_quality']['total_samples'] < 100:
            results['recommendations'].append(
                "Consider collecting more samples for robust feature analysis"
            )
        
        # Check if key features are highly correlated with targets
        for target in target_metrics:
            if target in results['correlation_analysis']:
                high_corr_features = [
                    f for f, corr in results['correlation_analysis'][target].items() 
                    if abs(corr) > 0.7
                ]
                if high_corr_features:
                    results['recommendations'].append(
                        f"High correlation features for {target}: {high_corr_features}"
                    )
        
        logger.info("Feature importance analysis completed successfully")
        
    except ImportError as e:
        logger.warning(f"Feature selection module not available: {e}")
        results['error'] = "Feature selection capabilities not available"
    except Exception as e:
        logger.error(f"Feature importance analysis failed: {e}")
        results['error'] = str(e)
    
    return results
